"""Usage:
python /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/perturb_sample_se2_oracle.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --checkpoint /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --idx 100 \
  --dy 1.0 \
  --dx 0.0 \
  --dtheta 0.0 \
  --out-dir outputs/se2_oracle_debug \
  --device cuda:0
"""

import argparse
import copy
import importlib
import json
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.utils import load_checkpoint
from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model


def make_se2_delta(dx: float, dy: float, dtheta: float) -> np.ndarray:
    c, s = np.cos(dtheta), np.sin(dtheta)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    delta[0, 3] = dx
    delta[1, 3] = dy
    return delta


def invert_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_from_rotmat(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def build_example_from_input_dict(dataset, input_dict):
    dataset.pre_pipeline(input_dict)
    return dataset.pipeline(input_dict)


def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def update_pose_fields(input_dict: dict, dx: float, dy: float, dtheta: float) -> dict:
    out = copy.deepcopy(input_dict)
    T_old = np.asarray(out['ego_pose'], dtype=np.float64)
    delta = make_se2_delta(dx, dy, dtheta)
    T_new = T_old @ delta
    T_new_inv = invert_pose(T_new)

    out['ego_pose'] = T_new.astype(np.float32)
    out['ego_pose_inv'] = T_new_inv.astype(np.float32)
    out['world2lidar'] = T_new_inv.astype(np.float32)
    out['ego_translation'] = T_new[:3, 3].astype(np.float32)
    out['ego_yaw'] = float(yaw_from_rotmat(T_new[:3, :3]))

    if 'can_bus' in out:
        can_bus = np.asarray(out['can_bus']).copy()
        yaw = out['ego_yaw']
        if yaw < 0:
            yaw += 2 * np.pi
        can_bus[:3] = out['ego_translation']
        can_bus[16] = yaw
        can_bus[17] = yaw / np.pi * 180.0
        out['can_bus'] = can_bus.astype(np.float32)

    if 'sensors' in out and 'LIDAR_TOP' in out['sensors']:
        out['sensors']['LIDAR_TOP']['world2lidar'] = T_new_inv.astype(np.float32)

    return out


def recompute_ego_fut_cmd(dataset, input_dict_pert: dict, raw_info: dict) -> None:
    cmd = np.zeros(140, dtype=np.float32)
    yaw = float(input_dict_pert['ego_yaw'])
    ego_xy = np.asarray(input_dict_pert['ego_translation'][:2], dtype=np.float32)
    far_xy_local = dataset.get_command_xy_in_local(raw_info['command_far_xy'], ego_xy, yaw)
    near_xy_local = dataset.get_command_xy_in_local(raw_info['command_near_xy'], ego_xy, yaw)
    cmd[0:6] = dataset.command2hot(raw_info['command_far'])
    cmd[6:70] = dataset.pos2posemb(far_xy_local)
    cmd[70:76] = dataset.command2hot(raw_info['command_near'])
    cmd[76:140] = dataset.pos2posemb(near_xy_local)
    input_dict_pert['ego_fut_cmd'] = cmd
    input_dict_pert['_debug_command_far_local'] = np.asarray(far_xy_local).tolist()
    input_dict_pert['_debug_command_near_local'] = np.asarray(near_xy_local).tolist()

def recompute_ego_future_labels(dataset, input_dict_pert: dict, index: int) -> None:
    cur_w2l = np.asarray(input_dict_pert['world2lidar'], dtype=np.float64)
    sample_rate = dataset.sample_interval_ego_fut
    fut_frames = dataset.future_frames_ego_fix_time
    full_adj_track = np.zeros((fut_frames, 2), dtype=np.float32)
    full_adj_mask = np.zeros((fut_frames,), dtype=np.float32)
    for j, adj_idx in enumerate(range(index + sample_rate, index + (fut_frames + 1) * sample_rate, sample_rate)):
        if not dataset.is_in_same_route(index, adj_idx):
            break
        adj_frame = dataset.get_data_by_index(adj_idx)
        w2l_adj = np.asarray(adj_frame['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float64)
        adj2cur = cur_w2l @ np.linalg.inv(w2l_adj)
        full_adj_track[j] = adj2cur[:2, 3]
        full_adj_mask[j] = 1.0
    full_adj_track[full_adj_mask == 0] = 0
    input_dict_pert['ego_fut_trajs_fix_time'] = full_adj_track
    input_dict_pert['ego_fut_masks_fix_time'] = full_adj_mask
    input_dict_pert['fut_valid_flag_fix_time'] = full_adj_mask[-1]

    # TODO: recompute ego_fut_trajs_fix_dist with perturbed world2lidar if needed.


def recompute_ego_history(dataset, input_dict_pert: dict, index: int) -> None:
    cur_w2l = np.asarray(input_dict_pert['world2lidar'], dtype=np.float64)
    sample_rate = dataset.sample_interval
    past_frames = dataset.past_frames
    full_track = np.zeros((past_frames + 1, 2), dtype=np.float32)
    full_mask = np.zeros((past_frames + 1,), dtype=np.float32)
    adj_list = range(index - past_frames * sample_rate, index + sample_rate, sample_rate)
    for j, adj_idx in enumerate(adj_list):
        if not dataset.is_in_same_route(index, adj_idx):
            break
        adj_frame = dataset.get_data_by_index(adj_idx)
        ego2world_adj = np.eye(4, dtype=np.float64)
        ego2world_adj[:2, 3] = adj_frame['ego_translation'][:2]
        yaw = adj_frame['ego_yaw']
        c, s = np.cos(yaw), np.sin(yaw)
        ego2world_adj[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        lidar2ego = np.asarray(adj_frame['sensors']['LIDAR_TOP']['lidar2ego'], dtype=np.float64)
        lidar2world_adj = ego2world_adj @ lidar2ego
        adj2cur = cur_w2l @ lidar2world_adj
        full_track[j] = adj2cur[:2, 3]
        full_mask[j] = 1.0
    offset = full_track[1:] - full_track[:-1]
    for j in range(past_frames - 2, -1, -1):
        if full_mask[j] == 0:
            offset[j] = offset[j + 1]
    input_dict_pert['ego_his_trajs'] = offset

def perturb_input_dict_se2_oracle(dataset, input_dict: dict, index: int, dx: float, dy: float, dtheta: float):
    raw_info = dataset.get_data_by_index(index)
    pert = update_pose_fields(input_dict, dx, dy, dtheta)
    recompute_ego_fut_cmd(dataset, pert, raw_info)
    recompute_ego_future_labels(dataset, pert, index)
    recompute_ego_history(dataset, pert, index)
    return pert

def _move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(_move_to_device(v, device) for v in data)
    return data

def reset_model_memory(model):
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "pts_bbox_head") and hasattr(m.pts_bbox_head, "reset_memory"):
        m.pts_bbox_head.reset_memory()
    if hasattr(m, "reset_memory"):
        m.reset_memory()

@torch.no_grad()
def run_model_forward(model, example, device):
    reset_model_memory(model)

    data = collate([example], samples_per_gpu=1)
    data = _move_to_device(data, device)
    return model(data, return_loss=False, rescale=True)

def to_key_tree(obj: Any):
    if isinstance(obj, dict):
        return {k: to_key_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_key_tree(v) for v in obj[:8]]
    if torch.is_tensor(obj):
        return {'type': 'tensor', 'shape': list(obj.shape)}
    if isinstance(obj, np.ndarray):
        return {'type': 'ndarray', 'shape': list(obj.shape)}
    return str(type(obj))


def _recursive_find(obj, path='root'):
    found = []
    keys = ['ego_fut_preds_fix_time', 'ego_fut_preds', 'ego_traj', 'planning', 'traj']
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}.{k}'
            if any(kk in k.lower() for kk in keys):
                found.append((p, v))
            found.extend(_recursive_find(v, p))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(_recursive_find(v, f'{path}[{i}]'))
    return found


def _select_mode(arr: np.ndarray):
    if arr.ndim == 3:
        return arr[0], {'mode_select': 'mode0_no_score'}
    return arr, {'mode_select': 'single'}


def extract_ego_pred_traj(outputs) -> Tuple[np.ndarray, dict]:
    candidates = _recursive_find(outputs)
    debug = {'candidates': [p for p, _ in candidates]}
    for p, v in candidates:
        arr = _to_np(v)
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            xy = arr[..., :2]
            if xy.ndim > 3:
                xy = xy.reshape(-1, xy.shape[-2], 2)
            xy2, md = _select_mode(xy)
            debug.update({'source_key': p, 'source_shape': list(arr.shape), **md})
            return xy2.astype(np.float32), debug
    raise RuntimeError('Cannot find ego trajectory in outputs. See key tree json for debug.')


def save_csv(path, arr):
    np.savetxt(path, arr, delimiter=',', fmt='%.6f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--idx', type=int, required=True)
    ap.add_argument('--dx', type=float, default=0.0)
    ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dtheta', type=float, default=0.0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--camera', default='CAM_FRONT')
    ap.add_argument('--score-mode', default='argmax')
    ap.add_argument('--traj-z', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().to(args.device)

    inp = dataset.get_data_info(args.idx)
    inp_pert = perturb_input_dict_se2_oracle(dataset, inp, args.idx, args.dx, args.dy, args.dtheta)
    ex = build_example_from_input_dict(dataset, inp)
    ex_pert = build_example_from_input_dict(dataset, inp_pert)

    with torch.no_grad():
        out = run_model_forward(model, ex, args.device)
        out_pert = run_model_forward(model, ex_pert, args.device)

    json.dump(to_key_tree(out), open(osp.join(args.out_dir, 'outputs_key_tree_original.json'), 'w'), indent=2)
    json.dump(to_key_tree(out_pert), open(osp.join(args.out_dir, 'outputs_key_tree_perturbed.json'), 'w'), indent=2)

    pred, dbg = extract_ego_pred_traj(out)
    pred_p, dbg_p = extract_ego_pred_traj(out_pert)

    gt = np.asarray(inp['ego_fut_trajs_fix_time'])
    gt_p = np.asarray(inp_pert['ego_fut_trajs_fix_time'])
    save_csv(osp.join(args.out_dir, 'traj_original_pred.csv'), pred)
    save_csv(osp.join(args.out_dir, 'traj_perturbed_pred.csv'), pred_p)
    save_csv(osp.join(args.out_dir, 'traj_original_gt.csv'), gt)
    save_csv(osp.join(args.out_dir, 'traj_perturbed_gt.csv'), gt_p)

    plt.figure(figsize=(8, 8))
    plt.plot(pred[:, 0], pred[:, 1], '-o', label='pred_orig')
    plt.plot(pred_p[:, 0], pred_p[:, 1], '-o', label='pred_pert')
    plt.plot(gt[:, 0], gt[:, 1], '--', label='gt_orig')
    plt.plot(gt_p[:, 0], gt_p[:, 1], '--', label='gt_pert')
    plt.scatter([0], [0], c='k', marker='x', label='ego_origin')
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.title(f'idx={args.idx} dx={args.dx} dy={args.dy} dtheta={args.dtheta}\nkey={dbg.get("source_key", "na")}\nimage unchanged=True')
    plt.savefig(osp.join(args.out_dir, 'fig_bev_original_vs_perturbed.png'), dpi=180)
    plt.close()

    # camera view
    cam_names = [args.camera] if args.camera != 'ALL' else [k for k in inp['sensors'].keys() if 'CAM' in k]
    proj_debug = {}
    for cam in cam_names:
        cam_info = inp['sensors'][cam]
        img_path = osp.join(dataset.data_root, cam_info['data_path'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        lidar2img = None
        for i, c in enumerate([k for k in inp['sensors'].keys() if 'CAM' in k]):
            if c == cam:
                lidar2img = np.asarray(inp['lidar2img'][i], dtype=np.float64)
        if lidar2img is None:
            continue

        def proj(tr):
            n = tr.shape[0]
            pts = np.concatenate([tr[:, :2], np.full((n, 1), args.traj_z), np.ones((n, 1))], axis=1)
            uvw = (lidar2img @ pts.T).T
            z = uvw[:, 2]
            valid_z = z > 1e-6
            uv = uvw[:, :2] / np.maximum(z[:, None], 1e-6)
            h, w = img.shape[:2]
            valid_in = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            valid = valid_z & valid_in
            return uv, valid, {'projected': int(valid.sum()), 'behind': int((~valid_z).sum()), 'oob': int((valid_z & (~valid_in)).sum())}

        uv1, m1, d1 = proj(pred)
        uv2, m2, d2 = proj(pred_p)
        for i in np.where(m1)[0]:
            cv2.circle(img, tuple(np.int32(uv1[i])), 3, (0, 255, 0), -1)
        for i in np.where(m2)[0]:
            cv2.circle(img, tuple(np.int32(uv2[i])), 3, (0, 0, 255), -1)
        cv2.putText(img, 'image is NOT re-rendered', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, 'projection uses original lidar2img', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imwrite(osp.join(args.out_dir, f'fig_camera_original_vs_perturbed_{cam}.png'), img)
        proj_debug[cam] = {'original': d1, 'perturbed': d2}

    old_origin_new = (np.asarray(inp_pert['world2lidar']) @ np.array([*inp['ego_translation'][:3], 1.0]))[:3]
    audit = {
        'idx': args.idx,
        'dx': args.dx,
        'dy': args.dy,
        'dtheta': args.dtheta,
        'experiment_type': 'planner_state_oracle_no_rerender',
        'image_rerendered': False,
        'updated_fields': ['ego_pose', 'ego_pose_inv', 'world2lidar', 'ego_translation', 'ego_yaw', 'can_bus', 'sensors.LIDAR_TOP.world2lidar', 'ego_fut_cmd', 'ego_fut_trajs_fix_time', 'ego_his_trajs'],
        'unchanged_fields': ['img', 'img_filename', 'cam_intrinsic', 'lidar2img'],
        'old_ego_origin_in_new_lidar': old_origin_new.tolist(),
        'command_far_local_before': dataset.get_command_xy_in_local(dataset.get_data_by_index(args.idx)['command_far_xy'], inp['ego_translation'][0:2], inp['ego_yaw']).tolist(),
        'command_far_local_after': inp_pert.get('_debug_command_far_local', []),
        'command_near_local_before': dataset.get_command_xy_in_local(dataset.get_data_by_index(args.idx)['command_near_xy'], inp['ego_translation'][0:2], inp['ego_yaw']).tolist(),
        'command_near_local_after': inp_pert.get('_debug_command_near_local', []),
        'original_pred_traj_shape': list(pred.shape),
        'perturbed_pred_traj_shape': list(pred_p.shape),
        'trajectory_output_key_original': dbg.get('source_key', ''),
        'trajectory_output_key_perturbed': dbg_p.get('source_key', ''),
        'projection_debug': proj_debug,
        'sanity_checks': {
            'L0_pose_inv_err': float(np.abs(np.asarray(inp_pert['ego_pose']) @ np.asarray(inp_pert['ego_pose_inv']) - np.eye(4)).max()),
            'L1_gt_diff_norm': float(np.linalg.norm(gt_p - gt, axis=-1).mean()),
            'L2_pred_mean_lat_diff': float((pred_p[:, 1] - pred[:, 1]).mean()),
            'L2_pred_first_lat_diff': float(pred_p[0, 1] - pred[0, 1]),
            'L2_pred_final_lat_diff': float(pred_p[-1, 1] - pred[-1, 1]),
        },
        'warnings': ['TODO: ego_fut_trajs_fix_dist is not recomputed in v1.'],
    }
    with open(osp.join(args.out_dir, 'audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)


if __name__ == '__main__':
    main()
