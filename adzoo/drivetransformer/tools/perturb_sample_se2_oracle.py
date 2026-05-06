"""Usage:
python /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/perturb_sample_se2_oracle.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --checkpoint /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --idx 100 \
  --dy 0.0 \
  --dx 1.0 \
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
from mmcv.parallel import DataContainer as DC
from mmcv.utils import load_checkpoint
from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.models import build_model
import types


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
    data = copy.deepcopy(input_dict)
    dataset.pre_pipeline(data)
    return dataset.pipeline(data)

def _to_np(x):
    x = unwrap_data(x)
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
    if isinstance(data, DC):
        if data.cpu_only:
            return data
        return DC(
            _move_to_device(data.data, device),
            stack=data.stack,
            padding_value=data.padding_value,
            cpu_only=data.cpu_only,
            pad_dims=data.pad_dims,
        )
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

def patch_swiglu_ffn_forward_compat(model):
    """
    DriveTransformerPreDecoderLayer calls:
        self.ffns[ffn_index](query, None)

    But DriveTransformer's custom SwiGLULayer.forward only accepts:
        forward(self, x)

    We cannot edit model source files in this debug script, so we monkey-patch
    SwiGLULayer instances to ignore extra positional/keyword args.
    """
    patched = 0

    for module in model.modules():
        # Avoid importing internal class name if possible.
        # Match by class name + key attributes to reduce accidental patching.
        is_swiglu_layer = (
            module.__class__.__name__ == "SwiGLULayer"
            and hasattr(module, "swiglu")
            and hasattr(module, "prenorm")
        )

        if not is_swiglu_layer:
            continue

        if getattr(module, "_se2_oracle_forward_patched", False):
            continue

        old_forward = module.forward

        def new_forward(self, x, *args, _old_forward=old_forward, **kwargs):
            # Ignore the extra identity=None argument passed by PreDecoderLayer.
            return _old_forward(x)

        module.forward = types.MethodType(new_forward, module)
        module._se2_oracle_forward_patched = True
        patched += 1

    print(f"[patch] patched {patched} SwiGLULayer.forward methods to ignore extra args")
    return patched

def sanitize_float32(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.float64:
            return obj.astype(np.float32)
        return obj
    if torch.is_tensor(obj):
        if obj.dtype == torch.float64:
            return obj.float()
        return obj
    if isinstance(obj, DC):
        return DC(
            sanitize_float32(obj.data),
            stack=obj.stack,
            padding_value=obj.padding_value,
            cpu_only=obj.cpu_only,
            pad_dims=obj.pad_dims,
        )
    if isinstance(obj, dict):
        return {k: sanitize_float32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_float32(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_float32(v) for v in obj)
    return obj

@torch.no_grad()
def run_model_forward(model, example, device):
    reset_model_memory(model)
    example = sanitize_float32(example)
    data = collate([example], samples_per_gpu=1)
    data = sanitize_float32(data)
    data = _move_to_device(data, device)
    data = sanitize_float32(data)
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

def _recursive_get_by_key(obj, target_key: str, path="root"):
    """
    Find exact key match in nested dict/list outputs.
    This is intentionally exact-match, because broad matching with 'traj'
    will accidentally pick agent trajs_3d.
    """
    obj = unwrap_data(obj)

    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key], f"{path}.{target_key}"
        for k, v in obj.items():
            found, found_path = _recursive_get_by_key(v, target_key, f"{path}.{k}")
            if found is not None:
                return found, found_path

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found, found_path = _recursive_get_by_key(v, target_key, f"{path}[{i}]")
            if found is not None:
                return found, found_path

    return None, None


def _choose_ego_mode(traj_arr: np.ndarray, score_arr=None, score_mode="argmax"):
    """
    Convert ego trajectory prediction to [T, 2].

    Expected possible shapes:
      [M, T, 2]
      [B, M, T, 2]
      [T, 2]
      [1, T, 2]

    If M modes exist, choose argmax by ego_traj_cls_scores if available.
    """
    arr = np.asarray(traj_arr)
    arr = np.squeeze(arr)

    # [T, 2]
    if arr.ndim == 2 and arr.shape[-1] >= 2:
        return arr[:, :2], {
            "mode_select": "single",
            "mode_idx": None,
            "traj_shape_after_squeeze": list(arr.shape),
        }

    # [1, T, 2] after partial squeeze sometimes becomes [T,2], handled above.
    # [M, T, 2]
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        mode_idx = 0
        debug = {
            "mode_select": "mode0_no_score",
            "mode_idx": 0,
            "traj_shape_after_squeeze": list(arr.shape),
        }

        if score_arr is not None and score_mode == "argmax":
            scores = np.asarray(score_arr)
            scores = np.squeeze(scores)

            # Flatten all score dims and choose global argmax.
            # Usually ego_traj_cls_scores is [M] or [1, M].
            if scores.size > 0:
                flat_scores = scores.reshape(-1)
                mode_idx = int(np.argmax(flat_scores))
                if mode_idx >= arr.shape[0]:
                    print(
                        f"[warn] score argmax={mode_idx} exceeds traj modes={arr.shape[0]}, fallback to 0"
                    )
                    mode_idx = 0
                else:
                    debug = {
                        "mode_select": "argmax_score",
                        "mode_idx": mode_idx,
                        "score_shape": list(scores.shape),
                        "traj_shape_after_squeeze": list(arr.shape),
                        "score_argmax_raw": int(np.argmax(flat_scores)),
                    }

        return arr[mode_idx, :, :2], debug

    # [B, M, T, 2] or higher
    if arr.ndim > 3 and arr.shape[-1] >= 2:
        old_shape = arr.shape

        # If batch dim exists and B=1, remove it first.
        if arr.shape[0] == 1:
            arr2 = arr[0]
        else:
            arr2 = arr.reshape(-1, arr.shape[-2], arr.shape[-1])

        return _choose_ego_mode(arr2, score_arr=score_arr, score_mode=score_mode)

    raise ValueError(f"ego traj has unsupported shape: {arr.shape}")


def extract_ego_pred_traj(outputs, score_mode="argmax") -> Tuple[np.ndarray, dict]:
    """
    Extract ego planner trajectory from DriveTransformer forward_test output.

    IMPORTANT:
    Do NOT use broad key matching like 'traj', because that picks agent trajs_3d.
    The correct ego planner output is ego_fut_preds_fix_time.
    """
    # 1. Exact-match ego planner trajectory.
    traj_obj, traj_path = _recursive_get_by_key(outputs, "ego_fut_preds_fix_time")
    if traj_obj is None:
        # Fallback only if fix_time not present.
        traj_obj, traj_path = _recursive_get_by_key(outputs, "ego_fut_preds")
    if traj_obj is None:
        raise RuntimeError(
            "Cannot find ego planner trajectory. Expected key "
            "'ego_fut_preds_fix_time'. Check outputs_key_tree_*.json."
        )

    # 2. Optional mode score.
    score_obj, score_path = _recursive_get_by_key(outputs, "ego_traj_cls_scores")

    traj_arr = _to_np(traj_obj)
    score_arr = _to_np(score_obj) if score_obj is not None else None

    traj_xy, md = _choose_ego_mode(traj_arr, score_arr=score_arr, score_mode=score_mode)

    debug = {
        "source_key": traj_path,
        "source_shape": list(np.asarray(traj_arr).shape),
        "score_key": score_path,
        "score_shape": None if score_arr is None else list(np.asarray(score_arr).shape),
        **md,
    }
    return traj_xy.astype(np.float32), debug

def save_csv(path, arr):
    np.savetxt(path, arr, delimiter=',', fmt='%.6f')

def unwrap_data(obj):
    """
    Recursively unwrap mmcv DataContainer / single-item wrappers.
    Intended for visualization and audit, not for model forward.
    """
    if isinstance(obj, DC):
        return unwrap_data(obj.data)

    # mmcv collate / pipeline often wraps single sample fields as [x]
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return unwrap_data(obj[0])

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()

    return obj

def as_traj_np(obj, name="traj") -> np.ndarray:
    """
    Handles DataContainer, Tensor, list wrappers, and extra batch/mode dims.
    """
    obj = unwrap_data(obj)

    if torch.is_tensor(obj):
        obj = obj.detach().cpu().numpy()
    arr = np.asarray(obj)
    # remove trivial singleton dimensions, but avoid destroying [T,2]
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        raise ValueError(f"{name} became 1D after squeeze: shape={arr.shape}")
    # If shape is [B, T, 2] and B=1
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    # If shape is [M, T, 2], choose mode 0 for visualization fallback
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        print(f"[warn] {name} has multiple modes or extra dim {arr.shape}; using arr[0]")
        arr = arr[0]
    # If shape is [..., T, 2], flatten leading dims and use first sequence
    if arr.ndim > 3 and arr.shape[-1] >= 2:
        old_shape = arr.shape
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
        print(f"[warn] {name} reshaped from {old_shape} to {arr.shape}")
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"{name} is not trajectory-like after unwrap: shape={arr.shape}, type={type(obj)}")
    return arr[:, :2].astype(np.float32)

def split_traj_lat_fwd(traj: np.ndarray, order="lat_fwd"):
    """
    Split trajectory into raw lateral/forward arrays.

    Empirically for DriveTransformer ego_fut_preds_fix_time on B2D:
      traj[:, 0] behaves like lateral
      traj[:, 1] behaves like forward

    But keep this configurable via --traj-order.
    """
    traj = as_traj_np(traj, "traj_for_split")

    if order == "lat_fwd":
        lat = traj[:, 0]
        fwd = traj[:, 1]
    elif order == "fwd_lat":
        fwd = traj[:, 0]
        lat = traj[:, 1]
    else:
        raise ValueError(order)

    return lat.astype(np.float32), fwd.astype(np.float32)


def traj_to_bev_axes(traj: np.ndarray, order="lat_fwd", lateral_positive="right"):
    """
    Convert trajectory to BEV display axes.

    Display convention:
      horizontal axis = lateral, left-positive
      vertical axis   = forward
    """
    lat, fwd = split_traj_lat_fwd(traj, order=order)

    # Convert raw lateral sign to display left-positive.
    if lateral_positive == "right":
        lat = -lat
    elif lateral_positive == "left":
        pass
    else:
        raise ValueError(lateral_positive)

    return lat, fwd


def traj_to_lidar_xy_for_projection(
    traj: np.ndarray,
    order="lat_fwd",
    lateral_positive="right",
):
    """
    Convert raw planner trajectory into the 2D coordinate expected by lidar2img.

    Important:
    This is for camera projection, not BEV display.

    For the current DriveTransformer/B2D setting, the model trajectory appears
    to use:
      raw dim0 = lateral
      raw dim1 = forward

    The lidar2img in this repo is built from the same dataset-local lidar frame,
    so we keep the raw coordinate order as [dim0, dim1] when order='lat_fwd'.

    If you later confirm the model output is standard MMDet3D lidar [x=front,y=left],
    run with --traj-order fwd_lat.
    """
    lat, fwd = split_traj_lat_fwd(traj, order=order)

    if order == "lat_fwd":
        # Dataset-local xy for this script: x-like dim = lateral, y-like dim = forward.
        x = lat
        y = fwd
    elif order == "fwd_lat":
        # Standard-ish lidar xy: x = forward, y = lateral.
        x = fwd
        y = lat
    else:
        raise ValueError(order)

    return np.stack([x, y], axis=1).astype(np.float32)


def make_traj_xyz1_for_projection(
    traj: np.ndarray,
    z: float,
    order="lat_fwd",
    lateral_positive="right",
):
    """
    Build homogeneous trajectory points for lidar2img projection.
    """
    xy = traj_to_lidar_xy_for_projection(
        traj,
        order=order,
        lateral_positive=lateral_positive,
    )
    n = xy.shape[0]
    pts = np.concatenate(
        [
            xy,
            np.full((n, 1), z, dtype=np.float32),
            np.ones((n, 1), dtype=np.float32),
        ],
        axis=1,
    )
    return pts.astype(np.float64)


def project_xyz1_to_image(pts_xyz1: np.ndarray, lidar2img: np.ndarray, img_shape):
    """
    Project homogeneous lidar-frame points to image.

    Returns:
      uv: [N,2] float pixel coordinates
      valid: bool mask for points with positive depth and inside image
      debug: count dict
    """
    uvw = (lidar2img @ pts_xyz1.T).T
    depth = uvw[:, 2]
    valid_depth = depth > 1e-6

    uv = uvw[:, :2] / np.maximum(depth[:, None], 1e-6)

    h, w = img_shape[:2]
    valid_in = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    valid = valid_depth & valid_in

    debug = {
        'projected': int(valid.sum()),
        'behind': int((~valid_depth).sum()),
        'oob': int((valid_depth & (~valid_in)).sum()),
        'depth_min': float(np.min(depth)) if len(depth) else None,
        'depth_max': float(np.max(depth)) if len(depth) else None,
    }
    return uv, valid, debug


def draw_projected_traj(
    img,
    uv,
    valid,
    color,
    point_radius=3,
    line_thickness=2,
):
    """
    Draw both scattered points and connected polyline.
    """
    uv_int = np.round(uv).astype(np.int32)
    idx = np.where(valid)[0]
    uv_valid = uv_int[idx]

    # Draw polyline first, then points on top.
    if len(uv_valid) >= 2:
        cv2.polylines(
            img,
            [uv_valid.reshape(-1, 1, 2)],
            isClosed=False,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    for p in uv_valid:
        cv2.circle(
            img,
            tuple(p),
            point_radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )

    return img

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
    ap.add_argument(
        '--traj-z',
        type=float,
        default=-1.8,
        help='Trajectory z in lidar frame for camera projection. '
             'Use a negative value to put trajectory on the ground plane.'
    )
    ap.add_argument(
        '--traj-order',
        default='lat_fwd',
        choices=['lat_fwd', 'fwd_lat'],
        help='How to interpret trajectory columns for visualization. '
             'lat_fwd means traj[:,0]=lateral, traj[:,1]=forward.'
    )
    ap.add_argument(
        '--lateral-positive',
        default='right',
        choices=['left', 'right'],
        help='Sign convention of raw trajectory lateral dimension.'
    )    
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    dataset = build_dataset(cfg.data.train)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().to(args.device)
    patch_swiglu_ffn_forward_compat(model)

    inp = dataset.get_data_info(args.idx)
    inp = sanitize_float32(inp)
    inp_raw_for_vis = copy.deepcopy(inp)

    inp_pert = perturb_input_dict_se2_oracle(dataset, inp, args.idx, args.dx, args.dy, args.dtheta)
    inp_pert = sanitize_float32(inp_pert)
    inp_pert_raw_for_vis = copy.deepcopy(inp_pert)
    
    ex = build_example_from_input_dict(dataset, inp)
    ex_pert = build_example_from_input_dict(dataset, inp_pert)

    with torch.no_grad():
        out = run_model_forward(model, ex, args.device)
        out_pert = run_model_forward(model, ex_pert, args.device)

    json.dump(to_key_tree(out), open(osp.join(args.out_dir, 'outputs_key_tree_original.json'), 'w'), indent=2)
    json.dump(to_key_tree(out_pert), open(osp.join(args.out_dir, 'outputs_key_tree_perturbed.json'), 'w'), indent=2)

    pred, dbg = extract_ego_pred_traj(out, score_mode=args.score_mode)
    pred_p, dbg_p = extract_ego_pred_traj(out_pert, score_mode=args.score_mode)
    pred = as_traj_np(pred, "pred_orig")
    pred_p = as_traj_np(pred_p, "pred_pert")

    gt = as_traj_np(inp_raw_for_vis['ego_fut_trajs_fix_time'], "gt_orig")
    gt_p = as_traj_np(inp_pert_raw_for_vis['ego_fut_trajs_fix_time'], "gt_pert")
    save_csv(osp.join(args.out_dir, 'traj_original_pred.csv'), pred)
    save_csv(osp.join(args.out_dir, 'traj_perturbed_pred.csv'), pred_p)
    save_csv(osp.join(args.out_dir, 'traj_original_gt.csv'), gt)
    save_csv(osp.join(args.out_dir, 'traj_perturbed_gt.csv'), gt_p)

    # bev view
    pred_lat, pred_fwd = traj_to_bev_axes(
        pred, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    predp_lat, predp_fwd = traj_to_bev_axes(
        pred_p, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    gt_lat, gt_fwd = traj_to_bev_axes(
        gt, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    gtp_lat, gtp_fwd = traj_to_bev_axes(
        gt_p, order=args.traj_order, lateral_positive=args.lateral_positive
    )

    plt.figure(figsize=(8, 8))

    plt.plot(pred_lat, pred_fwd, '-o', label='pred_orig', linewidth=2, markersize=5)
    plt.plot(predp_lat, predp_fwd, '-o', label='pred_pert', linewidth=2, markersize=5)
    plt.plot(gt_lat, gt_fwd, '--', label='gt_orig', linewidth=2)
    plt.plot(gtp_lat, gtp_fwd, '--', label='gt_pert', linewidth=2)

    plt.scatter([0], [0], c='k', marker='x', s=80, label='ego_origin')

    for i in range(len(pred_lat)):
        plt.text(pred_lat[i], pred_fwd[i], f"o{i}", fontsize=8)
    for i in range(len(predp_lat)):
        plt.text(predp_lat[i], predp_fwd[i], f"p{i}", fontsize=8)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.xlabel("lateral / m (left +)")
    plt.ylabel("forward / m")

    # Set a stable view range for planning.
    # Lateral range is narrow; forward range is long.
    plt.xlim(-10, 10)
    plt.ylim(-5, 35)

    plt.title(
        f'idx={args.idx} dx={args.dx} dy={args.dy} dtheta={args.dtheta}\n'
        f'key={dbg.get("source_key", "na")} '
        f'order={args.traj_order} lat_positive={args.lateral_positive}\n'
        f'image unchanged=True'
    )
    plt.savefig(osp.join(args.out_dir, 'fig_bev_original_vs_perturbed.png'), dpi=180)
    plt.close()

    # camera view
    vis_inp = inp_raw_for_vis
    cam_order = [k for k in vis_inp['sensors'].keys() if 'CAM' in k]
    if args.camera == 'ALL':
        cam_names = cam_order
    else:
        if args.camera not in cam_order:
            raise KeyError(f"Requested camera {args.camera} not found. Available cameras: {cam_order}")
        cam_names = [args.camera]

    lidar2img_all = unwrap_data(vis_inp['lidar2img'])
    lidar2img_all = np.asarray(lidar2img_all, dtype=np.float64)

    # Same delta convention as update_pose_fields:
    #   T_new = T_old @ delta
    # Therefore a point in the perturbed/new lidar frame can be mapped to the
    # original/old lidar frame by:
    #   p_old = delta @ p_new
    delta_lidar = make_se2_delta(args.dx, args.dy, args.dtheta)

    proj_debug = {}

    for cam in cam_names:
        cam_info = vis_inp['sensors'][cam]
        img_path = osp.join(dataset.data_root, cam_info['data_path'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"[warn] failed to read image: {img_path}")
            continue

        cam_idx = cam_order.index(cam)
        lidar2img = np.asarray(lidar2img_all[cam_idx], dtype=np.float64)

        # Original prediction is in original lidar frame.
        pts_orig = make_traj_xyz1_for_projection(
            pred,
            z=args.traj_z,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        )

        # Perturbed prediction is in perturbed lidar frame.
        # Convert it back to original lidar frame before projecting to original image.
        pts_pert_new = make_traj_xyz1_for_projection(
            pred_p,
            z=args.traj_z,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        )
        pts_pert_old = (delta_lidar @ pts_pert_new.T).T

        uv1, m1, d1 = project_xyz1_to_image(pts_orig, lidar2img, img.shape)
        uv2, m2, d2 = project_xyz1_to_image(pts_pert_old, lidar2img, img.shape)

        # Draw line + points.
        img = draw_projected_traj(img, uv1, m1, color=(0, 255, 0))  # green
        img = draw_projected_traj(img, uv2, m2, color=(0, 0, 255))  # red

        cv2.putText(
            img,
            'image is NOT re-rendered',
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f'projection: orig=L, pert=delta@Lprime, z={args.traj_z}',
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            img,
            'green=orig pred, red=pert pred, points+lines',
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            img,
            f'orig projected={d1["projected"]}, pert projected={d2["projected"]}',
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        out_img_path = osp.join(args.out_dir, f'fig_camera_original_vs_perturbed_{cam}.png')
        cv2.imwrite(out_img_path, img)
        print(f"[save] camera visualization: {out_img_path}")

        proj_debug[cam] = {
            'original': d1,
            'perturbed': d2,
            'cam_idx': int(cam_idx),
            'img_path': img_path,
            'traj_z': float(args.traj_z),
            'traj_order': args.traj_order,
            'lateral_positive': args.lateral_positive,
            'perturbed_projection_frame_conversion': 'p_old = delta_lidar @ p_new',
        }

    old_origin_new = (
        np.asarray(inp_pert_raw_for_vis['world2lidar'], dtype=np.float64)
        @ np.array([*inp_raw_for_vis['ego_translation'][:3], 1.0], dtype=np.float64)
    )[:3]
    raw_info_for_audit = dataset.get_data_by_index(args.idx)
    
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
        'command_far_local_before': dataset.get_command_xy_in_local(
            raw_info_for_audit['command_far_xy'],
            inp_raw_for_vis['ego_translation'][0:2],
            inp_raw_for_vis['ego_yaw'],
        ).tolist(),
        'command_far_local_after': inp_pert_raw_for_vis.get('_debug_command_far_local', []),
        'command_near_local_before': dataset.get_command_xy_in_local(
            raw_info_for_audit['command_near_xy'],
            inp_raw_for_vis['ego_translation'][0:2],
            inp_raw_for_vis['ego_yaw'],
        ).tolist(),
        'command_near_local_after': inp_pert_raw_for_vis.get('_debug_command_near_local', []),
        'original_pred_traj_shape': list(pred.shape),
        'perturbed_pred_traj_shape': list(pred_p.shape),
        'trajectory_output_key_original': dbg.get('source_key', ''),
        'trajectory_output_key_perturbed': dbg_p.get('source_key', ''),
        'projection_debug': proj_debug,
        'visualization_coord_convention': {
            'traj_order': args.traj_order,
            'lateral_positive_raw': args.lateral_positive,
            'bev_x_axis': 'lateral_left_positive',
            'bev_y_axis': 'forward',
            'camera_projection_z': args.traj_z,
            'camera_perturbed_projection': 'perturbed traj converted by p_old = delta_lidar @ p_new before original lidar2img',
        },
        'sanity_checks': {
            'L0_pose_inv_err': float(
                np.abs(
                    np.asarray(inp_pert_raw_for_vis['ego_pose'], dtype=np.float64)
                    @ np.asarray(inp_pert_raw_for_vis['ego_pose_inv'], dtype=np.float64)
                    - np.eye(4)
                ).max()
            ),
            'L1_gt_diff_norm': float(np.linalg.norm(gt_p - gt, axis=-1).mean()),
            'L2_pred_mean_display_lat_diff': float((predp_lat - pred_lat).mean()),
            'L2_pred_first_display_lat_diff': float(predp_lat[0] - pred_lat[0]),
            'L2_pred_final_display_lat_diff': float(predp_lat[-1] - pred_lat[-1]),
            'L2_pred_mean_forward_diff': float((predp_fwd - pred_fwd).mean()),
        },
        'warnings': ['TODO: ego_fut_trajs_fix_dist is not recomputed in v1.'],
    }
    with open(osp.join(args.out_dir, 'audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)


if __name__ == '__main__':
    main()
