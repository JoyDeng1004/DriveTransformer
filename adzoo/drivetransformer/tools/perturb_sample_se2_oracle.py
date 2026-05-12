"""Usage:
python /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/perturb_sample_se2_oracle.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --checkpoint /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --idx 100 \
  --dy 0.0 \
  --dx -1.0 \
  --dtheta 0.0 \
  --perturb-channels label \
  --out-dir outputs/se2_oracle_ablation/neg_round3 \
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

from adzoo.drivetransformer.utils.recovery_augment import RecoveryRefBuilder

PERTURB_CHANNEL_CHOICES = (
    'none',
    'history',
    'cmd',
    'pose_frame',
    'can_bus',
    'label',
    'full',
)

ABLATION_GROUPS = ('pose_frame', 'can_bus', 'ego_fut_cmd', 'future_label', 'ego_his_trajs')


def make_se2_delta(dx: float, dy: float, dtheta: float) -> np.ndarray:
    """
    Return old_lidar_from_new_lidar.

    In this Bench2Drive preprocessing, lidar xy is x=right, y=forward.
    dx/dy are the new lidar origin expressed in the original lidar frame.
    """
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


def compute_pose_frame_patch(input_dict: dict, dx: float, dy: float, dtheta: float) -> dict:
    T_old = np.asarray(input_dict['ego_pose'], dtype=np.float64)  # lidar_old -> world
    delta = make_se2_delta(dx, dy, dtheta)
    T_new = T_old @ delta  # lidar_new -> world
    T_new_inv = invert_pose(T_new)  # world -> lidar_new

    lidar2ego = np.eye(4, dtype=np.float64)
    if 'sensors' in input_dict and 'LIDAR_TOP' in input_dict['sensors']:
        lidar2ego = np.asarray(input_dict['sensors']['LIDAR_TOP'].get('lidar2ego', lidar2ego), dtype=np.float64)
    ego2world_new = T_new @ invert_pose(lidar2ego)
    ego_translation_new = ego2world_new[:3, 3]
    ego_yaw_new = yaw_from_rotmat(ego2world_new[:3, :3])

    return {
        'delta_lidar': delta.astype(np.float32),
        'ego_pose': T_new.astype(np.float32),
        'ego_pose_inv': T_new_inv.astype(np.float32),
        'world2lidar': T_new_inv.astype(np.float32),
        'ego_translation': ego_translation_new.astype(np.float32),
        'ego_yaw': float(ego_yaw_new),
    }


def patch_pose_frame(input_dict_pert: dict, pose_patch: dict) -> List[str]:
    input_dict_pert['_se2_delta_lidar'] = pose_patch['delta_lidar']
    input_dict_pert['ego_pose'] = pose_patch['ego_pose']
    input_dict_pert['ego_pose_inv'] = pose_patch['ego_pose_inv']
    input_dict_pert['world2lidar'] = pose_patch['world2lidar']
    input_dict_pert['ego_translation'] = pose_patch['ego_translation']
    input_dict_pert['ego_yaw'] = pose_patch['ego_yaw']

    patched = ['ego_pose', 'ego_pose_inv', 'world2lidar', 'ego_translation', 'ego_yaw']
    if 'sensors' in input_dict_pert and 'LIDAR_TOP' in input_dict_pert['sensors']:
        input_dict_pert['sensors']['LIDAR_TOP']['world2lidar'] = pose_patch['world2lidar']
        patched.append("sensors['LIDAR_TOP']['world2lidar']")
    return patched


def patch_can_bus(input_dict_pert: dict, pose_patch: dict) -> List[str]:
    if 'can_bus' not in input_dict_pert:
        return []

    can_bus = np.asarray(input_dict_pert['can_bus']).copy()
    yaw = float(pose_patch['ego_yaw'])
    can_bus[3:7] = np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=can_bus.dtype,
    )
    if yaw < 0:
        yaw += 2 * np.pi
    can_bus[:3] = np.asarray(pose_patch['ego_translation'])[:3]
    can_bus[16] = yaw
    can_bus[17] = yaw / np.pi * 180.0
    input_dict_pert['can_bus'] = can_bus.astype(np.float32)
    return ['can_bus']


def update_pose_fields(input_dict: dict, dx: float, dy: float, dtheta: float) -> dict:
    out = copy.deepcopy(input_dict)
    pose_patch = compute_pose_frame_patch(input_dict, dx, dy, dtheta)
    patch_pose_frame(out, pose_patch)

    return out


def recompute_ego_fut_cmd(dataset, input_dict_pert: dict, raw_info: dict, ego_xy=None, yaw=None) -> None:
    cmd = np.zeros(140, dtype=np.float32)
    if yaw is None:
        yaw = float(input_dict_pert['ego_yaw'])
    if ego_xy is None:
        ego_xy = np.asarray(input_dict_pert['ego_translation'][:2], dtype=np.float32)
    else:
        ego_xy = np.asarray(ego_xy, dtype=np.float32)
    far_xy_local = dataset.get_command_xy_in_local(raw_info['command_far_xy'], ego_xy, yaw)
    near_xy_local = dataset.get_command_xy_in_local(raw_info['command_near_xy'], ego_xy, yaw)
    cmd[0:6] = dataset.command2hot(raw_info['command_far'])
    cmd[6:70] = dataset.pos2posemb(far_xy_local)
    cmd[70:76] = dataset.command2hot(raw_info['command_near'])
    cmd[76:140] = dataset.pos2posemb(near_xy_local)
    input_dict_pert['ego_fut_cmd'] = cmd
    input_dict_pert['_debug_command_far_local'] = np.asarray(far_xy_local).tolist()
    input_dict_pert['_debug_command_near_local'] = np.asarray(near_xy_local).tolist()

def patch_route_command(dataset, input_dict_pert: dict, raw_info: dict, pose_patch: dict) -> List[str]:
    recompute_ego_fut_cmd(
        dataset,
        input_dict_pert,
        raw_info,
        ego_xy=np.asarray(pose_patch['ego_translation'])[:2],
        yaw=float(pose_patch['ego_yaw']),
    )
    return ['ego_fut_cmd']


def patch_recovery_ref_command(
    input_dict_old: dict,
    input_dict_pert: dict,
    pose_patch: dict,
    near_idx: int,
    far_idx: int,
    dx: float,
    dy: float,
    dtheta: float,
) -> List[str]:
    builder = RecoveryRefBuilder(near_idx=near_idx, far_idx=far_idx)
    recovery = builder.build(
        future_old_xy=np.asarray(input_dict_old['ego_fut_trajs_fix_time'], dtype=np.float64),
        T_old_lidar_to_world=np.asarray(input_dict_old['ego_pose'], dtype=np.float64),
        T_new_lidar_to_world=np.asarray(pose_patch['ego_pose'], dtype=np.float64),
    )
    input_dict_pert['ego_fut_cmd'] = recovery['ego_fut_cmd_new'].astype(np.float32)

    debug_info = dict(recovery['debug_info'])
    debug_info.update(
        {
            'cmd_source': 'recovery_ref',
            'dx': float(dx),
            'dy': float(dy),
            'dtheta': float(dtheta),
            'expected_sign_check': {
                'near_ref_new_x': float(recovery['near_ref_new'][0]),
                'expected_approximately_minus_dx': float(-dx),
                'near_ref_new_x_plus_dx': float(recovery['near_ref_new'][0] + dx),
            },
        }
    )
    input_dict_pert['_debug_recovery_ref'] = debug_info
    input_dict_pert['_debug_recovery_near_ref_new'] = recovery['near_ref_new'].astype(float).tolist()
    input_dict_pert['_debug_recovery_far_ref_new'] = recovery['far_ref_new'].astype(float).tolist()

    print(
        f"[recovery_ref] dx={dx:.6f}, dy={dy:.6f}, dtheta={dtheta:.6f}, "
        f"near_ref_new={input_dict_pert['_debug_recovery_near_ref_new']}, "
        f"far_ref_new={input_dict_pert['_debug_recovery_far_ref_new']}"
    )
    print(
        "[recovery_ref] identity mean/max diff: "
        f"{debug_info.get('identity_mean_point_error', 0.0):.9f}/"
        f"{debug_info.get('identity_max_point_error', 0.0):.9f}"
    )
    print(
        "[recovery_ref] expected sign check: "
        f"near_ref_new.x={float(recovery['near_ref_new'][0]):.6f}, "
        f"-dx={-dx:.6f}, x+dx={float(recovery['near_ref_new'][0] + dx):.6f}"
    )

    return ['ego_fut_cmd']


def recompute_ego_future_labels(dataset, input_dict_pert: dict, index: int, delta_lidar=None) -> None:
    raw_cur_frame = dataset.get_data_by_index(index)
    raw_cur_w2l = np.asarray(raw_cur_frame['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float64)
    if delta_lidar is None:
        delta_lidar = input_dict_pert['_se2_delta_lidar']
    delta_lidar = np.asarray(delta_lidar, dtype=np.float64)

    # Match B2D_DriveTransformer_Dataset.get_ego_future_trajs exactly:
    # future labels are future LIDAR_TOP origins expressed in the current
    # LIDAR_TOP frame. The raw dataset uses sensors.LIDAR_TOP.world2lidar for
    # this label, which is not always numerically identical to ego_pose_inv.
    # For a planner-state perturbation, current_new_from_world =
    # new_from_old @ current_old_from_world.
    cur_w2l = invert_pose(delta_lidar) @ raw_cur_w2l
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

def patch_future_label(dataset, input_dict_pert: dict, index: int, pose_patch: dict) -> List[str]:
    input_dict_pert['_se2_delta_lidar'] = pose_patch['delta_lidar']
    recompute_ego_future_labels(dataset, input_dict_pert, index, delta_lidar=pose_patch['delta_lidar'])
    return ['ego_fut_trajs_fix_time', 'ego_fut_masks_fix_time', 'fut_valid_flag_fix_time']


def recompute_ego_history(dataset, input_dict_pert: dict, index: int, cur_w2l=None) -> None:
    if cur_w2l is None:
        cur_w2l = input_dict_pert['world2lidar']
    cur_w2l = np.asarray(cur_w2l, dtype=np.float64)
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


def patch_history(dataset, input_dict_pert: dict, index: int, pose_patch: dict) -> List[str]:
    recompute_ego_history(dataset, input_dict_pert, index, cur_w2l=pose_patch['world2lidar'])
    return ['ego_his_trajs']


def _groups_for_perturb_channel(perturb_channels: str) -> List[str]:
    if perturb_channels == 'none':
        return []
    if perturb_channels == 'history':
        return ['ego_his_trajs']
    if perturb_channels == 'cmd':
        return ['ego_fut_cmd']
    if perturb_channels == 'pose_frame':
        return ['pose_frame']
    if perturb_channels == 'can_bus':
        return ['can_bus']
    if perturb_channels == 'label':
        return ['future_label']
    if perturb_channels == 'full':
        return ['pose_frame', 'can_bus', 'ego_fut_cmd', 'future_label', 'ego_his_trajs']
    raise ValueError(f"Unsupported perturb_channels={perturb_channels}")


def apply_perturbation_by_channels(
    dataset,
    input_dict: dict,
    index: int,
    dx: float,
    dy: float,
    dtheta: float,
    perturb_channels: str,
    cmd_source: str = 'route_command',
    recovery_near_idx: int = 1,
    recovery_far_idx: int = -1,
):
    raw_info = dataset.get_data_by_index(index)
    pert = copy.deepcopy(input_dict)
    pose_patch = compute_pose_frame_patch(input_dict, dx, dy, dtheta)
    groups = _groups_for_perturb_channel(perturb_channels)
    patched_fields = []

    if 'pose_frame' in groups:
        patched_fields.extend(patch_pose_frame(pert, pose_patch))
    if 'can_bus' in groups:
        patched_fields.extend(patch_can_bus(pert, pose_patch))
    if 'ego_fut_cmd' in groups:
        if cmd_source == 'route_command':
            patched_fields.extend(patch_route_command(dataset, pert, raw_info, pose_patch))
        elif cmd_source == 'recovery_ref':
            patched_fields.extend(
                patch_recovery_ref_command(
                    input_dict,
                    pert,
                    pose_patch,
                    near_idx=recovery_near_idx,
                    far_idx=recovery_far_idx,
                    dx=dx,
                    dy=dy,
                    dtheta=dtheta,
                )
            )
        else:
            raise ValueError(f"Unsupported cmd_source={cmd_source}")
    if 'future_label' in groups:
        patched_fields.extend(patch_future_label(dataset, pert, index, pose_patch))
    if 'ego_his_trajs' in groups:
        patched_fields.extend(patch_history(dataset, pert, index, pose_patch))

    not_patched = [g for g in ABLATION_GROUPS if g not in groups]
    print(f"[ablation] perturb_channels = {perturb_channels}")
    print(f"[ablation] patched: {', '.join(patched_fields) if patched_fields else 'none'}")
    print(f"[ablation] not patched: {', '.join(not_patched) if not_patched else 'none'}")

    info = {
        'perturb_channels': perturb_channels,
        'patched_groups': groups,
        'not_patched_groups': not_patched,
        'patched_fields': patched_fields,
        'pose_frame_patched': 'pose_frame' in groups,
        'future_label_patched': 'future_label' in groups,
        'cmd_source': cmd_source,
        'recovery_ref_debug': pert.get('_debug_recovery_ref', {}),
    }
    return pert, info


def perturb_input_dict_se2_oracle(dataset, input_dict: dict, index: int, dx: float, dy: float, dtheta: float):
    raw_info = dataset.get_data_by_index(index)
    pert = update_pose_fields(input_dict, dx, dy, dtheta)
    pose_patch = {
        'delta_lidar': pert['_se2_delta_lidar'],
        'world2lidar': pert['world2lidar'],
        'ego_translation': pert['ego_translation'],
        'ego_yaw': pert['ego_yaw'],
    }
    patch_can_bus(pert, pose_patch)
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
    # DriveTransformer.forward_test uses prev_scene_token to decide whether
    # prev_exists=0 or 1. Resetting only pts_bbox_head memory is not enough:
    # the second forward of the same sample would otherwise enter the temporal
    # path as if it had a previous frame, producing false ablation differences
    # even for --perturb-channels none/label.
    if hasattr(m, "prev_scene_token"):
        m.prev_scene_token = None
    if hasattr(m, "test_flag"):
        m.test_flag = False
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

def fmt_num_for_filename(x) -> str:
    return str(x).replace('/', '_')

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

    DriveTransformer/B2D fix-time ego labels are adj2cur_lidar[:2, 3].
    In this repo's preprocessed lidar frame, x is right and y is forward,
    so the default order is traj[:,0]=right/lateral, traj[:,1]=forward.
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

    Input trajectory is in a lidar-local frame.
    Display convention:
      horizontal axis = lateral, vehicle-right-positive
      vertical axis   = forward

    This keeps the BEV horizontal sign aligned with the preprocessed B2D lidar
    x axis. Camera image left/right is still perspective-dependent and should
    not be treated as a metric BEV axis.
    """
    lat, fwd = split_traj_lat_fwd(traj, order=order)

    # Convert raw lateral sign to display vehicle-right-positive.
    if lateral_positive == "right":
        pass
    elif lateral_positive == "left":
        lat = -lat
    else:
        raise ValueError(lateral_positive)

    return lat, fwd


def traj_pair_diff_stats(
    ref: np.ndarray,
    test: np.ndarray,
    order="lat_fwd",
    lateral_positive="right",
) -> dict:
    """
    Compare trajectories in lidar xy, where x=vehicle right and y=forward.
    """
    ref_xy = traj_to_lidar_xy_for_projection(
        ref,
        order=order,
        lateral_positive=lateral_positive,
    )
    test_xy = traj_to_lidar_xy_for_projection(
        test,
        order=order,
        lateral_positive=lateral_positive,
    )
    n = min(len(ref_xy), len(test_xy))
    ref_xy = ref_xy[:n]
    test_xy = test_xy[:n]
    diff = test_xy - ref_xy
    abs_diff = np.abs(diff)
    point_error = np.linalg.norm(diff, axis=1) if len(diff) else np.zeros((0,), dtype=np.float32)
    mid_idx = len(diff) // 2 if len(diff) else None
    return {
        'num_points_compared': int(n),
        'max_abs_diff_x_right_m': float(abs_diff[:, 0].max()) if len(abs_diff) else 0.0,
        'max_abs_diff_y_forward_m': float(abs_diff[:, 1].max()) if len(abs_diff) else 0.0,
        'mean_abs_diff_x_right_m': float(abs_diff[:, 0].mean()) if len(abs_diff) else 0.0,
        'mean_abs_diff_y_forward_m': float(abs_diff[:, 1].mean()) if len(abs_diff) else 0.0,
        'first_diff_xy_right_forward_m': diff[0].astype(float).tolist() if len(diff) else [],
        'mid_diff_xy_right_forward_m': diff[mid_idx].astype(float).tolist() if len(diff) else [],
        'last_diff_xy_right_forward_m': diff[-1].astype(float).tolist() if len(diff) else [],
        'final_point_error_m': float(point_error[-1]) if len(point_error) else 0.0,
        'mean_point_error_m': float(point_error.mean()) if len(point_error) else 0.0,
        'max_point_error_m': float(point_error.max()) if len(point_error) else 0.0,
    }


def traj_to_lidar_xy_for_projection(
    traj: np.ndarray,
    order="lat_fwd",
    lateral_positive="right",
):
    """
    Convert planner trajectory columns to this repo's lidar xy.

    Bench2Drive preprocessing used by DriveTransformer stores lidar xy as:
      x = right, y = forward.
    lidar2img is built for that same lidar frame, so BEV-only sign flips must
    not leak into camera projection.
    """
    lat, fwd = split_traj_lat_fwd(traj, order=order)

    if lateral_positive == "right":
        x = lat
    elif lateral_positive == "left":
        x = -lat
    else:
        raise ValueError(lateral_positive)
    y = fwd

    return np.stack([x, y], axis=1).astype(np.float32)


def transform_traj_lidar_frame(
    traj: np.ndarray,
    dst_from_src: np.ndarray,
    order="lat_fwd",
    lateral_positive="right",
):
    """
    Transform a 2D trajectory from one lidar frame to another.

    The returned trajectory uses the same column convention as the input.
    This is used to compare original and perturbed predictions in one BEV
    frame without reinterpreting the model output.
    """
    xy_src = traj_to_lidar_xy_for_projection(
        traj,
        order=order,
        lateral_positive=lateral_positive,
    )
    n = xy_src.shape[0]
    pts_src = np.concatenate(
        [
            xy_src,
            np.zeros((n, 1), dtype=np.float32),
            np.ones((n, 1), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float64)
    xy_dst = (dst_from_src @ pts_src.T).T[:, :2].astype(np.float32)
    if lateral_positive == "right":
        lat_dst = xy_dst[:, 0]
    elif lateral_positive == "left":
        lat_dst = -xy_dst[:, 0]
    else:
        raise ValueError(lateral_positive)
    fwd_dst = xy_dst[:, 1]
    if order == "lat_fwd":
        return np.stack([lat_dst, fwd_dst], axis=1).astype(np.float32)
    if order == "fwd_lat":
        return np.stack([fwd_dst, lat_dst], axis=1).astype(np.float32)
    raise ValueError(order)


def make_traj_xyz1_for_projection(
    traj: np.ndarray,
    z: float,
    order="lat_fwd",
    lateral_positive="right",
):
    """
    Build homogeneous points in the lidar frame consumed by lidar2img.

    z is lidar z. The default -1.8 m places the polyline near the road plane
    below the roof-mounted lidar; tune --traj-z if the sensor height differs.
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
    ap.add_argument('--dx', type=float, default=0.0,
                    help='New lidar origin x in original lidar frame; B2D lidar x is right.')
    ap.add_argument('--dy', type=float, default=0.0,
                    help='New lidar origin y in original lidar frame; B2D lidar y is forward.')
    ap.add_argument('--dtheta', type=float, default=0.0,
                    help='Yaw of new lidar frame relative to original lidar frame, radians.')
    ap.add_argument(
        '--perturb-channels',
        default='full',
        choices=PERTURB_CHANNEL_CHOICES,
        help='Planner-state ablation channel to perturb. '
             'none keeps planner-state unchanged; full preserves the original all-field behavior.'
    )
    ap.add_argument(
        '--cmd-source',
        default='route_command',
        choices=['route_command', 'recovery_ref'],
        help='Source used when the ego_fut_cmd perturb channel is patched.'
    )
    ap.add_argument(
        '--recovery-near-idx',
        type=int,
        default=1,
        help='Future trajectory index used as near recovery reference.'
    )
    ap.add_argument(
        '--recovery-far-idx',
        type=int,
        default=-1,
        help='Future trajectory index used as far recovery reference.'
    )
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--camera', default='CAM_FRONT')
    ap.add_argument('--score-mode', default='mode0')
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
             'Default B2D convention: lat_fwd means traj[:,0]=right/lateral, traj[:,1]=forward.'
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

    inp_pert, ablation_info = apply_perturbation_by_channels(
        dataset,
        inp,
        args.idx,
        args.dx,
        args.dy,
        args.dtheta,
        args.perturb_channels,
        cmd_source=args.cmd_source,
        recovery_near_idx=args.recovery_near_idx,
        recovery_far_idx=args.recovery_far_idx,
    )
    inp_pert = sanitize_float32(inp_pert)
    inp_pert_raw_for_vis = copy.deepcopy(inp_pert)
    run_tag = (
        f"{args.perturb_channels}_idx{args.idx}"
        f"_cmd{args.cmd_source}"
        f"_dx{fmt_num_for_filename(args.dx)}"
        f"_dy{fmt_num_for_filename(args.dy)}"
        f"_dtheta{fmt_num_for_filename(args.dtheta)}"
    )
    
    ex = build_example_from_input_dict(dataset, inp)
    ex_pert = build_example_from_input_dict(dataset, inp_pert)

    with torch.no_grad():
        out = run_model_forward(model, ex, args.device)
        out_pert = run_model_forward(model, ex_pert, args.device)

    json.dump(to_key_tree(out), open(osp.join(args.out_dir, f'outputs_key_tree_original_{run_tag}.json'), 'w'), indent=2)
    json.dump(to_key_tree(out_pert), open(osp.join(args.out_dir, f'outputs_key_tree_perturbed_{run_tag}.json'), 'w'), indent=2)

    pred, dbg = extract_ego_pred_traj(out, score_mode=args.score_mode)
    pred_p, dbg_p = extract_ego_pred_traj(out_pert, score_mode=args.score_mode)
    pred = as_traj_np(pred, "pred_orig")
    pred_p = as_traj_np(pred_p, "pred_pert")

    gt = as_traj_np(inp_raw_for_vis['ego_fut_trajs_fix_time'], "gt_orig")
    gt_p = as_traj_np(inp_pert_raw_for_vis['ego_fut_trajs_fix_time'], "gt_pert")
    save_csv(osp.join(args.out_dir, f'traj_original_pred_{run_tag}.csv'), pred)
    save_csv(osp.join(args.out_dir, f'traj_perturbed_pred_{run_tag}.csv'), pred_p)
    save_csv(osp.join(args.out_dir, f'traj_original_gt_{run_tag}.csv'), gt)
    save_csv(osp.join(args.out_dir, f'traj_perturbed_gt_{run_tag}.csv'), gt_p)

    # Same delta convention as update_pose_fields:
    #   lidar2world_new = lidar2world_old @ old_lidar_from_new_lidar
    # Therefore a point in the perturbed/new lidar frame maps to the original
    # lidar frame as p_old = old_lidar_from_new_lidar @ p_new.
    delta_lidar = make_se2_delta(args.dx, args.dy, args.dtheta)
    pred_dst_from_src = delta_lidar if ablation_info['pose_frame_patched'] else np.eye(4, dtype=np.float64)
    gt_dst_from_src = delta_lidar if ablation_info['future_label_patched'] else np.eye(4, dtype=np.float64)
    pred_frame_conversion = (
        'p_old = delta_lidar @ p_new because pose_frame was patched'
        if ablation_info['pose_frame_patched']
        else 'identity; pose_frame was not patched, so prediction is already treated as original lidar frame'
    )
    gt_frame_conversion = (
        'p_old = delta_lidar @ p_new because future_label was patched'
        if ablation_info['future_label_patched']
        else 'identity; future_label was not patched'
    )

    pred_p_old = transform_traj_lidar_frame(
        pred_p,
        pred_dst_from_src,
        order=args.traj_order,
        lateral_positive=args.lateral_positive,
    )
    gt_p_old = transform_traj_lidar_frame(
        gt_p,
        gt_dst_from_src,
        order=args.traj_order,
        lateral_positive=args.lateral_positive,
    )
    save_csv(osp.join(args.out_dir, f'traj_perturbed_pred_in_original_lidar_{run_tag}.csv'), pred_p_old)
    save_csv(osp.join(args.out_dir, f'traj_perturbed_gt_in_original_lidar_{run_tag}.csv'), gt_p_old)

    raw_info_for_audit = dataset.get_data_by_index(args.idx)
    lidar2ego = np.asarray(raw_info_for_audit['sensors']['LIDAR_TOP']['lidar2ego'], dtype=np.float64)
    ego2lidar = invert_pose(lidar2ego)
    sensor_w2l = np.asarray(inp_raw_for_vis['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float64)
    top_w2l = np.asarray(inp_raw_for_vis['world2lidar'], dtype=np.float64)
    pose_w2l = np.asarray(inp_raw_for_vis['ego_pose_inv'], dtype=np.float64)
    pose_lidar_origin_in_sensor_lidar = (
        sensor_w2l @ invert_pose(pose_w2l) @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )[:3]
    sensor_lidar_origin_in_pose_lidar = (
        pose_w2l @ invert_pose(sensor_w2l) @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )[:3]
    label_origin_debug = {
        'dataset_fix_time_label_source': 'sensors.LIDAR_TOP.world2lidar; future LIDAR_TOP origin in current LIDAR_TOP frame',
        'lidar2ego_translation_xyz': lidar2ego[:3, 3].astype(float).tolist(),
        'ego_origin_in_lidar_xyz': ego2lidar[:3, 3].astype(float).tolist(),
        'top_world2lidar_vs_sensor_world2lidar_max_abs': float(np.abs(top_w2l - sensor_w2l).max()),
        'ego_pose_inv_vs_sensor_world2lidar_max_abs': float(np.abs(pose_w2l - sensor_w2l).max()),
        'pose_lidar_origin_in_sensor_lidar_xyz': pose_lidar_origin_in_sensor_lidar.astype(float).tolist(),
        'sensor_lidar_origin_in_pose_lidar_xyz': sensor_lidar_origin_in_pose_lidar.astype(float).tolist(),
    }

    traj_sanity = {
        'frame': 'original_lidar',
        'axis_convention': 'x=vehicle_right, y=forward',
        'ablation': ablation_info,
        'pred_frame_conversion': pred_frame_conversion,
        'gt_frame_conversion': gt_frame_conversion,
        'label_origin_debug': label_origin_debug,
        'gt_p_old_minus_gt_orig': traj_pair_diff_stats(
            gt,
            gt_p_old,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        ),
        'pred_p_old_minus_pred_orig': traj_pair_diff_stats(
            pred,
            pred_p_old,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        ),
    }
    metrics = {
        'idx': args.idx,
        'dx': args.dx,
        'dy': args.dy,
        'dtheta': args.dtheta,
        'perturb_channels': args.perturb_channels,
        'cmd_source': args.cmd_source,
        'recovery_near_idx': args.recovery_near_idx,
        'recovery_far_idx': args.recovery_far_idx,
        'recovery_ref_debug': ablation_info.get('recovery_ref_debug', {}),
        'frame': 'original_lidar',
        'axis_convention': 'x=vehicle_right, y=forward',
        'pred_frame_conversion': pred_frame_conversion,
        'pred_p_old_minus_pred_orig': traj_sanity['pred_p_old_minus_pred_orig'],
    }
    metrics_path = osp.join(
        args.out_dir,
        f'ablation_metrics_{args.perturb_channels}'
        f'_cmd{args.cmd_source}'
        f'_dx{fmt_num_for_filename(args.dx)}'
        f'_dy{fmt_num_for_filename(args.dy)}'
        f'_dtheta{fmt_num_for_filename(args.dtheta)}'
        f'_idx{args.idx}.json',
    )
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[ablation] saved metrics: {metrics_path}")

    debug_json_path = osp.join(args.out_dir, f'trajectory_frame_sanity_{run_tag}.json')
    with open(debug_json_path, 'w') as f:
        json.dump(traj_sanity, f, indent=2)
    print(f"[sanity] saved trajectory frame sanity: {debug_json_path}")
    print("[sanity] label origin debug:", json.dumps(label_origin_debug, indent=2))
    print("[sanity] gt_p_old - gt_orig:", json.dumps(traj_sanity['gt_p_old_minus_gt_orig'], indent=2))
    print("[sanity] pred_p_old - pred_orig:", json.dumps(traj_sanity['pred_p_old_minus_pred_orig'], indent=2))

    # BEV comparison is drawn in the original lidar frame. The perturbed GT can
    # nearly overlap the original GT after this frame conversion; that means the
    # perturbation/reprojection is geometrically self-consistent, not inactive.
    pred_right, pred_fwd = traj_to_bev_axes(
        pred, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    predp_right, predp_fwd = traj_to_bev_axes(
        pred_p_old, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    gt_right, gt_fwd = traj_to_bev_axes(
        gt, order=args.traj_order, lateral_positive=args.lateral_positive
    )
    gtp_right, gtp_fwd = traj_to_bev_axes(
        gt_p_old, order=args.traj_order, lateral_positive=args.lateral_positive
    )

    fig, ax = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    colors = {
        'pred_orig': '#2563eb',
        'pred_pert': '#dc2626',
        'gt_orig': '#15803d',
        'gt_pert': '#f59e0b',
    }
    ax.plot(pred_right, pred_fwd, '-o', label='pred_orig', color=colors['pred_orig'],
            linewidth=2.3, markersize=4.5, zorder=4)
    ax.plot(predp_right, predp_fwd, '-o', label='pred_pert -> original frame',
            color=colors['pred_pert'], linewidth=2.3, markersize=4.5, zorder=5)
    ax.plot(gt_right, gt_fwd, '--', label='gt_orig', color=colors['gt_orig'],
            linewidth=2.1, alpha=0.9, zorder=2)
    ax.plot(gtp_right, gtp_fwd, ':', label='gt_pert -> original frame',
            color=colors['gt_pert'], linewidth=2.8, alpha=0.95, zorder=3)

    ax.scatter([0], [0], c='k', marker='x', s=80, label='ego_origin', zorder=6)
    ax.annotate('right +', xy=(3.0, -2.3), xytext=(0.4, -2.3),
                arrowprops=dict(arrowstyle='->', color='0.25', lw=1.4),
                fontsize=9, color='0.25', va='center')
    ax.annotate('forward +', xy=(-8.6, 5.2), xytext=(-8.6, 0.6),
                arrowprops=dict(arrowstyle='->', color='0.25', lw=1.4),
                fontsize=9, color='0.25', ha='center')

    if len(pred_right):
        ax.text(pred_right[0], pred_fwd[0], 'orig start', fontsize=8, color=colors['pred_orig'])
        ax.text(pred_right[-1], pred_fwd[-1], 'orig end', fontsize=8, color=colors['pred_orig'])
    if len(predp_right):
        ax.text(predp_right[0], predp_fwd[0], 'pert start', fontsize=8, color=colors['pred_pert'])
        ax.text(predp_right[-1], predp_fwd[-1], 'pert end', fontsize=8, color=colors['pred_pert'])

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='0.88', linewidth=0.8)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.set_xlabel("vehicle lateral / m (right +)")
    ax.set_ylabel("forward / m")

    # Stable planning view. BEV is metric; camera image left/right is perspective
    # projection and should not be compared to this axis as a literal coordinate.
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 35)
    ax.set_title(
        f'Unified BEV in original lidar frame | ch={args.perturb_channels}, idx={args.idx}, '
        f'dx={args.dx}, dy={args.dy}, dtheta={args.dtheta}\n'
        f'pred conversion: {pred_frame_conversion}; image/lidar2img unchanged',
        fontsize=11,
    )

    bev_jpg_path = osp.join(
        args.out_dir,
        f'bev_compare_{run_tag}.png',
    )
    fig.savefig(bev_jpg_path, dpi=220)
    plt.close(fig)
    print(f"[save] BEV visualization: {bev_jpg_path}")

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

        # Original image/lidar2img are unchanged, so project in original lidar.
        pts_orig = make_traj_xyz1_for_projection(
            pred,
            z=args.traj_z,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        )

        # The perturbed prediction is produced in the perturbed lidar frame.
        # Convert it back to original lidar before using the original lidar2img.
        pts_pert_new = make_traj_xyz1_for_projection(
            pred_p,
            z=args.traj_z,
            order=args.traj_order,
            lateral_positive=args.lateral_positive,
        )
        pts_pert_old = (pred_dst_from_src @ pts_pert_new.T).T

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
            f'projection: ch={args.perturb_channels}, z={args.traj_z}',
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

        out_img_path = osp.join(
            args.out_dir,
            f'cam_compare_{run_tag}_{cam}.png',
        )
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
            'perturbed_projection_frame_conversion': pred_frame_conversion,
        }

    old_origin_new = (
        np.asarray(inp_pert_raw_for_vis['world2lidar'], dtype=np.float64)
        @ np.array([*inp_raw_for_vis['ego_translation'][:3], 1.0], dtype=np.float64)
    )[:3]
    
    audit = {
        'idx': args.idx,
        'dx': args.dx,
        'dy': args.dy,
        'dtheta': args.dtheta,
        'perturb_channels': args.perturb_channels,
        'cmd_source': args.cmd_source,
        'recovery_near_idx': args.recovery_near_idx,
        'recovery_far_idx': args.recovery_far_idx,
        'recovery_ref_debug': ablation_info.get('recovery_ref_debug', {}),
        'ablation': ablation_info,
        'experiment_type': 'planner_state_oracle_no_rerender',
        'image_rerendered': False,
        'delta_semantics': 'old_lidar_from_new_lidar; dx/dy are new lidar origin in original lidar axes (x right, y forward)',
        'updated_fields': ablation_info['patched_fields'],
        'unchanged_field_groups': ablation_info['not_patched_groups'],
        'always_unchanged_fields': ['img', 'img_filename', 'cam_intrinsic', 'lidar2img', 'camera extrinsics', 'agent/map labels', 'ego_fut_trajs_fix_dist'],
        'label_origin_debug': label_origin_debug,
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
            'lidar_xy': 'x_right_y_forward',
            'bev_frame': 'original_lidar',
            'bev_x_axis': 'vehicle_right_positive',
            'bev_y_axis': 'forward',
            'bev_note': 'metric BEV right-positive is aligned with lidar x; camera image left/right is perspective-projected pixels, not a BEV coordinate axis',
            'camera_projection_z': args.traj_z,
            'camera_perturbed_projection': pred_frame_conversion,
        },
        'trajectory_frame_sanity': traj_sanity,
        'sanity_checks': {
            'L0_pose_inv_err': float(
                np.abs(
                    np.asarray(inp_pert_raw_for_vis['ego_pose'], dtype=np.float64)
                    @ np.asarray(inp_pert_raw_for_vis['ego_pose_inv'], dtype=np.float64)
                    - np.eye(4)
                ).max()
            ),
            'L1_gt_diff_norm': float(np.linalg.norm(gt_p - gt, axis=-1).mean()),
            'L1_gt_old_frame_diff_norm': float(np.linalg.norm(gt_p_old - gt, axis=-1).mean()),
            'L2_pred_mean_display_right_diff': float((predp_right - pred_right).mean()),
            'L2_pred_first_display_right_diff': float(predp_right[0] - pred_right[0]),
            'L2_pred_final_display_right_diff': float(predp_right[-1] - pred_right[-1]),
            'L2_pred_mean_forward_diff': float((predp_fwd - pred_fwd).mean()),
        },
        'warnings': ['ego_fut_trajs_fix_dist is not recomputed; this is acceptable for inference-only forward unless later loss/eval code consumes that label.'],
    }
    with open(osp.join(args.out_dir, f'audit_{run_tag}.json'), 'w') as f:
        json.dump(audit, f, indent=2)


if __name__ == '__main__':
    main()
