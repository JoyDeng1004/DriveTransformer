"""Recovery reference augmentation helpers for DriveTransformer.

This module keeps recovery-reference construction independent from the
single-sample debug script so the same code can be reused by a future dataset
or batch fine-tuning pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_array(name: str, value: Any, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def transform_xy_points(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to [N, 2] xy points."""
    T = _as_array("T", T)
    xy = _as_array("xy", xy)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape [4, 4], got {T.shape}")
    if xy.ndim != 2 or xy.shape[-1] != 2:
        raise ValueError(f"xy must have shape [N, 2], got {xy.shape}")

    xyz1 = np.concatenate(
        [
            xy,
            np.zeros((xy.shape[0], 1), dtype=np.float64),
            np.ones((xy.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )
    return (T @ xyz1.T).T[:, :2].astype(np.float32)


def select_near_far_ref(
    future_new_xy: np.ndarray,
    near_idx: int = 1,
    far_idx: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select near and far recovery reference points from future_new_xy."""
    future_new_xy = _as_array("future_new_xy", future_new_xy)
    if future_new_xy.ndim != 2 or future_new_xy.shape[-1] != 2:
        raise ValueError(f"future_new_xy must have shape [T, 2], got {future_new_xy.shape}")
    if future_new_xy.shape[0] == 0:
        raise ValueError("future_new_xy must contain at least one point")

    try:
        near_ref = future_new_xy[near_idx]
        far_ref = future_new_xy[far_idx]
    except IndexError as exc:
        raise IndexError(
            f"near_idx={near_idx}, far_idx={far_idx} are invalid for future length "
            f"{future_new_xy.shape[0]}"
        ) from exc

    return near_ref.astype(np.float32), far_ref.astype(np.float32)


def _pos2posemb(pos: np.ndarray, num_pos_feats: int = 32, temperature: int = 10000) -> np.ndarray:
    """Numpy copy of B2D_DriveTransformer_Dataset.pos2posemb."""
    pos = _as_array("pos", pos, dtype=np.float32)
    if pos.shape[-1] != 2:
        raise ValueError(f"pos must end with shape [2], got {pos.shape}")

    pos = pos * (2 * np.pi)
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_tmp = pos[..., None] / dim_t
    posemb = np.stack((np.sin(pos_tmp[..., 0::2]), np.cos(pos_tmp[..., 1::2])), axis=-1)
    return posemb.reshape(-1).astype(np.float32)


def build_ego_fut_cmd_from_refs(near_ref: np.ndarray, far_ref: np.ndarray) -> np.ndarray:
    """Encode recovery refs into ego_fut_cmd's existing position slots.

    The route-command one-hot slots are intentionally left zero. The recovery
    label itself must not be just future_new_xy; target generation belongs in a
    separate label builder, not in this command-conditioning helper.
    """
    near_ref = _as_array("near_ref", near_ref).reshape(2)
    far_ref = _as_array("far_ref", far_ref).reshape(2)

    cmd = np.zeros(140, dtype=np.float32)
    cmd[6:70] = _pos2posemb(far_ref)
    cmd[76:140] = _pos2posemb(near_ref)
    return cmd


def compute_recovery_ref_debug_stats(
    future_old_xy: np.ndarray,
    future_new_xy: np.ndarray,
    identity_future_new_xy: Optional[np.ndarray] = None,
    near_ref_new: Optional[np.ndarray] = None,
    far_ref_new: Optional[np.ndarray] = None,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dtheta: Optional[float] = None,
) -> Dict[str, Any]:
    """Build JSON-friendly debug stats for recovery reference construction."""
    future_old_xy = _as_array("future_old_xy", future_old_xy)
    future_new_xy = _as_array("future_new_xy", future_new_xy)
    n = min(len(future_old_xy), len(future_new_xy))
    diff = future_new_xy[:n] - future_old_xy[:n]
    point_error = np.linalg.norm(diff, axis=1) if n else np.zeros((0,), dtype=np.float64)

    stats: Dict[str, Any] = {
        "num_points": int(n),
        "mean_diff_xy_new_minus_old": diff.mean(axis=0).astype(float).tolist() if n else [],
        "max_abs_diff_xy_new_minus_old": np.abs(diff).max(axis=0).astype(float).tolist() if n else [],
        "mean_point_error_new_minus_old": float(point_error.mean()) if n else 0.0,
        "max_point_error_new_minus_old": float(point_error.max()) if n else 0.0,
    }

    if identity_future_new_xy is not None:
        identity_future_new_xy = _as_array("identity_future_new_xy", identity_future_new_xy)
        ni = min(len(future_old_xy), len(identity_future_new_xy))
        identity_diff = identity_future_new_xy[:ni] - future_old_xy[:ni]
        identity_error = np.linalg.norm(identity_diff, axis=1) if ni else np.zeros((0,), dtype=np.float64)
        stats.update(
            {
                "identity_num_points": int(ni),
                "identity_mean_diff_xy": identity_diff.mean(axis=0).astype(float).tolist() if ni else [],
                "identity_max_abs_diff_xy": np.abs(identity_diff).max(axis=0).astype(float).tolist() if ni else [],
                "identity_mean_point_error": float(identity_error.mean()) if ni else 0.0,
                "identity_max_point_error": float(identity_error.max()) if ni else 0.0,
            }
        )

    if near_ref_new is not None:
        near_ref_new = _as_array("near_ref_new", near_ref_new).reshape(2)
        stats["near_ref_new"] = near_ref_new.astype(float).tolist()
        if dx is not None:
            stats["expected_sign_near_x_plus_dx"] = float(near_ref_new[0] + float(dx))
    if far_ref_new is not None:
        stats["far_ref_new"] = _as_array("far_ref_new", far_ref_new).reshape(2).astype(float).tolist()
    if dx is not None:
        stats["dx"] = float(dx)
    if dy is not None:
        stats["dy"] = float(dy)
    if dtheta is not None:
        stats["dtheta"] = float(dtheta)

    return stats


@dataclass(frozen=True)
class RecoveryRefBuilder:
    near_idx: int = 1
    far_idx: int = -1

    def build(
        self,
        future_old_xy: np.ndarray,
        T_old_lidar_to_world: np.ndarray,
        T_new_lidar_to_world: np.ndarray,
    ) -> Dict[str, Any]:
        """Build recovery refs and ego_fut_cmd for a perturbed lidar frame."""
        future_old_xy = _as_array("future_old_xy", future_old_xy)
        T_old_lidar_to_world = _as_array("T_old_lidar_to_world", T_old_lidar_to_world)
        T_new_lidar_to_world = _as_array("T_new_lidar_to_world", T_new_lidar_to_world)

        if future_old_xy.ndim != 2 or future_old_xy.shape[-1] != 2:
            raise ValueError(f"future_old_xy must have shape [T, 2], got {future_old_xy.shape}")
        if T_old_lidar_to_world.shape != (4, 4):
            raise ValueError(f"T_old_lidar_to_world must have shape [4, 4], got {T_old_lidar_to_world.shape}")
        if T_new_lidar_to_world.shape != (4, 4):
            raise ValueError(f"T_new_lidar_to_world must have shape [4, 4], got {T_new_lidar_to_world.shape}")

        T_new_lidar_from_old_lidar = np.linalg.inv(T_new_lidar_to_world) @ T_old_lidar_to_world
        future_new_xy = transform_xy_points(T_new_lidar_from_old_lidar, future_old_xy)
        near_ref_new, far_ref_new = select_near_far_ref(future_new_xy, self.near_idx, self.far_idx)
        ego_fut_cmd_new = build_ego_fut_cmd_from_refs(near_ref_new, far_ref_new)

        identity_future_new_xy = transform_xy_points(
            np.linalg.inv(T_old_lidar_to_world) @ T_old_lidar_to_world,
            future_old_xy,
        )
        identity_max_err = (
            float(np.abs(identity_future_new_xy - future_old_xy).max())
            if len(future_old_xy)
            else 0.0
        )
        assert identity_max_err < 1e-5, f"identity transform changed future xy by {identity_max_err}"

        debug_info = compute_recovery_ref_debug_stats(
            future_old_xy=future_old_xy,
            future_new_xy=future_new_xy,
            identity_future_new_xy=identity_future_new_xy,
            near_ref_new=near_ref_new,
            far_ref_new=far_ref_new,
        )
        debug_info.update(
            {
                "near_idx": int(self.near_idx),
                "far_idx": int(self.far_idx),
                "identity_assert_max_abs_err": identity_max_err,
                "T_new_lidar_from_old_lidar": T_new_lidar_from_old_lidar.astype(float).tolist(),
                "ego_fut_cmd_layout": "far_ref posemb in [6:70], near_ref posemb in [76:140], route one-hot slots zero",
            }
        )

        return {
            "near_ref_new": near_ref_new,
            "far_ref_new": far_ref_new,
            "future_new_xy": future_new_xy,
            "ego_fut_cmd_new": ego_fut_cmd_new,
            "debug_info": debug_info,
        }
