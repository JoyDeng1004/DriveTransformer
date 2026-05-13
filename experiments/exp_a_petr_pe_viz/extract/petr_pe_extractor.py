from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .pe_base import PEExtractor


@dataclass
class PetrPEConfig:
    position_range: Sequence[float] = (-15.0, -30.0, -2.0, 15.0, 30.0, 2.0)
    depth_start: float = 1.0
    depth_num: int = 64
    lid: bool = True
    embed_dims: int = 512
    checkpoint_path: Optional[str] = None
    allow_untrained_mlp: bool = False
    device: str = "cpu"


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def build_depth_bins(cfg: PetrPEConfig) -> torch.Tensor:
    index = torch.arange(start=0, end=cfg.depth_num, step=1).float()
    x_max = float(cfg.position_range[3])
    if cfg.lid:
        index_1 = index + 1
        bin_size = (x_max - cfg.depth_start) / (cfg.depth_num * (1 + cfg.depth_num))
        return cfg.depth_start + bin_size * index * index_1
    bin_size = (x_max - cfg.depth_start) / cfg.depth_num
    return cfg.depth_start + bin_size * index


class PetrPEExtractor(PEExtractor):
    """Standalone extraction of DriveTransformer's PETR-style image-token PE."""

    def __init__(self, cfg: PetrPEConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.position_range = torch.tensor(cfg.position_range, dtype=torch.float32, device=self.device)
        self.coords_d = build_depth_bins(cfg).to(self.device)
        self.img_position_encoder = nn.Sequential(
            nn.Linear(cfg.depth_num * 3, cfg.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
        ).to(self.device)
        self.img_position_encoder.eval()
        self.loaded_checkpoint = None
        if cfg.checkpoint_path:
            self._load_checkpoint(cfg.checkpoint_path)
        elif not cfg.allow_untrained_mlp:
            raise ValueError(
                "checkpoint_path is required unless allow_untrained_mlp is true. "
                "The PETR PE MLP weights live in pts_bbox_head.img_position_encoder."
            )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        prefixes = (
            "module.pts_bbox_head.img_position_encoder.",
            "pts_bbox_head.img_position_encoder.",
            "img_position_encoder.",
        )
        own_state = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    own_state[key[len(prefix) :]] = value
                    break
        if not own_state:
            raise KeyError(f"No img_position_encoder weights found in {checkpoint_path}")
        missing_keys, extra_keys = self.img_position_encoder.load_state_dict(own_state, strict=False)
        if missing_keys or extra_keys:
            raise RuntimeError(
                f"Could not load img_position_encoder exactly. missing={missing_keys}, extra={extra_keys}"
            )
        self.loaded_checkpoint = checkpoint_path

    @torch.no_grad()
    def encode_image_tokens(
        self,
        lidar2img: np.ndarray,
        cam_intrinsic: np.ndarray,
        feature_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        lidar2img_t = torch.as_tensor(lidar2img, dtype=torch.float32, device=self.device)
        cam_intrinsic_t = torch.as_tensor(cam_intrinsic, dtype=torch.float32, device=self.device)
        if lidar2img_t.ndim == 3:
            lidar2img_t = lidar2img_t.unsqueeze(0)
        if cam_intrinsic_t.ndim == 3:
            cam_intrinsic_t = cam_intrinsic_t.unsqueeze(0)
        bsz, num_cam = lidar2img_t.shape[:2]
        h, w = feature_hw
        pad_h = float(h * 32)
        pad_w = float(w * 32)
        centers = self._locations(feature_hw, pad_h, pad_w)
        centers[..., 0] = centers[..., 0] * pad_w
        centers[..., 1] = centers[..., 1] * pad_h

        depth_num = self.coords_d.shape[0]
        num_tokens = h * w * num_cam
        memory_centers = centers.view(1, h * w, 1, 2).repeat(bsz, num_cam, depth_num, 1)
        coords_d = self.coords_d.view(1, 1, depth_num, 1).repeat(bsz, num_tokens, 1, 1)
        coords = torch.cat([memory_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        eps = 1e-5
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps
        )
        coords = coords.unsqueeze(-1)

        img2lidars = lidar2img_t.inverse()
        img2lidars = (
            img2lidars.view(bsz * num_cam, 1, 1, 4, 4)
            .repeat(1, h * w, depth_num, 1, 1)
            .view(bsz, num_tokens, depth_num, 4, 4)
        )
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d_norm = (coords3d - self.position_range[0:3]) / (
            self.position_range[3:6] - self.position_range[0:3]
        )
        pos_embed = inverse_sigmoid(coords3d_norm.reshape(bsz, -1, depth_num * 3))
        pe = self.img_position_encoder(pos_embed)
        pe = pe.view(bsz, num_cam, h, w, -1)
        coords3d_norm = coords3d_norm.view(bsz, num_cam, h, w, depth_num, 3)
        coords3d = coords3d.view(bsz, num_cam, h, w, depth_num, 3)
        intrinsic = torch.stack([cam_intrinsic_t[..., 0, 0], cam_intrinsic_t[..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        return {
            "pe": pe.detach().cpu().numpy(),
            "coords3d": coords3d.detach().cpu().numpy(),
            "coords3d_norm": coords3d_norm.detach().cpu().numpy(),
            "depth_bins": self.coords_d.detach().cpu().numpy(),
            "intrinsic_scale": intrinsic.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def encode_points(
        self,
        points_3d: np.ndarray,
        lidar2img: np.ndarray,
        cam_intrinsic: np.ndarray,
        image_hw: Tuple[int, int],
        camera_index: int = 0,
        feature_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        if feature_hw is None:
            feature_hw = (int(np.ceil(image_hw[0] / 32.0)), int(np.ceil(image_hw[1] / 32.0)))
        field = self.encode_image_tokens(lidar2img, cam_intrinsic, feature_hw)["pe"][0, camera_index]
        pts = np.asarray(points_3d, dtype=np.float64)
        uv, depth = project_points(pts, np.asarray(lidar2img)[camera_index])
        h_img, w_img = image_hw
        h_feat, w_feat = feature_hw
        ix = np.rint((uv[:, 0] / max(w_img, 1)) * w_feat - 0.5).astype(np.int64)
        iy = np.rint((uv[:, 1] / max(h_img, 1)) * h_feat - 0.5).astype(np.int64)
        valid = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w_img)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h_img)
            & (ix >= 0)
            & (ix < w_feat)
            & (iy >= 0)
            & (iy < h_feat)
        )
        pe = np.full((pts.shape[0], field.shape[-1]), np.nan, dtype=np.float32)
        pe[valid] = field[iy[valid], ix[valid]]
        return {
            "pe": pe,
            "uv": uv.astype(np.float32),
            "depth": depth.astype(np.float32),
            "valid": valid,
            "feature_xy": np.stack([ix, iy], axis=-1).astype(np.int64),
        }

    def _locations(self, hw: Tuple[int, int], pad_h: float, pad_w: float) -> torch.Tensor:
        h, w = hw
        stride = 32
        shifts_x = (torch.arange(0, stride * w, step=stride, dtype=torch.float32, device=self.device) + stride // 2) / pad_w
        shifts_y = (torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=self.device) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        return torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1).reshape(h, w, 2)

    def metadata(self) -> Dict[str, Any]:
        out = asdict(self.cfg)
        out["loaded_checkpoint"] = self.loaded_checkpoint
        out["depth_bins"] = self.coords_d.detach().cpu().numpy()
        return out


def project_points(points_3d: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_3d, dtype=np.float64)
    pts_h = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (np.asarray(lidar2img, dtype=np.float64) @ pts_h.T).T
    depth = proj[:, 2].copy()
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    mask = np.isfinite(depth) & (np.abs(depth) > 1e-12)
    uv[mask] = proj[mask, :2] / depth[mask, None]
    return uv, depth
