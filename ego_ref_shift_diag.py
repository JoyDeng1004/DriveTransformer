"""
Usage: python ego_ref_shift_diag.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --ckpt /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --sample-idx 96 \
  --device cuda:0

Pipeline: load model -> load sample 96 -> for each shift: register hook,
          forward, collect ego traj -> plot dual-view figure.

Output: ./pe_diagnosis/zero_ego_pe/fig1_trajectory_dual_view.png
"""

import os
import copy
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import inspect
import types

# ============================================================
# §0  Constants & Experiment Config
# ============================================================

# Position of ego_ref in pts_bbox_head.transformer's positional args.
# Confirmed from drivetransformer_head.py:
#   transformer(agent_query, map_query, ego_query, img_feats,
#               img_pos_embed, agent_temp_memory, agent_temp_pos,
#               map_temp_memory, map_temp_pos, ego_memory_embedding,
#               ego_temp_pos, agent_prep_ref, map_prep_ref,
#               map_prep_pts_coord, ego_ref, ...)
# 0-based index: ego_ref is the 15th positional arg => idx = 14
EGO_REF_ARG_IDX = 14

# Coordinate system (confirmed in Step 2a-bis):
#   +y = front, -y = back, +x = right, -x = left, units = meters
# point_cloud_range: x ∈ [-15, 15], y ∈ [-30, 30]
EXPERIMENTS = [
    # (group_name, shift_xyz, color, label)
    ("G0_baseline",  (0.0,  0.0, 0.0), "black",   "baseline (0,0,0)"),
    ("G1_front+5",   (0.0,  5.0, 0.0), "tab:blue",   "ego↑ front +5m"),
    ("G2_back-5",    (0.0, -5.0, 0.0), "tab:red",    "ego↓ back -5m"),
    ("G3_right+5",   (5.0,  0.0, 0.0), "tab:green",  "ego→ right +5m"),
]

SAMPLE_IDX = 96
OUTPUT_DIR = Path("./pe_diagnosis/zero_ego_pe")
OUTPUT_FIG = OUTPUT_DIR / "fig1_trajectory_dual_view.png"

# Front camera index in the 6-cam list
CAM_FRONT_IDX = 0


# ============================================================
# Model Loading
# ============================================================

def patch_single_arg_ffns_for_identity(model):
    """
    Compatibility patch for DriveTransformer layers.
    """
    patched = 0
    for module_name, module in model.named_modules():
        if not hasattr(module, "ffns"):
            continue

        ffns = getattr(module, "ffns")
        for i, ffn in enumerate(ffns):
            sig = inspect.signature(ffn.forward)
            num_params = len(sig.parameters)

            # Bound method:
            #   forward(x)                 -> 1 param
            #   forward(x, identity=None)  -> 2 params
            if num_params >= 2:
                continue

            old_forward = ffn.forward

            def new_forward(self, x, identity=None, _old_forward=old_forward):
                out = _old_forward(x)
                if identity is not None:
                    out = out + identity
                return out

            ffn.forward = types.MethodType(new_forward, ffn)
            patched += 1
            print(f"[patch] wrapped single-arg FFN: {module_name}.ffns[{i}]")

    print(f"[patch] total wrapped FFNs = {patched}")

def load_model(config_path: str, ckpt_path: str, device: str = "cuda:0"):
    """
    Load DriveTransformer in eval mode.
    """
    from mmcv import Config
    from mmcv.utils import load_checkpoint
    from mmcv.models import build_model
    # TODO[VERIFY]: DriveTransformer's custom modules (head, transformer,
    # dataset) need to be imported so mmcv registry picks them up.
    from adzoo.drivetransformer.mmdet3d_plugin.datasets.builder import build_dataloader
    cfg = Config.fromfile(config_path)

    # Disable any pretrained image backbone download (we have full ckpt)
    if cfg.model.get("img_backbone", {}).get("init_cfg") is not None:
        cfg.model.img_backbone.init_cfg = None

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    model = model.to(device).eval()
    # Compatibility patch for FFN forward(x) vs forward(x, identity)
    patch_single_arg_ffns_for_identity(model)
    return model


# ============================================================
# Data Loading
# ============================================================

def load_sample(config_path: str, sample_idx: int, device: str = "cuda:0"):
    """
    Load a single sample from B2D val set, ready to feed into model.
    """
    from mmcv import Config
    from mmcv.datasets import build_dataset
    from mmcv.parallel import collate
    from mmcv.parallel import DataContainer
    def unwrap_dc(x):
        """
        Recursively unwrap mmcv DataContainer.
        Keep list/dict structure.
        """
        while isinstance(x, DataContainer):
            x = x.data
        if isinstance(x, dict):
            return {k: unwrap_dc(v) for k, v in x.items()}
        if isinstance(x, list):
            return [unwrap_dc(v) for v in x]
        if isinstance(x, tuple):
            return tuple(unwrap_dc(v) for v in x)
        return x

    def move_to_device(x, device):
        if torch.is_tensor(x):
            x = x.to(device)
            if torch.is_floating_point(x):
                x = x.float()
            return x
        if isinstance(x, dict):
            return {k: move_to_device(v, device) for k, v in x.items()}
        if isinstance(x, list):
            return [move_to_device(v, device) for v in x]
        if isinstance(x, tuple):
            return tuple(move_to_device(v, device) for v in x)
        return x

    cfg = Config.fromfile(config_path)

    # Build val/test dataset
    data_cfg = cfg.data.val          # or cfg.data.test
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)

    sample = dataset[sample_idx]
    print("[debug] raw sample keys =", sorted(sample.keys()))

    batch = collate([sample], samples_per_gpu=1)
    print("[debug] collated batch keys =", sorted(batch.keys()))

    model_batch = {}
    for k, v in batch.items():
        v = unwrap_dc(v)

        if k == "img" and isinstance(v, list) and len(v) == 1:
            v = v[0]

        model_batch[k] = v

    # ---- normalize img ----
    img_tensor = model_batch["img"]
    if isinstance(img_tensor, list) and len(img_tensor) == 1:
        img_tensor = img_tensor[0]

    if not torch.is_tensor(img_tensor):
        img_tensor = torch.as_tensor(img_tensor)

    img_tensor = img_tensor.to(device)

    # Expected by DriveTransformer forward_test:
    #   img: Tensor [B, N_cam, 3, H, W]
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.dim() != 5:
        raise RuntimeError(f"unexpected img_tensor.shape = {tuple(img_tensor.shape)}")

    model_batch["img"] = img_tensor
    print(f"[debug] img_tensor.shape = {tuple(img_tensor.shape)}")

    # ---- normalize img_metas ----
    img_metas = model_batch["img_metas"]

    # Possible structures after collate / unwrap:
    #   [[meta0]] -> [meta0]
    #   [meta0]   -> [meta0]
    #   meta0     -> [meta0]   # because the generic len==1 unwrap may strip one level
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
        img_metas = img_metas[0]

    if isinstance(img_metas, dict):
        img_metas = [img_metas]

    assert isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], dict), (
        f"unexpected img_metas structure: {type(img_metas)}, "
        f"first={type(img_metas[0]) if isinstance(img_metas, list) and len(img_metas) > 0 else 'N/A'}"
    )

    model_batch["img_metas"] = img_metas
    metas_inner = img_metas
    meta0 = metas_inner[0]
    print(f"[debug] meta0 keys = {list(meta0.keys())}")

    # ---- ensure common geometry keys are tensor[B, N_cam, ...] ----
    for key in [
        "lidar2img",
        "cam_intrinsic",
        "lidar2cam",
        "cam2lidar",
        "ego2global",
        "lidar2ego",
        "ego2lidar",
    ]:
        if key not in model_batch:
            continue

        x = model_batch[key]

        # unwrap single-element list
        if isinstance(x, list) and len(x) == 1:
            x = x[0]

        # list of camera matrices -> ndarray [N_cam, ...]
        if isinstance(x, list):
            x = np.stack([np.asarray(xx) for xx in x], axis=0)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if torch.is_tensor(x):
            x = x.float().to(device)
            # camera-wise matrix: [N_cam, 4, 4] or [N_cam, 3, 3] -> [B, N_cam, ...]
            if x.dim() in (3, 4) and x.shape[0] == 6:
                x = x.unsqueeze(0)

        model_batch[key] = x

    # ---- fallback: recover cam_intrinsic if pipeline did not expose it ----
    if "cam_intrinsic" not in model_batch:
        cam_intrinsic = None

        # 1) try img_metas
        if "cam_intrinsic" in meta0:
            cam_intrinsic = meta0["cam_intrinsic"]

        # 2) try dataset info
        if cam_intrinsic is None:
            try:
                info = dataset.data_infos[sample_idx] \
                    if hasattr(dataset, "data_infos") \
                    else dataset.get_data_info(sample_idx)

                if "cam_intrinsic" in info:
                    cam_intrinsic = info["cam_intrinsic"]

                elif "cams" in info:
                    cams = info["cams"]
                    cam_names = [
                        "CAM_FRONT",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_RIGHT",
                        "CAM_BACK",
                        "CAM_BACK_LEFT",
                        "CAM_BACK_RIGHT",
                    ]

                    mats = []
                    for cam in cam_names:
                        if cam in cams and "cam_intrinsic" in cams[cam]:
                            mats.append(np.asarray(cams[cam]["cam_intrinsic"]))

                    if len(mats) == 6:
                        cam_intrinsic = np.stack(mats, axis=0)

            except Exception as e:
                print(f"[warn] failed to recover cam_intrinsic from dataset info: {e}")

        if cam_intrinsic is not None:
            if isinstance(cam_intrinsic, list):
                cam_intrinsic = np.stack([np.asarray(x) for x in cam_intrinsic], axis=0)

            cam_intrinsic = torch.as_tensor(
                cam_intrinsic,
                dtype=torch.float32,
                device=device,
            )

            # [6, 3, 3] -> [1, 6, 3, 3]
            if cam_intrinsic.dim() == 3 and cam_intrinsic.shape[0] == 6:
                cam_intrinsic = cam_intrinsic.unsqueeze(0)

            model_batch["cam_intrinsic"] = cam_intrinsic
            print("[debug] recovered cam_intrinsic.shape =", tuple(cam_intrinsic.shape))
        else:
            print("[warn] cam_intrinsic still missing after fallback")

    # ---- move remaining tensor fields to device ----
    model_batch = move_to_device(model_batch, device)

    for key in ["lidar2img", "cam_intrinsic", "lidar2cam", "cam2lidar"]:
        if key in model_batch:
            x = model_batch[key]
            if torch.is_tensor(x):
                print(f"[debug] model_batch['{key}'].shape = {tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
            else:
                print(f"[debug] model_batch['{key}'] type = {type(x)}")
        else:
            print(f"[warn] model_batch missing key: {key}")

    # Front camera image for the right panel
    img_norm = cfg.get("img_norm_cfg", {"mean": [123.675, 116.28, 103.53],
                                         "std":  [58.395, 57.12, 57.375],
                                         "to_rgb": True})
    img_np = img_tensor[0, CAM_FRONT_IDX].detach().cpu().numpy()  # [3, H, W]
    img_np = img_np.transpose(1, 2, 0)
    img_np = img_np * np.array(img_norm["std"]) + np.array(img_norm["mean"])
    if not img_norm.get("to_rgb", True):
        img_np = img_np[..., ::-1]
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    # Extract front-vehicle GT boxes for BEV reference
    front_boxes = []
    try:
        info = dataset.data_infos[sample_idx] \
               if hasattr(dataset, "data_infos") \
               else dataset.get_data_info(sample_idx)
        # B2D info dict typically has 'gt_boxes' / 'gt_names' or similar.
        # Each box: [cx, cy, cz, l, w, h, yaw] in lidar/ego frame.
        # Filter: y > 0 (front) and |x| < 5 (in lane)
        # Adapt the field names below to your B2D info schema.
        boxes_raw = info.get("gt_boxes", info.get("gt_bboxes_3d", None))
        if boxes_raw is not None:
            boxes_arr = np.asarray(boxes_raw)
            for b in boxes_arr:
                cx, cy = b[0], b[1]
                l, w = b[3], b[4]
                yaw = b[6]
                if cy > 0 and abs(cx) < 8:
                    front_boxes.append((cx, cy, l, w, yaw))
    except Exception as e:
        print(f"[warn] failed to extract GT boxes for plotting: {e}")
    # ------------------
    lidar2img_for_plot = model_batch["lidar2img"]
    if torch.is_tensor(lidar2img_for_plot):
        lidar2img_for_plot = lidar2img_for_plot.detach().cpu().numpy()
    if lidar2img_for_plot.ndim == 4 and lidar2img_for_plot.shape[0] == 1:
        lidar2img_for_plot = lidar2img_for_plot[0]
    gt_info = {
        "lidar2img": lidar2img_for_plot,
        "img_for_plot": img_np,
        "front_boxes_xywl_yaw": front_boxes,
    }

    for key in ["map_gt_bboxes_3d", "ego_his_trajs", "ego_fut_cmd", "ego_lcf_feat"]:
        if key in model_batch:
            v = model_batch[key]
            print(f"[debug] before forward {key}: type={type(v)}")
            if isinstance(v, list):
                print(f"[debug] before forward {key}: len={len(v)}, elem_type={type(v[0])}")
    return model_batch, gt_info

# ============================================================
# Hook Factory
# ============================================================

def make_ego_ref_perturb_hook(shift_xyz: tuple):
    """
    Closure factory: returns a forward_pre_hook that replaces ego_ref
    (positional arg #EGO_REF_ARG_IDX) with ego_ref + shift.
    """
    dx, dy, dz = shift_xyz

    def hook_fn(module, inputs):
        # 1. Sanity: inputs is a tuple long enough to contain ego_ref
        assert isinstance(inputs, tuple), \
            f"expected tuple inputs, got {type(inputs)}"
        assert len(inputs) > EGO_REF_ARG_IDX, \
            f"inputs has only {len(inputs)} args, need idx {EGO_REF_ARG_IDX}"

        ego_ref = inputs[EGO_REF_ARG_IDX]

        # 2. Sanity: shape [B, N_mode, 3] + (近似)全零
        assert torch.is_tensor(ego_ref), \
            f"ego_ref is not a tensor, got {type(ego_ref)}"
        assert ego_ref.dim() == 3 and ego_ref.shape[-1] == 3, \
            f"ego_ref shape {tuple(ego_ref.shape)} != [B, N_mode, 3]"
        max_abs = ego_ref.abs().max().item()
        assert max_abs < 1e-5, \
            f"ego_ref expected zeros but |max|={max_abs:.3e} — wrong arg index?"

        # 3. Build shift tensor matching ego_ref
        shift = torch.tensor([dx, dy, dz],
                             dtype=ego_ref.dtype,
                             device=ego_ref.device)
        # broadcast: [3] -> [B, N_mode, 3]
        new_ego_ref = ego_ref + shift.view(1, 1, 3)

        # 4. Return new tuple with replaced arg (no in-place mutation)
        new_inputs = list(inputs)
        new_inputs[EGO_REF_ARG_IDX] = new_ego_ref
        return tuple(new_inputs)

    return hook_fn


# ============================================================
# Experiment Loop
# ============================================================

@torch.no_grad()
def run_one_experiment(model, batch, shift_xyz: tuple):
    """
    Register hook with given shift, run forward, extract predicted ego traj
    (all modes) + best-mode index, remove hook.
    """
    # Locate target module (handle DDP wrapper)
    base = model.module if hasattr(model, "module") else model
    target_module = base.pts_bbox_head.transformer

    # Make each perturbation run independent.
    base.prev_scene_token = None
    base.pts_bbox_head.reset_memory()

    # Register hook
    hook = make_ego_ref_perturb_hook(shift_xyz)
    handle = target_module.register_forward_pre_hook(hook)

    try:
        captured = {}
        def capture_head_out(module, inputs, output):
            # head's forward returns the dict we want
            captured["out"] = output

        head_handle = base.pts_bbox_head.register_forward_hook(capture_head_out)
        try:
            _ = model(batch, return_loss=False, rescale=True)
        finally:
            head_handle.remove()

        head_out = captured["out"]
        # head_out is either a dict or a tuple/list — handle both
        if isinstance(head_out, (list, tuple)):
            head_out = head_out[0] if isinstance(head_out[0], dict) else head_out
        assert isinstance(head_out, dict), \
            f"unexpected head_out type {type(head_out)}; inspect manually"

    finally:
        handle.remove()

    traj_all = head_out["ego_fut_preds_fix_time"]
    cls_all = head_out["ego_traj_cls_scores"]

    # Last layer, batch 0
    traj_last = traj_all[-1, 0].detach().cpu().numpy()
    if cls_all is None:
        best_mode = 0
        print("[warn] ego_traj_cls_scores is None; use best_mode=0 for plotting")
    else:
        cls_last = cls_all[-1, 0].detach().cpu().numpy()
        best_mode = int(np.argmax(cls_last))

    return traj_last, best_mode


def run_all_experiments(model, batch):
    """
    Loop over EXPERIMENTS, return dict {group_name: (traj, best_mode)}.
    """
    results = {}
    for name, shift, _color, _label in EXPERIMENTS:
        print(f"[exp] running {name} with shift={shift} ...")
        traj, best_mode = run_one_experiment(model, batch, shift)
        results[name] = {"traj": traj, "best_mode": best_mode, "shift": shift}
        print(f"       traj shape={traj.shape}, best_mode={best_mode}")
    return results


# ============================================================
# Visualization
# ============================================================

def project_traj_to_image(traj_xy: np.ndarray, lidar2img: np.ndarray,
                          z: float = 0.0,
                          img_hw: tuple = None):
    """
    Project a 2D trajectory (in ego/lidar frame) onto image pixel coords.
    """
    T = traj_xy.shape[0]
    pts_h = np.concatenate(
        [traj_xy, np.full((T, 1), z), np.ones((T, 1))], axis=1
    )  # [T, 4]
    proj = (lidar2img @ pts_h.T).T  # [T, 4]

    depth = proj[:, 2]
    in_front = depth > 1e-3
    uv = np.zeros((T, 2))
    uv[in_front] = proj[in_front, :2] / depth[in_front, None]

    keep = in_front.copy()
    if img_hw is not None:
        H, W = img_hw
        in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < H)
        keep = keep & in_img

    return uv, keep


def _draw_box_bev(ax, box_xywl_yaw, **kw):
    """
    Draw a single 3D box footprint in BEV.
    Args:
        box_xywl_yaw: (cx, cy, l, w, yaw)  in lidar/ego frame
                      l = length (along yaw axis), w = width
    """
    cx, cy, l, w, yaw = box_xywl_yaw
    # 4 corners in box-local frame (length along x_box, width along y_box)
    corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    corners_world = (R @ corners.T).T + np.array([cx, cy])
    poly = plt.Polygon(corners_world, closed=True, fill=False, **kw)
    ax.add_patch(poly)


def plot_dual_view(results: dict, gt_info: dict, save_path: Path):
    """
    Two-panel figure: BEV (left) + CAM_FRONT (right).
    All N_mode trajectories drawn as thin lines; best_mode bolded.
    """
    fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(16, 7))

    # ---------- LEFT: BEV ----------
    # Convention: +y front (up), +x right
    ax_bev.set_title("BEV (ego frame)  +y=front, +x=right")
    ax_bev.set_xlabel("x  right (m)")
    ax_bev.set_ylabel("y  front (m)")
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)
    ax_bev.axhline(0, color="gray", lw=0.5)
    ax_bev.axvline(0, color="gray", lw=0.5)

    for name, shift, color, label in EXPERIMENTS:
        traj = results[name]["traj"]            # [N_mode, T, 2]
        best = results[name]["best_mode"]
        N_mode = traj.shape[0]

        # Determine ego "claimed" start position from shift
        sx, sy, _ = shift

        # All modes: thin lines
        for m in range(N_mode):
            xs = traj[m, :, 0]
            ys = traj[m, :, 1]
            lw = 2.5 if m == best else 0.6
            alpha = 1.0 if m == best else 0.4
            lbl = label if m == best else None
            # NOTE: traj is predicted in ego-original frame (network output).
            # Ego is "told" it's at (sx, sy) but its frame origin is still the
            # real ego. We plot traj in the original frame; the cross marker
            # at (sx, sy) shows where ego *thought* it was.
            ax_bev.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=lbl)

        # Mark "claimed" ego position
        ax_bev.plot(sx, sy, marker="+", color=color, ms=14, mew=2.5)

    # GT front-vehicle box(es)
    for box in gt_info.get("front_boxes_xywl_yaw", []):
        _draw_box_bev(ax_bev, box, edgecolor="orange", lw=1.5, ls="--")

    # Real ego marker
    ax_bev.plot(0, 0, marker="o", color="black", ms=8)
    ax_bev.text(0.3, -0.5, "real ego", fontsize=8)

    ax_bev.set_xlim(-15, 15)
    ax_bev.set_ylim(-5, 35)
    ax_bev.legend(loc="upper right", fontsize=8)

    # ---------- RIGHT: CAM_FRONT ----------
    ax_cam.set_title("CAM_FRONT projection")
    img = gt_info["img_for_plot"]              # [H, W, 3] uint8
    ax_cam.imshow(img)
    H, W = img.shape[:2]
    lidar2img_front = gt_info["lidar2img"][CAM_FRONT_IDX]   # [4, 4]

    for name, shift, color, label in EXPERIMENTS:
        traj = results[name]["traj"]
        best = results[name]["best_mode"]
        N_mode = traj.shape[0]
        for m in range(N_mode):
            uv, keep = project_traj_to_image(
                traj[m], lidar2img_front, z=-1.5, img_hw=(H, W)
            )
            if keep.sum() < 2:
                continue
            lw = 2.5 if m == best else 0.6
            alpha = 1.0 if m == best else 0.4
            ax_cam.plot(uv[keep, 0], uv[keep, 1],
                        color=color, lw=lw, alpha=alpha)

    ax_cam.set_xlim(0, W)
    ax_cam.set_ylim(H, 0)
    ax_cam.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="path to drivetransformer_large.py")
    parser.add_argument("--ckpt", required=True, help="path to .pth")
    parser.add_argument("--sample-idx", type=int, default=SAMPLE_IDX)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading model ...")
    model = load_model(args.config, args.ckpt, args.device)

    print(f"[2/4] loading sample {args.sample_idx} ...")
    batch, gt_info = load_sample(args.config, args.sample_idx, args.device)

    print("[3/4] running experiments ...")
    results = run_all_experiments(model, batch)

    print("[4/4] plotting ...")
    plot_dual_view(results, gt_info, OUTPUT_FIG)
    print(f"saved -> {OUTPUT_FIG}")


if __name__ == "__main__":
    main()