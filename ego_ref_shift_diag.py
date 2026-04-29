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
# §1  Model Loading
# ============================================================

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
    # Disable any test-time augmentation if present
    return model


# ============================================================
# §2  Data Loading
# ============================================================

def load_sample(config_path: str, sample_idx: int, device: str = "cuda:0"):
    """
    Load a single sample from B2D val set, ready to feed into model.

    Returns:
        batch:   dict for model(return_loss=False, **batch)
                 keys: img [1,N_cam,3,H,W], img_metas [[meta]]
        gt_info: dict for plotting
                 keys: lidar2img [N_cam,4,4], img_for_plot [H,W,3] uint8,
                       front_boxes_xywl_yaw list[(cx,cy,l,w,yaw)]
    """
    from mmcv import Config
    from mmcv.datasets import build_dataset
    from mmcv.parallel import collate

    cfg = Config.fromfile(config_path)

    # 1. Build val/test dataset
    # TODO[VERIFY]: which key holds val pipeline — `data.val` or `data.test`?
    data_cfg = cfg.data.val          # or cfg.data.test
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)

    sample = dataset[sample_idx]     # dict from pipeline (already has DataContainers)

    # 2. Collate single sample → batch (DataContainer-aware)
    batch = collate([sample], samples_per_gpu=1)

    # img is wrapped in DataContainer — unwrap and move
    img = batch["img"].data[0].to(device) if hasattr(batch["img"], "data") \
          else batch["img"].to(device)
    img_metas = batch["img_metas"].data[0] if hasattr(batch["img_metas"], "data") \
                else batch["img_metas"]

    model_batch = {"img": [img], "img_metas": [img_metas]}
    # NOTE: forward_test expects img and img_metas wrapped in *outer list*
    # (one entry per augmentation). With test_mode=True and no TTA, single entry.

    # 4. Build gt_info for plotting from img_metas[0]
    meta0 = img_metas[0]
    lidar2img = np.stack([np.asarray(m) for m in meta0["lidar2img"]], axis=0)
    # ^ shape [N_cam, 4, 4]

    # 5. Front camera image for the right panel
    # img tensor is normalized — need to denormalize for plotting.
    # mean/std typically in cfg.img_norm_cfg
    # TODO[VERIFY]
    img_norm = cfg.get("img_norm_cfg", {"mean": [123.675, 116.28, 103.53],
                                         "std":  [58.395, 57.12, 57.375],
                                         "to_rgb": True})
    img_np = img[0, CAM_FRONT_IDX].detach().cpu().numpy()  # [3, H, W]
    img_np = img_np.transpose(1, 2, 0)
    img_np = img_np * np.array(img_norm["std"]) + np.array(img_norm["mean"])
    if not img_norm.get("to_rgb", True):
        img_np = img_np[..., ::-1]
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    # 6. Extract front-vehicle GT boxes for BEV reference
    # TODO[VERIFY]: gt_bboxes_3d access path.
    # ----- FILL IN -----
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

    gt_info = {
        "lidar2img": lidar2img,
        "img_for_plot": img_np,
        "front_boxes_xywl_yaw": front_boxes,
    }
    return model_batch, gt_info


# ============================================================
# §3  Hook Factory
# ============================================================

def make_ego_ref_perturb_hook(shift_xyz: tuple):
    """
    Closure factory: returns a forward_pre_hook that replaces ego_ref
    (positional arg #EGO_REF_ARG_IDX) with ego_ref + shift.

    Args:
        shift_xyz: (dx, dy, dz) in meters, ego frame  (+y=front, +x=right)
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
# §4  Experiment Loop
# ============================================================

@torch.no_grad()
def run_one_experiment(model, batch, shift_xyz: tuple):
    """
    Register hook with given shift, run forward, extract predicted ego traj
    (all modes) + best-mode index, remove hook.

    Returns:
        ego_traj_xy:  np.ndarray [N_mode, T, 2]  in ego frame (+y front, +x right)
        best_mode:    int   argmax of ego_traj_cls_scores[-1, 0, :]
    """
    # 1. Locate target module (handle DDP wrapper)
    base = model.module if hasattr(model, "module") else model
    target_module = base.pts_bbox_head.transformer

    # 2. Register hook
    hook = make_ego_ref_perturb_hook(shift_xyz)
    handle = target_module.register_forward_pre_hook(hook)

    try:
        # 3. Forward.
        # NOTE: B2D / DriveTransformer's forward_test signature may differ
        captured = {}

        def capture_head_out(module, inputs, output):
            # head's forward returns the dict we want
            captured["out"] = output

        head_handle = base.pts_bbox_head.register_forward_hook(capture_head_out)
        try:
            _ = model(return_loss=False, rescale=True, **batch)
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

    # 4. Extract traj + cls
    # ego_fut_preds_fix_time: [N_layers, B, N_mode, T, 2]
    # ego_traj_cls_scores:     [N_layers, B, N_mode]
    traj_all = head_out["ego_fut_preds_fix_time"]
    cls_all = head_out["ego_traj_cls_scores"]

    # Last layer, batch 0
    traj_last = traj_all[-1, 0].detach().cpu().numpy()       # [N_mode, T, 2]
    cls_last = cls_all[-1, 0].detach().cpu().numpy()         # [N_mode]
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
# §5  Visualization
# ============================================================

def project_traj_to_image(traj_xy: np.ndarray, lidar2img: np.ndarray,
                          z: float = 0.0,
                          img_hw: tuple = None):
    """
    Project a 2D trajectory (in ego/lidar frame) onto image pixel coords.

    Args:
        traj_xy:   [T, 2]  in lidar frame  (+y front, +x right)
        lidar2img: [4, 4]
        z:         ground height in lidar frame (ego origin is roughly at
                   ground level after axis convention, so z=0 is fine for viz)
        img_hw:    optional (H, W) for additional in-bounds masking

    Returns:
        uv:   [T_visible, 2]
        keep: [T] bool mask of which timesteps survived
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
                traj[m], lidar2img_front, z=0.0, img_hw=(H, W)
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
# §6  Main
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