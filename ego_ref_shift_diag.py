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

from planner_sensitivity_report import run_basic_sensitivity_report

# ============================================================
# S0: Constants & Experiment Config
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
AGENT_QUERY_ARG_IDX = 0
MAP_QUERY_ARG_IDX = 1
EGO_QUERY_ARG_IDX = 2

# Coordinate system (confirmed in Step 2a-bis):
#   +y = front, -y = back, +x = right, -x = left, units = meters
# point_cloud_range: x ∈ [-15, 15], y ∈ [-30, 30]
EXPERIMENTS = [
    # name, ego_ref_shift_xyz, color, label
    ("G0_baseline",          (0.0, 0.0, 0.0), "black",      "G0 baseline"),
    ("G1_ego_ref_shift_x2",  (2.0, 0.0, 0.0), "tab:blue",   "G1 ego_ref x+2m"),
    ("G2_ego_his_zero",      (0.0, 0.0, 0.0), "tab:orange", "G2 ego_his zero"),
    ("G3_ego_lcf_zero",      (0.0, 0.0, 0.0), "tab:green",  "G3 ego_lcf zero"),
    ("G4_map_query_mask",    (0.0, 0.0, 0.0), "tab:red",    "G4 map query mask"),
    ("G5_agent_query_mask",  (0.0, 0.0, 0.0), "tab:purple", "G5 agent query mask"),
]

INTERVENTIONS = {
    "G0_baseline": {
        "type": "none",
    },
    "G1_ego_ref_shift_x2": {
        "type": "ego_ref_shift",
    },
    "G2_ego_his_zero": {
        "type": "batch_zero",
        "batch_key": "ego_his_trajs",
    },
    "G3_ego_lcf_zero": {
        "type": "batch_zero",
        "batch_key": "ego_lcf_feat",
    },
    "G4_map_query_mask": {
        "type": "query_mask",
        "arg_idx": MAP_QUERY_ARG_IDX,
        "target": "map_query",
    },
    "G5_agent_query_mask": {
        "type": "query_mask",
        "arg_idx": AGENT_QUERY_ARG_IDX,
        "target": "agent_query",
    },
}

SAMPLE_IDX = 96
OUTPUT_DIR = Path("./pe_diagnosis/zero_ego_pe")
OUTPUT_FIG1 = OUTPUT_DIR / "fig1_trajectory_dual_view.png"
OUTPUT_FIG2 = OUTPUT_DIR / "fig2_traj_shift_by_layer.png"
OUTPUT_FIG3 = OUTPUT_DIR / "fig3_ego_query_drift_by_layer.png"
OUTPUT_FIG4 = OUTPUT_DIR / "fig4_ego_agent_relation_shift_by_layer.png"

# Front camera index in the 6-cam list
CAM_FRONT_IDX = 0


# ============================================================
# S1: Model Loading
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
# S2: Data Loading
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

def clone_batch_for_intervention(x):
    """
    Clone batch recursively so each intervention run is independent.

    Important:
      - torch.Tensor is cloned.
      - dict/list/tuple are recursively cloned.
      - non-tensor objects are deep-copied.
    """
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_batch_for_intervention(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_batch_for_intervention(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_batch_for_intervention(v) for v in x)
    return copy.deepcopy(x)

def zero_like_nested(x):
    """
    Recursively zero tensors / arrays inside a nested object.
    """
    if torch.is_tensor(x):
        return torch.zeros_like(x)
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    if isinstance(x, dict):
        return {k: zero_like_nested(v) for k, v in x.items()}
    if isinstance(x, list):
        return [zero_like_nested(v) for v in x]
    if isinstance(x, tuple):
        return tuple(zero_like_nested(v) for v in x)
    return x


# ============================================================
# S3: Intervention Hooks
# ============================================================
def apply_batch_zero_intervention(batch, batch_key):
    """
    Zero out a top-level batch field, e.g. ego_his_trajs or ego_lcf_feat.
    """
    if batch_key not in batch:
        print(f"[warn] batch key '{batch_key}' not found; skip zero intervention.")
        print(f"[debug] available batch keys = {sorted(batch.keys())}")
        return batch

    old_v = batch[batch_key]
    batch[batch_key] = zero_like_nested(old_v)

    print(f"[intervention] zero-out batch['{batch_key}']")
    if torch.is_tensor(old_v):
        print(f"  old shape={tuple(old_v.shape)}, dtype={old_v.dtype}, device={old_v.device}")
    elif isinstance(old_v, list):
        print(f"  old type=list, len={len(old_v)}, elem_type={type(old_v[0]) if len(old_v) > 0 else None}")
    else:
        print(f"  old type={type(old_v)}")

    return batch

def infer_query_splits(head):
    """
    Infer query split sizes for concatenated decoder query:
        [agent_query, map_query, ego_query]

    Returns:
        agent_n, map_n, ego_n
    """
    # agent query num
    agent_candidates = [
        "num_query",
        "num_agent_query",
        "agent_query_num",
    ]
    map_candidates = [
        "num_map_query",
        "map_query_num",
        "num_vec",
    ]
    ego_candidates = [
        "ego_query_num",
        "num_ego_query",
    ]

    def get_first_attr(obj, names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if isinstance(v, int):
                    return v, n
        return None, None

    agent_n, agent_name = get_first_attr(head, agent_candidates)
    map_n, map_name = get_first_attr(head, map_candidates)
    ego_n, ego_name = get_first_attr(head, ego_candidates)

    # Fallbacks from embedding shapes
    if ego_n is None:
        ego_n = 1
        ego_name = "fallback_planning_query_token_num"

    if agent_n is None and hasattr(head, "agent_reference_points"):
        agent_n = head.agent_reference_points.weight.shape[0]
        agent_name = "agent_reference_points.weight.shape[0]"

    if map_n is None and hasattr(head, "map_reference_points"):
        map_n = head.map_reference_points.weight.shape[0]
        map_name = "map_reference_points.weight.shape[0]"

    #   agent_n = num_query - map_n
    if agent_n is not None and map_n is not None and agent_n > map_n:
        if hasattr(head, "num_query") and agent_name == "num_query":
            old_agent_n = agent_n
            agent_n = agent_n - map_n
            print(
                f"[fix] head.num_query appears to be agent+map total: "
                f"{old_agent_n} - map_n({map_n}) = agent_n({agent_n})"
            )

    # Your Zero Ego PE experiment uses N_mode ego queries.
    # If no attr exists, infer later as tail size = total - agent_n - map_n.
    print(f"[debug] query split attrs: agent={agent_n}({agent_name}), map={map_n}({map_name}), ego={ego_n}({ego_name})")
    return agent_n, map_n, ego_n

def make_transformer_input_capture_hook(captures: dict):
    """
    Capture task queries at pts_bbox_head.transformer forward input.

    This is more reliable for ego_query than decoder layer inputs, because
    ego_query may not be passed as a normal tensor into each decoder layer.
    """
    def hook_fn(module, inputs):
        assert isinstance(inputs, tuple), f"expected tuple inputs, got {type(inputs)}"

        # agent_query: arg0, [B, N_agent, D]
        if len(inputs) > AGENT_QUERY_ARG_IDX and torch.is_tensor(inputs[AGENT_QUERY_ARG_IDX]):
            captures["transformer_input_agent_query"] = (
                inputs[AGENT_QUERY_ARG_IDX].detach().float().cpu().numpy()
            )

        # map_query: arg1, [B, N_map, D]
        if len(inputs) > MAP_QUERY_ARG_IDX and torch.is_tensor(inputs[MAP_QUERY_ARG_IDX]):
            captures["transformer_input_map_query"] = (
                inputs[MAP_QUERY_ARG_IDX].detach().float().cpu().numpy()
            )

        # ego_query: arg2, [B, N_ego, D] or maybe [B, D]
        if len(inputs) > EGO_QUERY_ARG_IDX and torch.is_tensor(inputs[EGO_QUERY_ARG_IDX]):
            ego_q = inputs[EGO_QUERY_ARG_IDX]
            captures["transformer_input_ego_query"] = (
                ego_q.detach().float().cpu().numpy()
            )
            print(f"[debug] transformer input ego_query shape = {tuple(ego_q.shape)}")
        else:
            print("[warn] transformer input ego_query not found at arg_idx=2")

        return None

    return hook_fn

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

def make_query_mask_hook(arg_idx: int, target_name: str):
    """
    Return a forward_pre_hook that zeros one transformer input query.

    Example:
        arg_idx = 0 -> agent_query
        arg_idx = 1 -> map_query

    Note:
        This masks the initial task query content passed into transformer.
        It is a first-stage coarse ablation, not yet a perfect removal of
        all map/agent information.
    """
    def hook_fn(module, inputs):
        assert isinstance(inputs, tuple), f"expected tuple inputs, got {type(inputs)}"
        assert len(inputs) > arg_idx, f"inputs has only {len(inputs)} args, need idx {arg_idx}"

        q = inputs[arg_idx]
        assert torch.is_tensor(q), f"{target_name} is not tensor, got {type(q)}"

        print(f"[intervention] mask {target_name}: shape={tuple(q.shape)}")

        new_inputs = list(inputs)
        new_inputs[arg_idx] = torch.zeros_like(q)
        return tuple(new_inputs)

    return hook_fn

# ============================================================
# S4: Query Capture Hooks
# ============================================================

def tensor_from_layer_output(output):
    """
    Robustly extract the main query tensor from a decoder layer output.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (list, tuple)):
        for x in output:
            if torch.is_tensor(x) and x.dim() == 3:
                return x

    if isinstance(output, dict):
        for x in output.values():
            if torch.is_tensor(x) and x.dim() == 3:
                return x

    return None

def find_tensor_with_token_num(obj, token_num):
    """
    Recursively search tensor whose shape looks like:
        [B, token_num, D] or [token_num, B, D]
    """
    if token_num is None:
        return None

    if torch.is_tensor(obj) and obj.dim() == 3:
        if obj.shape[1] == token_num:
            return obj
        if obj.shape[0] == token_num:
            return obj.transpose(0, 1)

    if isinstance(obj, (list, tuple)):
        for x in obj:
            found = find_tensor_with_token_num(x, token_num)
            if found is not None:
                return found

    if isinstance(obj, dict):
        for x in obj.values():
            found = find_tensor_with_token_num(x, token_num)
            if found is not None:
                return found

    return None


def register_decoder_query_capture_hooks(base, agent_n=None, map_n=None, ego_n=None):
    """
    Capture per-layer decoder queries.

    DriveTransformerDecoderLayer.forward() returns a tuple:
        (agent_query[B,900,768], map_query[B,100,768], ego_query[B,1,768])

    We directly index output[0], output[1], output[2] for reliable capture.
    Fallback to find_tensor_with_token_num only if output is not a 3-tuple.
    """
    decoder = base.pts_bbox_head.transformer.decoder
    layers = decoder.layers

    captures = {
        "raw": {},
        "agent": {},
        "map": {},
        "ego": {},
    }
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            # DriveTransformerDecoderLayer returns (agent_query, map_query, ego_query)
            if isinstance(output, (list, tuple)) and len(output) >= 3:
                # -------- agent from output[0] --------
                q_agent = output[0]
                if torch.is_tensor(q_agent) and q_agent.dim() == 3:
                    q_agent_cpu = q_agent.detach().float().cpu()
                    captures["agent"][layer_idx] = q_agent_cpu
                    if layer_idx == 0:
                        print(f"[debug] layer {layer_idx} output agent_query shape = {tuple(q_agent_cpu.shape)}")

                # -------- map from output[1] --------
                q_map = output[1]
                if torch.is_tensor(q_map) and q_map.dim() == 3:
                    q_map_cpu = q_map.detach().float().cpu()
                    captures["map"][layer_idx] = q_map_cpu
                    if layer_idx == 0:
                        print(f"[debug] layer {layer_idx} output map_query shape = {tuple(q_map_cpu.shape)}")

                # -------- ego from output[2] --------
                q_ego = output[2]
                if torch.is_tensor(q_ego) and q_ego.dim() == 3:
                    q_ego_cpu = q_ego.detach().float().cpu()
                    captures["ego"][layer_idx] = q_ego_cpu
                    if layer_idx == 0:
                        print(f"[debug] layer {layer_idx} output ego_query shape = {tuple(q_ego_cpu.shape)}")

                # Also store raw (agent) for backward compat
                captures["raw"][layer_idx] = captures["agent"].get(layer_idx)

            else:
                # Fallback: output is not a 3-tuple (unexpected architecture)
                q_out = tensor_from_layer_output(output)
                if q_out is not None and q_out.dim() == 3:
                    q_cpu = q_out.detach().float().cpu()
                    captures["raw"][layer_idx] = q_cpu
                    if layer_idx == 0:
                        print(f"[debug] layer {layer_idx} output q shape = {tuple(q_cpu.shape)} (fallback)")

                # Try to find agent/map from inputs
                q_agent = find_tensor_with_token_num(inputs, agent_n)
                if q_agent is not None:
                    captures["agent"][layer_idx] = q_agent.detach().float().cpu()
                elif layer_idx == 0:
                    print(f"[warn] layer {layer_idx}: cannot find agent query (fallback)")

                q_map = find_tensor_with_token_num(inputs, map_n)
                if q_map is not None:
                    captures["map"][layer_idx] = q_map.detach().float().cpu()
                elif layer_idx == 0:
                    print(f"[warn] layer {layer_idx}: cannot find map query (fallback)")

                # ego not capturable in fallback path
                if layer_idx == 0:
                    print(f"[warn] layer {layer_idx}: cannot find ego query (output not a 3-tuple)")

        return hook_fn

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    return captures, handles

def stack_layer_dict(layer_dict):
    """
    Convert {layer_idx: [B, N, D]} to [L, B, N, D].

    If empty, return None.
    """
    if layer_dict is None or len(layer_dict) == 0:
        return None

    layers = sorted(layer_dict.keys())
    arrs = [layer_dict[l] for l in layers]
    return np.stack(arrs, axis=0)


# ============================================================
# S5: Experiment Runner
# ============================================================

@torch.no_grad()
def run_one_experiment(model, batch, name: str, shift_xyz: tuple, intervention: dict):
    """
    Run one G0-G5 intervention experiment.

    Collect:
      - traj: final-layer trajectory, [N_mode, T, 2]
      - traj_layers: all-layer trajectory, [N_layer, N_mode, T, 2]
      - ego_query / agent_query / map_query: stacked decoder-layer queries
      - legacy ego_query_layers / agent_query_layers / map_query_layers
    """
    base = model.module if hasattr(model, "module") else model
    head = base.pts_bbox_head
    target_module = head.transformer

    # Make each perturbation run independent.
    base.prev_scene_token = None
    head.reset_memory()

    agent_n, map_n, ego_n = infer_query_splits(head)

    # Clone batch so batch-level interventions do not contaminate later runs.
    run_batch = clone_batch_for_intervention(batch)

    intervention_type = intervention.get("type", "none")
    handles = []

    # 1) Batch-level intervention
    if intervention_type == "batch_zero":
        batch_key = intervention["batch_key"]
        run_batch = apply_batch_zero_intervention(run_batch, batch_key)

    # 2) ego_ref shift intervention
    # For G0/G2/G3/G4/G5 shift is usually zero, but using the hook is fine
    # because the hook also sanity-checks that ego_ref is really zero.
    if intervention_type == "ego_ref_shift" or shift_xyz != (0.0, 0.0, 0.0):
        hook = make_ego_ref_perturb_hook(shift_xyz)
        handles.append(target_module.register_forward_pre_hook(hook))

    # 3) Query-level intervention: mask agent_query or map_query at transformer input.
    if intervention_type == "query_mask":
        arg_idx = intervention["arg_idx"]
        target_name = intervention.get("target", f"arg_{arg_idx}")
        hook = make_query_mask_hook(arg_idx=arg_idx, target_name=target_name)
        handles.append(target_module.register_forward_pre_hook(hook))

    # 4) Capture transformer input queries, especially ego_query.
    transformer_input_captures = {}
    input_capture_hook = make_transformer_input_capture_hook(transformer_input_captures)
    handles.append(target_module.register_forward_pre_hook(input_capture_hook))

    # Register decoder query capture hooks
    query_captures, query_handles = register_decoder_query_capture_hooks(
        base, agent_n=agent_n, map_n=map_n, ego_n=ego_n
    )

    handles.extend(query_handles)

    try:
        captured = {}

        def capture_head_out(module, inputs, output):
            captured["out"] = output

        head_handle = head.register_forward_hook(capture_head_out)

        try:
            _ = model(run_batch, return_loss=False, rescale=True)
        finally:
            head_handle.remove()

        head_out = captured["out"]

        if isinstance(head_out, (list, tuple)):
            head_out = head_out[0] if isinstance(head_out[0], dict) else head_out

        assert isinstance(head_out, dict), \
            f"unexpected head_out type {type(head_out)}; inspect manually"

    finally:
        for h in handles:
            h.remove()

    # ---- Extract trajectory predictions ----
    # Generation chain for ego_fut_preds_fix_time:
    #   ego_query (from decoder output[2]) -> ego_planning_head (MLP)
    #   -> ego_fut_preds_fix_time [N_layer, B, N_mode, T, 2]
    #
    # The planning head decodes ego_query at each decoder layer into
    # multi-mode future trajectories (x,y offsets at T timesteps).
    # cls_scores rank the modes by confidence.
    # ---- Trajectory extraction ----
    # ego_fut_preds_fix_time is generated by:
    #   DriveTransformerHead.forward() -> DriveTransformerDecoder.forward()
    #   -> ego_query passes through N decoder layers -> ego_planning_head MLP
    #   -> output shape: [N_layer, B, N_mode, T, 2]  (x,y in ego frame)
    # See: drivetransformer_head.py, drivetransformer_decoder.py
    traj_all = head_out["ego_fut_preds_fix_time"]
    cls_all = head_out.get("ego_traj_cls_scores", None)

    # [N_layer, B, N_mode, T, 2] -> [N_layer, N_mode, T, 2]
    traj_layers = traj_all[:, 0].detach().float().cpu().numpy()
    traj_last = traj_layers[-1]

    if cls_all is None:
        best_mode = 0
        print("[warn] ego_traj_cls_scores is None; use best_mode=0 for plotting")
    else:
        cls_last = cls_all[-1, 0].detach().float().cpu().numpy()
        best_mode = int(np.argmax(cls_last))

    # Convert captured tensors to numpy dicts
    ego_query_layers = {
        k: v.numpy() for k, v in query_captures["ego"].items()
    }
    agent_query_layers = {
        k: v.numpy() for k, v in query_captures["agent"].items()
    }
    map_query_layers = {
        k: v.numpy() for k, v in query_captures["map"].items()
    }

    print(
        f"[debug] captured decoder layers: "
        f"ego={sorted(ego_query_layers.keys())}, "
        f"agent={sorted(agent_query_layers.keys())}, "
        f"map={sorted(map_query_layers.keys())}"
    )

    # These stacked versions are for planner_sensitivity_report.py
    ego_query = stack_layer_dict(ego_query_layers)
    agent_query = stack_layer_dict(agent_query_layers)
    map_query = stack_layer_dict(map_query_layers)

    # Fallback: ego_query is often not present in decoder layer inputs.
    # Use transformer input ego_query so fig_D1 can still summarize ego-query sensitivity.
    ego_query_is_layerwise = ego_query is not None
    ego_query_source = "decoder_layer_output[2]" if ego_query_is_layerwise else None

    if ego_query is None:
        ego_query = transformer_input_captures.get("transformer_input_ego_query", None)
        if ego_query is not None:
            ego_query_source = "transformer_input_fallback"
            print(f"[fix] use transformer input ego_query as fallback: shape={ego_query.shape}")

    # Transformer input ego_query (always captured for reference)
    transformer_input_ego_query = transformer_input_captures.get("transformer_input_ego_query", None)

    return {
        # New report-compatible keys
        "traj": traj_last,                  # [N_mode, T, 2]
        "ego_query": ego_query,             # [L, B, N_ego, D] or None
        "agent_query": agent_query,         # [L, B, N_agent, D] or None
        "map_query": map_query,             # [L, B, N_map, D] or None
        "ego_query_is_layerwise": ego_query_is_layerwise,
        "ego_query_source": ego_query_source,
        "transformer_input_ego_query": transformer_input_ego_query,
        "meta": {
            "label": dict((e[0], e[3]) for e in EXPERIMENTS).get(name, name),
            "color": dict((e[0], e[2]) for e in EXPERIMENTS).get(name, None),
            "perturb_type": intervention_type,
        },

        # Legacy keys for old figures
        "traj_layers": traj_layers,         # [N_layer, N_mode, T, 2]
        "best_mode": best_mode,
        "shift": shift_xyz,
        "ego_query_layers": ego_query_layers,
        "agent_query_layers": agent_query_layers,
        "map_query_layers": map_query_layers,
    }

def run_all_experiments(model, batch):
    """
    Loop over G0-G5 experiments, return dict {group_name: result_dict}.
    """
    results = {}

    for name, shift, color, label in EXPERIMENTS:
        intervention = INTERVENTIONS.get(name, {"type": "none"})

        print("=" * 80)
        print(f"[exp] running {name}")
        print(f"      label={label}")
        print(f"      shift={shift}")
        print(f"      intervention={intervention}")

        out = run_one_experiment(
            model=model,
            batch=batch,
            name=name,
            shift_xyz=shift,
            intervention=intervention,
        )

        # Make sure meta is aligned with EXPERIMENTS.
        out["meta"] = {
            "label": label,
            "color": color,
            "perturb_type": intervention.get("type", "none"),
        }
        out["shift"] = shift

        results[name] = out

        # Sanity-check logging
        _eq = out['ego_query']
        _aq = out['agent_query']
        _mq = out['map_query']
        _tieq = out.get('transformer_input_ego_query')
        print(
            f"[done] {name}:\n"
            f"  traj={out['traj'].shape}\n"
            f"  traj_layers={out['traj_layers'].shape}\n"
            f"  layerwise_ego_query={None if _eq is None else _eq.shape}, "
            f"source={out.get('ego_query_source', 'N/A')}\n"
            f"  layerwise_agent_query={None if _aq is None else _aq.shape}\n"
            f"  layerwise_map_query={None if _mq is None else _mq.shape}\n"
            f"  transformer_input_ego_query={None if _tieq is None else _tieq.shape}"
        )

    return results

# ============================================================
# S8: Legacy Plots
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

def plot_traj_shift_by_layer(results: dict, save_path: Path):
    """
    Fig2:
      x: decoder layer index
      y: ||traj_perturbed - traj_baseline||

    Norm over all modes, timesteps, xy.
    """
    base_name = "G0_baseline"
    base_traj = results[base_name]["traj_layers"]  # [L, N_mode, T, 2]
    L = base_traj.shape[0]
    xs = np.arange(L)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, shift, color, label in EXPERIMENTS:
        traj = results[name]["traj_layers"]

        # layer-wise Frobenius norm
        vals = []
        for l in range(L):
            diff = traj[l] - base_traj[l]
            vals.append(np.linalg.norm(diff))

        ax.plot(xs, vals, marker="o", color=color, label=label)

    ax.set_title("Fig2: trajectory drift vs decoder layer")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(r"$||traj_{perturbed} - traj_{baseline}||_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_ego_query_drift_by_layer(results: dict, save_path: Path):
    """
    Fig3:
      x: decoder layer index
      y: ||ego_query_perturbed - ego_query_baseline||

    Norm over batch, ego modes, embedding dim.
    """
    base_name = "G0_baseline"
    base_q = results[base_name]["ego_query_layers"]

    common_layers = sorted(base_q.keys())
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, shift, color, label in EXPERIMENTS:
        q_dict = results[name]["ego_query_layers"]
        layers = [l for l in common_layers if l in q_dict]

        vals = []
        for l in layers:
            diff = q_dict[l] - base_q[l]
            vals.append(np.linalg.norm(diff))

        ax.plot(layers, vals, marker="o", color=color, label=label)

    ax.set_title("Fig3: ego query embedding drift vs decoder layer")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(r"$||ego\_query_{perturbed} - ego\_query_{baseline}||_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def ego_agent_relation_distribution(ego_q, agent_q, temperature=1.0):
    """
    ego_q:   [B, N_ego, D]
    agent_q: [B, N_agent, D]

    Return:
        prob: [B, N_ego, N_agent]
    """
    eps = 1e-8
    ego = ego_q / (np.linalg.norm(ego_q, axis=-1, keepdims=True) + eps)
    agent = agent_q / (np.linalg.norm(agent_q, axis=-1, keepdims=True) + eps)

    # [B, N_ego, D] x [B, D, N_agent] -> [B, N_ego, N_agent]
    sim = np.matmul(ego, np.swapaxes(agent, -1, -2)) / temperature
    prob = softmax_np(sim, axis=-1)
    return prob


def kl_divergence_np(p, q, eps=1e-8):
    """
    Mean KL(p || q) over B and ego modes.
    p/q: [B, N_ego, N_agent]
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)  # [B, N_ego]
    return float(np.mean(kl))


def plot_ego_agent_relation_shift_by_layer(results: dict, save_path: Path):
    """
    Fig4:
      x: decoder layer index
      y: KL between perturbed and baseline ego-agent relation distributions.

    This is a proxy, not raw attention weights.
    """
    base_name = "G0_baseline"
    base_ego = results[base_name]["ego_query_layers"]
    base_agent = results[base_name]["agent_query_layers"]

    common_layers = sorted(set(base_ego.keys()) & set(base_agent.keys()))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, shift, color, label in EXPERIMENTS:
        ego_dict = results[name]["ego_query_layers"]
        agent_dict = results[name]["agent_query_layers"]

        layers = [
            l for l in common_layers
            if l in ego_dict and l in agent_dict
        ]

        vals = []
        for l in layers:
            p_base = ego_agent_relation_distribution(base_ego[l], base_agent[l])
            p_pert = ego_agent_relation_distribution(ego_dict[l], agent_dict[l])
            vals.append(kl_divergence_np(p_pert, p_base))

        ax.plot(layers, vals, marker="o", color=color, label=label)

    ax.set_title("Fig4: ego-agent relation redistribution vs decoder layer")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("KL(perturbed || baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def ego_self_relation_distribution(ego_q, temperature=1.0):
    """
    ego_q: [B, N_ego, D]
    Return:
        prob: [B, N_ego, N_ego]
    """
    eps = 1e-8
    ego = ego_q / (np.linalg.norm(ego_q, axis=-1, keepdims=True) + eps)
    sim = np.matmul(ego, np.swapaxes(ego, -1, -2)) / temperature
    prob = softmax_np(sim, axis=-1)
    return prob


def plot_ego_self_relation_shift_by_layer(results: dict, save_path: Path):
    """
    Fallback Fig4:
      x: decoder layer index
      y: KL between perturbed and baseline ego-mode relation distributions.

    This is not ego-agent attention.
    It measures whether Zero Ego PE changes relation among ego modes.
    """
    base_name = "G0_baseline"
    base_ego = results[base_name]["ego_query_layers"]

    common_layers = sorted(base_ego.keys())
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, shift, color, label in EXPERIMENTS:
        ego_dict = results[name]["ego_query_layers"]
        layers = [l for l in common_layers if l in ego_dict]

        vals = []
        for l in layers:
            p_base = ego_self_relation_distribution(base_ego[l])
            p_pert = ego_self_relation_distribution(ego_dict[l])
            vals.append(kl_divergence_np(p_pert, p_base))

        ax.plot(layers, vals, marker="o", color=color, label=label)

    ax.set_title("Fig4 fallback: ego-mode relation redistribution vs decoder layer")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("KL(perturbed || baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# S9: Main
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

    print("[3.5/4] generating planner sensitivity report ...")
    sensitivity_out_dir = OUTPUT_DIR / "planner_sensitivity_g0_g5"

    rows, metrics_by_name = run_basic_sensitivity_report(
        results=results,
        out_dir=str(sensitivity_out_dir),
        baseline_key="G0_baseline",
        mode_idx=0,
        reduce_mode="select",
        sort_by="final_point_error",
        attach_meta=True,
        save_results_pickle=True,
        make_query_drift=True,
    )

    print(f"[report] saved planner sensitivity report -> {sensitivity_out_dir}")

    print("[4/4] plotting ...")
    plot_dual_view(results, gt_info, OUTPUT_FIG1)
    print(f"saved -> {OUTPUT_FIG1}")

    plot_traj_shift_by_layer(results, OUTPUT_FIG2)
    print(f"saved -> {OUTPUT_FIG2}")

    # Guard: only plot layerwise ego_query drift if capture was from decoder layers
    baseline_ego_layerwise = results.get("G0_baseline", {}).get("ego_query_is_layerwise", False)
    if baseline_ego_layerwise:
        plot_ego_query_drift_by_layer(results, OUTPUT_FIG3)
        print(f"saved -> {OUTPUT_FIG3}")
    else:
        print(f"[SKIP] fig3 ego_query drift: ego_query_is_layerwise=False")

    # plot_ego_agent_relation_shift_by_layer(results, OUTPUT_FIG4)
    plot_ego_self_relation_shift_by_layer(results, OUTPUT_FIG4)
    print(f"saved -> {OUTPUT_FIG4}")


if __name__ == "__main__":
    main()