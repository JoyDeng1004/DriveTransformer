"""
TODO 4.5: Ego-position carrier probe for DriveTransformer.

Goal
----
Probe which ego-related variable is a plausible carrier for "ego position / ego offset"
information in the planner path.

This script intentionally separates two kinds of interventions:

1. Meter-space coordinate shift
   - Only valid for coordinate-like tensors with last dim == 3, e.g. ego_ref.

2. Embedding-space ablation
   - Valid for token / PE tensors such as ego_query, ego_temp_pos, ego_pose_pe.
   - We zero them out instead of adding "+2m", because they are not meter-space tensors.

Experiments
-----------
H0_baseline
H1_ego_ref_x2
H2_ego_query_zero
H3_ego_temp_pos_zero
H4_ego_pose_pe_zero

Typical usage
-------------
Put this file in the same directory as ego_ref_shift_diag.py and run:

python ego_position_carrier_probe.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --ckpt /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --sample-idxs 0:50:1 \
  --device cuda:0 \
  --out-dir /gs/bs/tga-RLA/qdeng/DriveTransformer/pe_diagnosis/ego_position_carrier_probe

Outputs
-------
out_dir/
  samples/sample_000000/
    results.pkl
    sensitivity_summary.csv/json
    carrier_audit.csv/json
    fig_A/B/C...
  all_sample_sensitivity_summary.csv
  aggregate_summary.csv/json
  rank1_frequency.csv
  rank1_by_sample.csv
  pairwise_stability.csv
  per_sample_heatmap_final_point_error.png
  multi_sample_bar_mean_std.png
  rank_frequency_bar.png
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Make local imports robust when this script is launched from another cwd.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ego_ref_shift_diag import (  # noqa: E402
    load_model,
    load_sample,
    clone_batch_for_intervention,
)
from planner_sensitivity_single import run_basic_sensitivity_report  # noqa: E402

# ============================================================
# S0. Transformer argument positions
# ============================================================

# Confirmed from ego_ref_shift_diag.py / drivetransformer_head.py call:
# transformer(agent_query, map_query, ego_query, img_feats,
#             img_pos_embed, agent_temp_memory, agent_temp_pos,
#             map_temp_memory, map_temp_pos, ego_memory_embedding,
#             ego_temp_pos, agent_prep_ref, map_prep_ref,
#             map_prep_pts_coord, ego_ref, ...)
AGENT_QUERY_ARG_IDX = 0
MAP_QUERY_ARG_IDX = 1
EGO_QUERY_ARG_IDX = 2
EGO_TEMP_POS_ARG_IDX = 10
EGO_REF_ARG_IDX = 14

BASELINE_KEY = "H0_baseline"
DEFAULT_RANK_METRIC = "final_point_error"

# name, kind, color, label
PROBE_EXPERIMENTS = [
    ("H0_baseline", "none", "black", "H0 baseline"),
    ("H1_ego_ref_x2", "ego_ref_shift_x2", "tab:blue", "H1 ego_ref x+2m"),
    ("H2_ego_query_zero", "ego_query_zero", "tab:orange", "H2 ego_query zero"),
    ("H3_ego_temp_pos_zero", "ego_temp_pos_zero", "tab:green", "H3 ego_temp_pos zero"),
    ("H4_ego_pose_pe_zero", "ego_pose_pe_zero", "tab:red", "H4 ego_pose_pe zero"),
]

INTERVENTION_SHORT = {
    "H0_baseline": "baseline",
    "H1_ego_ref_x2": "ego_ref_x+2m",
    "H2_ego_query_zero": "ego_query_zero",
    "H3_ego_temp_pos_zero": "ego_temp_pos_zero",
    "H4_ego_pose_pe_zero": "ego_pose_pe_zero",
}

PLOT_ORDER = [
    "H1_ego_ref_x2",
    "H2_ego_query_zero",
    "H3_ego_temp_pos_zero",
    "H4_ego_pose_pe_zero",
]


# ============================================================
# S1. General utilities
# ============================================================

def parse_sample_indices(s: str) -> List[int]:
    """
    Parse sample index spec.

    Supported forms:
      - "96"
      - "0,5,10"
      - "0:50:1"
      - "0:50"  # step defaults to 1
    """
    s = str(s).strip()
    if not s:
        raise ValueError("empty --sample-idxs")

    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"bad range spec: {s}")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, end, step))

    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    return [int(s)]


def ensure_jsonable(x: Any) -> Any:
    """Convert numpy scalars/arrays and paths to JSON-serializable objects."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): ensure_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ensure_jsonable(v) for v in x]
    return x


def nested_zero_like(x: Any) -> Any:
    """Zero a nested output structure while preserving structure."""
    if torch.is_tensor(x):
        return torch.zeros_like(x)
    if isinstance(x, tuple):
        return tuple(nested_zero_like(v) for v in x)
    if isinstance(x, list):
        return [nested_zero_like(v) for v in x]
    if isinstance(x, dict):
        return {k: nested_zero_like(v) for k, v in x.items()}
    return x


def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def tensor_stats_np(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    flat = arr.reshape(-1) if arr.size > 0 else arr
    out = {
        "shape": str(tuple(arr.shape)),
        "ndim": int(arr.ndim),
        "numel": int(arr.size),
        "mean": float(np.nanmean(flat)) if arr.size > 0 else np.nan,
        "std": float(np.nanstd(flat)) if arr.size > 0 else np.nan,
        "min": float(np.nanmin(flat)) if arr.size > 0 else np.nan,
        "max": float(np.nanmax(flat)) if arr.size > 0 else np.nan,
        "l2_norm": float(np.linalg.norm(flat)) if arr.size > 0 else np.nan,
        "nan_count": int(np.isnan(flat).sum()) if arr.size > 0 else 0,
        "is_coordinate_like": bool(arr.ndim >= 2 and arr.shape[-1] in (2, 3)),
        "can_meter_shift": bool(arr.ndim >= 2 and arr.shape[-1] == 3),
    }
    return out


def append_tensor_audit(
    audit_rows: List[Dict[str, Any]],
    experiment: str,
    variable: str,
    value: torch.Tensor | np.ndarray,
    where: str,
    note: str = "",
) -> None:
    arr = detach_to_numpy(value) if torch.is_tensor(value) else np.asarray(value)
    row = {
        "experiment": experiment,
        "intervention_short": INTERVENTION_SHORT.get(experiment, experiment),
        "variable": variable,
        "where": where,
        "note": note,
    }
    row.update(tensor_stats_np(arr))
    audit_rows.append(row)


# ============================================================
# S2. Hooks
# ============================================================

def make_transformer_arg_zero_hook(arg_idx: int, name: str):
    """
    Zero one positional transformer argument.

    Use this for embedding-space tensors such as ego_query and ego_temp_pos.
    Do NOT interpret this as a geometric +2m shift.
    """
    def hook_fn(module, inputs):
        assert isinstance(inputs, tuple), f"expected tuple inputs, got {type(inputs)}"
        assert len(inputs) > arg_idx, f"inputs has only {len(inputs)} args, need idx {arg_idx}"

        x = inputs[arg_idx]
        assert torch.is_tensor(x), f"{name} is not tensor, got {type(x)}"

        new_inputs = list(inputs)
        new_inputs[arg_idx] = torch.zeros_like(x)
        print(f"[intervention] zero {name}: arg_idx={arg_idx}, shape={tuple(x.shape)}")
        return tuple(new_inputs)

    return hook_fn


def make_transformer_arg_shift_hook(arg_idx: int, name: str, shift_xyz: Tuple[float, float, float]):
    """
    Shift a coordinate-like transformer argument by a meter-space xyz offset.

    Safety rule:
      Only tensors with shape [..., 3] can be shifted in meters.
    """
    def hook_fn(module, inputs):
        assert isinstance(inputs, tuple), f"expected tuple inputs, got {type(inputs)}"
        assert len(inputs) > arg_idx, f"inputs has only {len(inputs)} args, need idx {arg_idx}"

        x = inputs[arg_idx]
        assert torch.is_tensor(x), f"{name} is not tensor, got {type(x)}"
        if not (x.dim() >= 2 and x.shape[-1] == 3):
            raise RuntimeError(
                f"Refuse to meter-shift {name}: shape={tuple(x.shape)} is not coordinate-like [..., 3]. "
                f"Use zero-out / upstream-coordinate perturbation instead."
            )

        shift = torch.tensor(shift_xyz, dtype=x.dtype, device=x.device)
        view_shape = [1] * (x.dim() - 1) + [3]
        new_x = x + shift.view(*view_shape)

        new_inputs = list(inputs)
        new_inputs[arg_idx] = new_x
        print(f"[intervention] shift {name}: arg_idx={arg_idx}, shift={shift_xyz}, shape={tuple(x.shape)}")
        return tuple(new_inputs)

    return hook_fn


def make_transformer_carrier_capture_hook(
    captures: Dict[str, np.ndarray],
    audit_rows: List[Dict[str, Any]],
    experiment: str,
):
    """
    Capture ego-related transformer inputs after preceding pre-hooks.

    Register intervention hooks first, then this capture hook. That way the audit
    records the effective tensor that the transformer will receive.
    """
    def hook_fn(module, inputs):
        assert isinstance(inputs, tuple), f"expected tuple inputs, got {type(inputs)}"

        to_capture = [
            ("transformer_input_ego_query", EGO_QUERY_ARG_IDX, "transformer.arg2"),
            ("transformer_input_ego_temp_pos", EGO_TEMP_POS_ARG_IDX, "transformer.arg10"),
            ("transformer_input_ego_ref", EGO_REF_ARG_IDX, "transformer.arg14"),
        ]

        for var_name, idx, where in to_capture:
            if len(inputs) <= idx or not torch.is_tensor(inputs[idx]):
                print(f"[warn] {var_name} not found at arg_idx={idx}")
                audit_rows.append({
                    "experiment": experiment,
                    "intervention_short": INTERVENTION_SHORT.get(experiment, experiment),
                    "variable": var_name,
                    "where": where,
                    "note": f"missing at arg_idx={idx}",
                })
                continue

            x = inputs[idx]
            captures[var_name] = detach_to_numpy(x)
            append_tensor_audit(
                audit_rows=audit_rows,
                experiment=experiment,
                variable=var_name,
                value=x,
                where=where,
                note="effective transformer input after pre-hooks",
            )
            print(
                f"[capture] {var_name}: shape={tuple(x.shape)}, "
                f"mean={x.float().mean().item():.6f}, std={x.float().std().item():.6f}, "
                f"min={x.float().min().item():.6f}, max={x.float().max().item():.6f}"
            )

        return None

    return hook_fn


def register_ego_pose_pe_hook(
    head: torch.nn.Module,
    captures: Dict[str, np.ndarray],
    audit_rows: List[Dict[str, Any]],
    experiment: str,
    zero_output: bool,
):
    """
    Register a forward hook on head.ego_pose_pe if it exists and is an nn.Module.

    If zero_output=True, the hook returns zeros with the same nested structure.
    This is an embedding-space ablation, not a meter-space shift.
    """
    if not hasattr(head, "ego_pose_pe"):
        print("[warn] head has no attribute ego_pose_pe; skip ego_pose_pe hook")
        audit_rows.append({
            "experiment": experiment,
            "intervention_short": INTERVENTION_SHORT.get(experiment, experiment),
            "variable": "ego_pose_pe",
            "where": "head.ego_pose_pe",
            "note": "missing attribute",
        })
        return None

    module = getattr(head, "ego_pose_pe")
    if not isinstance(module, torch.nn.Module):
        print(f"[warn] head.ego_pose_pe is not nn.Module: {type(module)}; skip hook")
        audit_rows.append({
            "experiment": experiment,
            "intervention_short": INTERVENTION_SHORT.get(experiment, experiment),
            "variable": "ego_pose_pe",
            "where": "head.ego_pose_pe",
            "note": f"attribute exists but is not nn.Module: {type(module)}",
        })
        return None

    def hook_fn(mod, inputs, output):
        # Capture inputs if tensor-like.
        if isinstance(inputs, tuple):
            for i, x in enumerate(inputs):
                if torch.is_tensor(x):
                    key = f"ego_pose_pe_input_{i}"
                    captures[key] = detach_to_numpy(x)
                    append_tensor_audit(
                        audit_rows=audit_rows,
                        experiment=experiment,
                        variable=key,
                        value=x,
                        where=f"head.ego_pose_pe.input[{i}]",
                        note="input to ego_pose_pe module",
                    )

        # Capture output.
        if torch.is_tensor(output):
            captures["ego_pose_pe_output"] = detach_to_numpy(output)
            append_tensor_audit(
                audit_rows=audit_rows,
                experiment=experiment,
                variable="ego_pose_pe_output",
                value=output,
                where="head.ego_pose_pe.output",
                note="output before optional zeroing",
            )
            print(
                f"[capture] ego_pose_pe output: shape={tuple(output.shape)}, "
                f"mean={output.float().mean().item():.6f}, std={output.float().std().item():.6f}"
            )
        elif isinstance(output, (tuple, list)):
            for i, x in enumerate(output):
                if torch.is_tensor(x):
                    key = f"ego_pose_pe_output_{i}"
                    captures[key] = detach_to_numpy(x)
                    append_tensor_audit(
                        audit_rows=audit_rows,
                        experiment=experiment,
                        variable=key,
                        value=x,
                        where=f"head.ego_pose_pe.output[{i}]",
                        note="output before optional zeroing",
                    )

        if zero_output:
            print("[intervention] zero ego_pose_pe output")
            return nested_zero_like(output)

        return None

    return module.register_forward_hook(hook_fn)


# ============================================================
# S3. Forward runner
# ============================================================

@torch.no_grad()
def run_one_probe_experiment(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    name: str,
    kind: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one H0-H4 carrier probe experiment."""
    base = model.module if hasattr(model, "module") else model
    head = base.pts_bbox_head
    target_module = head.transformer

    # Keep every intervention independent, matching the G0-G5 sensitivity protocol.
    if hasattr(base, "prev_scene_token"):
        base.prev_scene_token = None
    if hasattr(head, "reset_memory"):
        head.reset_memory()

    run_batch = clone_batch_for_intervention(batch)
    handles = []
    captures: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, Any]] = []

    # 1) Register intervention hooks first.
    if kind == "none":
        pass
    elif kind == "ego_ref_shift_x2":
        handles.append(
            target_module.register_forward_pre_hook(
                make_transformer_arg_shift_hook(EGO_REF_ARG_IDX, "ego_ref", (2.0, 0.0, 0.0))
            )
        )
    elif kind == "ego_query_zero":
        handles.append(
            target_module.register_forward_pre_hook(
                make_transformer_arg_zero_hook(EGO_QUERY_ARG_IDX, "ego_query")
            )
        )
    elif kind == "ego_temp_pos_zero":
        handles.append(
            target_module.register_forward_pre_hook(
                make_transformer_arg_zero_hook(EGO_TEMP_POS_ARG_IDX, "ego_temp_pos")
            )
        )
    elif kind == "ego_pose_pe_zero":
        h = register_ego_pose_pe_hook(
            head=head,
            captures=captures,
            audit_rows=audit_rows,
            experiment=name,
            zero_output=True,
        )
        if h is not None:
            handles.append(h)
    else:
        raise ValueError(f"Unknown probe kind: {kind}")

    # 2) Always capture ego_pose_pe for baseline / other experiments too, if possible.
    if kind != "ego_pose_pe_zero":
        h = register_ego_pose_pe_hook(
            head=head,
            captures=captures,
            audit_rows=audit_rows,
            experiment=name,
            zero_output=False,
        )
        if h is not None:
            handles.append(h)

    # 3) Capture transformer effective inputs after intervention pre-hooks.
    handles.append(
        target_module.register_forward_pre_hook(
            make_transformer_carrier_capture_hook(captures, audit_rows, experiment=name)
        )
    )

    captured_head = {}

    def capture_head_out(module, inputs, output):
        captured_head["out"] = output

    head_handle = head.register_forward_hook(capture_head_out)
    handles.append(head_handle)

    try:
        _ = model(run_batch, return_loss=False, rescale=True)
        head_out = captured_head.get("out")
        if isinstance(head_out, (list, tuple)):
            head_out = head_out[0] if len(head_out) > 0 and isinstance(head_out[0], dict) else head_out

        if not isinstance(head_out, dict):
            raise RuntimeError(f"unexpected head_out type={type(head_out)}; cannot extract trajectory")

        traj_all = head_out["ego_fut_preds_fix_time"]
        cls_all = head_out.get("ego_traj_cls_scores", None)

        # [N_layer, B, N_mode, T, 2] -> [N_layer, N_mode, T, 2]
        traj_layers = traj_all[:, 0].detach().float().cpu().numpy()
        traj_last = traj_layers[-1]

        if cls_all is None:
            best_mode = 0
        else:
            cls_last = cls_all[-1, 0].detach().float().cpu().numpy()
            best_mode = int(np.argmax(cls_last))

        color = dict((x[0], x[2]) for x in PROBE_EXPERIMENTS).get(name, None)
        label = dict((x[0], x[3]) for x in PROBE_EXPERIMENTS).get(name, name)

        result = {
            "traj": traj_last,
            "traj_layers": traj_layers,
            "best_mode": best_mode,
            "meta": {
                "label": label,
                "color": color,
                "perturb_type": kind,
                "intervention_short": INTERVENTION_SHORT.get(name, name),
            },
            "shift": (2.0, 0.0, 0.0) if kind == "ego_ref_shift_x2" else (0.0, 0.0, 0.0),
            # Report-compatible optional query fields.
            "ego_query": captures.get("transformer_input_ego_query"),
            "transformer_input_ego_query": captures.get("transformer_input_ego_query"),
            "transformer_input_ego_temp_pos": captures.get("transformer_input_ego_temp_pos"),
            "transformer_input_ego_ref": captures.get("transformer_input_ego_ref"),
            "ego_pose_pe_output": captures.get("ego_pose_pe_output"),
            "carrier_captures": captures,
        }

        print(
            f"[done] {name}: traj={traj_last.shape}, best_mode={best_mode}, "
            f"captured={sorted(captures.keys())}"
        )
        return result, audit_rows

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


@torch.no_grad()
def run_probe_for_one_sample(
    model: torch.nn.Module,
    config_path: str,
    sample_idx: int,
    sample_out_dir: Path,
    device: str,
    reduce_mode: str = "select",
    mode_idx: int = 0,
    make_query_drift: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load one sample, run H0-H4, save report and carrier audit."""
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"[sample] {sample_idx} -> {sample_out_dir}")
    batch, _gt_info = load_sample(config_path, sample_idx, device)

    results: Dict[str, Any] = {}
    all_audit_rows: List[Dict[str, Any]] = []

    for name, kind, _color, label in PROBE_EXPERIMENTS:
        print("=" * 80)
        print(f"[probe] running {name}: {label}, kind={kind}")
        result, audit_rows = run_one_probe_experiment(
            model=model,
            batch=batch,
            name=name,
            kind=kind,
        )
        results[name] = result
        for row in audit_rows:
            row["sample_idx"] = sample_idx
        all_audit_rows.extend(audit_rows)

    # Save raw probe results explicitly. run_basic_sensitivity_report may also save results.pkl;
    # this one is kept to make the script self-contained.
    with open(sample_out_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    audit_df = pd.DataFrame(all_audit_rows)
    audit_df.to_csv(sample_out_dir / "carrier_audit.csv", index=False)
    with open(sample_out_dir / "carrier_audit.json", "w") as f:
        json.dump(ensure_jsonable(all_audit_rows), f, indent=2)

    rows, _metrics_by_name = run_basic_sensitivity_report(
        results=results,
        out_dir=str(sample_out_dir),
        baseline_key=BASELINE_KEY,
        mode_idx=mode_idx,
        reduce_mode=reduce_mode,
        sort_by=DEFAULT_RANK_METRIC,
        attach_meta=True,
        save_results_pickle=True,
        make_query_drift=make_query_drift,
    )
    print(f"[report] saved sample report -> {sample_out_dir}")

    return results, audit_df


# ============================================================
# S4. Aggregation and global plots
# ============================================================

def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make intervention columns robust across planner_sensitivity_report versions."""
    df = df.copy()

    if "intervention_key" not in df.columns:
        # Common fallback names.
        for c in ["name", "key", "experiment"]:
            if c in df.columns:
                df["intervention_key"] = df[c].astype(str)
                break
    if "intervention_key" not in df.columns:
        raise RuntimeError(f"Cannot find intervention key column in: {list(df.columns)}")

    if "intervention_short" not in df.columns:
        df["intervention_short"] = df["intervention_key"].map(INTERVENTION_SHORT).fillna(df["intervention_key"])

    df["intervention_key"] = df["intervention_key"].astype(str)
    df["intervention_short"] = df["intervention_short"].astype(str)
    return df


def collect_sample_summaries(out_root: Path) -> pd.DataFrame:
    rows = []
    sample_root = out_root / "samples"
    for csv_path in sorted(sample_root.glob("sample_*/sensitivity_summary.csv")):
        sample_dir = csv_path.parent
        sample_idx = int(sample_dir.name.replace("sample_", ""))
        df = pd.read_csv(csv_path)
        df = _normalize_summary_df(df)
        df["sample_idx"] = sample_idx
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No sensitivity_summary.csv found under {sample_root}")

    return pd.concat(rows, ignore_index=True)


def collect_carrier_audits(out_root: Path) -> Optional[pd.DataFrame]:
    rows = []
    for csv_path in sorted((out_root / "samples").glob("sample_*/carrier_audit.csv")):
        df = pd.read_csv(csv_path)
        rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def add_rank_columns(df: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    df = df.copy()
    if rank_metric not in df.columns:
        raise RuntimeError(f"rank_metric={rank_metric} not found. columns={list(df.columns)}")

    df["is_baseline"] = df["intervention_key"].eq(BASELINE_KEY)
    df["rank_by_metric"] = np.nan
    valid = ~df["is_baseline"]
    df.loc[valid, "rank_by_metric"] = (
        df.loc[valid]
        .groupby("sample_idx")[rank_metric]
        .rank(method="min", ascending=False)
    )
    return df


def compute_aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in ["mean_point_error", "final_point_error", "max_point_error", "total_l2"]
        if c in df.columns
    ]
    if not metric_cols:
        raise RuntimeError("No expected metric columns found in sensitivity summary")

    valid = df[~df["intervention_key"].eq(BASELINE_KEY)].copy()
    valid["intervention_key"] = valid["intervention_key"].astype(str)
    valid["intervention_short"] = valid["intervention_short"].astype(str)

    agg = (
        valid
        .groupby(["intervention_key", "intervention_short"], observed=True)[metric_cols]
        .agg(["mean", "std", "median", "count"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    return agg


def compute_rank1_frequency(df: pd.DataFrame, rank_metric: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[~df["intervention_key"].eq(BASELINE_KEY)].copy()
    idx = valid.groupby("sample_idx")[rank_metric].idxmax()
    rank1 = valid.loc[idx, ["sample_idx", "intervention_key", "intervention_short", rank_metric]].copy()

    total = rank1["sample_idx"].nunique()
    freq = (
        rank1.groupby(["intervention_key", "intervention_short"], observed=True)
        .size()
        .reset_index(name="rank1_count")
    )
    freq["rank1_ratio"] = freq["rank1_count"] / max(total, 1)

    # Include zero-frequency interventions for readability.
    all_interventions = [k for k in INTERVENTION_SHORT.keys() if k != BASELINE_KEY]
    full = pd.DataFrame({"intervention_key": all_interventions})
    full["intervention_short"] = full["intervention_key"].map(INTERVENTION_SHORT)
    freq = full.merge(freq, on=["intervention_key", "intervention_short"], how="left")
    freq["rank1_count"] = freq["rank1_count"].fillna(0).astype(int)
    freq["rank1_ratio"] = freq["rank1_ratio"].fillna(0.0)
    return freq, rank1


def compute_pairwise_stability(df: pd.DataFrame, rank_metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="sample_idx",
        columns="intervention_key",
        values=rank_metric,
        aggfunc="first",
    )

    comparisons = [
        ("H1_ego_ref_x2", "H2_ego_query_zero"),
        ("H1_ego_ref_x2", "H3_ego_temp_pos_zero"),
        ("H1_ego_ref_x2", "H4_ego_pose_pe_zero"),
    ]

    rows = []
    for weak_key, strong_key in comparisons:
        if weak_key not in pivot.columns or strong_key not in pivot.columns:
            rows.append({
                "comparison": f"{INTERVENTION_SHORT.get(weak_key, weak_key)} < {INTERVENTION_SHORT.get(strong_key, strong_key)}",
                "weak_key": weak_key,
                "strong_key": strong_key,
                "true_count": 0,
                "total": 0,
                "ratio": np.nan,
                "note": "missing one or both columns",
            })
            continue
        valid = pivot[[weak_key, strong_key]].dropna()
        true_count = int((valid[weak_key] < valid[strong_key]).sum())
        total = int(len(valid))
        rows.append({
            "comparison": f"{INTERVENTION_SHORT.get(weak_key, weak_key)} < {INTERVENTION_SHORT.get(strong_key, strong_key)}",
            "weak_key": weak_key,
            "strong_key": strong_key,
            "true_count": true_count,
            "total": total,
            "ratio": true_count / total if total > 0 else np.nan,
            "note": "",
        })

    return pd.DataFrame(rows)


def plot_multi_sample_bar_mean_std(agg: pd.DataFrame, out_root: Path, metric: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    plot_df = agg.set_index("intervention_key").reindex(PLOT_ORDER).reset_index()
    plot_df = plot_df.dropna(subset=[mean_col])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot_df))
    y = plot_df[mean_col].to_numpy(dtype=float)
    yerr = plot_df[std_col].fillna(0).to_numpy(dtype=float)
    labels = plot_df["intervention_short"].tolist()

    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Ego-position carrier probe: {metric} mean ± std")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "multi_sample_bar_mean_std.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_frequency(freq: pd.DataFrame, out_root: Path) -> None:
    plot_df = freq.set_index("intervention_key").reindex(PLOT_ORDER).reset_index()
    plot_df = plot_df.dropna(subset=["intervention_short"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_df))
    y = plot_df["rank1_ratio"].to_numpy(dtype=float)
    labels = plot_df["intervention_short"].tolist()

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rank-1 frequency")
    ax.set_title("Rank-1 frequency by intervention")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "rank_frequency_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_heatmap(df: pd.DataFrame, out_root: Path, metric: str) -> None:
    valid = df[~df["intervention_key"].eq(BASELINE_KEY)].copy()
    pivot = valid.pivot_table(
        index="sample_idx",
        columns="intervention_key",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=PLOT_ORDER)
    col_labels = [INTERVENTION_SHORT.get(c, c) for c in pivot.columns]

    fig_h = max(5, min(18, 0.28 * len(pivot) + 2.5))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index.tolist()])
    ax.set_xlabel("intervention")
    ax.set_ylabel("sample_idx")
    ax.set_title(f"Per-sample heatmap: {metric}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(out_root / f"per_sample_heatmap_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_probe_results(out_dir: str, rank_metric: str = DEFAULT_RANK_METRIC) -> None:
    out_root = Path(out_dir)
    df = collect_sample_summaries(out_root)
    df = add_rank_columns(df, rank_metric=rank_metric)
    df.to_csv(out_root / "all_sample_sensitivity_summary.csv", index=False)

    agg = compute_aggregate_summary(df)
    agg.to_csv(out_root / "aggregate_summary.csv", index=False)
    with open(out_root / "aggregate_summary.json", "w") as f:
        json.dump(ensure_jsonable(agg.to_dict(orient="records")), f, indent=2)

    freq, rank1 = compute_rank1_frequency(df, rank_metric=rank_metric)
    freq.to_csv(out_root / "rank1_frequency.csv", index=False)
    rank1.to_csv(out_root / "rank1_by_sample.csv", index=False)

    pairwise = compute_pairwise_stability(df, rank_metric=rank_metric)
    pairwise.to_csv(out_root / "pairwise_stability.csv", index=False)

    audit_df = collect_carrier_audits(out_root)
    if audit_df is not None:
        audit_df.to_csv(out_root / "all_sample_carrier_audit.csv", index=False)

    plot_multi_sample_bar_mean_std(agg, out_root, metric=rank_metric)
    plot_rank_frequency(freq, out_root)
    plot_per_sample_heatmap(df, out_root, metric=rank_metric)

    print(f"[aggregate] saved global probe results -> {out_root}")


# ============================================================
# S5. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TODO4.5 ego-position carrier probe")
    parser.add_argument("--config", required=True, help="path to drivetransformer config")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pth")
    parser.add_argument("--sample-idxs", required=True, help="e.g. 96 or 0,5,10 or 0:50:1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        default="/gs/bs/tga-RLA/qdeng/DriveTransformer/pe_diagnosis/ego_position_carrier_probe",
    )
    parser.add_argument("--force", action="store_true", help="rerun samples even if sensitivity_summary.csv exists")
    parser.add_argument("--continue-on-error", action="store_true", help="continue when a sample fails")
    parser.add_argument("--mode-idx", type=int, default=0)
    parser.add_argument("--reduce-mode", default="select", choices=["select", "best", "mean"])
    parser.add_argument("--rank-metric", default=DEFAULT_RANK_METRIC)
    parser.add_argument("--make-query-drift", action="store_true")
    args = parser.parse_args()

    sample_indices = parse_sample_indices(args.sample_idxs)
    out_root = Path(args.out_dir)
    sample_root = out_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "run_config.json", "w") as f:
        run_config = vars(args).copy()
        run_config["sample_indices"] = sample_indices
        json.dump(ensure_jsonable(run_config), f, indent=2)
        print("[1/3] loading model ...")
        model = load_model(args.config, args.ckpt, args.device)

    failed = []
    print(f"[2/3] running ego-position carrier probe for {len(sample_indices)} samples ...")
    for sample_idx in sample_indices:
        sample_out_dir = sample_root / f"sample_{sample_idx:06d}"
        summary_path = sample_out_dir / "sensitivity_summary.csv"

        if summary_path.exists() and not args.force:
            print(f"[skip] sample {sample_idx}: {summary_path} exists. Use --force to rerun.")
            continue

        try:
            run_probe_for_one_sample(
                model=model,
                config_path=args.config,
                sample_idx=sample_idx,
                sample_out_dir=sample_out_dir,
                device=args.device,
                reduce_mode=args.reduce_mode,
                mode_idx=args.mode_idx,
                make_query_drift=args.make_query_drift,
            )
        except Exception as e:
            print(f"[ERROR] sample {sample_idx} failed: {repr(e)}")
            failed.append({"sample_idx": sample_idx, "error": repr(e)})
            if not args.continue_on_error:
                raise

    if failed:
        with open(out_root / "failed_samples.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"[warn] failed samples saved -> {out_root / 'failed_samples.json'}")

    print("[3/3] aggregating probe results ...")
    aggregate_probe_results(args.out_dir, rank_metric=args.rank_metric)


if __name__ == "__main__":
    main()
