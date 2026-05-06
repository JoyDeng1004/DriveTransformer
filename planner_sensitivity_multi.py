"""
Multi-sample planner sensitivity ranking for DriveTransformer G0-G5 experiments.

Recommended usage:
python planner_sensitivity_multi.py \
  --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
  --ckpt /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
  --sample-idxs 0:50:1 \
  --device cuda:0 \
  --out-dir /gs/bs/tga-RLA/qdeng/DriveTransformer/pe_diagnosis/planner_sensitivity_g0_g5_multi
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from planner_sensitivity_single import run_basic_sensitivity_report

# Reuse the already-debugged single-sample experiment logic.
from ego_ref_shift_diag import (
    CAM_FRONT_IDX,
    EXPERIMENTS,
    load_model,
    run_all_experiments,
)


DEFAULT_OUT_DIR = (
    "/gs/bs/tga-RLA/qdeng/DriveTransformer/pe_diagnosis/"
    "planner_sensitivity_g0_g5_multi"
)

BASELINE_KEY = "G0_baseline"
RANK_METRIC = "final_point_error"

# Stable order used in plots/tables. Keep these names aligned with ego_ref_shift_diag.EXPERIMENTS.
INTERVENTION_ORDER = [
    "G1_ego_ref_shift_x2",
    "G2_ego_his_zero",
    "G3_ego_lcf_zero",
    "G4_map_query_mask",
    "G5_agent_query_mask",
]

LABEL_BY_KEY = {name: label for name, _shift, _color, label in EXPERIMENTS}
COLOR_BY_KEY = {name: color for name, _shift, color, _label in EXPERIMENTS}

SHORT_LABEL_BY_KEY = {
    "G0_baseline": "baseline",
    "G1_ego_ref_shift_x2": "ego_ref_x+2m",
    "G2_ego_his_zero": "ego_his_zero",
    "G3_ego_lcf_zero": "ego_lcf_zero",
    "G4_map_query_mask": "map_query_mask",
    "G5_agent_query_mask": "agent_query_mask",
}


# ============================================================
# A0. CLI utilities
# ============================================================

def parse_sample_indices(spec: str) -> List[int]:
    """
    Parse sample index specification.

    Supported forms:
      - "100,150,200"
      - "100:600:50"  -> range(100, 600, 50)
      - "100:600"     -> range(100, 600, 1)
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("--sample-idxs is empty")

    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) == 2:
            start, end = map(int, parts)
            step = 1
        elif len(parts) == 3:
            start, end, step = map(int, parts)
        else:
            raise ValueError(f"Bad range spec: {spec}")
        if step == 0:
            raise ValueError("range step cannot be 0")
        return list(range(start, end, step))

    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def to_jsonable(obj):
    """Convert numpy/pandas scalar containers into JSON-serializable objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, np.ndarray)) else False:
        return None
    return obj


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


# ============================================================
# A1. Dataset/sample loading, built once
# ============================================================

def build_dataset_from_config(config_path: str):
    """Build DriveTransformer validation dataset once."""
    from mmcv import Config
    from mmcv.datasets import build_dataset

    cfg = Config.fromfile(config_path)
    data_cfg = copy.deepcopy(cfg.data.val)
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    return dataset, cfg


def unwrap_datacontainer(x):
    """Recursively unwrap mmcv DataContainer while keeping list/dict structure."""
    from mmcv.parallel import DataContainer

    while isinstance(x, DataContainer):
        x = x.data
    if isinstance(x, dict):
        return {k: unwrap_datacontainer(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unwrap_datacontainer(v) for v in x]
    if isinstance(x, tuple):
        return tuple(unwrap_datacontainer(v) for v in x)
    return x


def move_to_device(x, device: str):
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


def normalize_model_batch(batch: dict, dataset, cfg, sample_idx: int, device: str) -> dict:
    """
    Convert collated mmcv batch into the format expected by DriveTransformer forward.

    This is a lightly factored version of ego_ref_shift_diag.load_sample(), kept here
    so the multi-sample runner can build the dataset only once.
    """
    model_batch = {}
    for k, v in batch.items():
        v = unwrap_datacontainer(v)
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
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.dim() != 5:
        raise RuntimeError(f"unexpected img_tensor.shape = {tuple(img_tensor.shape)}")
    model_batch["img"] = img_tensor

    # ---- normalize img_metas ----
    img_metas = model_batch["img_metas"]
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
        img_metas = img_metas[0]
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    assert isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], dict), (
        f"unexpected img_metas structure: {type(img_metas)}"
    )
    model_batch["img_metas"] = img_metas
    meta0 = img_metas[0]

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
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        if isinstance(x, list):
            x = np.stack([np.asarray(xx) for xx in x], axis=0)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.float().to(device)
            if x.dim() in (3, 4) and x.shape[0] == 6:
                x = x.unsqueeze(0)
        model_batch[key] = x

    # ---- fallback: recover cam_intrinsic if pipeline did not expose it ----
    if "cam_intrinsic" not in model_batch:
        cam_intrinsic = None
        if "cam_intrinsic" in meta0:
            cam_intrinsic = meta0["cam_intrinsic"]

        if cam_intrinsic is None:
            try:
                info = dataset.data_infos[sample_idx] if hasattr(dataset, "data_infos") else dataset.get_data_info(sample_idx)
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
            cam_intrinsic = torch.as_tensor(cam_intrinsic, dtype=torch.float32, device=device)
            if cam_intrinsic.dim() == 3 and cam_intrinsic.shape[0] == 6:
                cam_intrinsic = cam_intrinsic.unsqueeze(0)
            model_batch["cam_intrinsic"] = cam_intrinsic
        else:
            print("[warn] cam_intrinsic still missing after fallback")

    model_batch = move_to_device(model_batch, device)
    return model_batch


def load_sample_from_dataset(dataset, cfg, sample_idx: int, device: str) -> dict:
    """Load one sample from an already-built dataset."""
    from mmcv.parallel import collate

    sample = dataset[sample_idx]
    batch = collate([sample], samples_per_gpu=1)
    model_batch = normalize_model_batch(batch, dataset=dataset, cfg=cfg, sample_idx=sample_idx, device=device)
    return model_batch


# ============================================================
# A2. Single-sample wrapper
# ============================================================

def reset_model_memory(model) -> None:
    """Reset DriveTransformer temporal memory at sample boundaries."""
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "prev_scene_token"):
        base.prev_scene_token = None
    if hasattr(base, "pts_bbox_head") and hasattr(base.pts_bbox_head, "reset_memory"):
        base.pts_bbox_head.reset_memory()


@torch.no_grad()
def run_g0_g5_for_one_sample(
    model,
    dataset,
    cfg,
    sample_idx: int,
    out_dir: Path,
    device: str,
    mode_idx: int = 0,
    reduce_mode: str = "select",
    sort_by: str = RANK_METRIC,
    make_query_drift: bool = True,
    force: bool = False,
) -> Path:
    """
    Run G0-G5 for one sample and save its per-sample report.

    Output directory:
        out_dir / sample_xxxxxx /
          - results.pkl
          - sensitivity_summary.csv/json
          - fig_A/B/C...
    """
    sample_out_dir = out_dir / "samples" / f"sample_{sample_idx:06d}"
    summary_csv = sample_out_dir / "sensitivity_summary.csv"

    if summary_csv.exists() and not force:
        print(f"[skip] sample {sample_idx}: found {summary_csv}")
        return sample_out_dir

    sample_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"[sample] idx={sample_idx} -> {sample_out_dir}")

    reset_model_memory(model)
    batch = load_sample_from_dataset(dataset, cfg=cfg, sample_idx=sample_idx, device=device)

    reset_model_memory(model)
    results = run_all_experiments(model, batch)

    # Save an explicit pickle here even if planner_sensitivity_report also saves it.
    with (sample_out_dir / "results.pkl").open("wb") as f:
        pickle.dump(results, f)

    rows, metrics_by_name = run_basic_sensitivity_report(
        results=results,
        out_dir=str(sample_out_dir),
        baseline_key=BASELINE_KEY,
        mode_idx=mode_idx,
        reduce_mode=reduce_mode,
        sort_by=sort_by,
        attach_meta=True,
        save_results_pickle=True,
        make_query_drift=make_query_drift,
    )

    # Extra lightweight metadata for reproducibility.
    save_json(
        sample_out_dir / "sample_run_meta.json",
        {
            "sample_idx": sample_idx,
            "mode_idx": mode_idx,
            "reduce_mode": reduce_mode,
            "sort_by": sort_by,
            "baseline_key": BASELINE_KEY,
            "experiment_order": [name for name, *_ in EXPERIMENTS],
            "rows_returned_by_report": rows,
            "metrics_by_name_keys": list(metrics_by_name.keys()) if isinstance(metrics_by_name, dict) else None,
        },
    )

    print(f"[done] sample {sample_idx}: report saved -> {sample_out_dir}")
    return sample_out_dir


# ============================================================
# A3. Aggregation helpers
# ============================================================

def infer_intervention_key(row: pd.Series) -> str:
    """
    Robustly infer the experiment/intervention key from different possible
    planner_sensitivity_report.py CSV schemas.
    """
    candidates = [
        "intervention_key",
        "experiment_key",
        "name",
        "group",
        "key",
        "intervention",
        "label",
    ]
    values = []
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            values.append(str(row[c]))

    joined = " | ".join(values)
    for key in [BASELINE_KEY] + INTERVENTION_ORDER:
        if key in joined:
            return key

    # Match readable labels as fallback.
    for key, label in LABEL_BY_KEY.items():
        if label in joined:
            return key
    for key, label in SHORT_LABEL_BY_KEY.items():
        if label in joined:
            return key

    # Last resort: preserve the most likely column value.
    if values:
        return values[0]
    return "UNKNOWN"


def normalize_summary_df(df: pd.DataFrame, sample_idx: int) -> pd.DataFrame:
    """Add sample_idx, intervention_key, and stable labels to a per-sample summary."""
    df = df.copy()
    df["sample_idx"] = sample_idx

    if "intervention_key" not in df.columns:
        df["intervention_key"] = df.apply(infer_intervention_key, axis=1)
    else:
        df["intervention_key"] = df["intervention_key"].astype(str)

    df["intervention_label"] = df["intervention_key"].map(LABEL_BY_KEY).fillna(df["intervention_key"])
    df["intervention_short"] = df["intervention_key"].map(SHORT_LABEL_BY_KEY).fillna(df["intervention_key"])

    # Ensure numeric metrics are numeric when present.
    for col in ["mean_point_error", "final_point_error", "max_point_error", "total_l2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def collect_sample_summaries(out_root: Path) -> pd.DataFrame:
    rows = []
    sample_dirs = sorted((out_root / "samples").glob("sample_*"))

    for sample_dir in sample_dirs:
        csv_path = sample_dir / "sensitivity_summary.csv"
        if not csv_path.exists():
            print(f"[warn] missing summary: {csv_path}")
            continue

        try:
            sample_idx = int(sample_dir.name.replace("sample_", ""))
        except ValueError:
            print(f"[warn] cannot parse sample idx from {sample_dir.name}; skip")
            continue

        df = pd.read_csv(csv_path)
        rows.append(normalize_summary_df(df, sample_idx=sample_idx))

    if not rows:
        raise RuntimeError(f"No sensitivity_summary.csv found under {out_root / 'samples'}")

    all_df = pd.concat(rows, ignore_index=True)
    return all_df


def add_rank_columns(df: pd.DataFrame, rank_metric: str = RANK_METRIC) -> pd.DataFrame:
    """Rank interventions per sample by descending rank_metric. Baseline is excluded."""
    if rank_metric not in df.columns:
        raise KeyError(f"rank metric '{rank_metric}' not found in columns: {list(df.columns)}")

    df = df.copy()
    df["is_baseline"] = df["intervention_key"].eq(BASELINE_KEY) | df["intervention_short"].eq("baseline")
    df["rank_by_final"] = np.nan

    mask = ~df["is_baseline"]
    df.loc[mask, "rank_by_final"] = (
        df.loc[mask]
        .groupby("sample_idx")[rank_metric]
        .rank(method="min", ascending=False)
    )
    return df


def compute_aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median for each intervention."""
    metric_cols = [
        c for c in ["mean_point_error", "final_point_error", "max_point_error", "total_l2"]
        if c in df.columns
    ]
    if not metric_cols:
        raise KeyError("No expected metric columns found in summary CSV files")

    valid = df[~df["is_baseline"]].copy()
    valid["intervention_key"] = pd.Categorical(
        valid["intervention_key"], categories=INTERVENTION_ORDER, ordered=True
    )

    agg = (
        valid
        .groupby(["intervention_key", "intervention_short"], observed=False)[metric_cols]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return agg


def compute_rank1_frequency(df: pd.DataFrame, rank_metric: str = RANK_METRIC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count which intervention is rank-1 for each sample."""
    valid = df[~df["is_baseline"]].copy()
    valid = valid.dropna(subset=[rank_metric])

    idx = valid.groupby("sample_idx")[rank_metric].idxmax()
    rank1 = valid.loc[idx, ["sample_idx", "intervention_key", "intervention_short", rank_metric]].copy()
    rank1 = rank1.sort_values("sample_idx")

    total_samples = rank1["sample_idx"].nunique()
    freq = (
        rank1["intervention_key"]
        .value_counts()
        .rename_axis("intervention_key")
        .reset_index(name="rank1_count")
    )
    freq["rank1_ratio"] = freq["rank1_count"] / max(total_samples, 1)
    freq["intervention_short"] = freq["intervention_key"].map(SHORT_LABEL_BY_KEY).fillna(freq["intervention_key"])

    # Add zero rows for interventions that never win.
    existing = set(freq["intervention_key"])
    missing_rows = []
    for key in INTERVENTION_ORDER:
        if key not in existing:
            missing_rows.append({
                "intervention_key": key,
                "rank1_count": 0,
                "rank1_ratio": 0.0,
                "intervention_short": SHORT_LABEL_BY_KEY.get(key, key),
            })
    if missing_rows:
        freq = pd.concat([freq, pd.DataFrame(missing_rows)], ignore_index=True)

    freq["order"] = freq["intervention_key"].map({k: i for i, k in enumerate(INTERVENTION_ORDER)})
    freq = freq.sort_values("order").drop(columns=["order"])
    return freq, rank1


def compute_pairwise_stability(df: pd.DataFrame, rank_metric: str = RANK_METRIC) -> pd.DataFrame:
    """Check whether G1 ego_ref shift is consistently weaker than G3/G5."""
    pivot = df.pivot_table(
        index="sample_idx",
        columns="intervention_key",
        values=rank_metric,
        aggfunc="first",
    )

    comparisons = [
        ("G1_ego_ref_shift_x2", "G3_ego_lcf_zero"),
        ("G1_ego_ref_shift_x2", "G5_agent_query_mask"),
    ]

    rows = []
    for weak_key, strong_key in comparisons:
        if weak_key not in pivot.columns or strong_key not in pivot.columns:
            rows.append({
                "comparison": f"{SHORT_LABEL_BY_KEY.get(weak_key, weak_key)} < {SHORT_LABEL_BY_KEY.get(strong_key, strong_key)}",
                "weak_key": weak_key,
                "strong_key": strong_key,
                "true_count": 0,
                "total": 0,
                "ratio": np.nan,
                "note": "missing column",
            })
            continue

        valid = pivot[[weak_key, strong_key]].dropna()
        is_weaker = valid[weak_key] < valid[strong_key]
        rows.append({
            "comparison": f"{SHORT_LABEL_BY_KEY.get(weak_key, weak_key)} < {SHORT_LABEL_BY_KEY.get(strong_key, strong_key)}",
            "weak_key": weak_key,
            "strong_key": strong_key,
            "true_count": int(is_weaker.sum()),
            "total": int(len(valid)),
            "ratio": float(is_weaker.mean()) if len(valid) > 0 else np.nan,
            "note": "",
        })

    return pd.DataFrame(rows)


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten pandas MultiIndex columns after groupby agg."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in out.columns]
    return out


# ============================================================
# A4. Global plots
# ============================================================

def _ordered_nonbaseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df[~df["is_baseline"]].copy()
    out["intervention_key"] = pd.Categorical(
        out["intervention_key"], categories=INTERVENTION_ORDER, ordered=True
    )
    return out.sort_values(["intervention_key", "sample_idx"])


def plot_multi_sample_bar_mean_std(df: pd.DataFrame, out_path: Path, metric: str = RANK_METRIC) -> None:
    valid = _ordered_nonbaseline(df).dropna(subset=[metric])
    stats = valid.groupby("intervention_key", observed=False)[metric].agg(["mean", "std"]).reindex(INTERVENTION_ORDER)
    labels = [SHORT_LABEL_BY_KEY.get(k, k) for k in stats.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stats))
    y = stats["mean"].to_numpy(dtype=float)
    yerr = stats["std"].fillna(0.0).to_numpy(dtype=float)
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Multi-sample planner sensitivity: mean ± std of {metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_frequency_bar(freq: pd.DataFrame, out_path: Path) -> None:
    freq = freq.copy()
    freq["intervention_key"] = pd.Categorical(
        freq["intervention_key"], categories=INTERVENTION_ORDER, ordered=True
    )
    freq = freq.sort_values("intervention_key")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(freq))
    ax.bar(x, freq["rank1_ratio"].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(freq["intervention_short"].tolist(), rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rank-1 frequency")
    ax.set_title("Which intervention is most often rank-1?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_sample_heatmap(df: pd.DataFrame, out_path: Path, metric: str = RANK_METRIC) -> None:
    valid = df[~df["is_baseline"]].copy()
    pivot = valid.pivot_table(
        index="sample_idx",
        columns="intervention_key",
        values=metric,
        aggfunc="first",
    ).reindex(columns=INTERVENTION_ORDER)

    fig_h = max(4, 0.35 * len(pivot) + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(np.arange(len(INTERVENTION_ORDER)))
    ax.set_xticklabels([SHORT_LABEL_BY_KEY.get(k, k) for k in INTERVENTION_ORDER], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("intervention")
    ax.set_ylabel("sample_idx")
    ax.set_title(f"Per-sample heatmap: {metric}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_figures(df: pd.DataFrame, rank1_freq: pd.DataFrame, out_root: Path, metric: str = RANK_METRIC) -> None:
    plot_multi_sample_bar_mean_std(df, out_root / "multi_sample_bar_mean_std.png", metric=metric)
    plot_rank_frequency_bar(rank1_freq, out_root / "rank_frequency_bar.png")
    plot_per_sample_heatmap(df, out_root / f"per_sample_heatmap_{metric}.png", metric=metric)


# ============================================================
# A5. Aggregate entry
# ============================================================

def aggregate_multi_sample_results(out_dir: str, rank_metric: str = RANK_METRIC) -> Dict[str, Path]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_df = collect_sample_summaries(out_root)
    all_df = add_rank_columns(all_df, rank_metric=rank_metric)

    all_csv = out_root / "all_sample_sensitivity_summary.csv"
    all_df.to_csv(all_csv, index=False)

    agg = compute_aggregate_summary(all_df)
    agg_flat = flatten_multiindex_columns(agg)
    agg_csv = out_root / "aggregate_summary.csv"
    agg_json = out_root / "aggregate_summary.json"
    agg_flat.to_csv(agg_csv, index=False)
    save_json(agg_json, agg_flat.to_dict(orient="records"))

    rank1_freq, rank1_by_sample = compute_rank1_frequency(all_df, rank_metric=rank_metric)
    rank1_freq_csv = out_root / "rank1_frequency.csv"
    rank1_by_sample_csv = out_root / "rank1_by_sample.csv"
    rank1_freq.to_csv(rank1_freq_csv, index=False)
    rank1_by_sample.to_csv(rank1_by_sample_csv, index=False)

    pairwise = compute_pairwise_stability(all_df, rank_metric=rank_metric)
    pairwise_csv = out_root / "pairwise_stability.csv"
    pairwise.to_csv(pairwise_csv, index=False)

    plot_global_figures(all_df, rank1_freq, out_root, metric=rank_metric)

    print("=" * 100)
    print("[aggregate] saved global files:")
    for p in [
        all_csv,
        agg_csv,
        agg_json,
        rank1_freq_csv,
        rank1_by_sample_csv,
        pairwise_csv,
        out_root / "multi_sample_bar_mean_std.png",
        out_root / "rank_frequency_bar.png",
        out_root / f"per_sample_heatmap_{rank_metric}.png",
    ]:
        print(f"  - {p}")

    return {
        "all_csv": all_csv,
        "aggregate_csv": agg_csv,
        "aggregate_json": agg_json,
        "rank1_frequency_csv": rank1_freq_csv,
        "rank1_by_sample_csv": rank1_by_sample_csv,
        "pairwise_stability_csv": pairwise_csv,
    }


# ============================================================
# A6. Main runner
# ============================================================

def run_all_samples(args) -> None:
    out_root = Path(args.out_dir)
    (out_root / "samples").mkdir(parents=True, exist_ok=True)

    print("[1/4] loading model once ...")
    model = load_model(args.config, args.ckpt, args.device)

    print("[2/4] building dataset once ...")
    dataset, cfg = build_dataset_from_config(args.config)
    print(f"[debug] dataset length = {len(dataset)}")

    print(f"[3/4] running samples: {args.sample_indices}")
    failed = []
    for sample_idx in args.sample_indices:
        try:
            run_g0_g5_for_one_sample(
                model=model,
                dataset=dataset,
                cfg=cfg,
                sample_idx=sample_idx,
                out_dir=out_root,
                device=args.device,
                mode_idx=args.mode_idx,
                reduce_mode=args.reduce_mode,
                sort_by=args.rank_metric,
                make_query_drift=not args.no_query_drift,
                force=args.force,
            )
        except Exception as e:
            print(f"[ERROR] sample {sample_idx} failed: {repr(e)}")
            failed.append({"sample_idx": sample_idx, "error": repr(e)})
            if not args.continue_on_error:
                raise

    if failed:
        save_json(out_root / "failed_samples.json", failed)
        print(f"[warn] failed samples saved -> {out_root / 'failed_samples.json'}")

    print("[4/4] aggregating multi-sample results ...")
    aggregate_multi_sample_results(str(out_root), rank_metric=args.rank_metric)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to drivetransformer config")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pth")
    parser.add_argument("--sample-idxs", required=True, help="e.g. '100,150,200' or '100:600:50'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode-idx", type=int, default=0)
    parser.add_argument("--reduce-mode", default="select", choices=["select", "mean", "min"], help="passed to planner_sensitivity_report")
    parser.add_argument("--rank-metric", default=RANK_METRIC, help="usually final_point_error")
    parser.add_argument("--force", action="store_true", help="rerun samples even if summary already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="continue if one sample fails")
    parser.add_argument("--no-query-drift", action="store_true", help="disable per-sample query drift plots")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.sample_indices = parse_sample_indices(args.sample_idxs)

    run_all_samples(args)


if __name__ == "__main__":
    main()
