"""
Expected input format
---------------------
results = {
    "G0_baseline": {
        "traj": np.ndarray,              # [T, 2] or [N_mode, T, 2]
        "ego_query": np.ndarray,         # optional, any of:
                                         # [N_ego, D]
                                         # [L, N_ego, D]
                                         # [B, N_ego, D]
                                         # [L, B, N_ego, D]
        "agent_query": np.ndarray,       # optional
        "map_query": np.ndarray,         # optional
        "map_ref": np.ndarray,           # optional, [N_map, 2]
        "agent_ref": np.ndarray,         # optional, [N_agent, 2]
        "meta": {
            "label": "baseline",
            "color": "black",
            "perturb_type": "none",
        },
    },
    "G1_ego_ref_shift_x2": {...},
    ...
}

Usage: python -m planner_sensitivity_report \
  --results ./outputs/results.pkl \
  --out-dir ./outputs/planner_sensitivity_g0_g5 \
  --baseline-key G0_baseline \
  --mode-idx 0
"""

from __future__ import annotations

import os
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]
ResultDict = Dict[str, Dict[str, Any]]


# =============================================================================
# Basic IO
# =============================================================================
def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if path is not None and path != "":
        os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, path: str) -> None:
    """Save Python object as pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load Python object from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_jsonable(obj: Any) -> Any:
    """
    Convert numpy-heavy objects into JSON-serializable objects.

    This is mainly used for saving summary rows and metric dicts.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """Save object as JSON after converting numpy values."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=indent)


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """
    Save summary rows as CSV.

    Uses pandas if available; otherwise falls back to Python csv.
    """
    ensure_dir(os.path.dirname(path))

    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)
    except ImportError:
        import csv

        if len(rows) == 0:
            with open(path, "w") as f:
                f.write("")
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


# =============================================================================
# Trajectory utilities
# =============================================================================


def standardize_traj(traj: ArrayLike) -> np.ndarray:
    """
    Convert trajectory to shape [N_mode, T, 2].

    Args:
        traj:
            np.ndarray with shape [T, 2] or [N_mode, T, 2]

    Returns:
        traj_std:
            np.ndarray with shape [N_mode, T, 2]
    """
    traj = np.asarray(traj)

    if traj.ndim == 2:
        assert traj.shape[-1] == 2, f"Expected [T, 2], got {traj.shape}"
        traj = traj[None, ...]  # [1, T, 2]

    elif traj.ndim == 3:
        assert traj.shape[-1] == 2, f"Expected [N_mode, T, 2], got {traj.shape}"

    else:
        raise ValueError(f"Unsupported traj shape: {traj.shape}")

    return traj.astype(np.float64)


def select_traj_mode(
    traj: ArrayLike,
    mode_idx: int = 0,
    reduce_mode: str = "select",
) -> np.ndarray:
    """
    Select or reduce trajectory mode for visualization.

    Args:
        traj:
            [T, 2] or [N_mode, T, 2]
        mode_idx:
            selected mode if reduce_mode == "select"
        reduce_mode:
            "select": select one mode
            "mean": average all modes

    Returns:
        display_traj:
            [T, 2]
    """
    traj = standardize_traj(traj)

    if reduce_mode == "select":
        mode_idx = min(max(mode_idx, 0), traj.shape[0] - 1)
        return traj[mode_idx]

    if reduce_mode == "mean":
        return traj.mean(axis=0)

    raise ValueError(f"Unknown reduce_mode: {reduce_mode}")


def compute_traj_effect(
    base_traj: ArrayLike,
    pert_traj: ArrayLike,
) -> Dict[str, Any]:
    """
    Compute trajectory difference metrics between perturbed and baseline traj.

    Args:
        base_traj:
            [T, 2] or [N_mode, T, 2]
        pert_traj:
            [T, 2] or [N_mode, T, 2]

    Returns:
        metrics:
            dict containing:
                total_l2
                mean_point_error
                final_point_error
                max_point_error
                per_timestep_error
                per_mode_mean_error
                per_mode_final_error
    """
    base = standardize_traj(base_traj)
    pert = standardize_traj(pert_traj)

    if base.shape != pert.shape:
        raise ValueError(f"Shape mismatch: base={base.shape}, pert={pert.shape}")

    diff = pert - base                         # [N_mode, T, 2]
    point_l2 = np.linalg.norm(diff, axis=-1)   # [N_mode, T]

    metrics = {
        "total_l2": float(np.linalg.norm(diff)),
        "mean_point_error": float(point_l2.mean()),
        "final_point_error": float(point_l2[:, -1].mean()),
        "max_point_error": float(point_l2.max()),
        "per_timestep_error": point_l2.mean(axis=0),       # [T]
        "per_mode_mean_error": point_l2.mean(axis=1),      # [N_mode]
        "per_mode_final_error": point_l2[:, -1],           # [N_mode]
    }

    return metrics


def summarize_sensitivity(
    results: ResultDict,
    baseline_key: str = "G0_baseline",
    sort_by: str = "final_point_error",
    descending: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Summarize trajectory sensitivity for all experiments.

    Args:
        results:
            experiment result dict
        baseline_key:
            key of baseline experiment
        sort_by:
            metric used for sorting rows
        descending:
            whether to sort from large to small

    Returns:
        rows:
            list of summary rows
        metrics_by_name:
            dict from experiment name to full metrics dict
    """
    if baseline_key not in results:
        raise KeyError(f"baseline_key={baseline_key} not found in results.")

    base_traj = results[baseline_key]["traj"]

    rows: List[Dict[str, Any]] = []
    metrics_by_name: Dict[str, Dict[str, Any]] = {}

    for name, item in results.items():
        if name == baseline_key:
            continue

        if "traj" not in item:
            print(f"[WARN] Skip {name}: missing key 'traj'.")
            continue

        metrics = compute_traj_effect(base_traj, item["traj"])
        metrics_by_name[name] = metrics

        meta = item.get("meta", {})
        rows.append(
            {
                "name": name,
                "label": meta.get("label", name),
                "perturb_type": meta.get("perturb_type", "unknown"),
                "total_l2": metrics["total_l2"],
                "mean_point_error": metrics["mean_point_error"],
                "final_point_error": metrics["final_point_error"],
                "max_point_error": metrics["max_point_error"],
            }
        )

    if len(rows) > 0:
        if sort_by not in rows[0]:
            raise KeyError(f"sort_by={sort_by} not found in summary rows.")
        rows = sorted(rows, key=lambda x: x[sort_by], reverse=descending)

    return rows, metrics_by_name


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    """
    Print sensitivity rows in a compact text table.
    """
    if len(rows) == 0:
        print("[WARN] Empty summary rows.")
        return

    headers = [
        "rank",
        "name",
        "perturb_type",
        "mean_point_error",
        "final_point_error",
        "max_point_error",
        "total_l2",
    ]

    print("\n=== Planner Sensitivity Ranking ===")
    print(
        f"{headers[0]:<6} {headers[1]:<32} {headers[2]:<16} "
        f"{headers[3]:>18} {headers[4]:>18} {headers[5]:>16} {headers[6]:>12}"
    )
    print("-" * 126)

    for i, row in enumerate(rows, start=1):
        print(
            f"{i:<6} "
            f"{row['name']:<32} "
            f"{row['perturb_type']:<16} "
            f"{row['mean_point_error']:>18.6f} "
            f"{row['final_point_error']:>18.6f} "
            f"{row['max_point_error']:>16.6f} "
            f"{row['total_l2']:>12.6f}"
        )
    print()


# =============================================================================
# Query utilities
# =============================================================================
def standardize_query(query: ArrayLike, take_last_layer: bool = True) -> np.ndarray:
    """
    Convert query tensor into [N, D] for metric / relation computation.
    """
    if query is None:
        raise ValueError("query is None; probably this query was not captured.")

    q = np.asarray(query)

    if q.ndim == 4:
        # [L, B, N, D]
        return q[-1, 0].astype(np.float64)

    if q.ndim == 3:
        if take_last_layer:
            # [L, N, D]
            return q[-1].astype(np.float64)
        # [B, N, D]
        return q[0].astype(np.float64)

    if q.ndim == 2:
        return q.astype(np.float64)

    raise ValueError(f"Unsupported query shape: {q.shape}, type={type(query)}")

def compute_query_drift(
    base_query: ArrayLike,
    pert_query: ArrayLike,
    take_last_layer: bool = True,
) -> Dict[str, float]:
    """
    Compute embedding drift between baseline and perturbed query.

    Args:
        base_query:
            baseline query embedding
        pert_query:
            perturbed query embedding
        take_last_layer:
            see standardize_query

    Returns:
        drift metrics
    """
    base = standardize_query(base_query, take_last_layer=take_last_layer)
    pert = standardize_query(pert_query, take_last_layer=take_last_layer)

    if base.shape != pert.shape:
        raise ValueError(f"Query shape mismatch: base={base.shape}, pert={pert.shape}")

    diff = pert - base
    token_l2 = np.linalg.norm(diff, axis=-1)

    return {
        "query_total_l2": float(np.linalg.norm(diff)),
        "query_mean_token_l2": float(token_l2.mean()),
        "query_max_token_l2": float(token_l2.max()),
    }


def summarize_query_drift(
    results: ResultDict,
    query_key: str,
    baseline_key: str = "G0_baseline",
    take_last_layer: bool = True,
) -> List[Dict[str, Any]]:
    """
    Summarize query drift for one query type, e.g. ego_query / agent_query / map_query.

    Args:
        results:
            experiment results
        query_key:
            "ego_query", "agent_query", or "map_query"
        baseline_key:
            baseline key
        take_last_layer:
            see standardize_query

    Returns:
        rows:
            list of drift summary rows
    """
    if baseline_key not in results:
        raise KeyError(f"baseline_key={baseline_key} not found.")

    if query_key not in results[baseline_key]:
        print(f"[WARN] baseline missing query_key={query_key}. Skip query drift.")
        return []

    base_query = results[baseline_key].get(query_key, None)

    # Critical fix:
    # key exists but value may be None if capture failed.
    if base_query is None:
        print(f"[WARN] baseline query_key={query_key} is None. Skip query drift.")
        return []

    rows: List[Dict[str, Any]] = []

    for name, item in results.items():
        if name == baseline_key:
            continue

        pert_query = item.get(query_key, None)

        # Critical fix:
        # skip experiments where this query was not captured.
        if pert_query is None:
            print(f"[WARN] {name}: query_key={query_key} is None. Skip this item.")
            continue

        try:
            metrics = compute_query_drift(
                base_query,
                pert_query,
                take_last_layer=take_last_layer,
            )
        except Exception as e:
            print(
                f"[WARN] failed to compute query drift for "
                f"name={name}, query_key={query_key}: {e}"
            )
            continue

        meta = item.get("meta", {})
        rows.append(
            {
                "name": name,
                "label": meta.get("label", name),
                "perturb_type": meta.get("perturb_type", "unknown"),
                **metrics,
            }
        )

    rows = sorted(rows, key=lambda x: x["query_mean_token_l2"], reverse=True)
    return rows

# =============================================================================
# Plot helpers
# =============================================================================


def _get_label(item: Dict[str, Any], name: str) -> str:
    return item.get("meta", {}).get("label", name)


def _get_color(item: Dict[str, Any]) -> Optional[str]:
    return item.get("meta", {}).get("color", None)


def _save_or_show(save_path: Optional[str]) -> None:
    plt.tight_layout()
    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# =============================================================================
# Plot 1: trajectory overlay
# =============================================================================


def plot_traj_overlay(
    results: ResultDict,
    baseline_key: str = "G0_baseline",
    mode_idx: int = 0,
    reduce_mode: str = "select",
    save_path: Optional[str] = None,
    title: str = "G0-G5 Trajectory Overlay",
    xlabel: str = "x",
    ylabel: str = "y",
    draw_origin: bool = True,
    draw_final_point: bool = True,
) -> None:
    """
    Plot all trajectories in one BEV figure.

    This answers:
        "How does each perturbation change the predicted ego trajectory?"
    """
    plt.figure(figsize=(7, 7))

    for name, item in results.items():
        if "traj" not in item:
            continue

        traj = select_traj_mode(
            item["traj"],
            mode_idx=mode_idx,
            reduce_mode=reduce_mode,
        )

        label = _get_label(item, name)
        color = _get_color(item)

        is_base = name == baseline_key
        linewidth = 3.0 if is_base else 2.0
        linestyle = "-" if is_base else "--"
        alpha = 1.0 if is_base else 0.85

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label,
            alpha=alpha,
        )

        if draw_final_point:
            plt.scatter(
                traj[-1, 0],
                traj[-1, 1],
                s=60 if is_base else 45,
                color=color,
                alpha=alpha,
            )

    if draw_origin:
        plt.scatter([0], [0], marker="x", s=100, label="ego origin")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    _save_or_show(save_path)


# =============================================================================
# Plot 2: per-timestep error curve
# =============================================================================


def plot_timestep_error_curve(
    results: ResultDict,
    metrics_by_name: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Per-timestep Trajectory Error",
    xlabel: str = "future timestep",
    ylabel: str = "L2 error vs baseline",
) -> None:
    """
    Plot per-timestep trajectory error.

    This answers:
        "Does the perturbation affect short-term dynamics or long-term planning?"
    """
    plt.figure(figsize=(8, 5))

    for name, metrics in metrics_by_name.items():
        if "per_timestep_error" not in metrics:
            continue

        item = results[name]
        label = _get_label(item, name)
        color = _get_color(item)

        y = np.asarray(metrics["per_timestep_error"])
        x = np.arange(len(y))

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    _save_or_show(save_path)


# =============================================================================
# Plot 3: sensitivity bar chart
# =============================================================================


def plot_sensitivity_bar(
    rows: List[Dict[str, Any]],
    metric_name: str = "final_point_error",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    rotation: int = 30,
) -> None:
    """
    Plot variable sensitivity ranking as a bar chart.

    This answers:
        "Which variable causes the largest trajectory change?"
    """
    if len(rows) == 0:
        print("[WARN] Empty rows, skip plot_sensitivity_bar.")
        return

    labels = [r["label"] for r in rows]
    values = [r[metric_name] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=rotation, ha="right")
    plt.ylabel(metric_name)

    if title is None:
        title = f"Sensitivity Ranking by {metric_name}"
    plt.title(title)

    plt.grid(True, axis="y", alpha=0.3)
    _save_or_show(save_path)


def plot_multi_metric_bar(
    rows: List[Dict[str, Any]],
    metric_names: Tuple[str, ...] = (
        "mean_point_error",
        "final_point_error",
        "max_point_error",
    ),
    save_path: Optional[str] = None,
    title: str = "Sensitivity Ranking: Multiple Metrics",
    rotation: int = 30,
) -> None:
    """
    Plot grouped bar chart for multiple trajectory sensitivity metrics.
    """
    if len(rows) == 0:
        print("[WARN] Empty rows, skip plot_multi_metric_bar.")
        return

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.8 / len(metric_names)

    plt.figure(figsize=(11, 5))

    for i, metric_name in enumerate(metric_names):
        values = [r[metric_name] for r in rows]
        offset = (i - (len(metric_names) - 1) / 2.0) * width
        plt.bar(x + offset, values, width=width, label=metric_name)

    plt.xticks(x, labels, rotation=rotation, ha="right")
    plt.ylabel("error")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    _save_or_show(save_path)


# =============================================================================
# Optional Plot 4: query drift bar
# =============================================================================


def plot_query_drift_bar(
    rows: List[Dict[str, Any]],
    metric_name: str = "query_mean_token_l2",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    rotation: int = 30,
) -> None:
    """
    Plot query embedding drift ranking.

    This does NOT replace trajectory sensitivity.
    It only helps interpret whether query representations changed.
    """
    if len(rows) == 0:
        print("[WARN] Empty rows, skip plot_query_drift_bar.")
        return

    labels = [r["label"] for r in rows]
    values = [r[metric_name] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=rotation, ha="right")
    plt.ylabel(metric_name)

    if title is None:
        title = f"Query Drift by {metric_name}"
    plt.title(title)

    plt.grid(True, axis="y", alpha=0.3)
    _save_or_show(save_path)


# =============================================================================
# Optional Plot 5: BEV sensitivity heatmap
# =============================================================================


def plot_bev_sensitivity_heatmap(
    grid_x: ArrayLike,
    grid_y: ArrayLike,
    values: ArrayLike,
    save_path: Optional[str] = None,
    title: str = "BEV Sensitivity Heatmap",
    xlabel: str = "x shift",
    ylabel: str = "y shift",
    value_label: str = "final_point_error",
) -> None:
    """
    Plot BEV sensitivity heatmap.

    Args:
        grid_x:
            1D array of x-shift values, shape [W]
        grid_y:
            1D array of y-shift values, shape [H]
        values:
            2D array, shape [H, W]
            values[j, i] corresponds to grid_y[j], grid_x[i]

    This is useful after you run a 2D shift scan.
    """
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)
    values = np.asarray(values)

    if values.shape != (len(grid_y), len(grid_x)):
        raise ValueError(
            f"values shape should be [len(grid_y), len(grid_x)] = "
            f"{(len(grid_y), len(grid_x))}, got {values.shape}"
        )

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        values,
        origin="lower",
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        aspect="auto",
    )
    plt.colorbar(im, label=value_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    _save_or_show(save_path)


# =============================================================================
# Optional relation proxy
# =============================================================================


def cosine_similarity_matrix(a: ArrayLike, b: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of vectors.

    Args:
        a:
            [N, D]
        b:
            [M, D]

    Returns:
        sim:
            [N, M]
    """
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got a={a.shape}, b={b.shape}")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Dim mismatch: a={a.shape}, b={b.shape}")

    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)

    return a_norm @ b_norm.T


def compute_ego_to_other_relation(
    ego_query: ArrayLike,
    other_query: ArrayLike,
    ego_reduce: str = "mean",
    take_last_layer: bool = True,
) -> np.ndarray:
    """
    Compute ego-to-map or ego-to-agent cosine relation proxy.

    Args:
        ego_query:
            ego query embedding
        other_query:
            map or agent query embedding
        ego_reduce:
            "mean": average ego queries first, output [N_other]
            "max": max over ego queries, output [N_other]
            "all": output [N_ego, N_other]
        take_last_layer:
            see standardize_query

    Returns:
        relation:
            [N_other] or [N_ego, N_other]
    """
    ego = standardize_query(ego_query, take_last_layer=take_last_layer)
    other = standardize_query(other_query, take_last_layer=take_last_layer)

    sim = cosine_similarity_matrix(ego, other)  # [N_ego, N_other]

    if ego_reduce == "mean":
        return sim.mean(axis=0)
    if ego_reduce == "max":
        return sim.max(axis=0)
    if ego_reduce == "all":
        return sim

    raise ValueError(f"Unknown ego_reduce: {ego_reduce}")


def plot_relation_scatter_bev(
    ref_xy: ArrayLike,
    relation: ArrayLike,
    save_path: Optional[str] = None,
    title: str = "Ego-to-token Relation Proxy",
    xlabel: str = "x",
    ylabel: str = "y",
    value_label: str = "cosine relation",
    point_size: float = 40.0,
    draw_origin: bool = True,
) -> None:
    """
    Plot relation score on BEV token reference points.

    Args:
        ref_xy:
            [N, 2]
        relation:
            [N]
    """
    ref_xy = np.asarray(ref_xy)
    relation = np.asarray(relation)

    if ref_xy.ndim != 2 or ref_xy.shape[-1] != 2:
        raise ValueError(f"Expected ref_xy [N, 2], got {ref_xy.shape}")
    if relation.ndim != 1 or relation.shape[0] != ref_xy.shape[0]:
        raise ValueError(
            f"Expected relation [N] aligned with ref_xy [N, 2], "
            f"got ref_xy={ref_xy.shape}, relation={relation.shape}"
        )

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        ref_xy[:, 0],
        ref_xy[:, 1],
        c=relation,
        s=point_size,
        alpha=0.85,
    )
    plt.colorbar(sc, label=value_label)

    if draw_origin:
        plt.scatter([0], [0], marker="x", s=100, label="ego origin")
        plt.legend(fontsize=8)

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _save_or_show(save_path)


# =============================================================================
# Default metadata helper
# =============================================================================


def attach_default_meta(results: ResultDict) -> ResultDict:
    """
    Attach default meta info for common G0-G5 experiments if missing.

    This modifies `results` in-place and also returns it.
    """
    default_meta = {
        "G0_baseline": {
            "label": "G0 baseline",
            "color": "black",
            "perturb_type": "none",
        },
        "G1_ego_ref_shift_x2": {
            "label": "G1 ego_ref x+2m",
            "color": "tab:blue",
            "perturb_type": "ego_ref",
        },
        "G2_ego_his_zero": {
            "label": "G2 ego_his zero",
            "color": "tab:orange",
            "perturb_type": "ego_his_trajs",
        },
        "G3_ego_lcf_zero": {
            "label": "G3 ego_lcf zero",
            "color": "tab:green",
            "perturb_type": "ego_lcf_feat",
        },
        "G4_map_query_mask": {
            "label": "G4 map query mask",
            "color": "tab:red",
            "perturb_type": "map_query",
        },
        "G5_agent_query_mask": {
            "label": "G5 agent query mask",
            "color": "tab:purple",
            "perturb_type": "agent_query",
        },
    }

    for name, item in results.items():
        if "meta" not in item:
            item["meta"] = {}

        if name in default_meta:
            for k, v in default_meta[name].items():
                item["meta"].setdefault(k, v)
        else:
            item["meta"].setdefault("label", name)
            item["meta"].setdefault("perturb_type", "unknown")

    return results


# =============================================================================
# Main report function
# =============================================================================


def run_basic_sensitivity_report(
    results: ResultDict,
    out_dir: str,
    baseline_key: str = "G0_baseline",
    mode_idx: int = 0,
    reduce_mode: str = "select",
    sort_by: str = "final_point_error",
    attach_meta: bool = True,
    save_results_pickle: bool = True,
    make_query_drift: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Generate basic planner sensitivity report.

    Outputs:
        out_dir/
            results.pkl                         optional
            sensitivity_summary.json
            sensitivity_summary.csv
            metrics_by_name.json
            fig_A_traj_overlay.png
            fig_B_timestep_error.png
            fig_C1_bar_mean_point_error.png
            fig_C2_bar_final_point_error.png
            fig_C3_bar_max_point_error.png
            fig_C4_bar_multi_metric.png
            query_drift_ego_query.json          optional
            query_drift_agent_query.json        optional
            query_drift_map_query.json          optional
            fig_D1_ego_query_drift.png          optional
            fig_D2_agent_query_drift.png        optional
            fig_D3_map_query_drift.png          optional

    Returns:
        rows:
            sorted sensitivity summary rows
        metrics_by_name:
            full metric dict per experiment
    """
    ensure_dir(out_dir)

    if attach_meta:
        results = attach_default_meta(results)

    if save_results_pickle:
        save_pickle(results, os.path.join(out_dir, "results.pkl"))

    rows, metrics_by_name = summarize_sensitivity(
        results,
        baseline_key=baseline_key,
        sort_by=sort_by,
        descending=True,
    )

    print_summary_table(rows)

    save_json(rows, os.path.join(out_dir, "sensitivity_summary.json"))
    save_csv(rows, os.path.join(out_dir, "sensitivity_summary.csv"))
    save_json(metrics_by_name, os.path.join(out_dir, "metrics_by_name.json"))

    plot_traj_overlay(
        results,
        baseline_key=baseline_key,
        mode_idx=mode_idx,
        reduce_mode=reduce_mode,
        save_path=os.path.join(out_dir, "fig_A_traj_overlay.png"),
        title="G0-G5 Trajectory Overlay",
    )

    plot_timestep_error_curve(
        results,
        metrics_by_name,
        save_path=os.path.join(out_dir, "fig_B_timestep_error.png"),
        title="Per-timestep Trajectory Error",
    )

    plot_sensitivity_bar(
        rows,
        metric_name="mean_point_error",
        save_path=os.path.join(out_dir, "fig_C1_bar_mean_point_error.png"),
        title="Sensitivity Ranking by Mean Point Error",
    )

    plot_sensitivity_bar(
        rows,
        metric_name="final_point_error",
        save_path=os.path.join(out_dir, "fig_C2_bar_final_point_error.png"),
        title="Sensitivity Ranking by Final Point Error",
    )

    plot_sensitivity_bar(
        rows,
        metric_name="max_point_error",
        save_path=os.path.join(out_dir, "fig_C3_bar_max_point_error.png"),
        title="Sensitivity Ranking by Max Point Error",
    )

    plot_multi_metric_bar(
        rows,
        save_path=os.path.join(out_dir, "fig_C4_bar_multi_metric.png"),
        title="Sensitivity Ranking: Mean / Final / Max Error",
    )

    if make_query_drift:
        # Check if ego_query is layerwise (captured from decoder layers)
        # vs fallback (transformer input only). Skip layerwise drift plot if not layerwise.
        baseline_item = results.get(baseline_key, {})
        ego_query_is_layerwise = baseline_item.get("ego_query_is_layerwise", True)

        for query_key, fig_name in [
            ("ego_query", "fig_D1_layerwise_ego_query_drift.png"),
            ("agent_query", "fig_D2_layerwise_agent_query_drift.png"),
            ("map_query", "fig_D3_layerwise_map_query_drift.png"),
        ]:
            # Guard: skip ego_query drift plot if capture was not layerwise
            if query_key == "ego_query" and not ego_query_is_layerwise:
                print(
                    f"[WARN] Skipping {fig_name}: ego_query_is_layerwise=False "
                    f"(using transformer input fallback, not per-layer decoder output)"
                )
                continue

            drift_rows = summarize_query_drift(
                results,
                query_key=query_key,
                baseline_key=baseline_key,
                take_last_layer=True,
            )

            if len(drift_rows) == 0:
                continue

            save_json(
                drift_rows,
                os.path.join(out_dir, f"query_drift_{query_key}.json"),
            )

            plot_query_drift_bar(
                drift_rows,
                metric_name="query_mean_token_l2",
                save_path=os.path.join(out_dir, fig_name),
                title=f"{query_key} Drift",
            )

    return rows, metrics_by_name


# =============================================================================
# Optional CLI
# =============================================================================


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DriveTransformer planner sensitivity report from a results.pkl file."
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.pkl. The object should be a results dict.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for summary and figures.",
    )

    parser.add_argument(
        "--baseline-key",
        type=str,
        default="G0_baseline",
        help="Baseline key in results dict.",
    )

    parser.add_argument(
        "--mode-idx",
        type=int,
        default=0,
        help="Trajectory mode index to visualize.",
    )

    parser.add_argument(
        "--reduce-mode",
        type=str,
        default="select",
        choices=["select", "mean"],
        help="How to display multi-modal trajectories.",
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        default="final_point_error",
        choices=[
            "total_l2",
            "mean_point_error",
            "final_point_error",
            "max_point_error",
        ],
        help="Metric used to sort sensitivity rows.",
    )

    parser.add_argument(
        "--no-query-drift",
        action="store_true",
        help="Disable query drift summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    results = load_pickle(args.results)

    run_basic_sensitivity_report(
        results=results,
        out_dir=args.out_dir,
        baseline_key=args.baseline_key,
        mode_idx=args.mode_idx,
        reduce_mode=args.reduce_mode,
        sort_by=args.sort_by,
        attach_meta=True,
        save_results_pickle=False,
        make_query_drift=not args.no_query_drift,
    )


if __name__ == "__main__":
    main()