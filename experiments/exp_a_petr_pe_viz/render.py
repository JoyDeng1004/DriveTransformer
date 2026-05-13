#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXP_DIR = Path(__file__).resolve().parent


def load_sample(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["_path"] = path
    return out


def as_bool(x: Any) -> bool:
    return bool(np.asarray(x).item())


def point_style(point_types: Sequence[Any]) -> List[str]:
    return ["circle" if str(t) == "static" else "triangle-up" for t in point_types]


def color_values(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32)


def add_camera_image(fig: go.Figure, sample: Dict[str, Any], t: int, row: int, col: int, cols: int) -> None:
    if "camera_images" not in sample:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    img = np.asarray(sample["camera_images"][t])
    h, w = img.shape[:2]
    idx = (row - 1) * cols + col
    suffix = "" if idx == 1 else str(idx)
    fig.add_layout_image(
        dict(
            source=Image.fromarray(img.astype(np.uint8)),
            xref=f"x{suffix}",
            yref=f"y{suffix}",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            layer="below",
        )
    )
    fig.update_xaxes(range=[0, w], row=row, col=col)
    fig.update_yaxes(range=[h, 0], row=row, col=col)


def canonical_points(sample: Dict[str, Any]) -> np.ndarray:
    ego0_inv = np.linalg.inv(np.asarray(sample["ego_pose"][0], dtype=np.float64))
    pts_world = np.asarray(sample["points_world"], dtype=np.float64)
    out = np.full_like(pts_world, np.nan, dtype=np.float64)
    for t in range(pts_world.shape[0]):
        valid = np.isfinite(pts_world[t]).all(axis=1)
        if valid.any():
            hom = np.concatenate([pts_world[t, valid], np.ones((valid.sum(), 1))], axis=1)
            out[t, valid] = (ego0_inv @ hom.T).T[:, :3]
    return out


def add_point_trace(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    sample: Dict[str, Any],
    row: int,
    col: int,
    name: str,
    showlegend: bool = False,
) -> None:
    ids = [str(v) for v in sample["point_ids"]]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=color_values(len(ids)),
                colorscale="Turbo",
                symbol=point_style(sample["point_types"]),
                size=9,
                line=dict(width=0.5, color="black"),
            ),
            text=ids,
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def panel_a_html(sample: Dict[str, Any], out_html: Path) -> None:
    t_count = sample["points_3d"].shape[0]
    titles = []
    for row_name in ("Camera", "Current Ego BEV", "Canonical BEV"):
        titles.extend([f"{row_name} t{i}" for i in range(t_count)])
    fig = make_subplots(rows=3, cols=t_count, subplot_titles=titles, vertical_spacing=0.08)
    canon = canonical_points(sample)
    for t in range(t_count):
        uv = sample["camera_uv"][t]
        add_camera_image(fig, sample, t, 1, t + 1, t_count)
        add_point_trace(fig, uv[:, 0], uv[:, 1], sample, 1, t + 1, f"camera_t{t}", t == 0)
        pts = sample["points_3d"][t]
        add_point_trace(fig, pts[:, 0], pts[:, 1], sample, 2, t + 1, f"ego_t{t}", False)
        add_point_trace(fig, canon[t, :, 0], canon[t, :, 1], sample, 3, t + 1, f"canonical_t{t}", False)
        fig.update_yaxes(autorange="reversed", row=1, col=t + 1)
        fig.update_xaxes(title_text="u", row=1, col=t + 1)
        fig.update_yaxes(title_text="v", row=1, col=t + 1)
        for r in (2, 3):
            fig.update_xaxes(title_text="x", scaleanchor=f"y{(r - 1) * t_count + t + 1}", scaleratio=1, row=r, col=t + 1)
            fig.update_yaxes(title_text="y", row=r, col=t + 1)
    fig.update_layout(title="Panel A: Synchronized Views", height=900, width=max(1200, 240 * t_count))
    fig.write_html(out_html)


def panel_a_png(sample: Dict[str, Any], out_png: Path) -> None:
    t_count = sample["points_3d"].shape[0]
    fig, axes = plt.subplots(3, t_count, figsize=(4 * t_count, 10), squeeze=False)
    c = color_values(len(sample["point_ids"]))
    canon = canonical_points(sample)
    for t in range(t_count):
        if "camera_images" in sample:
            axes[0, t].imshow(sample["camera_images"][t])
        axes[0, t].scatter(sample["camera_uv"][t, :, 0], sample["camera_uv"][t, :, 1], c=c)
        if "camera_images" not in sample:
            axes[0, t].invert_yaxis()
        axes[0, t].set_title(f"Camera t{t}")
        axes[1, t].scatter(sample["points_3d"][t, :, 0], sample["points_3d"][t, :, 1], c=c)
        axes[1, t].set_title(f"Current Ego BEV t{t}")
        axes[1, t].set_aspect("equal", adjustable="datalim")
        axes[2, t].scatter(canon[t, :, 0], canon[t, :, 1], c=c)
        axes[2, t].set_title(f"Canonical BEV t{t}")
        axes[2, t].set_aspect("equal", adjustable="datalim")
    fig.suptitle("Panel A: Synchronized Views")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def shared_pe_limits(samples: Sequence[Dict[str, Any]], fixed_t: int) -> Tuple[Optional[float], Optional[float]]:
    vals = np.concatenate([np.ravel(s["pe_vectors"][fixed_t]) for s in samples])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print("Panel B PE heatmap: no finite PE values for shared color limits")
        return None, None
    return float(vals.min()), float(vals.max())


def panel_b_html(baseline: Dict[str, Any], perturbed: Dict[str, Any], out_html: Path) -> None:
    fixed_t = baseline["points_3d"].shape[0] // 2
    zmin, zmax = shared_pe_limits([baseline, perturbed], fixed_t)
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=[
            "Baseline Camera",
            "Perturbed Camera",
            "Baseline Current Ego BEV",
            "Perturbed Current Ego BEV",
            "Baseline Canonical BEV",
            "Perturbed Canonical BEV",
            "Baseline PETR PE",
            "Perturbed PETR PE",
            "",
            "",
        ],
        vertical_spacing=0.06,
    )
    for col, sample in enumerate((baseline, perturbed), start=1):
        canon = canonical_points(sample)
        add_camera_image(fig, sample, fixed_t, 1, col, 2)
        add_point_trace(fig, sample["camera_uv"][fixed_t, :, 0], sample["camera_uv"][fixed_t, :, 1], sample, 1, col, "camera", col == 1)
        add_point_trace(fig, sample["points_3d"][fixed_t, :, 0], sample["points_3d"][fixed_t, :, 1], sample, 2, col, "ego", False)
        add_point_trace(fig, canon[fixed_t, :, 0], canon[fixed_t, :, 1], sample, 3, col, "canonical", False)
        fig.add_trace(
            go.Heatmap(z=sample["pe_vectors"][fixed_t], zmin=zmin, zmax=zmax, colorscale="Viridis", colorbar=dict(title="PE") if col == 2 else None),
            row=4,
            col=col,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.add_annotation(text="Reserved for later extractor output", xref="x9 domain", yref="y9 domain", x=0.5, y=0.5, showarrow=False)
    fig.add_annotation(text="Reserved for later extractor output", xref="x10 domain", yref="y10 domain", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title="Panel B: Baseline vs Perturbed", height=1300, width=1100)
    fig.write_html(out_html)


def panel_b_png(baseline: Dict[str, Any], perturbed: Dict[str, Any], out_png: Path) -> None:
    fixed_t = baseline["points_3d"].shape[0] // 2
    zmin, zmax = shared_pe_limits([baseline, perturbed], fixed_t)
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), squeeze=False)
    for col, sample in enumerate((baseline, perturbed)):
        c = color_values(len(sample["point_ids"]))
        canon = canonical_points(sample)
        if "camera_images" in sample:
            axes[0, col].imshow(sample["camera_images"][fixed_t])
        axes[0, col].scatter(sample["camera_uv"][fixed_t, :, 0], sample["camera_uv"][fixed_t, :, 1], c=c)
        if "camera_images" not in sample:
            axes[0, col].invert_yaxis()
        axes[1, col].scatter(sample["points_3d"][fixed_t, :, 0], sample["points_3d"][fixed_t, :, 1], c=c)
        axes[1, col].set_aspect("equal", adjustable="datalim")
        axes[2, col].scatter(canon[fixed_t, :, 0], canon[fixed_t, :, 1], c=c)
        axes[2, col].set_aspect("equal", adjustable="datalim")
        im = axes[3, col].imshow(sample["pe_vectors"][fixed_t], aspect="auto", vmin=zmin, vmax=zmax)
        fig.colorbar(im, ax=axes[3, col], fraction=0.046)
        axes[4, col].axis("off")
        label = "Baseline" if col == 0 else "Perturbed"
        for row, row_name in enumerate(("Camera", "Current Ego BEV", "Canonical BEV", "PETR PE", "")):
            axes[row, col].set_title(f"{label} {row_name}".strip())
    fig.suptitle("Panel B: Baseline vs Perturbed")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite_rows = np.isfinite(x).all(axis=1)
    out = np.full((x.shape[0], x.shape[0]), np.nan, dtype=np.float64)
    if finite_rows.any():
        xv = x[finite_rows]
        norm = np.linalg.norm(xv, axis=1)
        valid_norm = norm > 0
        if (~valid_norm).any():
            print(f"cosine_matrix: {int((~valid_norm).sum())} finite vectors have zero norm")
        xv = xv[valid_norm]
        idx = np.where(finite_rows)[0][valid_norm]
        if xv.shape[0]:
            sim = (xv @ xv.T) / (np.linalg.norm(xv, axis=1)[:, None] * np.linalg.norm(xv, axis=1)[None, :])
            out[np.ix_(idx, idx)] = sim
    return out


def pca2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full((x.shape[0], 2), np.nan, dtype=np.float64)
    finite_rows = np.isfinite(x).all(axis=1)
    if finite_rows.sum() < 2:
        print("PCA: fewer than two finite PE vectors")
        return out
    xv = x[finite_rows]
    centered = xv - xv.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.shape[0] < 2 or s[1] <= 0:
        print(f"PCA: singular values={s.tolist()}")
    coords = u[:, :2] * s[:2]
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    out[finite_rows] = coords[:, :2]
    return out


def time_cosine(sample: Dict[str, Any], ref_t: int, ref_i: int) -> np.ndarray:
    pe = np.asarray(sample["pe_vectors"], dtype=np.float64)
    ref = pe[ref_t, ref_i]
    out = np.full((pe.shape[0],), np.nan, dtype=np.float64)
    if not np.isfinite(ref).all() or np.linalg.norm(ref) == 0:
        print("C3: reference PE is non-finite or has zero norm")
        return out
    for t in range(pe.shape[0]):
        cur = pe[t, ref_i]
        if np.isfinite(cur).all() and np.linalg.norm(cur) > 0:
            out[t] = float(cur @ ref / (np.linalg.norm(cur) * np.linalg.norm(ref)))
    return out


def panel_c_html(sample: Dict[str, Any], out_html: Path, ref_point: int = 0, ref_t: Optional[int] = None) -> None:
    if ref_t is None:
        ref_t = sample["pe_vectors"].shape[0] // 2
    pe = sample["pe_vectors"][ref_t]
    sim = cosine_matrix(pe)
    pca = pca2(pe)
    curve = time_cosine(sample, ref_t, ref_point)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["C1 Cosine Matrix", "C2 PCA 2D", "C3 Time Cosine"])
    fig.add_trace(go.Heatmap(z=sim, colorscale="Viridis"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=pca[:, 0],
            y=pca[:, 1],
            mode="markers",
            marker=dict(color=color_values(len(sample["point_ids"])), colorscale="Turbo", size=9),
            text=[str(v) for v in sample["point_ids"]],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=np.arange(curve.shape[0]), y=curve, mode="lines+markers", name="cosine"), row=1, col=3)
    fig.update_layout(title="Panel C: PE Structure Observation", height=520, width=1400)
    fig.write_html(out_html)


def panel_c_png(sample: Dict[str, Any], out_png: Path, ref_point: int = 0, ref_t: Optional[int] = None) -> None:
    if ref_t is None:
        ref_t = sample["pe_vectors"].shape[0] // 2
    pe = sample["pe_vectors"][ref_t]
    sim = cosine_matrix(pe)
    pca = pca2(pe)
    curve = time_cosine(sample, ref_t, ref_point)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im = axes[0].imshow(sim, aspect="auto")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_title("C1 Cosine Matrix")
    axes[1].scatter(pca[:, 0], pca[:, 1], c=color_values(len(sample["point_ids"])))
    axes[1].set_title("C2 PCA 2D")
    axes[2].plot(np.arange(curve.shape[0]), curve, marker="o")
    axes[2].set_title("C3 Time Cosine")
    fig.suptitle("Panel C: PE Structure Observation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment A PETR PE panels from .npz samples.")
    parser.add_argument("npz", nargs="+", help="One or more samples .npz files. Pass baseline then perturbed for Panel B.")
    parser.add_argument("--name", default="default")
    parser.add_argument("--out-dir", default=str(EXP_DIR / "outputs"))
    parser.add_argument("--ref-point", type=int, default=0)
    parser.add_argument("--ref-t", type=int, default=None)
    args = parser.parse_args()

    samples = [load_sample(p) for p in args.npz]
    out_dir = Path(args.out_dir)
    html_dir = out_dir / "html"
    png_dir = out_dir / "png"
    html_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    panel_a_html(samples[0], html_dir / f"panel_A_{args.name}.html")
    panel_a_png(samples[0], png_dir / f"panel_A_{args.name}.png")
    panel_c_html(samples[0], html_dir / f"panel_C_{args.name}.html", args.ref_point, args.ref_t)
    panel_c_png(samples[0], png_dir / f"panel_C_{args.name}.png", args.ref_point, args.ref_t)
    print(f"wrote Panel A and Panel C for {samples[0]['_path']}")

    if len(samples) >= 2:
        baseline = next((s for s in samples if not as_bool(s["perturbed"])), samples[0])
        perturbed = next((s for s in samples if as_bool(s["perturbed"])), samples[1])
        panel_b_html(baseline, perturbed, html_dir / f"panel_B_{args.name}.html")
        panel_b_png(baseline, perturbed, png_dir / f"panel_B_{args.name}.png")
        print("wrote Panel B")


if __name__ == "__main__":
    main()
