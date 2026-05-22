"""Compare recovery behavior before/after planning-side fine-tuning.

This wrapper runs perturb_sample_se2_oracle.py for a baseline checkpoint and a
fine-tuned checkpoint, then computes metrics in the perturbed raw lidar frame and
in the original lidar frame. The recovery question is answered primarily by
raw_gt_error and raw_compensation_error, not by the coordinate-transform-induced
old-frame shift.
"""

import argparse
import csv
import glob
import os
import os.path as osp
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np


def fmt_num_for_filename(x) -> str:
    return str(x).replace('/', '_')


def make_se2_delta(dx: float, dy: float, dtheta: float) -> np.ndarray:
    c, s = np.cos(dtheta), np.sin(dtheta)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    delta[0, 3] = dx
    delta[1, 3] = dy
    return delta


def invert_pose(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_xy(points_xy: np.ndarray, dst_from_src: np.ndarray) -> np.ndarray:
    n = points_xy.shape[0]
    pts = np.concatenate(
        [points_xy, np.zeros((n, 1), dtype=np.float64), np.ones((n, 1), dtype=np.float64)],
        axis=1,
    )
    return (dst_from_src @ pts.T).T[:, :2].astype(np.float32)


def read_traj(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=',')
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :2]


def write_traj(path: str, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=',', fmt='%.6f')


def find_one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    if len(matches) > 1:
        raise RuntimeError(f'Pattern matched multiple files: {pattern}\n' + '\n'.join(matches))
    return matches[0]


def run_diagnostic(args, checkpoint_name: str, checkpoint_path: str, idx: int, dx: float) -> str:
    ckpt_out_dir = osp.join(args.out_dir, checkpoint_name)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        args.diagnostic_script,
        '--config', args.config,
        '--checkpoint', checkpoint_path,
        '--idx', str(idx),
        '--dx', str(dx),
        '--dy', str(args.dy),
        '--dtheta', str(args.dtheta),
        '--perturb-channels', 'full',
        '--cmd-source', 'route_command',
        '--out-dir', ckpt_out_dir,
        '--device', args.device,
        '--score-mode', args.score_mode,
    ]
    if not args.skip_run:
        print('[run]', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return ckpt_out_dir


def load_run_outputs(run_dir: str, idx: int, dx: float, dy: float, dtheta: float) -> Dict[str, np.ndarray]:
    suffix = f'full_idx{idx}_*dx{fmt_num_for_filename(dx)}_dy{fmt_num_for_filename(dy)}_dtheta{fmt_num_for_filename(dtheta)}.csv'
    paths = {
        'pred_orig_old': find_one(osp.join(run_dir, 'traj_original_pred_' + suffix)),
        'pred_pert_new': find_one(osp.join(run_dir, 'traj_perturbed_pred_' + suffix)),
        'gt_orig_old': find_one(osp.join(run_dir, 'traj_original_gt_' + suffix)),
        'gt_pert_new': find_one(osp.join(run_dir, 'traj_perturbed_gt_' + suffix)),
        'pred_pert_in_old': find_one(osp.join(run_dir, 'traj_perturbed_pred_in_original_lidar_' + suffix)),
        'gt_pert_in_old': find_one(osp.join(run_dir, 'traj_perturbed_gt_in_original_lidar_' + suffix)),
    }
    out = {k: read_traj(v) for k, v in paths.items()}
    delta = make_se2_delta(dx, dy, dtheta)  # old_from_new
    new_from_old = invert_pose(delta)
    out['expected_compensated_new'] = transform_xy(out['pred_orig_old'], new_from_old)
    expected_path = osp.join(
        run_dir,
        f'traj_expected_compensated_new_full_idx{idx}_dx{fmt_num_for_filename(dx)}_dy{fmt_num_for_filename(dy)}_dtheta{fmt_num_for_filename(dtheta)}.csv',
    )
    write_traj(expected_path, out['expected_compensated_new'])
    return out


def align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def metric_row(error_name: str, error: np.ndarray) -> Dict[str, float]:
    if error.size == 0:
        point = np.zeros((0,), dtype=np.float32)
    else:
        point = np.linalg.norm(error, axis=1)
    return {
        'error_name': error_name,
        'mean_point_error': float(point.mean()) if len(point) else 0.0,
        'final_point_error': float(point[-1]) if len(point) else 0.0,
        'mean_x_error': float(error[:, 0].mean()) if len(error) else 0.0,
        'final_x_error': float(error[-1, 0]) if len(error) else 0.0,
        'mean_y_error': float(error[:, 1].mean()) if len(error) else 0.0,
        'final_y_error': float(error[-1, 1]) if len(error) else 0.0,
    }


def compute_rows(checkpoint_name: str, idx: int, dx: float, dy: float, dtheta: float, out: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    pairs = {
        'raw_gt_error': ('pred_pert_new', 'gt_pert_new'),
        'raw_compensation_error': ('pred_pert_new', 'expected_compensated_new'),
        'old_frame_error': ('pred_pert_in_old', 'gt_orig_old'),
        'raw_shift': ('pred_pert_new', 'pred_orig_old'),
        'expected_raw_shift': ('expected_compensated_new', 'pred_orig_old'),
    }
    rows = []
    for name, (lhs_key, rhs_key) in pairs.items():
        lhs, rhs = align(out[lhs_key], out[rhs_key])
        row = metric_row(name, lhs - rhs)
        row.update({
            'checkpoint': checkpoint_name,
            'idx': idx,
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
        })
        rows.append(row)
    return rows


def write_summary(path: str, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    os.makedirs(osp.dirname(path), exist_ok=True)
    fieldnames = [
        'checkpoint', 'idx', 'dx', 'dy', 'dtheta', 'error_name',
        'mean_point_error', 'final_point_error',
        'mean_x_error', 'final_x_error', 'mean_y_error', 'final_y_error',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py')
    ap.add_argument('--diagnostic-script', default='adzoo/drivetransformer/tools/perturb_sample_se2_oracle.py')
    ap.add_argument('--baseline-checkpoint', required=True)
    ap.add_argument('--finetuned-checkpoint', required=True)
    ap.add_argument('--baseline-name', default='baseline')
    ap.add_argument('--finetuned-name', default='finetuned')
    ap.add_argument('--idx', nargs='+', type=int, default=[100])
    ap.add_argument('--dx', nargs='+', type=float, default=[1.0, -1.0])
    ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dtheta', type=float, default=0.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--score-mode', default='mode0')
    ap.add_argument('--out-dir', default='outputs/recovery_finetune/eval_compare')
    ap.add_argument('--summary-csv', default=None)
    ap.add_argument('--skip-run', action='store_true')
    args = ap.parse_args()

    checkpoints = [
        (args.baseline_name, args.baseline_checkpoint),
        (args.finetuned_name, args.finetuned_checkpoint),
    ]
    all_rows = []
    for checkpoint_name, checkpoint_path in checkpoints:
        for idx in args.idx:
            for dx in args.dx:
                run_dir = run_diagnostic(args, checkpoint_name, checkpoint_path, idx, dx)
                outputs = load_run_outputs(run_dir, idx, dx, args.dy, args.dtheta)
                all_rows.extend(compute_rows(checkpoint_name, idx, dx, args.dy, args.dtheta, outputs))

    summary_csv = args.summary_csv or osp.join(args.out_dir, 'recovery_summary.csv')
    write_summary(summary_csv, all_rows)
    print(f'[summary] wrote {summary_csv}')


if __name__ == '__main__':
    main()
