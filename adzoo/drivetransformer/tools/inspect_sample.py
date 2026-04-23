"""
usage: python -m adzoo.drivetransformer.tools.inspect_sample --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py --idx 100
"""
import argparse, pickle, copy
import numpy as np
import torch
import importlib
from mmcv.utils import Config
from mmcv.datasets import build_dataset
from mmcv.parallel import DataContainer


def dump_sample(sample, tag):
    """Package the key values into a flat dictionary to facilitate subsequent diffing."""
    d = {}
    for i, k in enumerate(sample.keys()):
        v = sample[k]
        print(f"  [{i:02d}] {k:30s} type={type(v)}")
        if hasattr(v, 'data'):
            print(f"       └─ data type = {type(v.data)}")
    img_metas = sample['img_metas'].data  # dict of per-frame metas
    last = img_metas[max(img_metas.keys())]  # meta of the last frame
    
    d['ego_pose'] = sample['ego_pose'].data.numpy()        # (T, 4, 4)
    d['ego_pose_inv'] = sample['ego_pose_inv'].data.numpy()
    d['can_bus_last'] = last['can_bus'].copy()                  # 18,
    
    for k in ['ego_his_trajs', 'ego_fut_trajs_fix_time',
              'ego_fut_trajs_fix_dist', 'ego_fut_cmd', 'ego_lcf_feat']:
        if k in sample:
            v = sample[k]
            if isinstance(v, DataContainer):
                d[k] = v.data.numpy()
            else:
                d[k] = np.asarray(v)
    
    gt = sample['gt_bboxes_3d'].data.tensor.numpy()  # (N, 9 or 10)
    d['gt_boxes_xy'] = gt[:, :2]
    d['gt_boxes_yaw'] = gt[:, 6]
    d['gt_labels']   = sample['gt_labels_3d'].data.numpy()
    d['lidar2img']   = sample['lidar2img'].data.numpy()
    
    print(f"\n===== [{tag}] idx sample summary =====")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:30s} shape={str(v.shape):20s} "
                  f"min={v.min():+.3f} max={v.max():+.3f} mean={v.mean():+.3f}")
    
    # --- camera image paths (for viz_scene cross-check) ---
    meta_keys = list(last.keys())
    # Bench2Drive里叫 'filename'
    fname_key = next((k for k in ('filename', 'img_filename') if k in last), None)
    if fname_key is None:
        print(f"  [warn] no filename-like key in img_metas. keys = {meta_keys}")
    else:
        paths = list(last[fname_key])
        d['img_filenames'] = paths
        print(f"  img_filenames ({fname_key}, n={len(paths)})")
        for p in paths:
            print(f"    {p}")

    return d

def sanity_checks(d):
    """Geometric self-consistent assert"""
    ep, epi = d['ego_pose'], d['ego_pose_inv']

    # float32 - relative error
    I_approx = np.einsum('tij,tjk->tik', ep, epi)
    abs_err = np.abs(I_approx - np.eye(4)[None]).max()
    scale = np.abs(ep[..., :3, 3]).max()
    rel_err = abs_err / max(scale, 1.0)
    print(f"  [check] abs_err={abs_err:.2e}, scale={scale:.1f}m, "
          f"rel_err={rel_err:.2e}")
    assert rel_err < 1e-6, "The relative error is too large"
    
    # float64 - re-calculate inv
    ep64 = ep.astype(np.float64)
    epi_recomp = np.linalg.inv(ep64[0])
    recomp_diff = np.abs(epi_recomp - epi[0].astype(np.float64)).max()
    print(f"  [check] The maximum difference between the recalculated inv value and the stored value for float64. = {recomp_diff:.2e}")

def identity_perturb_regression(dataset, idx):
    """Run the same idx twice to confirm that the dataset output is completely deterministic."""
    np.random.seed(0); torch.manual_seed(0)
    s1 = dataset[idx]
    np.random.seed(0); torch.manual_seed(0)
    s2 = dataset[idx]
    
    d1 = dump_sample(s1, 'run1')
    d2 = dump_sample(s2, 'run2')
    
    print("\n===== [E1] identity regression diff =====")
    for k in d1:
        if isinstance(d1[k], np.ndarray) and d1[k].shape == d2[k].shape:
            diff = np.abs(d1[k] - d2[k]).max()
            flag = "✓" if diff < 1e-8 else "✗"
            print(f"  {flag} {k:30s} max|diff|={diff:.2e}")
    return d1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--idx', type=int, default=100)
    ap.add_argument('--dump', default='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/sample_baseline.pkl')
    args = ap.parse_args()
    
    cfg = Config.fromfile(args.config)
    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    
    dataset = build_dataset(cfg.data.train)
    print(f"dataset len = {len(dataset)}")
    
    d = identity_perturb_regression(dataset, args.idx)
    sanity_checks(d)
    
    with open(args.dump, 'wb') as f:
        pickle.dump(d, f)
    print(f"\n[saved] baseline dump -> {args.dump}")


if __name__ == '__main__':
    main()