# Experiment A: PETR-style Sensor PE Visualization

## Setup

Run commands from the repository root:

```bash
cd /gs/bs/tga-RLA/qdeng/DriveTransformer
```

Python dependencies used by these scripts:

- `numpy`
- `torch`
- `matplotlib`
- `plotly`
- `pyyaml` for YAML configs

The PETR-style PE path is extracted from:

- `adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py`
- `DriveTransformerlHead.__init__`: `coords_d`, `position_range`, `img_position_encoder`
- `DriveTransformerlHead.img_3d_position_embedding`: image-token grid, depth bins, `lidar2img.inverse()`, normalized 3D coordinates, `inverse_sigmoid`, MLP

Set `pe.checkpoint_path` in the config to a DriveTransformer checkpoint containing `pts_bbox_head.img_position_encoder.*`. The sampler refuses to use an untrained MLP unless `pe.allow_untrained_mlp: true` is set.

## Usage

Bench2Drive baseline sampling:

```bash
/gs/bs/tga-RLA/qdeng/anaconda3/envs/drivetransformer/bin/python \
  experiments/exp_a_petr_pe_viz/sample.py \
  --config experiments/exp_a_petr_pe_viz/configs/bench2drive.yaml \
  --override '{"pe": {"checkpoint_path": "path/to/drivetransformer.pth"}}'
```

Bench2Drive perturbed sampling:

```bash
/gs/bs/tga-RLA/qdeng/anaconda3/envs/drivetransformer/bin/python \
  experiments/exp_a_petr_pe_viz/sample.py \
  --config experiments/exp_a_petr_pe_viz/configs/perturb_examples/bench2drive_dx0_dy05_dyaw5.yaml \
  --override '{"pe": {"checkpoint_path": "path/to/drivetransformer.pth"}}'
```

Render Bench2Drive panels:

```bash
/gs/bs/tga-RLA/qdeng/anaconda3/envs/drivetransformer/bin/python \
  experiments/exp_a_petr_pe_viz/render.py \
  experiments/exp_a_petr_pe_viz/outputs/data/samples_baseline.npz \
  experiments/exp_a_petr_pe_viz/outputs/data/samples_perturb_dx0_dy0.5_dyaw5.npz \
  --name b2d_anchor
```

nuScenes baseline sampling:

```bash
python experiments/exp_a_petr_pe_viz/sample.py \
  --config experiments/exp_a_petr_pe_viz/configs/default.yaml \
  --override '{"pe": {"checkpoint_path": "path/to/drivetransformer.pth"}}'
```

Perturbed sampling:

```bash
python experiments/exp_a_petr_pe_viz/sample.py \
  --config experiments/exp_a_petr_pe_viz/configs/perturb_examples/dx0_dy05_dyaw5.yaml \
  --override '{"pe": {"checkpoint_path": "path/to/drivetransformer.pth"}}'
```

Render panels:

```bash
python experiments/exp_a_petr_pe_viz/render.py \
  experiments/exp_a_petr_pe_viz/outputs/data/samples_baseline.npz \
  experiments/exp_a_petr_pe_viz/outputs/data/samples_perturb_dx0_dy0.5_dyaw5.npz \
  --name anchor_default
```

Sampling modes are configured by `sample.mode`:

- `mode_anchor`: static anchors from `sample.static_anchors`
- `mode_dynamic`: GT centers selected by `gt_ids`
- `mode_ray`: depth samples along one configured camera pixel

For Bench2Drive split metadata, `dataset.ann_file` can point to `data/infos/b2d_infos_v1_val_drivetransformer_meta.pkl`; the sampler expands route pickle files from `dataset.info_root / infos_dir_name`. Set `dataset.data_root` to the directory containing `v1/<route>/camera/...` if you want camera image backgrounds stored in the `.npz`.

The sampler writes baseline and perturbed runs as separate `.npz` files. It does not compare two runs.

## Output structure

```text
experiments/exp_a_petr_pe_viz/
├── extract/
│   ├── petr_pe_extractor.py
│   └── pe_base.py
├── sample.py
├── render.py
├── configs/
│   ├── default.yaml
│   └── perturb_examples/
├── outputs/
│   ├── data/
│   │   ├── samples_baseline.npz
│   │   └── samples_perturb_*.npz
│   ├── html/
│   │   ├── panel_A_<config>.html
│   │   ├── panel_B_<config>.html
│   │   └── panel_C_<config>.html
│   └── png/
└── README.md
```

Each `.npz` stores:

- `points_3d`: `[T, N, 3]`
- `points_world`: `[T, N, 3]`
- `pe_vectors`: `[T, N, C]`
- `point_ids`: `[N]`
- `point_types`: `[N]`
- `camera_uv`: `[T, N, 2]`
- `ego_pose`: `[T, 4, 4]`
- `perturbed`: scalar bool
- `perturb_config`: dict object
- `frame_meta`: object array of per-frame dicts

## Observations

[本节由实验者填写,Codex 不要预填任何内容]
