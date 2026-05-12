#!/usr/bin/env bash
set -euo pipefail

### Usage:
# cd /gs/bs/tga-RLA/qdeng/DriveTransformer
# bash adzoo/drivetransformer/tools/planner_ablation.sh

channels=(none history cmd pose_frame can_bus label full)

for ch in "${channels[@]}"; do
      python /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/perturb_sample_se2_oracle.py \
      --config /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py \
      --checkpoint /gs/bs/tga-RLA/qdeng/DriveTransformer/ckpts/drivetransformer_large.pth \
      --idx 0 \
      --dx -1.0 \
      --dy 0.0 \
      --dtheta 0.0 \
      --perturb-channels "$ch" \
      --out-dir outputs/se2_oracle_ablation/neg_idx0 \
      --device cuda:0
done
