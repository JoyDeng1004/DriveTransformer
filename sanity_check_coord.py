"""坐标系验证脚本：把 dataset 里一个 sample 的 GT box在 BEV 和 CAM_FRONT 上同时画出来对照。"""
import os, sys
sys.path.insert(0, os.getcwd())
import adzoo.drivetransformer.mmdet3d_plugin

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.lines import Line2D

# ============ Config ============
CONFIG_PATH = 'adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py'
SAMPLE_IDX  = 96
OUT_PATH    = './sanity_check_coord.png'
PC_RANGE    = [-15, -30, -2, 15, 30, 2]
CAM_FRONT_IDX = 0
EGO_LWH     = (4.5, 2.0, 1.5)

# ============ Setup ============
sys.path.insert(0, os.getcwd())
from mmcv import Config
from mmcv.datasets import build_dataset

cfg = Config.fromfile(CONFIG_PATH)
# 用 train pipeline 但只跑前几个 transform 拿到原始数据
dataset = build_dataset(cfg.data.val)
sample = dataset[SAMPLE_IDX]

# ============ 提取关键字段 ============
def unwrap(x):
    return x.data if hasattr(x, 'data') else x

gt_boxes_3d = unwrap(sample['gt_bboxes_3d'])  # LiDARInstance3DBoxes
gt_corners  = gt_boxes_3d.corners.numpy()      # [N, 8, 3]，8 个角点
gt_centers  = gt_boxes_3d.gravity_center.numpy()  # [N, 3]
gt_yaws     = gt_boxes_3d.tensor[:, 6].numpy()  # [N]

img_metas_raw   = unwrap(sample['img_metas'])
img_metas = img_metas_raw[0] if isinstance(img_metas_raw, dict) and 0 in img_metas_raw else img_metas_raw
for key, value in img_metas.items():
    print(f"key: {key:}, value: {value}")

# 找一个前方车多的 sample
# for idx in range(100):
#     s = dataset[idx]
#     centers = unwrap(s['gt_bboxes_3d']).gravity_center.numpy()
#     n_front = ((centers[:, 1] > 5) & (centers[:, 1] < 25) & (np.abs(centers[:, 0]) < 5)).sum()
#     if n_front >= 1:
#         print(f'sample {idx}: {n_front} agents in front')

lidar2img = unwrap(img_metas['lidar2img']).numpy()  # [num_cam, 4, 4]
img       = unwrap(sample['img']).numpy()     # [num_cam, C, H, W] or [N, num_cam, C, H, W]
if img.ndim == 5:
    img = img[0]
img_front   = img[CAM_FRONT_IDX].transpose(1, 2, 0)  # [H, W, 3]
# 反归一化（如果 dataset 做了 normalize）
img_norm_cfg = cfg.img_norm_cfg
mean, std = np.array(img_norm_cfg['mean']), np.array(img_norm_cfg['std'])
img_front = (img_front * std + mean).clip(0, 255).astype(np.uint8)

# GT boxes
gt_boxes_3d = unwrap(sample['gt_bboxes_3d'])
gt_corners  = gt_boxes_3d.corners.numpy()      # [N, 8, 3]
gt_centers  = gt_boxes_3d.gravity_center.numpy()
print(f'Loaded {len(gt_centers)} GT boxes')
print(f'GT centers (first 5):\n{gt_centers[:5]}')
print(f'GT center xy range: x=[{gt_centers[:,0].min():.1f}, {gt_centers[:,0].max():.1f}], '
      f'y=[{gt_centers[:,1].min():.1f}, {gt_centers[:,1].max():.1f}]')

# ============ 画图 ============
fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(16, 8))

# ---- BEV ----
# ★ 注意：BEV 画图时通常把 x（前方）放在画面"上方"，y（左）放在画面"左方"
ax_bev.set_xlim(PC_RANGE[0], PC_RANGE[3])  # x: ±15
ax_bev.set_ylim(PC_RANGE[1], PC_RANGE[4])  # y: ±30
ax_bev.set_xlabel('lidar x (m)')
ax_bev.set_ylabel('lidar y (m)')
ax_bev.set_title(f'BEV (lidar coord, sample {SAMPLE_IDX})')
ax_bev.set_aspect('equal')
ax_bev.grid(alpha=0.3)
ax_bev.axhline(0, color='k', linewidth=0.5)
ax_bev.axvline(0, color='k', linewidth=0.5)

# 画 ego（黑色矩形 + 朝向箭头）
ego_size = 1.5
ego_rect = Rectangle((-ego_size/2, -ego_size/2), ego_size, ego_size,
                     fill=True, facecolor='black', alpha=0.8)
ax_bev.add_patch(ego_rect)
ax_bev.text(0, -2.5, 'ego', ha='center', fontsize=10, fontweight='bold')
# 画显眼的方向箭头：+x 红色，+y 蓝色
ax_bev.annotate('', xy=(8, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax_bev.text(9, 0, '+x', color='red', fontsize=12, fontweight='bold', va='center')
ax_bev.annotate('', xy=(0, 8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
ax_bev.text(0, 9, '+y', color='blue', fontsize=12, fontweight='bold', ha='center')

# 找最近的5个agent
dists = np.linalg.norm(gt_centers[:, :2], axis=1)
near_idx = np.argsort(dists)[:5]
colors = ['tab:orange', 'tab:cyan', 'tab:green', 'tab:purple', 'tab:brown']

# 画所有 GT box
for i, corners in enumerate(gt_corners):
    is_near = i in near_idx
    rank = list(near_idx).index(i) if is_near else -1
    color = colors[rank] if is_near else 'gray'
    lw = 2.5 if is_near else 0.6
    alpha = 1.0 if is_near else 0.4
    bot = corners[:4, :]
    poly_x = bot[:, 0]
    poly_y = bot[:, 1]
    ax_bev.fill(poly_x, poly_y, facecolor=color, alpha=alpha*0.3)
    ax_bev.plot(np.append(poly_x, poly_x[0]),
                np.append(poly_y, poly_y[0]),
                color=color, linewidth=lw)
    if is_near:
        cx, cy = gt_centers[i, 0], gt_centers[i, 1]
        ax_bev.text(cx, cy + 1.0, f'A{rank}',
                    color=color, fontsize=12, fontweight='bold',
                    ha='center',
                    bbox=dict(facecolor='white', alpha=0.7,
                              edgecolor=color, pad=1))
# ---- CAM_FRONT ----
ax_cam.imshow(img_front)
ax_cam.set_title('CAM_FRONT')
ax_cam.axis('off')

# 把 GT box 投影到前视图
def project_to_img(pts_3d, lidar2img_mat):
    """ pts_3d: [N, 3] in lidar coord -> [N, 2] in image """
    N = pts_3d.shape[0]
    pts_h = np.concatenate([pts_3d, np.ones((N, 1))], axis=1)  # [N, 4]
    pts_img = pts_h @ lidar2img_mat.T  # [N, 4]
    pts_2d = pts_img[:, :2] / np.maximum(pts_img[:, 2:3], 1e-5)
    in_front = pts_img[:, 2] > 0.1
    return pts_2d, in_front

mat = lidar2img[CAM_FRONT_IDX]
H_img, W_img = img_front.shape[:2]
projected_count = 0
for rank, i in enumerate(near_idx):
    color = colors[rank]
    corners = gt_corners[i]
    pts_2d, in_front = project_to_img(corners, mat)
    if not in_front.all():
        print(f'  A{rank} (center={gt_centers[i,:2]}): not all corners in front, skip')
        continue
    # 检查至少有部分点在图像范围内
    in_img = (pts_2d[:, 0] > -200) & (pts_2d[:, 0] < W_img + 200) & \
             (pts_2d[:, 1] > -200) & (pts_2d[:, 1] < H_img + 200)
    if not in_img.any():
        print(f'  A{rank} (center={gt_centers[i,:2]}): outside image bounds')
        continue
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for s, e in edges:
        x0, y0 = pts_2d[s]
        x1, y1 = pts_2d[e]
        ax_cam.plot([x0, x1], [y0, y1], color=color, linewidth=2)
    cx, cy = pts_2d.mean(axis=0)
    if 0 < cx < W_img and 0 < cy < H_img:
        ax_cam.text(cx, cy, f'A{rank}',
                    color='white', fontsize=14, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='white'))
        projected_count += 1
    print(f'  A{rank} (center=({gt_centers[i,0]:.1f},{gt_centers[i,1]:.1f})): '
          f'projected to img center ({cx:.0f}, {cy:.0f})')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved to {OUT_PATH}')