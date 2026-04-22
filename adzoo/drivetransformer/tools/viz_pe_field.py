"""V4: PE 空间场热力图
采样一个 ego 位姿网格，对每个位姿算 PE，可视化 PE 在 (x,y) 上的响应。

这里有个关键选择点：DriveTransformer 的 'ego_pose_pe' 具体是什么？
看 train.py 里 torch.compile 列表，它是 pts_bbox_head 的一个成员。
你需要先 grep 找到它的实现，确认：
  - 输入是 ego_pose (4x4) 还是 ego_translation (3,) 还是 can_bus?
  - 输出维度？是 sin/cos PE 还是 MLP？
下面是个占位实现，把它替换成真实的 encoder。"""

import numpy as np, torch, matplotlib.pyplot as plt

def build_pe_encoder(cfg_path):
    # TODO: 从 config 加载模型，取出 model.pts_bbox_head.ego_pose_pe
    # from mmcv.models import build_model
    # cfg = Config.fromfile(cfg_path)
    # model = build_model(cfg.model)
    # return model.pts_bbox_head.ego_pose_pe
    raise NotImplementedError("请先 grep ego_pose_pe 的实现")

def sweep_pe_field(encoder, grid_range=(-10, 10), n=41):
    """采样 n×n 个位姿，算 PE，返回 (n, n, C) 张量"""
    xs = np.linspace(*grid_range, n)
    ys = np.linspace(*grid_range, n)
    pes = []
    with torch.no_grad():
        for y in ys:
            row = []
            for x in xs:
                pose = torch.eye(4)
                pose[0, 3] = x; pose[1, 3] = y
                pe = encoder(pose.unsqueeze(0))   # (1, C) 或 (1, C, ...)
                row.append(pe.squeeze().cpu().numpy())
            pes.append(row)
    return np.array(pes), xs, ys

def plot_pe_field(pe_grid, xs, ys, out='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/pe_field.png'):
    """画若干维的 PE 随 (x,y) 变化的热力图 + 1D cut"""
    C = pe_grid.shape[-1]
    # 选几个代表性 channel（低频、中频、高频）
    chans = [0, 1, C // 4, C // 2, C - 2, C - 1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), chans):
        im = ax.imshow(pe_grid[..., c], extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       origin='lower', cmap='RdBu')
        ax.set_title(f'PE channel {c}')
        plt.colorbar(im, ax=ax)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    
    # 关键指标：最小可分辨位移
    # 在 x=y=0 处算 PE，然后算 ||PE(dx, 0) - PE(0, 0)|| 作为 dx 的函数
    n = pe_grid.shape[0]; c = n // 2
    center = pe_grid[c, c]
    dx_diffs = [np.linalg.norm(pe_grid[c, c+k] - center) / np.linalg.norm(center)
                for k in range(1, n - c)]
    print(f"\n[V4] PE 相对响应幅度（沿 +x 方向）:")
    for k, d in enumerate(dx_diffs[:10], 1):
        print(f"  dx={xs[c+k]-xs[c]:+.2f}m  rel_diff={d:.4f}")