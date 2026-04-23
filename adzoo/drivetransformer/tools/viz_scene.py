"""BEV + 6 camera images in one figure."""
import argparse, pickle, os
import numpy as np
import matplotlib.pyplot as plt
import cv2

CAM_SOURCE_ORDER = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT',
]

CAM_LAYOUT = {
    'CAM_FRONT_LEFT':  (0, 0),
    'CAM_FRONT':       (0, 1),
    'CAM_FRONT_RIGHT': (0, 2),
    'CAM_BACK_LEFT':   (1, 0),
    'CAM_BACK':        (1, 1),
    'CAM_BACK_RIGHT':  (1, 2),
}

CLASSES = ['car','van','truck','bicycle','t_sign',
           't_cone','t_light','ped','other']
DYNAMIC_IDS = {0, 1, 2, 3, 7}

def plot_scene(d, data_root, out_path):
    fig = plt.figure(figsize=(22, 10))
    gs  = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2])

    # right: camera view
    paths = d.get('img_filenames', [])
    for i, p in enumerate(paths[:6]):
        name = CAM_SOURCE_ORDER[i] 
        r, c = CAM_LAYOUT[name]
        ax = fig.add_subplot(gs[r, c])

        full = p if (os.path.isabs(p) or os.path.exists(p)) else os.path.join(data_root, p)
        img = cv2.imread(full)
        if img is None:
            ax.text(0.5, 0.5, f'missing\n{p}', ha='center', va='center')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9)
        ax.axis('off')

    # right: BEV view
    ax = fig.add_subplot(gs[:, 3])
    _draw_bev(ax, d)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"[saved] {out_path}")


def _draw_bev(ax, d):
    # ego
    ax.plot(0, 0, 'o', color='red', markersize=12, zorder=5, label='ego')
    ax.arrow(0, 0, 0, 2, head_width=0.4, color='red', zorder=5)

    xy, yaw, lbl = d['gt_boxes_xy'], d['gt_boxes_yaw'], d['gt_labels']
    # 两类先用代理对象注册一次 label,避免每个 agent 都进 legend
    dyn_proxy_registered = False
    sta_proxy_registered = False

    for i in range(len(xy)):
        cid = int(lbl[i])
        is_dyn = cid in DYNAMIC_IDS
        name = CLASSES[cid] if 0 <= cid < len(CLASSES) else '?'

        # marker
        marker, ec = ('o', 'steelblue') if is_dyn else ('s', 'darkorange')
        ax.plot(xy[i,0], xy[i,1], marker, markersize=7,
                markerfacecolor='none', markeredgecolor=ec, mew=1.5)
        ax.annotate(name, (xy[i,0]+0.3, xy[i,1]), fontsize=7, color=ec)

        # arrow
        if is_dyn:
            # 动态物体:运动朝向(vehicle 分支公式)
            dx_a, dy_a = -np.sin(yaw[i]), -np.cos(yaw[i])
            color, label = 'steelblue', 'dynamic: heading' if not dyn_proxy_registered else None
            dyn_proxy_registered = True
        else:
            # 静态物体:面朝法向(static 分支公式)
            dx_a, dy_a = np.cos(yaw[i]), np.sin(yaw[i])
            color, label = 'darkorange', 'static: face normal' if not sta_proxy_registered else None
            sta_proxy_registered = True

        ax.arrow(xy[i,0], xy[i,1], 1.4*dx_a, 1.4*dy_a,
                 head_width=0.35, color=color, alpha=0.75,
                 length_includes_head=True)
        if label is not None:
            ax.plot([], [], color=color, marker='>', linestyle='-',
                    markersize=6, label=label)

    # his/fut
    his = np.asarray(d['ego_his_trajs']).reshape(-1, 2)
    rev_cum = np.cumsum(his[::-1], axis=0)[::-1]
    his_pts = np.vstack([-rev_cum, np.zeros((1, 2))])
    ax.plot(his_pts[:,0], his_pts[:,1], 'o-', color='darkred',
            markersize=3, label='ego history', alpha=0.8)
    fut = d['ego_fut_trajs_fix_time'].reshape(-1, 2)
    ax.plot(fut[:,0], fut[:,1], 'o-', color='green',
            markersize=3, label='ego future (fix_time)')

    ax.add_patch(plt.Rectangle((-15, -30), 30, 60, fill=False,
                                edgecolor='gray', linestyle='--'))
    ax.set_xlim(-20, 20); ax.set_ylim(-35, 35)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('lidar x (ego right)')
    ax.set_ylabel('lidar y (ego forward)')
    ax.legend(fontsize=8, loc='upper right')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/sample_baseline.pkl')
    ap.add_argument('--data-root', default='data/bench2drive')
    ap.add_argument('--out', default='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/scene.png')
    args = ap.parse_args()
    d = pickle.load(open(args.dump, 'rb'))
    plot_scene(d, args.data_root, args.out)