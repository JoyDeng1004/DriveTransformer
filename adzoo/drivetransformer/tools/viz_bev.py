"""Read the pkl saved in inspect_sample.py and draw a BEV overview diagram."""
import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_bev(d, out_path='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/bev.png'):
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # ego at origin with heading arrow
    ax.plot(0, 0, 'o', color='red', markersize=14, label='ego', zorder=5)
    ax.arrow(0, 0, 2, 0, head_width=0.3, color='orange', zorder=5)
    ax.arrow(0, 0, 0, 2, head_width=0.3, color='red', zorder=5)
    ax.text(2.2, 0, '+x', color='orange')
    ax.text(0.2, 2.2, '+y', color='red')
    
    # agents
    xy = d['gt_boxes_xy']
    yaw = d['gt_boxes_yaw']
    lbl = d['gt_labels']
    for i in range(len(xy)):
        ax.plot(xy[i,0], xy[i,1], 's', markersize=6, color='steelblue')
        ax.arrow(xy[i,0], xy[i,1],
                 1.5*np.cos(yaw[i]), 1.5*np.sin(yaw[i]),
                 head_width=0.3, color='steelblue', alpha=0.6)
        ax.annotate(str(lbl[i]), (xy[i,0], xy[i,1]), fontsize=7)
    
    # histories/futures
    if 'ego_his_trajs' in d:
        his = np.asarray(d['ego_his_trajs']).reshape(-1, 2)   # differential offsets
        # reconstruct historical points in CURRENT lidar frame
        # offsets satisfy: d[j] = p[j+1] - p[j], with p[-1] = current = (0,0)
        rev_cum = np.cumsum(his[::-1], axis=0)[::-1]
        his_pts = np.vstack([-rev_cum, np.zeros((1, 2))])
        ax.plot(his_pts[:, 0], his_pts[:, 1], 'o-', color='darkred',
                markersize=3, label='his', alpha=0.8)
    if 'ego_fut_trajs_fix_time' in d:
        # ego_fut_trajs_fix_time是当前lidar系下的未来绝对点序列
        print('ego_fut_trajs_fix_time raw =', d['ego_fut_trajs_fix_time'][:5])
        fut = d['ego_fut_trajs_fix_time'].reshape(-1, 2)
        ax.plot(fut[:,0], fut[:,1], 'o-', color='green',
                markersize=3, label='fut_time')
    if 'ego_fut_trajs_fix_dist' in d:
        print('ego_fut_trajs_fix_dist raw =', d['ego_fut_trajs_fix_dist'][:5])
        fd = d['ego_fut_trajs_fix_dist']
        # fix_dist 输出是 angle-only (use_angle_as_dis_traj=True)，
        print(f"[V1] fix_dist shape={fd.shape}")
    
    # point cloud range 边框
    for x, y, w, h in [(-15, -30, 30, 60)]:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                    edgecolor='gray', linestyle='--'))
    
    ax.set_xlim(-20, 20); ax.set_ylim(-35, 35)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('lidar x (right)'); ax.set_ylabel('lidar y (front)')
    ax.legend()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[saved] {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/sample_baseline.pkl')
    ap.add_argument('--out', default='/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/bev.png')
    args = ap.parse_args()
    d = pickle.load(open(args.dump, 'rb'))
    plot_bev(d, args.out)