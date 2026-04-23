"""V4-a (修订版): 真实 memory_ego_motion 15 维结构下的 pos2posemb 响应场
输入维度语义:
  [0]      timestamp
  [1..12]  ego_pose[:3, :].flatten()  = [R(3x3)_row0, t_x, R_row1, t_y, R_row2, t_z]
           索引映射: [1,2,3, 4=tx, 5,6,7, 8=ty, 9,10,11, 12=tz]
  [13..14] memory_velo (vx, vy)
"""
import numpy as np, matplotlib.pyplot as plt

def pos2posemb(pos, num_pos_feats=12, temperature=10000):
    scale = 2 * np.pi
    pos = pos * scale
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_tmp = pos[..., None] / dim_t
    emb = np.stack((np.sin(pos_tmp[..., 0::2]),
                    np.cos(pos_tmp[..., 1::2])), axis=-1)
    return emb.reshape(*pos.shape[:-1], -1)


# --- 实验 1: 单维扰动响应图 ---
def sweep_single_dim(base_vec, dim_idx, delta_range, n=201):
    """固定其他 14 维 = base_vec,扫描第 dim_idx 维 ∈ base + delta_range"""
    deltas = np.linspace(*delta_range, n)
    embs = []
    for d in deltas:
        v = base_vec.copy()
        v[dim_idx] = base_vec[dim_idx] + d
        embs.append(pos2posemb(v, 12))
    return deltas, np.stack(embs)     # (n, 180)


def plot_single_dim_response(deltas, embs, base_val, title, ax):
    center = embs[len(embs) // 2]
    diff = np.linalg.norm(embs - center, axis=1)
    rel  = diff / (np.linalg.norm(center) + 1e-9)
    ax.plot(deltas, rel)
    ax.set_title(f'{title}\n(base={base_val:.1f})')
    ax.set_xlabel('Δ')
    ax.set_ylabel('rel L2')
    ax.grid(alpha=0.3)


def build_realistic_base(tx=2600, ty=-1800, tz=0):
    """模拟一个真实 ego_pose:车头朝 +y(来自 BEV 确认的 scene=100 场景)
    R = Rz(π/2) 的形式:
      [[0, -1, 0],
       [1,  0, 0],
       [0,  0, 1]]
    整体 3×4 flatten → [0,-1,0, tx, 1,0,0, ty, 0,0,1, tz]
    加上 timestamp=0 和 velo=(0, 9.8):
    """
    base = np.array([
        0.0,                     # timestamp
        0.0, -1.0, 0.0, tx,      # R_row0, t_x
        1.0,  0.0, 0.0, ty,      # R_row1, t_y
        0.0,  0.0, 1.0, tz,      # R_row2, t_z
        0.0, 9.8,                # velo
    ], dtype=np.float32)
    return base


# --- 实验 2: t_x 在不同绝对值下的"可分辨度" ---
def compare_scales(tx_bases=(0, 100, 1000, 2700)):
    """相同 1m 扰动,在不同绝对位置下 pos2posemb 的响应。
    若平移不变 → 四条线重合;若不平移不变 → 线形不同。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for tx in tx_bases:
        base = build_realistic_base(tx=tx)
        d, e = sweep_single_dim(base, dim_idx=4,          # index 4 = t_x
                                 delta_range=(-2.0, 2.0), n=201)
        plot_single_dim_response(d, e, tx, f'tx base={tx}m', ax)
    ax.legend([f'tx={t}' for t in tx_bases])
    ax.set_title('The response of pos2posemb to a t_x perturbation')
    plt.tight_layout()
    plt.savefig('/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/pe_scale_invariance.png', dpi=120)
    print('[saved] /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/pe_scale_invariance.png')


# --- 实验 3: 15 维各自的敏感度对比 ---
def per_dim_sensitivity(base, delta_mag=0.5):
    """对每一维施加 ±delta_mag 的扰动,看 ||ΔPE|| 有多大。
    找出"哪几维对 PE 变化贡献最大" """
    center = pos2posemb(base, 12)
    sens = []
    for i in range(15):
        v = base.copy()
        v[i] += delta_mag
        sens.append(np.linalg.norm(pos2posemb(v, 12) - center))
    return sens

def response_curve(delta_range=(-5, 5), n=2001):
    """单张图:||ΔPE||_rel 作为 Δ 的函数,揭示 PE 的周期结构"""
    base = np.zeros(1, dtype=np.float32)
    deltas = np.linspace(*delta_range, n)
    center = pos2posemb(base, 12)
    resp = []
    for d in deltas:
        v = base.copy(); v[0] = d
        resp.append(np.linalg.norm(pos2posemb(v, 12) - center))
    resp = np.array(resp) / np.linalg.norm(center)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(deltas, resp)
    ax.set_xlabel('Δ'); ax.set_ylabel('rel L2')
    ax.set_title('pos2posemb response function')
    ax.grid(alpha=0.3)
    # 标注局部极大/极小
    # 手动 mark 几个你扰动设计可能用到的幅度
    for mark in [0.25, 0.5, 1.0, 1.5, 2.0]:
        idx = np.argmin(np.abs(deltas - mark))
        ax.axvline(mark, color='r', alpha=0.2)
        ax.annotate(f'Δ={mark}\nrel={resp[idx]:.3f}',
                    (mark, resp[idx]), fontsize=8)
    plt.tight_layout()
    plt.savefig('/gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/pe_response_curve.png', dpi=120)
    print('[saved] /gs/bs/tga-RLA/qdeng/DriveTransformer/adzoo/drivetransformer/tools/pe_response_curve.png')

if __name__ == '__main__':
    # --- 实验 2 + 3 --- 
    compare_scales(tx_bases=(0, 100, 1000, 2700))
    
    base = build_realistic_base(tx=2600, ty=-1800)
    s = per_dim_sensitivity(base, delta_mag=0.5)
    response_curve(delta_range=(-5, 5), n=2001)

    labels = ['ts', 'R00','R01','R02','tx', 'R10','R11','R12','ty',
              'R20','R21','R22','tz', 'vx','vy']
    print('\n每维施加 +0.5 扰动后 ||ΔPE||:')
    for i, (lbl, v) in enumerate(zip(labels, s)):
        bar = '█' * int(v * 20)
        print(f'  [{i:2d}] {lbl:4s} {v:8.4f}  {bar}')