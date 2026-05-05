import numpy as np
from mmcv.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from .bench2drive_drivetransformer_dataset import B2D_DriveTransformer_Dataset
import copy
from pyquaternion import Quaternion
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

@DATASETS.register_module()
class B2D_DriveTransformer_Perturb_Dataset(B2D_DriveTransformer_Dataset):
    """自洽重规划增广。Δ=0 必须与父类逐元素等同。"""

    def __init__(self,
                 perturb_enabled=False,
                 perturb_scale_xy=0.4,
                 perturb_scale_yaw=0.1,
                 perturb_prob=0.5,
                 perturb_seed=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_enabled = perturb_enabled
        self.perturb_scale_xy = float(perturb_scale_xy)
        self.perturb_scale_yaw = float(perturb_scale_yaw)
        self.perturb_prob = float(perturb_prob)
        self._rng = (np.random.RandomState(perturb_seed)
                     if perturb_seed is not None else np.random)

    def _sample_perturbation(self):
        if not self.perturb_enabled:
            return np.zeros(2, dtype=np.float64), 0.0
        if self._rng.rand() > self.perturb_prob:
            return np.zeros(2, dtype=np.float64), 0.0
        dxy = self._rng.uniform(-self.perturb_scale_xy,
                                 self.perturb_scale_xy, size=2).astype(np.float64)
        dyaw = float(self._rng.uniform(-self.perturb_scale_yaw,
                                        self.perturb_scale_yaw))
        return dxy, dyaw

    def _detach_from_cache(self, input_dict):
        """对与缓存 info 共享的可变对象断引用,避免扰动污染 get_data_by_index 的 cache。"""
        if 'sensors' in input_dict:
            input_dict['sensors'] = copy.deepcopy(input_dict['sensors'])
        return input_dict

    def _perturb_world_to_lidar_chain(self, input_dict, dxy, dyaw):
        """就地更新 ego_translation/ego_yaw/ego_pose/ego_pose_inv/world2lidar。
        float64 内部计算,写回时 cast 回原 dtype。"""
        lidar2ego = np.asarray(input_dict['sensors']['LIDAR_TOP']['lidar2ego'], dtype=np.float64)

        t_old = np.asarray(input_dict['ego_translation'], dtype=np.float64)
        t_new = t_old.copy()
        t_new[0:2] += np.asarray(dxy, dtype=np.float64)
        yaw_new = float(input_dict['ego_yaw']) + float(dyaw)

        ego2world = np.eye(4, dtype=np.float64)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=yaw_new).rotation_matrix
        ego2world[0:3, 3] = t_new
        lidar2world = ego2world @ lidar2ego
        world2lidar = self.invert_pose(lidar2world)

        et_dt = input_dict['ego_translation'].dtype
        ep_dt = input_dict['ego_pose'].dtype
        epi_dt = input_dict['ego_pose_inv'].dtype
        w2l_dt = np.asarray(input_dict['sensors']['LIDAR_TOP']['world2lidar']).dtype

        input_dict['ego_translation'] = t_new.astype(et_dt)
        input_dict['ego_yaw'] = yaw_new
        input_dict['ego_pose'] = lidar2world.astype(ep_dt)
        input_dict['ego_pose_inv'] = world2lidar.astype(epi_dt)
        new_w2l = world2lidar.astype(w2l_dt)
        input_dict['sensors']['LIDAR_TOP']['world2lidar'] = new_w2l
        input_dict['world2lidar'] = new_w2l   # 顶层别名也要同步

    def _perturb_can_bus(self, input_dict):
        """基于已更新的 ego_translation / ego_yaw 重建 can_bus 中的 world 相关项。
        严格复刻原版归一化:yaw<0 则 +2π,不处理 ≥2π。"""
        can_bus = input_dict['can_bus']  # 长度 18 的 ndarray,来自 get_data_info
        assert can_bus.shape == (18,), f'unexpected can_bus shape: {can_bus.shape}'

        t_new = np.asarray(input_dict['ego_translation'])
        yaw_new = float(input_dict['ego_yaw'])

        # 重算四元数(与原版 list(Quaternion(...)) 一致,顺序 [w,x,y,z])
        rot = list(Quaternion(axis=[0, 0, 1], radians=yaw_new))

        # 严格复刻原版归一化
        yaw_norm = yaw_new
        if yaw_norm < 0:
            yaw_norm += 2 * np.pi
        yaw_in_degree = yaw_norm / np.pi * 180

        can_bus[0:3] = t_new
        can_bus[3:7] = rot
        can_bus[16] = yaw_norm
        can_bus[17] = yaw_in_degree
        # [7:10] ego_vel, [10:13] ego_accel, [13:16] ego_rotation_rate 不变

    def _perturb_gt_boxes_and_ann(self, input_dict):
        """基于新 world2lidar 与 npc2world 重算 box 在 lidar 系的 xyz/yaw/(vx,vy)。
        npc2world 从 raw info 读以保 float64 原始精度(input_dict['npc2world'] 被降级为 float32)。"""
        gt_boxes = input_dict['gt_boxes'].copy()
        if len(gt_boxes) == 0:
            input_dict['gt_boxes'] = gt_boxes
            return

        info = self.get_data_by_index(input_dict['index'])
        npc2world = np.asarray(info['npc2world'], dtype=np.float64)                  # (N,4,4) float64
        w2l = np.asarray(input_dict['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float64)
        box2lidar = np.einsum('ij,njk->nik', w2l, npc2world)                         # (N,4,4)

        new_xyz       = box2lidar[:, 0:3, 3]
        new_yaw_local = np.arctan2(box2lidar[:, 1, 0], box2lidar[:, 0, 0])
        new_yaw_box   = -new_yaw_local - np.pi / 2

        vx, vy = gt_boxes[:, 7].astype(np.float64), gt_boxes[:, 8].astype(np.float64)
        speed  = np.sqrt(vx * vx + vy * vy)
        new_vx = speed * np.cos(new_yaw_local)
        new_vy = speed * np.sin(new_yaw_local)

        dt = gt_boxes.dtype
        gt_boxes[:, 0:3] = new_xyz.astype(dt)
        gt_boxes[:, 6]   = new_yaw_box.astype(dt)
        gt_boxes[:, 7]   = new_vx.astype(dt)
        gt_boxes[:, 8]   = new_vy.astype(dt)
        input_dict['gt_boxes'] = gt_boxes

        mask = (info['num_points'] != 0)
        arr = gt_boxes[mask]
        if not self.with_velocity:
            arr = arr[:, 0:7]
        input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            arr, box_dim=arr.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

    def _perturb_ego_his_trajs(self, input_dict):
        """重算 ego_his_trajs:历史帧 world pose 不变,但投影进入的当前 lidar 系已被扰动。
        完全复刻 get_ego_past_trajs 的语义,只替换 w2l_cur 为扰动后的值。"""
        idx = input_dict['index']
        sample_rate = self.sample_interval
        past_frames = self.past_frames
        adj_idx_list = range(idx - past_frames * sample_rate, idx + sample_rate, sample_rate)

        # 扰动后的 world→lidar(当前帧),float64 保精度
        w2l_cur = np.asarray(input_dict['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float64)

        # 历史帧的 lidar2ego 约定与 raw info 一致,故可以直接取 raw info 的 lidar2ego
        cur_info = self.get_data_by_index(idx)
        lidar2ego = np.asarray(cur_info['sensors']['LIDAR_TOP']['lidar2ego'], dtype=np.float64)

        full_track = np.zeros((past_frames + 1, 2), dtype=np.float64)
        full_mask = np.zeros(past_frames + 1, dtype=np.float64)
        for j, adj_idx in enumerate(adj_idx_list):
            if not self.is_in_same_route(idx, adj_idx):
                break
            adj_info = self.get_data_by_index(adj_idx)
            ego2world_adj = np.eye(4, dtype=np.float64)
            ego2world_adj[0:2, 3] = adj_info['ego_translation'][0:2]
            ego2world_adj[0:3, 0:3] = Quaternion(axis=[0, 0, 1],
                                                radians=adj_info['ego_yaw']).rotation_matrix
            lidar2world_adj = ego2world_adj @ lidar2ego
            adj2cur_lidar = w2l_cur @ lidar2world_adj
            full_track[j, 0:2] = adj2cur_lidar[0:2, 3]
            full_mask[j] = 1

        offset_track = full_track[1:] - full_track[:-1]
        for j in range(past_frames - 2, -1, -1):
            if full_mask[j] == 0:
                offset_track[j] = offset_track[j + 1]

        dt = input_dict['ego_his_trajs'].dtype
        input_dict['ego_his_trajs'] = offset_track.astype(dt)

    def _apply_perturbation(self, input_dict, dxy, dyaw):
        if np.allclose(dxy, 0.0) and abs(dyaw) < 1e-12:
            return input_dict
        input_dict = self._detach_from_cache(input_dict)
        self._perturb_world_to_lidar_chain(input_dict, dxy, dyaw)
        self._perturb_can_bus(input_dict)
        self._perturb_gt_boxes_and_ann(input_dict)
        self._perturb_ego_his_trajs(input_dict)  
        raise NotImplementedError("TODO 8-10 未实现")
    
    def prepare_train_data(self, index, aug_config):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        input_dict["aug_config"] = aug_config

        dxy, dyaw = self._sample_perturbation()
        input_dict = self._apply_perturbation(input_dict, dxy, dyaw)

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        gt_labels, gt_bboxes = self.get_map_info(index)
        example['map_gt_labels_3d'] = DC(gt_labels, cpu_only=False)
        example['map_gt_bboxes_3d'] = DC(gt_bboxes, cpu_only=True)
        if self.filter_empty_gt and (
                example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return self.union2one([example])