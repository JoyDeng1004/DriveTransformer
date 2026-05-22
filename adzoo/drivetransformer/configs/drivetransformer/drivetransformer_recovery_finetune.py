_base_ = './drivetransformer_large.py'

# Exploratory planning-side recovery fine-tune. Images are not re-rendered for
# the perturbed ego pose, so the visual feature extractor is frozen and only
# ego-planning-side modules are trainable.

load_from = 'ckpts/drivetransformer_large.pth'
work_dir = 'outputs/recovery_finetune/route_command_dx1'

# Keep this intentionally short for an exploratory closed-loop test. Increase
# max_iters after the smoke run if the recovery metrics move in the right
# direction.
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-2)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-2)

# Only these prefixes remain trainable. Everything else, including the visual
# backbone/neck and shared perception heads, has requires_grad=False before the
# optimizer is built.
recovery_freeze = dict(
    trainable_param_prefixes=[
        'pts_bbox_head.ego_lcf_encoder',
        'pts_bbox_head.ego_traj_ref_fix_time_embedding',
        'pts_bbox_head.ego_traj_ref_fix_dist_embedding',
        'pts_bbox_head.ego_traj_branches_fix_time',
        'pts_bbox_head.ego_traj_branches_fix_dist',
        'pts_bbox_head.ego_traj_cls_branches',
    ],
    frozen_param_prefixes=[
        'img_backbone',
        'img_neck',
    ])

model = dict(
    pts_bbox_head=dict(
        # Planning-only objective for the first exploratory run. Non-planning
        # losses are left in the graph but have zero weight, so we do not need
        # to make all perception labels physically self-consistent yet.
        loss_cls=dict(loss_weight=0.0),
        loss_bbox=dict(loss_weight=0.0),
        loss_traj=dict(loss_weight=0.0),
        loss_traj_cls=dict(loss_weight=0.0),
        loss_map_cls=dict(loss_weight=0.0),
        loss_map_pts=dict(loss_weight=0.0),
        loss_map_dir=dict(loss_weight=0.0),
        loss_plan_reg_fix_time=dict(loss_weight=3.5),
        loss_plan_reg_fix_dist=dict(loss_weight=10.0),
        loss_plan_cls=dict(loss_weight=20.0)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='B2D_DriveTransformer_Perturb_Dataset',
        perturb_enabled=True,
        perturb_prob=1.0,
        perturb_scale_x=1.0,
        perturb_scale_y=0.0,
        perturb_scale_yaw=0.08726646259971647,  # 5 degrees
        perturb_seed=None))
