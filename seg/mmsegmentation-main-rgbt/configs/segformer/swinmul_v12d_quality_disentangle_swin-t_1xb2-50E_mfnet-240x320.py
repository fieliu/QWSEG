_base_ = [
    '../_base_/datasets/mfnet_240x320.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.swinmul_v12d_quality_disentangle',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (240, 320)
num_classes = 9
depths = [2, 2, 6, 2]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

swin_common = dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=depths,
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    frozen_stages=-1,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'))

swin_private = dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=depths,
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    frozen_stages=-1,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'))

_segformer_head = dict(
    type='SegformerHead',
    in_channels=[96, 192, 384, 768],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
    ])

_segformer_aux_head = dict(
    type='SegformerHead',
    in_channels=[96, 192, 384, 768],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
    ])

model = dict(
    type='SwinMulV12DQualityDisentangle',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=swin_common,
    private_branch_rgb=swin_private,
    private_branch_t=swin_private,
    quality_pyramid_net=dict(
        type='QualityAwarePyramidNet',
        in_channels=3,
        mid_channels=64,
        num_stages=4),
    decode_head=_segformer_head,
    common_decode_head=_segformer_aux_head,
    rgb_private_decode_head=_segformer_aux_head,
    t_private_decode_head=_segformer_aux_head,
    quality_threshold=0.1,
    loss_align_weight=0.5,
    loss_invariant_weight=1.0,
    loss_distill_weight=1.0,
    aux_loss_weight=0.3,
    missing_ratio=0.3,
    global_deg_ratio=0.3,
    local_deg_ratio=0.4,
    quality_pretrained='./pretrain/mfnet_best_quality_pyramid_net.pth',
    quality_freeze_epochs=30,
    test_cfg=dict(mode='whole'))

backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_rgb': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_t': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'quality_pyramid_net': dict(lr_mult=0.5, decay_mult=1.0),
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'private_branch_rgb.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_rgb.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'private_branch_t.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_t.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=2),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=2,
        end=50,
        by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=10,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [dict(type='TrainVisHook', interval=2, num_samples=2)]
