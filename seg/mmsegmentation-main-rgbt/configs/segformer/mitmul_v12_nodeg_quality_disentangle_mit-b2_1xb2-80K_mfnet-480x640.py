_base_ = [
    '../_base_/datasets/mfnet_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v12_nodeg_quality_disentangle',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (480, 640)
num_classes = 9
num_layers = [3, 4, 6, 3]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

mit_b2 = dict(
    type='MixVisionTransformer',
    in_channels=3,
    embed_dims=64,
    num_stages=4,
    num_layers=[3, 4, 6, 3],
    num_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    out_indices=(0, 1, 2, 3),
    mlp_ratio=4,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN', eps=1e-6),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/segformer_mit-b2_512x512_160k_ade20k.pth',
        prefix='backbone.'))

_segformer_head = dict(
    type='SegformerHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
    ])

_segformer_aux_head = dict(
    type='SegformerHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
    ])

model = dict(
    type='MiTMulV12QualityDisentangleNoDeg',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=mit_b2,
    private_branch_rgb=mit_b2,
    private_branch_t=mit_b2,
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
    aux_loss_weight=0.3,
    quality_pretrained='./pretrain/mfnet_best_quality_pyramid_net.pth',
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_rgb': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_t': dict(lr_mult=0.1, decay_mult=1.0),
    'quality_pyramid_net': dict(lr_mult=0.5, decay_mult=1.0),
}
custom_keys.update({
    f'backbone.layers.{stage_id}.0.norm': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})
custom_keys.update({
    f'backbone.layers.{stage_id}.1.{block_id}.norm1': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layers.{stage_id}.2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})
custom_keys.update({
    f'private_branch_rgb.layers.{stage_id}.0.norm': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})
custom_keys.update({
    f'private_branch_rgb.layers.{stage_id}.1.{block_id}.norm1': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_rgb.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_rgb.layers.{stage_id}.2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})
custom_keys.update({
    f'private_branch_t.layers.{stage_id}.0.norm': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})
custom_keys.update({
    f'private_branch_t.layers.{stage_id}.1.{block_id}.norm1': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_t.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_t.layers.{stage_id}.2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id in range(4)
})

optimizer = dict(
    type='AdamW', lr=0.00006, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=10),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=10,
        end=200,
        by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))

custom_hooks = [dict(type='TrainVisHook', interval=5, num_samples=2)]

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
