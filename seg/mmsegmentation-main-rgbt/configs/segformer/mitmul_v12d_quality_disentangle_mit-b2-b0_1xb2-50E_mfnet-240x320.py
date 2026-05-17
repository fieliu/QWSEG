_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v12d_quality_disentangle',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (240, 320)
num_classes = 9

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

dataset_type = 'MFNetDataset'
data_root = '/home/lh/code/data/MFNet'

train_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='val.txt',
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='test.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

mit_common = dict(
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

mit_private = dict(
    type='MixVisionTransformer',
    in_channels=3,
    embed_dims=32,
    num_stages=4,
    num_layers=[2, 2, 2, 2],
    num_heads=[1, 2, 4, 8],
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
        checkpoint='./pretrain/segformer_mit-b0_512x512_160k_ade20k.pth',
        prefix='backbone.'))

model = dict(
    type='MiTMulV12DQualityDisentangle',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=mit_common,
    private_branch_rgb=mit_private,
    private_branch_t=mit_private,
    quality_pyramid_net=dict(
        type='QualityAwarePyramidNet',
        in_channels=3,
        mid_channels=64,
        num_stages=4),
    decode_head=dict(
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
        ]),
    common_decode_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_classes=num_classes,
        num_convs=2,
        in_index=-1,
        concat_input=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce'),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_name='loss_dice'),
        ]),
    rgb_private_decode_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_classes=num_classes,
        num_convs=2,
        in_index=-1,
        concat_input=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce'),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_name='loss_dice'),
        ]),
    t_private_decode_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_classes=num_classes,
        num_convs=2,
        in_index=-1,
        concat_input=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce'),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_name='loss_dice'),
        ]),
    quality_threshold=0.3,
    loss_align_weight=0.5,
    loss_invariant_weight=1.0,
    loss_distill_weight=1.0,
    aux_loss_weight=0.3,
    missing_ratio=0.3,
    global_deg_ratio=0.3,
    local_deg_ratio=0.4,
    quality_pretrained='./pretrain/mfnet_best_quality_pyramid_net.pth',
    test_cfg=dict(mode='whole'))

num_layers_common = [3, 4, 6, 3]
num_layers_private = [2, 2, 2, 2]

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
    for stage_id, num_blocks in enumerate(num_layers_common)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers_common)
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
    for stage_id, num_blocks in enumerate(num_layers_private)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_rgb.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers_private)
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
    for stage_id, num_blocks in enumerate(num_layers_private)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_t.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
    for stage_id, num_blocks in enumerate(num_layers_private)
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
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=5),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=50,
        by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

custom_hooks = [dict(type='TrainVisHook', interval=5, num_samples=2)]
