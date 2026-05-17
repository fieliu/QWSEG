_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v9_quality_gated',
             'mmseg.engine',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (480, 640)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

dataset_type = 'FMBDataset'
data_root = '/home/lh/code/data/FMB_ALL/FMB'

train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(640, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
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
        data_prefix=dict(img_path='images/training',
                         seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation',
                         seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

model = dict(
    type='MiTMulV9QualityGated',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
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
            prefix='backbone.')),
    private_branch_rgb=dict(
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
            prefix='backbone.')),
    private_branch_t=dict(
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
            prefix='backbone.')),
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
        num_classes=15,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    universal_embed_dims=[64, 128, 320, 512],
    private_embed_dims=[32, 64, 128, 256],
    learnable_threshold=True,
    init_threshold=0.1,
    loss_align_weight=0.5,
    quality_pretrained=None,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 426)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'quality_pyramid_net': dict(lr_mult=0.1),
            'private_branch_rgb': dict(lr_mult=0.1),
            'private_branch_t': dict(lr_mult=0.1),
            'private_inject_rgb': dict(lr_mult=0.5),
            'private_inject_t': dict(lr_mult=0.5),
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=3000,
        end=80000,
        by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))

custom_hooks = [dict(type='TrainVisHook', interval=5, num_samples=2)]
