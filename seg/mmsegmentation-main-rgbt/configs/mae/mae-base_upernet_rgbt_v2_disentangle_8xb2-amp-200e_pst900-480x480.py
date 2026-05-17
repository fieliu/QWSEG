"""
RGB-T语义分割 - V2 Disentangle (通用/私有解耦) - PST900 480x480

架构说明：
- 通用编码器：M-SpecGene (MAE ViT-B)，RGB和Thermal共享权重，提取通用特征zc
- 私有编码器：LightweightMAEBranch (ViT-Tiny结构)，分别提取RGB/Thermal私有特征zp
- 融合：跨模态注意力融合 (zc + zp -> fused features)
- 分割头：UPerNet解码器 (主) + FCN (zc辅助)
- 损失：Focal+Dice (主分割) + HSIC (解耦) + InfoNCE (模态) + MSE (不变性) + zc辅助分割
- 训练数据：仅干净数据
- 数据集：PST900_RGBT_Dataset (5类)
- 输入尺寸：480x480
"""

_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.backbones.lightweight_mae_branch',
             'mmseg.models.segmentors.rgbt_v1_baseline',
             'mmseg.models.segmentors.rgbt_v2_disentangle',
             'mmseg.datasets.pst900',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 480)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='RGBTv2Disentangle',
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',

    universal_branch=dict(
        type='RGBTv1Baseline',
        backbone=dict(
            type='MAE',
            img_size=(480, 480),
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            init_values=1.0,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11]),
        neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
        decode_head=dict(
            type='UPerHead',
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=5,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=[
                dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                     alpha=0.25, loss_weight=1.0, loss_name='loss_focal'),
                dict(type='DiceLoss', use_sigmoid=True, activate=True,
                     naive_dice=False, loss_weight=1.0, loss_name='loss_dice'),
            ]),
        use_lora=True,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_target_modules=['qkv', 'proj'],
        freeze_backbone=True),

    private_branch_rgb=dict(
        type='LightweightMAEBranch',
        embed_dims=192,
        num_layers=12,
        num_heads=3,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        drop_path_rate=0.1,
        universal_embed_dims=768,
        img_size=(480, 480)),

    private_branch_t=dict(
        type='LightweightMAEBranch',
        embed_dims=192,
        num_layers=12,
        num_heads=3,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        drop_path_rate=0.1,
        universal_embed_dims=768,
        img_size=(480, 480)),

    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),

    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                loss_name='loss_focal'),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=False,
                loss_weight=1.0,
                loss_name='loss_dice'),
        ]),

    zc_seg_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                loss_name='loss_ce'),
        ]),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.4,
                loss_name='loss_focal'),
            dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=False,
                loss_weight=0.4,
                loss_name='loss_dice'),
        ]),

    loss_seg_zc_weight=0.3,
    loss_modal_weight=0.2,
    loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
    loss_invariance_weight=0.01,

    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(213, 213)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'universal_branch': dict(lr_mult=0.1),
        }))

warmup_epochs = 4
max_epochs = 200

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=warmup_epochs),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

fp16 = dict(loss_scale='dynamic')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=True)

dataset_type = 'PST900Dataset'
data_root = '/home/lh/code/data/PST900_RGBT_Dataset'

train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='PST900_RGBT_Dataset', ir_replace_dst='PST900_RGBT_Dataset_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1600, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='PST900_RGBT_Dataset', ir_replace_dst='PST900_RGBT_Dataset_T'),
    dict(type='Resize', scale=(1600, 480), keep_ratio=True),
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
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
