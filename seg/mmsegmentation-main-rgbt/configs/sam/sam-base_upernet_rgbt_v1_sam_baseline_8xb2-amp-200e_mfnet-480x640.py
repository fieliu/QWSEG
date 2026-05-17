"""
RGB-T语义分割 - V1 SAM Baseline (SAM ViT-B + LoRA + 特征相加融合) - MFNet 480x640

配置文件说明：
- 通用编码器：SAM ViT-B (ImageEncoderViT)，RGB和Thermal共享权重
- 微调方式：LoRA (rank=4, alpha=4.0) - qkv, proj
- 融合：特征相加 (RGB feat + IR feat -> element-wise addition)
- 分割头：UPerNet解码器
- 损失：FocalLoss + DiceLoss
- 训练数据：仅干净数据
- 数据集：MFNet (9类)
- 输入尺寸：480x640
- 推理：全图推理
"""

_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.rgbt_v1_sam_baseline',
             'mmseg.models.backbones.sam_vit',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

crop_size = (480, 640)
data_root = '/home/lh/code/data/MFNet'
dataset_type = 'MFNetDataset'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

train_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(800, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='Resize', scale=(800, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
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

model = dict(
    type='RGBTv1SAMBaseline',
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/sam_vit_b_01ec64.pth',
    backbone=dict(
        type='SAMViT',
        img_size=(480, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
        out_indices=(3, 5, 7, 11)),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=9,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
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

    use_lora=True,
    lora_rank=4,
    lora_alpha=4.0,
    lora_dropout=0.1,
    lora_target_modules=['q_proj', 'v_proj', 'proj'],
    freeze_backbone=True,

    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'lora_A': dict(lr_mult=1.0),
            'lora_B': dict(lr_mult=1.0),
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
