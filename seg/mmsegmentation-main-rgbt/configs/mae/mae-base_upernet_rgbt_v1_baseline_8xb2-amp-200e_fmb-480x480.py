"""
RGB-T语义分割 - V1 Baseline (LoRA + FocalLoss + DiceLoss) - FMB 480x480

配置文件说明：
- 通用编码器：M-SpecGene (MAE ViT-B)，RGB和Thermal共享权重
- 微调方式：LoRA (rank=16, alpha=16.0) - qkv, proj, fc1, fc2
- 融合：通道拼接 (RGB feat + IR feat -> concat channel)
- 分割头：UPerNet解码器
- 损失：FocalLoss + DiceLoss
- 训练数据：仅干净数据
- 数据集：FMB_ALL (15类)
- 输入尺寸：480x480
- 推理：滑动窗口
"""

_base_ = [
    '../_base_/datasets/fmb_480x480.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.rgbt_v1_baseline',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading'],
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
    type='RGBTv1Baseline',
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
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
    neck=dict(type='Feature2Pyramid', embed_dim=768 * 2, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768 * 2, 768 * 2, 768 * 2, 768 * 2],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=15,
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
        in_channels=768 * 2,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=15,
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
    lora_target_modules=['qkv', 'proj'],
    freeze_backbone=True,

    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(213, 213)))

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

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
