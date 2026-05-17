"""
RGB-T语义分割 - 验证实验: MAE冻结Backbone + 只训练解码器

目的: 验证MAE预训练编码器(M-SpecGene)的原始特征提取能力
方法: 使用原始EncoderDecoder架构，冻结backbone全部参数，只训练neck+decode_head+auxiliary_head
预期: 
  - 如果mIoU接近LoRA微调版本，说明编码器特征质量好，LoRA是瓶颈
  - 如果mIoU远低于LoRA微调，说明编码器特征本身不够好，需要更多微调

架构: 与原项目完全一致 (EncoderDecoder)
  - 输入: 6通道(RGB+Thermal)，backbone内batch拼接共享参数
  - Backbone: MAE ViT-B (in_channels=3, 共享权重处理RGB和IR)
  - 融合: 通道拼接 (view reshape: [2B,C,H,W] -> [B,2C,H,W])
  - 解码器: UPerNet
  - 无LoRA，backbone完全冻结

训练: 40 epoch (解码器收敛很快)
"""

_base_ = [
    '../_base_/datasets/fmb_480x480.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.datasets.fmb',
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
    type='EncoderDecoder',
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
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(213, 213)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0, decay_mult=0),
        }))

max_epochs = 40

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=2),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2,
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
