_base_ = [
    '../_base_/datasets/mfnet_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.amdanet',
        'mmseg.models.backbones.rgbx_transformer',
        'mmseg.models.decode_heads.seg_decoder_head',
        'mmseg.models.decode_heads.fuse_decoder_head',
        'mmseg.models.losses.fusion_loss',
        'mmseg.engine',
        'mmseg.datasets.mfnet',
        'mmseg.datasets.transforms.loading',
    ],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (480, 640)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 127.0, 127.0, 127.0],
    std=[58.395, 57.12, 57.375, 60.0, 60.0, 60.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='AMDANet',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='RGBXTransformer',
        img_size=480,
        in_chans=3,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrain/segformer_mit-b4_512x512_160k_ade20k.pth')),

    decode_head=dict(
        type='SegDecoderHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        embed_dim=768,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    fuse_head=dict(
        type='FuseDecoderHead',
        in_channels=[64, 128, 320, 512],
        embed_dim=16,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_fusion=dict(type='FusionLoss', loss_weight=1.0,
                         mean=[123.675, 116.28, 103.53],
                         std=[58.395, 57.12, 57.375],
                         ir_mean=[127.0, 127.0, 127.0],
                         ir_std=[60.0, 60.0, 60.0])),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=117600,
        by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=5))

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
