_base_ = [
    '../_base_/datasets/mfnet_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'mmseg.models.segmentors.encoder_decoder_mult',
        'mmseg.models.backbones.bi_mix_vision_transformer',
        'mmseg.models.decode_heads.segformer_head_mult',
        'mmseg.models.losses.akd_loss',
        'mmseg.models.losses.rl1_loss',
        'mmseg.models.losses.m_cross_entropy_loss',
        'mmseg.engine',
        'mmseg.datasets.mfnet',
        'mmseg.datasets.transforms.loading',
    ],
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

model = dict(
    type='EncoderDecoderMult',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='BIMixVisionTransformer',
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
            checkpoint='./pretrain/segformer_mit-b2_512x512_160k_ade20k.pth')),

    decode_head=dict(
        type='SegformerHeadMult',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
        ],
        loss_decode_modal=[
            dict(type='M_CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ],
        loss_decode_akd=[
            dict(type='AKDLoss', loss_weight=0.5),
        ],
        loss_decode_head=[
            dict(type='RegionL1', loss_weight=1.0, tau=1.0, N_cls=9),
        ]),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.0),
        }),
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=1e-5,
        by_epoch=True,
        convert_to_iter_based=True),
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

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
