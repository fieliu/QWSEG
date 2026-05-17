_base_ = [
    '../_base_/datasets/mfnet_240x320.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.swin_dual_add',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

crop_size = (240, 320)
num_classes = 9

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

swin_t = dict(
    type='SwinTransformer',
    pretrain_img_size=224,
    in_channels=3,
    embed_dims=96,
    patch_size=4,
    window_size=7,
    mlp_ratio=4,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    strides=(4, 2, 2, 2),
    out_indices=(0, 1, 2, 3),
    qkv_bias=True,
    qk_scale=None,
    patch_norm=True,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    use_abs_pos_embed=False,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN', requires_grad=True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'))

model = dict(
    type='SwinDualAdd',
    data_preprocessor=data_preprocessor,
    backbone=swin_t,
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(160, 213)))

custom_keys = {
    'absolute_pos_embed': dict(decay_mult=0.),
    'relative_position_bias_table': dict(decay_mult=0.),
    'norm': dict(decay_mult=0.),
}

optimizer = dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys=custom_keys))

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
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
