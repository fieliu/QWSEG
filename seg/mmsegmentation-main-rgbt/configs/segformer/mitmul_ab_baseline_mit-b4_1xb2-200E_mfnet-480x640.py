_base_ = [
    '../_base_/datasets/mfnet_ab_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_ablation',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 640)
num_classes = 9
num_layers = [3, 8, 27, 3]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

mit_b4 = dict(
    type='MixVisionTransformer',
    in_channels=3,
    embed_dims=64,
    num_stages=4,
    num_layers=[3, 8, 27, 3],
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
        checkpoint='./pretrain/segformer_mit-b4_512x512_160k_ade20k.pth',
        prefix='backbone.'))

_segformer_head = dict(
    type='SegformerHead',
    in_channels=[64, 128, 320, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=num_classes,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
    ])

model = dict(
    type='MiTMulABBaseline',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=mit_b4,
    decode_head=_segformer_head,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

custom_keys = {
    'backbone_rgb': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone_t': dict(lr_mult=0.1, decay_mult=1.0),
}
for branch in ['backbone_rgb', 'backbone_t']:
    custom_keys.update({
        f'{branch}.layers.{stage_id}.0.norm': dict(lr_mult=0.1, decay_mult=0.0)
        for stage_id in range(4)
    })
    custom_keys.update({
        f'{branch}.layers.{stage_id}.1.{block_id}.norm1': dict(lr_mult=0.1, decay_mult=0.0)
        for stage_id, num_blocks in enumerate(num_layers)
        for block_id in range(num_blocks)
    })
    custom_keys.update({
        f'{branch}.layers.{stage_id}.1.{block_id}.norm2': dict(lr_mult=0.1, decay_mult=0.0)
        for stage_id, num_blocks in enumerate(num_layers)
        for block_id in range(num_blocks)
    })
    custom_keys.update({
        f'{branch}.layers.{stage_id}.2': dict(lr_mult=0.1, decay_mult=0.0)
        for stage_id in range(4)
    })

optimizer = dict(
    type='AdamW', lr=0.00006, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
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
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [dict(type='TrainVisHook', interval=5, num_samples=2)]
