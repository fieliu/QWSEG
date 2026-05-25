_base_ = [
    '../_base_/datasets/mfnet_ab_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mit_quality_mamba',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 640)
num_classes = 9
num_layers = [3, 4, 6, 3]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

mit_b2 = dict(
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
    with_cp=True,
    drop_path_rate=0.1,
    act_cfg=dict(type='GELU'),
    norm_cfg=dict(type='LN', eps=1e-6),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/segformer_mit-b2_512x512_160k_ade20k.pth',
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
    type='QualityGatedMiTMamba',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=mit_b2,
    private_branch_rgb=mit_b2,
    private_branch_t=mit_b2,
    decode_head=_segformer_head,
    common_decode_head=_segformer_head,
    rgb_private_decode_head=_segformer_head,
    t_private_decode_head=_segformer_head,
    total_epochs=200,
    loss_align_weight=0.1,
    contrast_tau=0.07,
    contrast_num_samples=512,
    loss_invariant_weight=0.03,
    loss_q_guide_weight=0.1,
    loss_distill_weight=0.3,
    distill_temperature=4.0,
    aux_loss_weight=0.3,
    missing_ratio=0.3,
    global_deg_ratio=0.3,
    local_deg_ratio=0.4,
    tau=0.3,
    alpha=10.0,
    fuse_epsilon=1e-6,
    fuse_beta=3.0,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_rgb': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_t': dict(lr_mult=0.1, decay_mult=1.0),
}
for branch in ['backbone', 'private_branch_rgb', 'private_branch_t']:
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

predictor_keys = {}
for pred_name in ['predictors_common_rgb', 'predictors_common_t',
                  'predictors_priv_rgb', 'predictors_priv_t']:
    predictor_keys[f'{pred_name}'] = dict(lr_mult=5.0, decay_mult=1.0)
    for stage_id in range(4):
        for sub in ['ctx_proj', 'norm1', 'conv1', 'norm2', 'conv2',
                    'score_head']:
            predictor_keys[f'{pred_name}.{stage_id}.{sub}'] = dict(lr_mult=5.0, decay_mult=1.0)

custom_keys.update(predictor_keys)

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
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [dict(type='TrainVisHook', interval=5, num_samples=2, mask_threshold=0.3),
                dict(type='ValDegradationHook')]
