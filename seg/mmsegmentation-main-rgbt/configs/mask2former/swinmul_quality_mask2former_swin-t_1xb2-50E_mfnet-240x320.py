_base_ = [
    '../_base_/datasets/mfnet_ab_240x320.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.swin_quality_mask2former',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

crop_size = (240, 320)
num_classes = 9
depths = [2, 2, 6, 2]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

swin_common = dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=depths,
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    frozen_stages=-1,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'))

swin_private = dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=depths,
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    frozen_stages=-1,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'))

_mask2former_head = dict(
    type='Mask2FormerHead',
    in_channels=[96, 192, 384, 768],
    strides=[4, 8, 16, 32],
    feat_channels=256,
    out_channels=256,
    num_classes=num_classes,
    num_queries=100,
    num_transformer_feat_level=3,
    align_corners=False,
    pixel_decoder=dict(
        type='mmdet.MSDeformAttnPixelDecoder',
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    num_levels=3,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=True,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        positional_encoding=dict(
            num_feats=128, normalize=True),
        init_cfg=None),
    enforce_decoder_input_project=False,
    positional_encoding=dict(
        num_feats=128, normalize=True),
    transformer_decoder=dict(
        return_intermediate=True,
        num_layers=9,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                dropout_layer=None,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.0,
                dropout_layer=None,
                add_identity=True)),
        init_cfg=None),
    loss_cls=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=2.0,
        reduction='mean',
        class_weight=[1.0] * num_classes + [0.1]),
    loss_mask=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=5.0),
    loss_dice=dict(
        type='mmdet.DiceLoss',
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=5.0),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.ClassificationCost', weight=2.0),
                dict(
                    type='mmdet.CrossEntropyLossCost',
                    weight=5.0,
                    use_sigmoid=True),
                dict(
                    type='mmdet.DiceCost',
                    weight=5.0,
                    pred_act=True,
                    eps=1.0)
            ]),
        sampler=dict(type='mmdet.MaskPseudoSampler')))

model = dict(
    type='QualityGatedSwinMask2Former',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=swin_common,
    private_branch_rgb=swin_private,
    private_branch_t=swin_private,
    decode_head=_mask2former_head,
    common_decode_head=dict(
        type='SegformerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
        ]),
    rgb_private_decode_head=dict(
        type='SegformerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
        ]),
    t_private_decode_head=dict(
        type='SegformerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, activate=True, loss_weight=1.0),
        ]),
    retention_min=0.4,
    retention_max=0.98,
    phase1_epochs=10,
    phase2_epochs=20,
    loss_align_weight=0.1,
    contrast_tau=0.07,
    contrast_num_samples=512,
    loss_invariant_weight=0.03,
    loss_q_guide_weight=0.1,
    loss_distill_weight=0.3,
    distill_temperature=4.0,
    aux_loss_weight=0.3,
    total_epochs=50,
    missing_ratio=0.3,
    global_deg_ratio=0.3,
    local_deg_ratio=0.4,
    tau=0.3,
    alpha=10.0,
    tau_hard=0.2,
    fuse_epsilon=1e-3,
    fuse_beta=6.0,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(160, 214)))

backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_rgb': dict(lr_mult=0.1, decay_mult=1.0),
    'private_branch_t': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'private_branch_rgb.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_rgb.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
custom_keys.update({
    f'private_branch_t.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'private_branch_t.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
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
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=10.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=2000,
        end=29400,
        by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=10,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [dict(type='TrainVisHook', interval=2, num_samples=2, mask_threshold=0.3)]
