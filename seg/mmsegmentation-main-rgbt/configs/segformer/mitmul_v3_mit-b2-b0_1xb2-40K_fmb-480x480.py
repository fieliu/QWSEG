_base_ = [
    '../_base_/datasets/fmb_480x480.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v3_degradation',
             'mmseg.models.backbones.lightweight_mit_branch',
             'mmseg.models.backbones.mae',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)
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
    type='MiTMulV3Degradation',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
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
        drop_path_rate=0.1,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrain/mit_b2.pth')),
    private_branch_rgb=dict(
        type='LightweightMITBranch',
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
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
        universal_embed_dims_list=[64, 128, 320, 512]),
    private_branch_t=dict(
        type='LightweightMITBranch',
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
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
        universal_embed_dims_list=[64, 128, 320, 512]),
    quality_network=dict(
        type='QualityNetworkV3',
        embed_dim=512,
        global_norm_hidden=128,
        local_attn_mid=128),
    pseudo_label_vit=dict(
        type='MAE',
        img_size=(480, 480),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        init_values=1.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN'),
        final_norm=True),
    pseudo_label_pretrained='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=15,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    zc_seg_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=15,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ]),
    fusion_embed_dims=[64, 128, 320, 512],
    private_embed_dims=[32, 64, 128, 256],
    loss_seg_zc_weight=0.3,
    loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
    loss_invariance_weight=0.1,
    loss_ranking_weight=0.5,
    loss_level_consistency_weight=0.3,
    loss_intra_rank_weight=0.5,
    intra_rank_margin=0.1,
    level_margin=0.1,
    max_deg_quality=0.1,
    num_degradation_levels=5,
    quality_prune_threshold=0.3,
    pseudo_label_perturb_sigma=0.02,
    pseudo_label_normality_gamma=1.0,
    pseudo_label_alpha=5.0,
    pseudo_label_beta=0.0,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(213, 213)))

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
        end=40000,
        by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
