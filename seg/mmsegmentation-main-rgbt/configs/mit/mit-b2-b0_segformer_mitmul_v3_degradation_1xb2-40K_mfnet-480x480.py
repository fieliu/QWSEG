"""
RGB-T语义分割 - MiT V3 Degradation (MiT-B2通用分支 + 解耦 + 在线退化增强) - MFNet 480x480

架构说明：
- 与V2 MiT完全相同的模型架构
- 区别：训练时使用在线退化数据增强管道
- 使用RGBTModalDegradation统一管理退化策略
- 训练iters: 40000
- 数据集：MFNet (9类)
- 输入尺寸：480x480
"""

_base_ = [
    './mit-b2-b0_segformer_mitmul_v2_disentangle_1xb2-40K_mfnet-480x480.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v3_degradation',
             'mmseg.models.backbones.lightweight_mit_branch',
             'mmseg.models.backbones.mae',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 480)

model = dict(
    type='MiTMulV3Degradation',
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
)

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
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))
