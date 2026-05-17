"""
RGB-T语义分割 - V4 SAM QualityPruning (SAM通用分支 + 解耦 + 退化增强 + 质量网络) - MFNet 480x640

架构说明：
- 在V3 SAM基础上加入质量网络 (QualityNetwork)
- 质量网络：双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
- 损失：V2全部损失 + 质量锚定损失(MSE) + 空间多样性损失 + 高退化天花板损失
- 训练时使用RGBTModalDegradation退化数据增强管道
- 训练epoch: 300
- 数据集：MFNet (9类)
- 输入尺寸：480x640
"""

_base_ = [
    './sam-base_upernet_rgbt_v2_sam_disentangle_8xb2-amp-200e_mfnet-480x640.py',
]

custom_imports = dict(
    imports=['mmseg.models.backbones.lightweight_mae_branch',
             'mmseg.models.backbones.sam_vit',
             'mmseg.models.segmentors.rgbt_v1_sam_baseline',
             'mmseg.models.segmentors.rgbt_v2_sam_disentangle',
             'mmseg.models.segmentors.rgbt_v4_sam_quality_pruning',
             'mmseg.models.utils.quality_network',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 640)

model = dict(
    type='RGBTv4SAMQualityPruning',
    quality_network=dict(
        type='QualityNetwork',
        embed_dim=768,
        proj_dim=128,
        num_heads=4,
        num_scales=3),
    loss_quality_weight=0.1,
    loss_anchor_weight=0.5,
    quality_prune_threshold=0.3,
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

max_epochs = 300

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=4),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=4,
        end=max_epochs,
        by_epoch=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)
