"""
RGB-T语义分割 - MiT V4 QualityPruning (MiT-B2通用分支 + 解耦 + 退化增强 + 质量网络) - FMB 480x480

架构说明：
- 在V2 MiT基础上加入质量网络 (QualityNetwork)
- 质量网络：双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
- 损失：V2全部损失 + 质量锚定损失(MSE) + 空间多样性损失 + 高退化天花板损失
- 训练iters: 40000
- 数据集：FMB_ALL (15类)
- 输入尺寸：480x480
"""

_base_ = [
    './mit-b2-b0_segformer_mitmul_v2_disentangle_1xb2-40K_fmb-480x480.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mitmul_v4_quality_pruning',
             'mmseg.models.backbones.lightweight_mit_branch',
             'mmseg.models.utils.quality_network',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 480)

model = dict(
    type='MiTMulV4QualityPruning',
    quality_network=dict(
        type='QualityNetwork',
        embed_dim=512,
        num_heads=8,
        num_layers=2),
    loss_quality_weight=0.1,
    loss_anchor_weight=0.5,
    quality_prune_threshold=0.3,
)

train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB',
         ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1600, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))
