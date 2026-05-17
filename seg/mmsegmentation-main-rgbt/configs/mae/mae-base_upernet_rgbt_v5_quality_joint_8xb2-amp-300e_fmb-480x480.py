"""
RGB-T语义分割 - V5 QualityJoint (解耦 + 退化增强 + 质量网络联合微调) - FMB 480x480

架构说明：
- 在V4基础上实现真正的质量网络联合微调
- 质量网络V2：双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
- 渐进式token剪枝：软掩码(早期) → Gumbel-Softmax(中期) → 掩码注意力(后期)
- 质量引导模态对齐（替代InfoNCE）
- 质量感知加权不变性损失
- 质量锚定损失(MSE) + 空间多样性损失 + 高退化天花板损失
- 训练epoch: 300
- 数据集：FMB_ALL (15类)
- 输入尺寸：480x480
"""

_base_ = [
    './mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-480x480.py',
]

custom_imports = dict(
    imports=['mmseg.models.backbones.lightweight_mae_branch',
             'mmseg.models.segmentors.rgbt_v1_baseline',
             'mmseg.models.segmentors.rgbt_v2_disentangle',
             'mmseg.models.segmentors.rgbt_v5_quality_joint',
             'mmseg.models.utils.quality_network_v2',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (480, 480)

model = dict(
    type='RGBTv5QualityJoint',
    quality_network=dict(
        type='QualityNetworkV2',
        embed_dim=768,
        proj_dim=128,
        num_heads=4,
        num_scales=3),
    quality_pretrained=None,
    loss_seg_zc_weight=0.3,
    loss_modal_weight=0.2,
    loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
    loss_invariance_weight=0.001,
    loss_quality_reg_weight=0.05,
    quality_prune_threshold=0.3,
    early_epoch=30,
    mid_epoch=60,
    gumbel_initial_tau=1.0,
    gumbel_min_tau=0.1,
    gumbel_anneal_rate=0.01,
    invariance_stage_weights=(0.5, 0.7, 1.0, 1.5),
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
