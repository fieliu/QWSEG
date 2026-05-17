"""
RGB-T语义分割 - V3 Degradation (解耦 + 在线退化增强) - MFNet 768x768

架构说明：
- 与V2完全相同的模型架构
- 区别：训练时使用在线退化数据增强管道
- 使用RGBTModalDegradation统一管理退化策略（与鲁棒性测试一致的退化类型）
- 训练epoch: 300
- 数据集：MFNet (9类)
"""

_base_ = [
    './mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_mfnet-768x768.py',
]

custom_imports = dict(
    imports=['mmseg.models.backbones.lightweight_mae_branch',
             'mmseg.models.segmentors.rgbt_v1_baseline',
             'mmseg.models.segmentors.rgbt_v2_disentangle',
             'mmseg.models.segmentors.rgbt_v3_degradation',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmseg.datasets.transforms.rgbt_augmentation'],
    allow_failed_imports=False)

crop_size = (768, 768)

model = dict(
    type='RGBTv3Degradation',
)

train_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(1280, 768),
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
