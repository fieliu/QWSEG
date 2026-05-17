"""
RGB-T语义分割 - SAM2 Hiera-L + LoRA + MLP融合 + 文本解码器 - MFNet 240x320

配置文件说明：
- Backbone: SAM2 Hiera (large), RGB和Thermal共享权重
- 微调方式: LoRA (rank=4, alpha=4.0) - q_proj, v_proj
- 融合: 拼接 + MLP降维 (Concat + Conv1x1)
- 解码头: 文本驱动解码器 (余弦相似度)
- 损失: CrossEntropyLoss + DiceLoss
- 数据集: MFNet (9类)
- 输入尺寸: 240x320
"""

_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.backbones.hiera_lora',
             'mmseg.models.decode_heads.sam2_text_head',
             'mmseg.models.segmentors.rgbt_v1_sam2',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

crop_size = (240, 320)
data_root = '/home/lh/code/data/MFNet'
dataset_type = 'MFNetDataset'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

train_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(400, 240),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='Resize', scale=(400, 240), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='val.txt',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        ann_file='test.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

model = dict(
    type='RGBTv1SAM2',
    data_preprocessor=data_preprocessor,
    pretrained='/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt/pretrain/sam2.1_hiera_large.pt',
    backbone=dict(
        type='HieraLoRA',
        arch='l',
        use_lora=True,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_modules=['q_proj', 'v_proj'],
        freeze_backbone=True,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        type='SAM2TextHead',
        in_channels=256,
        channels_list=(256, 256, 256, 256),
        text_embed_dim=768,
        num_classes=9,
        out_scale_factor=4,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=False,
                loss_weight=1.0,
                ignore_index=255),
        ],
        ignore_index=255,
        align_corners=False,
    ),
    num_classes=9,
    label_feature_path='/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt/pretrain/mf_class_embedding.pt',
    fusion_type='mlp')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=8e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'lora_A': dict(lr_mult=1.0),
            'lora_B': dict(lr_mult=1.0),
        }))

warmup_epochs = 4
max_epochs = 200

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=warmup_epochs),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

fp16 = dict(loss_scale='dynamic')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=True)
