# Swin-L common + Swin-T private, 480x640 / 200 epochs.
# Upper-bound run aligned with the Swin-T main result
# (swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640).
#
# Inherits the full MODEL definition (backbone/head/losses/optimizer/
# custom_keys) from the 240x320 LT config. The dataset pipelines below
# OVERRIDE the inherited 240x320 ones to 480x640.
_base_ = ['./swinmul_quality_lt_mask2former_swin-l+t_1xb2-50E_mfnet-240x320.py']

crop_size = (480, 640)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(
        type='Pretrained',
        checkpoint='./pretrain/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth')),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

# ---- 480x640 dataset pipelines (override inherited 240x320) ----
train_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(640, 480), ratio_range=(0.5, 1.75),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# 200-epoch schedule, identical iter math to the Swin-T 480x640 main config
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=117600,
         by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,
                    save_best='mIoU'))

custom_hooks = [
    dict(type='TrainVisHook', interval=5, num_samples=2, mask_threshold=0.3),
    dict(type='MissingModalityEvalHook', interval=1),
]
