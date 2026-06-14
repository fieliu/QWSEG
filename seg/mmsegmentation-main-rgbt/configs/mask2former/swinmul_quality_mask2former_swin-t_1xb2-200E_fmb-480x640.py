# FMB baseline for QualityGatedSwinMask2Former (Swin-T, 480x640).
# Inherits model/optimizer/schedule from the MFNet baseline; overrides the
# dataset to FMB (15 classes, FMB loader) and all num_classes-dependent
# fields. This is the M0 (full model) reference for FMB ablations.
_base_ = ['./swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.swin_quality_mask2former',
             'mmseg.engine',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

num_classes = 15

model = dict(
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    common_decode_head=dict(num_classes=num_classes),
    rgb_private_decode_head=dict(num_classes=num_classes),
    t_private_decode_head=dict(num_classes=num_classes))

# ---- FMB dataset (override inherited MFNet dataset) ----
dataset_type = 'FMBDataset'
data_root = '/home/lh/code/data/FMB_ALL/FMB'
crop_size = (480, 640)

train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(640, 480), ratio_range=(0.5, 2.0),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='images/training',
                         seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='images/validation',
                         seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
