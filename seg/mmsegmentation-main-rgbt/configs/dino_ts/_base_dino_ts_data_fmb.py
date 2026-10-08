# FMB data for the DINO-TS pipeline (RGB = FMB, thermal = FMB_T).
#
# Two differences from the stock FMB configs:
#
# 1. Background is already gone.  FMBDataset sets reduce_zero_label=True, which
#    maps raw label 0 (Background) to ignore (255) and shifts 1..14 -> 0..13.
#
# 2. Bicycle is removed as well (FMBDataset13 + RemapLabels).  Across the 1,220
#    training images it occupies 9,115 pixels in 12 images, and in the 280-image
#    validation split it has exactly zero pixels.  Its IoU is undefined there,
#    and a spurious prediction would give a non-empty union with an empty
#    intersection and pull the 13-class mean down.  After reduce_zero_label the
#    raw label 13 (Bicycle) is index 12 and Pole is index 13, so the mapping is
#    {12: 255, 13: 12} -- dropping 12 alone would leave Pole at 13, outside the
#    13-output model.
#
# Resolution: FMB images are 800x600.  600 is not a multiple of the ViT patch
# size 16, so the native image is fed as-is and the data preprocessor pads
# height 600 -> 608 (size_divisor=16) with seg_pad_val=255, which keeps the
# padded rows out of the loss.  This preserves the native pixels exactly; the
# alternative -- resizing to 608x800 -- would distort by 1.3%, and simply
# feeding 600 would make the patch-embed convolution silently discard the
# bottom 8 rows.

data_root = '{{$FMB_ROOT:/root/autodl-tmp/data/FMB_ALL/FMB}}'
dataset_type = 'FMBDataset13'
crop_size = (600, 800)

train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RemapLabels', mapping={12: 255, 13: 12}),
    # Resize upward only, so the 600x800 crop always fits without padding and
    # the augmentation is a genuine random crop at varying scale.
    dict(type='RandomResize', scale=(800, 600), ratio_range=(1.0, 1.5),
         keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RemapLabels', mapping={12: 255, 13: 12}),
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
        data_prefix=dict(img_path='images/training',
                         seg_map_path='annotations/training'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/validation',
                         seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
