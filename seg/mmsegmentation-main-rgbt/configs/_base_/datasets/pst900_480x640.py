dataset_type = 'PST900Dataset'
data_root = '/home/lh/code/data/PST900_RGBT_Dataset'
crop_size = (480, 640)
train_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_color_type='unchanged',
         ir_replace_src='/rgb/', ir_replace_dst='/thermal/'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(640, 480),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile',
         ir_color_type='unchanged',
         ir_replace_src='/rgb/', ir_replace_dst='/thermal/'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/rgb', seg_map_path='train/labels'),
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
        data_prefix=dict(img_path='test/rgb', seg_map_path='test/labels'),
        ann_file='test.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
