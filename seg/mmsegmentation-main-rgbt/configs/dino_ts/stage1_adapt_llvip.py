"""Stage-1 adaptation on LLVIP train, selected on held-out LLVIP test pairs."""

_base_ = ['stage1_adapt_mfnet.py']

llvip_root = '{{$LLVIP_ROOT:/root/autodl-tmp/data/LLVIP}}'

model = dict(
    backbone=dict(with_cp=False),
    # Relation alignment is primary.  A weak pooled-region cosine term removes
    # coordinate ambiguity; exact patch equality remains disabled.
    lambda_patch=0.0,
    lambda_region=0.25,
    lambda_relation=1.0,
    relation_temperature=0.2)

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=llvip_root,
        data_prefix=dict(img_path='visible/train'),
        img_suffix='.jpg',
        ann_file='',
        pipeline=[
            dict(
                type='LoadRGBTImageFromFile',
                ir_replace_src='/visible/',
                ir_replace_dst='/infrared/',
                ir_color_type='color'),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(480, 640)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs'),
        ]))

# LLVIP has an official train/test split but no separate validation split.
# Train never reads test pairs; the official test pairs are used only to select
# the lowest label-free alignment loss for this auxiliary pretraining stage.
val_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=llvip_root,
        data_prefix=dict(img_path='visible/test'),
        img_suffix='.jpg',
        ann_file='',
        test_mode=False,
        pipeline=[
            dict(
                type='LoadRGBTImageFromFile',
                ir_replace_src='/visible/',
                ir_replace_dst='/infrared/',
                ir_color_type='color'),
            # Exact size avoids padding tokens biasing the alignment metric.
            dict(type='Resize', scale=(640, 480), keep_ratio=False),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(_delete_=True, type='Stage1AlignmentMetric')

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(_delete_=True, type='ValLoop')
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=2,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=48, eta_min=1e-6, by_epoch=True,
         begin=2, end=50, convert_to_iter_based=True),
]
custom_hooks = []
default_hooks = dict(
    logger=dict(interval=50, log_metric_by_epoch=True),
    checkpoint=dict(
        by_epoch=True, interval=5, max_keep_ckpts=2, save_last=True,
        save_best='stage1/align_loss', rule='less'))
randomness = dict(seed=42)
