"""Stage-1 modality adaptation on FMB train pairs, selected on FMB val pairs.

Label-free: Stage 1 aligns thermal tokens to the frozen RGB DINO teacher and
never reads a segmentation map, so the pipeline has no LoadAnnotations at all.
The FMB training split is used purely as a bank of registered RGB-T pairs, which
removes the cross-domain transfer the LLVIP variant would otherwise introduce
(LLVIP is surveillance footage of pedestrians; FMB is driving scenes).

Resolution: 800x600 native.  600 is not a multiple of the patch size 16, so the
train pipeline crops to 608x800 -- the crop always fits because the resize is
constrained to keep the shorter side at 627 or more.  Validation resizes to
exactly 608x800 with keep_ratio=False so that no padding tokens can bias the
alignment metric (the same reason the LLVIP variant does this).
"""

_base_ = ['stage1_adapt_mfnet.py']

fmb_root = '{{$FMB_ROOT:/root/autodl-tmp/data/FMB_ALL/FMB}}'

model = dict(
    backbone=dict(with_cp=False),
    # Relation alignment is primary.  A weak pooled-region cosine term removes
    # coordinate ambiguity; exact patch equality remains disabled.
    lambda_patch=0.0,
    lambda_region=0.25,
    lambda_relation=1.0,
    relation_temperature=0.2)

_ir = dict(ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T')

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=fmb_root,
        data_prefix=dict(img_path='images/training'),
        img_suffix='.png',
        seg_map_suffix='.png',
        ann_file='',
        pipeline=[
            dict(type='LoadRGBTImageFromFile', **_ir),
            # scale=(880, 660): the shorter side maps to 660, and ratio_range
            # keeps it at 627 or above, so the 608x800 crop always fits.
            dict(type='RandomResize', scale=(880, 660),
                 ratio_range=(0.95, 1.25), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(608, 800)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs'),
        ]))

# Held-out FMB pairs; never seen by the alignment loss during training.
val_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=fmb_root,
        data_prefix=dict(img_path='images/validation'),
        img_suffix='.png',
        seg_map_suffix='.png',
        ann_file='',
        test_mode=False,
        pipeline=[
            dict(type='LoadRGBTImageFromFile', **_ir),
            # Exact size: padding tokens would bias the alignment metric.
            dict(type='Resize', scale=(800, 608), keep_ratio=False),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(_delete_=True, type='Stage1AlignmentMetric')

train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
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
    dict(type='CosineAnnealingLR', T_max=98, eta_min=1e-6, by_epoch=True,
         begin=2, end=100, convert_to_iter_based=True),
]
custom_hooks = []
default_hooks = dict(
    logger=dict(interval=50, log_metric_by_epoch=True),
    checkpoint=dict(
        by_epoch=True, interval=5, max_keep_ckpts=1, save_last=True,
        save_best='stage1/align_loss', rule='less'))
randomness = dict(seed=42)
