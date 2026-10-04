"""Four-step GPU integration check, not a scientific training configuration.

Uses real MFNet samples at the target resolution, validates four samples and
saves a checkpoint. Clean-only input isolates runtime/weight-loading issues;
robustness and later-stage checks must be run separately before full training.
"""
_base_ = ['stage2a_dense_mfnet.py']

model = dict(
    backbone=dict(backbone_ckpt='/root/autodl-tmp/pretrain/dinov3-vitb16'),
    lambda_deg=0.0,
    lambda_missing=0.0,
    test_cfg=dict(mode='whole'))

data_root = '/root/autodl-tmp/data/MFNet'
pipeline = [
    dict(type='LoadRGBTImageFrom4Channel'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=data_root, indices=list(range(8)), pipeline=pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=data_root, ann_file='val.txt', indices=list(range(4))))
test_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False,
    dataset=dict(data_root=data_root, ann_file='test.txt', indices=list(range(4))))

train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=4, val_interval=4)
param_scheduler = []
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', loss_scale=dict(init_scale=128.0),
    optimizer=dict(type='AdamW', lr=3e-5, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))
custom_hooks = []
default_hooks = dict(
    logger=dict(interval=1, log_metric_by_epoch=False),
    checkpoint=dict(by_epoch=False, interval=4, max_keep_ckpts=1, save_best=None))
log_processor = dict(by_epoch=False)
randomness = dict(seed=42)
