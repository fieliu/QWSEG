"""Four-step LLVIP modality-adaptation check; no semantic labels are consumed."""
_base_ = ['stage1_adapt_mfnet.py']

model = dict(backbone=dict(
    backbone_ckpt='/root/autodl-tmp/pretrain/dinov3-vitb16'))
train_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=0, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset', data_root='/root/autodl-tmp/data/LLVIP',
        data_prefix=dict(img_path='visible/train'),
        img_suffix='.jpg', ann_file='', indices=list(range(8)),
        pipeline=[
            dict(type='LoadRGBTImageFromFile', ir_replace_src='/visible/',
                 ir_replace_dst='/infrared/', ir_color_type='color'),
            dict(type='Resize', scale=(640, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(480, 640)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs'),
        ]))
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=4)
param_scheduler = []
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))
custom_hooks = []
default_hooks = dict(
    logger=dict(interval=1, log_metric_by_epoch=False),
    checkpoint=dict(by_epoch=False, interval=4, max_keep_ckpts=1))
randomness = dict(seed=42)
log_processor = dict(by_epoch=False)
