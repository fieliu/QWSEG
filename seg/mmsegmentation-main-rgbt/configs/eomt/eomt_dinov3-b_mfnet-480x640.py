# EoMT single-modal (RGB-only) smoke-test config on MFNet.
# Deliverable 1: confirm EoMT runs inside mmseg end to end.
_base_ = [
    '../_base_/datasets/mfnet_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.eomt_segmentor',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

crop_size = (480, 640)   # 480/16=30, 640/16=40 -> divisible by DINOv3 patch 16
num_classes = 9

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=16))

# Path to a locally-saved HuggingFace DINOv3 ViT-B/16 model dir.
# Leave None to attempt online download (requires transformers + internet).
backbone_ckpt = None

model = dict(
    type='EoMTSegmentor',
    data_preprocessor=data_preprocessor,
    img_size=crop_size,
    num_classes=num_classes,
    backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    backbone_ckpt=backbone_ckpt,
    patch_size=16,
    num_q=100,
    num_blocks=4,
    masked_attn_enabled=True,
    local_files_only=True,
    mask_coefficient=5.0,
    dice_coefficient=5.0,
    class_coefficient=2.0,
    no_object_coefficient=0.1,
    ignore_index=255,
)

# Layer-wise lr decay is common for ViT fine-tuning; keep it simple here.
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=10.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'network.encoder.backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'network.q': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=117600,
         by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
