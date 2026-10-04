# Stage 2A: Dense Robust Warm-up (doc section 7) on MFNet 480x640.
# Trains a full-token (dense) shared-ViT anchor/extra segmentor with three-view
# supervision: clean + degraded C(x) + missing-modality Drop(x), all vs real GT.
# This is the first trainable segmentation entry; its accepted weights seed 2B.
_base_ = [
    '_base_dino_ts_m2f.py',
    '_base_dino_ts_data.py',
    '../_base_/default_runtime.py',
]

crop_size = (480, 640)

model = dict(
    type='DinoTSDense',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='dense',
    lambda_deg=1.0,
    lambda_missing=1.0,
    degradation={{_base_.degradation_policy}},
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

# AdamW + layer-wise decay: ViT blocks decayed, adapters/fusion/head at full lr.
optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=117600,
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

custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='PartialDegradeEvalHook', interval=5, num_samples=50),
]
