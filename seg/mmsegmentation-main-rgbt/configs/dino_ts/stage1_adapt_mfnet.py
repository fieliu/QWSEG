# Stage 1: Modality Adaptation (doc section 6) on MFNet 480x640.
# Feature-alignment pretext: freeze RGB path + shared DINO + fusion/head, train
# ONLY the Thermal PatchEmbed / modality embed / shallow Thermal adapters /
# student projector, aligning Thermal tokens to the frozen RGB DINO teacher.
# No segmentation output (doc: exit criteria use a linear probe, not seg mIoU).
#
# NOTE: the doc recommends running Stage 1 on well-registered UNLABELED RGB-T
# pairs (LLVIP / KAIST, deduped). Here we reuse the MFNet train split as a
# stand-in so the pipeline is runnable; swap the dataloader to the paired
# unlabeled set for the real run.
_base_ = [
    '_base_dino_ts_m2f.py',
    '_base_dino_ts_data.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DinoTSStage1Adapt',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='adapt',
    # Primary: preserve RGB-like spatial relations between coarse regions.
    # Weak region cosine fixes the shared coordinate system; exact patch
    # matching is disabled to retain thermal-specific information.
    lambda_patch=0.0,
    lambda_region=0.25,
    lambda_relation=1.0,
    region=2,
    relation_temperature=0.2,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=58800,
         by_epoch=False),
]

# alignment pretext: no validation loop (no segmentation metric)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=100000)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=10,
        max_keep_ckpts=2, save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'))

custom_hooks = [
    dict(type='EpochSyncHook'),
]

# Scalar losses/LR are written both to the normal log and TensorBoard.  Stage 1
# has no semantic predictions to render because it is an alignment pretext.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
