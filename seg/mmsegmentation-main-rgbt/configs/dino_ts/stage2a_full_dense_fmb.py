# Stage 2A on FMB: clean Dense Teacher, 13 foreground classes, native 800x600.
#
# Inherits the FMB model fragment (13 classes, size_divisor=16 padding) and the
# FMB data fragment (FMBDataset13 + RemapLabels, which removes Bicycle and
# closes the label gap it would otherwise leave).
#
# Warm start: run tools/extract_dino_ts_stage1.py on the Stage-1 FMB checkpoint
# and point load_from at the merged result.  Never load a full Stage-1
# checkpoint here -- it carries randomly initialised fusion/decoder weights that
# would overwrite the correct Stage-2 initialisation.
_base_ = [
    '_base_dino_ts_m2f_fmb.py',
    '_base_dino_ts_data_fmb.py',
    '../_base_/default_runtime.py',
]

crop_size = (600, 800)

# Stage-1 thermal transfer only: 24 tensors / 893,379 params
# (thermal PatchEmbed + thermal modality_embed + the three shallow thermal
# adapters).  Loaded with strict=False into the DINOv3-initialised model, so the
# RGB path and the shared trunk keep their pretrained weights while the thermal
# branch starts from the FMB-adapted values.  The full Stage-1 checkpoint must
# never be used here: it also carries randomly initialised fusion/decoder
# weights that would overwrite the correct Stage-2 initialisation.
# NOTE: mmengine's env-var regex cannot cross a line break, keep this on one line.
load_from = '{{$FMB_STAGE1_TRANSFER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/fmb_stage1_thermal_transfer.pth}}'

# Two samples per step; FMB images are all exactly 800x600 so the batch needs no
# cross-sample padding, only the 600 -> 608 divisibility pad.
train_dataloader = dict(batch_size=2)

model = dict(
    type='DinoTSDense',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='dense',
    lambda_deg=0.0,
    lambda_missing=0.0,
    degradation={{_base_.degradation_policy}},
    train_cfg=dict(),
    # Every FMB image is exactly 800x600, so sliding windows buy nothing and
    # would only run the backbone twice for a 608-tall padded input with a
    # 600-tall crop.  Whole inference on the padded image is both correct and
    # half the cost.
    test_cfg=dict(mode='whole'))

# AdamW + layer-wise decay: ViT blocks decayed, adapters/fusion/head at full lr.
optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    accumulative_counts=1,
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=1.0))

# 100 epochs.  Stage 2A on MFNet plateaued around epoch 60 of a 200-epoch
# schedule (~35k iterations) and FMB has a comparable image count, so 100
# epochs is generous headroom; the run can be stopped early once the curve
# flattens.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=100,
         by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        max_keep_ckpts=1, save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False))

custom_hooks = [dict(type='EpochSyncHook')]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
