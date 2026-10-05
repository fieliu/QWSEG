# Stage 3: Dense-to-Sparse Distillation (doc section 9) on MFNet 480x640.
# Sparse student <- Frozen Dense Robust Teacher (the accepted Stage-2B EMA
# weights). Student adds the Utility Router and prunes extras to a fixed budget
# K via a real gather; teacher stays dense and fully frozen. Distillation:
# L_compression (same C(x)) + L_robust (student C(x) vs teacher clean) +
# labeled L_logit + standard seg loss.
_base_ = [
    '_base_dino_ts_m2f.py',
    '_base_dino_ts_data.py',
    '../_base_/default_runtime.py',
]

crop_size = (480, 640)
# N = (480/16) * (640/16) = 30 * 40 = 1200 anchor positions.
# target_k = 0.5 N = 600 extras kept at final budget (doc 9.7).
target_k = 600

# Frozen dense teacher: a DinoTSDense with the SAME backbone/neck/head, loaded
# from the Stage-2B EMA checkpoint. Kept dense (all extras) and frozen.
teacher_cfg = dict(
    type='DinoTSDense',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='dense',
    test_cfg=dict(mode='whole'))

model = dict(
    type='DinoTSSparse',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='sparse_hard',   # eval path uses real Top-K gather
    target_k=target_k,
    budget_schedule=(1.0, 0.75, 0.5),  # K ramps N -> .75N -> .5N -> target_k
    budget_warmup_epochs=30,
    soft_to_hard_epoch=30,        # soft gate for warmup, then hard gather
    lambda_comp=1.0,
    lambda_rob=1.0,
    lambda_logit=1.0,
    logit_temperature=2.0,
    teacher_cfg=teacher_cfg,
    # set on the CLI:
    #   --cfg-options model.teacher_ckpt=work_dirs/stage2b/best_mIoU.pth
    teacher_ckpt=None,
    init_from_teacher=True,
    degradation={{_base_.degradation_policy}},
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    accumulative_counts=2,
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=1.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=5, end=200,
         by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5,
        max_keep_ckpts=2, save_last=True, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))

custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='PartialDegradeEvalHook', interval=5, num_samples=50),
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
