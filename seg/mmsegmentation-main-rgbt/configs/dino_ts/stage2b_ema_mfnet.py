# Stage 2B: Dense EMA Robust Self-Distillation (doc section 8) on MFNet 480x640.
# Inherits Stage 2A (dense three-view seg + degradation) and adds an EMA teacher
# + per-position anchor consistency (online x_strong vs stop-grad EMA x_weak).
# Warm-start from an accepted Stage-2A checkpoint via --cfg-options load_from=...
_base_ = ['stage2a_dense_mfnet.py']

crop_size = (480, 640)

model = dict(
    type='DinoTSDenseEMA',
    forward_mode='dense',
    ema_momentum_base=0.996,
    ema_momentum_final=0.9999,
    total_epochs=200,
    lambda_anchor=1.0,
    lambda_global=0.0,   # optional DINO-style global CE; off by default
    # Inherit exactly the Stage-2A policy. The EMA target sees clean pixels.
    )

# Warm-start the online model from the accepted Stage-2A weights (the EMA
# teacher is registered before DDP and synchronized after checkpoint loading).
# Set on the CLI:  --cfg-options load_from=work_dirs/stage2a/best_mIoU.pth
load_from = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)

custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='EMAUpdateHook'),
    dict(type='PartialDegradeEvalHook', interval=5, num_samples=50),
]
