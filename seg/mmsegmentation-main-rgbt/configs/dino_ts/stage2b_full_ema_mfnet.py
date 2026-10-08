# Stage 2B: full Dense EMA robust self-distillation run on MFNet.
#
# Starts from the accepted clean Dense Teacher (Stage 2A, mIoU 62.07) and adds
# three-view robustness plus an EMA teacher with per-position anchor
# consistency (doc section 8).
#
# Schedule note: the inherited 200-epoch schedule is not affordable here.  A
# single iteration costs ~7.7x a Stage-2A iteration (three supervised
# segmentation passes -- clean / degraded / missing-modality -- plus the online
# anchor pass and the no-grad EMA anchor pass, plus the EMA update), measured
# at 3.15 s/iter vs 0.41 s/iter.  200 epochs would be ~103 h.
#
# 60 epochs is chosen because Stage 2A plateaued around epoch 60 of its
# 200-epoch schedule: 60 epochs is where this model/dataset combination
# settles.  That is ~31 h here.
#
# The PolyLR horizon must move with the schedule.  Leaving end=200 while
# running 60 epochs would stop the LR at ~2.1e-5 (70% of peak) and the model
# would never converge.
_base_ = ['./stage2b_ema_mfnet.py']

randomness = dict(seed=471605737, deterministic=False)

# The accepted Stage-2A clean Dense Teacher.
# NOTE: mmengine substitutes ``{{$VAR:default}}`` with a regex that cannot
# cross a line break, so the whole template must stay on one line.
load_from = '{{$DINO_TS_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_dense_teacher_e115_mIoU62.07.pth}}'

# EMA momentum ramps 0.996 -> 0.9999 over this many epochs (doc 8.1), so it
# must track max_epochs.
model = dict(total_epochs=60)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=60,
         by_epoch=True, convert_to_iter_based=True),
]

# Keep one rolling checkpoint (also what last_checkpoint points at, so --resume
# works) plus the best.  Stage-2B checkpoints are ~2x Stage-2A ones because the
# EMA teacher is part of the state dict.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best='mIoU', rule='greater'))
