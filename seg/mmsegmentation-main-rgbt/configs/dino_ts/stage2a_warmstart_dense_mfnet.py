# Stage 2A warm start: 5-epoch Dense bimodal fine-tune on MFNet.
#
# This is a short diagnostic run, not the final Stage 2A schedule.  It exists to
# answer one question: starting from "RGB-only best (epoch 3, mIoU 52.52) +
# selective Stage1 thermal transfer", does the Dense 2400-token path recover and
# then beat the RGB-only 52.52 baseline?  Only if it does do the longer Stage 2A
# schedule, Stage 2B robustness training and Stage 3 distillation make sense.
#
# The seed is pinned to 471605737, the seed mmengine generated for the RGB-only
# diagnostic run (work_dirs/dino_ts/diagnostic_rgb_only_trainval_e1), so the
# data order and augmentation stream match and the two curves are paired.
#
# The init weights are the merged checkpoint written by the Stage1 selective
# transfer step, NOT the full Stage1 checkpoint: only the 24 thermal tensors
# (893,379 params) may cross from Stage 1 to Stage 2.
_base_ = ['./stage2a_dense_mfnet.py']

randomness = dict(seed=471605737, deterministic=False)

# NOTE: mmengine substitutes ``{{$VAR:default}}`` with a regex that cannot
# cross a line break, so the whole template must stay on one line.
load_from = '{{$DINO_TS_WARMSTART:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/rgb_e3_plus_stage1_thermal.pth}}'

# Short diagnostic schedule: validate and checkpoint every epoch so the
# warm-start curve can be read off directly.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1,
        save_last=True, save_best='mIoU', rule='greater'))
