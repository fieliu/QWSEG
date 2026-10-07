# Stage 2A: full clean Dense Teacher schedule (200 epochs) on MFNet.
#
# This is the single-variable extension of the 5-epoch diagnostic
# (stage2a_warmstart_dense_mfnet.py): identical warm start, identical seed,
# identical clean-only objective and identical optimizer/scheduler.  Only the
# schedule length and the checkpoint retention differ, so the two runs and the
# RGB-only baseline stay directly comparable.
#
# The 5-epoch diagnostic answered its question -- the Dense 2400-token path
# recovers from the warm start and beats RGB-only (57.98 vs 52.52 at best) --
# but it stopped while the LR was still at its 3e-5 peak (PolyLR runs to epoch
# 200), so it was never a converged number.
#
# The seed is pinned to 471605737, the seed mmengine generated for the RGB-only
# diagnostic run, so the data order and augmentation stream match.
_base_ = ['./stage2a_dense_mfnet.py']

randomness = dict(seed=471605737, deterministic=False)

# Same merged checkpoint as the diagnostic run: RGB-only best (epoch 3,
# mIoU 52.52) + selective Stage1 thermal (24 tensors / 893,379 params).
# NOTE: mmengine substitutes ``{{$VAR:default}}`` with a regex that cannot
# cross a line break, so the whole template must stay on one line.
load_from = '{{$DINO_TS_WARMSTART:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/rgb_e3_plus_stage1_thermal.pth}}'

# Retain exactly one rolling checkpoint (the latest, which is also what
# ``last_checkpoint`` points at, so --resume still works) plus the best-mIoU
# one.  The base config keeps 2 rolling checkpoints (~1.8 GB each).
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best='mIoU', rule='greater'))
