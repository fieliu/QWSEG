# Stage 3: full Dense-to-Sparse distillation run on MFNet.
#
# Student keeps all 1200 anchors and prunes the 1200 extras to a fixed budget
# K via a real Top-K gather; the dense teacher stays frozen.  Losses are
# L_compression (student vs teacher on the SAME corrupted view), L_robust
# (student on C(x) vs teacher on clean x), labeled L_logit and the ordinary
# segmentation loss.
#
# Changes from stage3_sparse_mfnet.py, all needed before the first run:
#
#   * teacher_ckpt set.  The base leaves it None, so the run would have had no
#     frozen teacher at all.
#   * PartialDegradeEvalHook dropped and RobustSelectionHook used instead.  The
#     former evaluates 50 samples and averages only over the classes present in
#     them, which on MFNet drops guardrail (4 test images) and turns the
#     9-class metric into an 8-class one -- its numbers are not comparable to
#     the headline mIoU.
#   * 60 epochs instead of 200.  Stage 2A plateaued around epoch 60 of its own
#     200-epoch schedule, and Stage 2C used 60.  The PolyLR horizon moves with
#     it; leaving it at 200 would stop the LR at ~2.1e-5 and never converge.
#   * Seed pinned, so the run is comparable with the rest of the pipeline.
#   * max_keep_ckpts 2 -> 1 (the server is down to ~7 GB free).
_base_ = ['stage3_sparse_mfnet.py']

randomness = dict(seed=471605737, deterministic=False)

# The accepted Stage-2C EMA teacher.  best_robust_score_epoch_60 is preferred
# over best_mIoU_epoch_35: it gives up 0.16 clean mIoU for 0.61 corrupted mIoU,
# and Stage 3's whole purpose is robustness at a reduced token budget.
# NOTE: mmengine's env-var regex cannot cross a line break.
model = dict(
    teacher_ckpt='{{$STAGE2C_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2c_full_ema/weight/stage2c_full_ema/best_robust_score_epoch_60.pth}}',
    # The base config ramps K over 30 epochs and only switches from the soft
    # gate to the real Top-K gather at epoch 30.  With max_epochs=30 that leaves
    # ZERO epochs of hard-gather training -- the student would never be trained
    # in the regime inference actually uses.  Halving both gives 15 epochs of
    # soft ramp and 15 epochs of hard gather.
    budget_warmup_epochs=15,
    soft_to_hard_epoch=15)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=5, end=30,
         by_epoch=True, convert_to_iter_based=True),
]

custom_hooks = [
    dict(type='EpochSyncHook'),
    # priority NORMAL is above CheckpointHook's VERY_LOW, so the injected
    # robust_score is visible to save_best.
    dict(type='RobustSelectionHook', priority='NORMAL',
         cases=['gaussian_noise/3', 't_gaussian_noise/3',
                'rgb_missing/1', 't_missing/1'],
         seed=42, clean_weight=0.5),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best=['robust_score', 'mIoU'],
        rule=['greater', 'greater']),
    visualization=dict(type='SegVisualizationHook', draw=False))
