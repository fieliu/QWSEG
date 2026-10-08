# Stage 2C: Dense EMA robust self-distillation WITH direct supervision on the
# strong (degraded) view.
#
# Difference from Stage 2B (stage2b_full_ema_mfnet.py) is one term:
#
#   2B:  L = L_stage2A(clean)                 + lambda_anchor * L_anchor
#   2C:  L = L_stage2A(clean)
#          + lambda_deg * L_seg(strong)       <-- added here
#          + lambda_anchor * L_anchor
#
# Rationale: 2B makes the student robust only through feature-level agreement
# (the backbone is pulled towards the clean teacher's anchors).  Nothing ever
# tells the model what the correct segmentation of a degraded image is, so the
# decoder is never trained on the feature distribution it will actually meet
# under degradation.  2C adds that supervision directly.
#
# This costs nothing extra: the strong view is already forwarded once for
# L_anchor, and 2C simply attaches the segmentation loss to that same pass
# (the loss implementation reuses the anchors from it rather than forwarding
# again).  Same 3 backbone passes / 2 backwards as 2B.
#
# The degradation policy is unchanged and still sampled ONCE per sample, so
# degrade_prob=0.8, modality_probs, scope_probs, severity_range and
# missing_prob=0.1 all keep their intended meaning.  Note that ~20% of samples
# therefore arrive clean, which under 2C means the strong-view segmentation
# loss is itself computed on a clean image about one time in five.
#
# 2B and 2C are a clean single-variable comparison and share the same code
# path: lambda_deg = 0 gives 2B, lambda_deg > 0 gives 2C.
_base_ = ['./stage2b_full_ema_mfnet.py']

model = dict(lambda_deg=1.0)

# ---------------------------------------------------------------------------
# Robustness-aware checkpoint selection.
#
# The stock CheckpointHook selects on clean mIoU, which for Stage 2 picks the
# *least* robustified checkpoint -- exactly the wrong criterion.  The proxy
# benchmark below runs 4 corrupted cases over the FULL 393 test images every
# validation (~3-4 min) and injects robust_score = 0.5*clean + 0.5*corrupted,
# so CheckpointHook can select on it.
#
# Image count is deliberately not reduced: guardrail appears in only 4 test
# images, bump in 17, color_cone in 61, and any class missing from a subset is
# dropped from the mean, which would make the proxy incomparable to the
# headline 9-class mIoU.
#
# PartialDegradeEvalHook is dropped: it evaluates 50 samples and averages only
# over classes present there, so its numbers are not comparable to anything.
#
# Both best checkpoints are kept (robust_score and clean mIoU) plus the latest,
# so the clean-vs-robust trade-off can be inspected after the run instead of
# being collapsed into one scalar up front.
# ---------------------------------------------------------------------------
custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='EMAUpdateHook'),
    # priority='NORMAL' is above CheckpointHook's 'VERY_LOW', so this runs
    # first and the injected robust_score is visible to save_best.
    dict(type='RobustSelectionHook', priority='NORMAL',
         cases=['gaussian_noise/3', 't_gaussian_noise/3',
                'rgb_missing/1', 't_missing/1'],
         seed=42,
         clean_weight=0.5),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best=['robust_score', 'mIoU'],
        rule=['greater', 'greater']))
