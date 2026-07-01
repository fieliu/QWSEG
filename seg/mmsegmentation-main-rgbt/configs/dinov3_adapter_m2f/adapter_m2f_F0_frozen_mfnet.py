# DINOv3-Adapter + M2F F0: FROZEN-backbone + FROZEN-baseline-fusion, NO quality.
# Warm-start from trained clean baseline (Mask2FormerRGBTCrossAttn), freeze
# backbone + baseline fusion (cross_fusions + cs_fusions), train ONLY decode head.
# Quality modules (quality_predictors + quality_cross_fusions) are NOT added
# (use_quality_gate=False, use_quality_merge=False).
# Lower bound for the frozen-backbone increment study:
#   F0 = freeze all, no quality -> only decode head trains
#   F1 = freeze backbone + baseline fusion, train quality modules + head
#   F1-F0 gain = quality mechanism contribution (pure increment value)
_base_ = ['./adapter_m2f_quality_rgbt_mfnet.py']

model = dict(
    freeze_backbone=True,         # freeze DINOv3 + Adapter
    freeze_fusion=True,           # freeze baseline fusion (cross_fusions + cs_fusions)
    use_quality_gate=False,       # no quality-gated attention
    use_quality_merge=False,      # no quality-weighted residual
    quality_loss_weight=0.0,
    deg_ceiling_weight=0.0,
    clean_floor_weight=0.0,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)
