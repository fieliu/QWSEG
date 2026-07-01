# DINOv3-Adapter + M2F F1: FROZEN backbone + FROZEN baseline fusion + FULL quality.
# Warm-start from clean baseline, freeze backbone + baseline fusion,
# train ONLY the quality increment (quality_predictors + quality_cross_fusions)
# + decode head.
# F1 - F0 = the quality plugin's pure robustness increment on a frozen clean
# model (no backbone/baseline-fusion change).
_base_ = ['./adapter_m2f_quality_rgbt_mfnet.py']

model = dict(
    freeze_backbone=True,
    freeze_fusion=True,           # freeze baseline fusion (cross_fusions + cs_fusions)
    use_quality_gate=True,        # quality-gated attention ON
    use_quality_merge=True,       # quality-weighted residual ON
    quality_loss_weight=1.0,
    deg_ceiling_weight=0.1,
    clean_floor_weight=0.1,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)
