# DINOv3-Adapter + M2F F1b: FROZEN backbone + FROZEN baseline fusion + MERGE ONLY.
# = quality prediction + quality-weighted residual merge,
# but no quality key-gating in quality_cross_fusions (plain cross-attn).
# This isolates the contribution of "trust repaired features by quality" vs
# F0 (no quality) and vs F1 (full quality).
# F1b - F0 = merge-only increment
# F1 - F1b = gate-on-top-of-merge increment
_base_ = ['./adapter_m2f_quality_rgbt_mfnet.py']

model = dict(
    freeze_backbone=True,
    freeze_fusion=True,           # freeze baseline fusion
    use_quality_gate=False,       # no key-gating (plain quality cross-attn)
    use_quality_merge=True,       # quality-weighted residual ON
    quality_loss_weight=1.0,
    deg_ceiling_weight=0.1,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)
