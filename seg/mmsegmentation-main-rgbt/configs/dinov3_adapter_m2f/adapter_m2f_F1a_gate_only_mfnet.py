# DINOv3-Adapter + M2F F1a: FROZEN backbone + FROZEN baseline fusion + GATE ONLY.
# = quality prediction + quality-gated cross-attn (suppress low-quality keys),
# but no quality-weighted residual (use_quality_merge=False).
# This isolates the contribution of "don't listen to bad tokens" vs
# F0 (no quality) and vs F1 (full quality).
# F1a - F0 = gate-only increment
# F1 - F1a = merge-on-top-of-gate increment
_base_ = ['./adapter_m2f_quality_rgbt_mfnet.py']

model = dict(
    freeze_backbone=True,
    freeze_fusion=True,           # freeze baseline fusion
    use_quality_gate=True,        # quality-gated attention ON
    use_quality_merge=False,      # no quality-weighted residual
    quality_loss_weight=1.0,
    deg_ceiling_weight=0.1,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)
