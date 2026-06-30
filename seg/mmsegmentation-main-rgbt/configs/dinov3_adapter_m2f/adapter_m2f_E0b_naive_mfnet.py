# DINOv3-Adapter + M2F E0b: NAIVE degraded training (no quality mechanism).
# = "the model has merely SEEN degraded data" baseline.
# Uses DINOv3AdapterM2FQuality with quality OFF, so:
#   - No quality repair step (quality_predictors / quality_cross_fusions unused)
#   - Features go directly to baseline fusion (CrossAttn + ChannelSpatialAttn)
#   - Degradation is still applied (model sees degraded data without any defense)
# All parameters update (no freezing).
_base_ = ['./adapter_m2f_quality_rgbt_mfnet.py']

model = dict(
    use_quality_gate=False,       # no quality-gated attention
    use_quality_merge=False,      # no quality-weighted residual
    quality_loss_weight=0.0,      # no quality supervision
    deg_ceiling_weight=0.0,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)
