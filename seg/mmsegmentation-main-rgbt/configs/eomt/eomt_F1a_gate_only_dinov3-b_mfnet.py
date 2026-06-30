# EoMT RGB-T F1a: FROZEN backbone + GATE ONLY (self-attn bias, no compensation).
# = quality prediction + quality-injected self-attn bias (suppress low-quality tokens),
# but NO cross-modal compensation merge.
# F1a - F0 = gate-only increment
# F1 - F1a = compensation-on-top-of-gate increment
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_self_attn_bias=True,      # gate ON: quality bias in self-attn
    use_compensation=False,       # no cross-modal compensation
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
