# EoMT RGB-T F1b: FROZEN backbone + MERGE ONLY (compensation, no self-attn bias).
# = quality prediction + cross-modal compensation merge,
# but NO quality bias in self-attn (plain self-attn at fusion points).
# F1b - F0 = merge-only increment
# F1 - F1b = gate-on-top-of-merge increment
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_self_attn_bias=False,     # no gate: plain self-attn at fusion points
    use_compensation=True,        # merge ON: cross-modal compensation
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
