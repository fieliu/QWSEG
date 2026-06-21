# EoMT RGB-T F1: FROZEN backbone + FULL quality mechanism, NO distillation,
# started from the CLEAN model. Adapter paradigm: warm-start from clean fusion
# baseline (teacher weights = starting point), freeze backbone, train quality
# (bias + compensation) + fusion + decode head only.
# F1 - F0 = the quality plugin's pure robustness increment on a frozen clean
# model (no backbone change). Provide clean ckpt:
#   --cfg-options model.teacher_ckpt=<clean fusion best.pth>
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_self_attn_bias=True,
    use_compensation=True,
    distill_loss_weight=0.0,      # NO distillation (isolate quality increment)
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
