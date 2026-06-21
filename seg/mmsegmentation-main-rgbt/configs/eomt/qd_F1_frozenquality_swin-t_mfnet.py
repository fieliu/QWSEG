# Swin RGB-T F1: FROZEN backbone + FULL quality mechanism, NO distillation,
# started from the CLEAN model. Warm-start from clean qd_teacher, freeze
# backbone, train quality (bias + compensation) + fusion + decode head only.
# F1 - F0 = the quality plugin's pure robustness increment on a frozen clean
# model. Provide clean ckpt: --cfg-options model.teacher_ckpt=<best.pth>
# DINOv3 mirror: eomt_F1_frozenquality.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_compensation=True,
    distill_loss_weight=0.0,      # NO distillation (isolate quality increment)
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
