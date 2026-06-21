# Swin RGB-T F1d: FROZEN backbone + FULL quality mechanism + DISTILLATION,
# started from the CLEAN model. Same as F1 but distillation ON: clean teacher
# (clean input) guides the frozen-backbone student (degraded input). With the
# backbone frozen the student has fewer free params, so distillation toward the
# clean output may help more than in full-finetune.
# F1d - F1 = the distillation increment under the frozen-backbone paradigm.
# Provide clean ckpt: --cfg-options model.teacher_ckpt=<best.pth>
# DINOv3 mirror: eomt_F1d_frozenquality_distill.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_compensation=True,
    distill_loss_weight=1.0,      # distillation ON (clean-output target)
    output_distill_weight=1.0,
    quality_loss_weight=1.0,
)
