# EoMT RGB-T F1d: FROZEN backbone + FULL quality mechanism + DISTILLATION,
# started from the CLEAN model. Same as F1 but turns distillation ON: the clean
# teacher (clean input) guides the frozen-backbone student (degraded input) to
# output like the clean result. With the backbone frozen the student has fewer
# free params, so the distillation target may help MORE than in the full-
# finetune setting.
# F1d - F1 = the distillation increment under the frozen-backbone paradigm
# (answers: does distilling toward the clean output help when the backbone is
# frozen?). Provide clean ckpt: --cfg-options model.teacher_ckpt=<best.pth>
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_self_attn_bias=True,
    use_compensation=True,
    distill_loss_weight=1.0,      # distillation ON (clean-output target)
    output_distill_weight=1.0,
    quality_loss_weight=1.0,
)
