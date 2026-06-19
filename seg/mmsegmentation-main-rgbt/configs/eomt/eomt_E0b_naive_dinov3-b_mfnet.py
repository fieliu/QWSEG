# EoMT RGB-T E0b: NAIVE degraded training. No quality mechanism, no distillation.
# = "the model has merely SEEN degraded data" baseline. DINOv3 mirror of qd_E0b.
# Every other mechanism (distillation in E1, quality in E2) must beat THIS to
# claim an independent contribution.
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    use_quality=False,            # no bias, no compensation, no quality loss
    teacher_cfg=None,             # no teacher
    teacher_ckpt=None,
    distill_loss_weight=0.0,      # no feature distillation
    output_distill_weight=0.0,    # no output distillation
    quality_loss_weight=0.0,
)
