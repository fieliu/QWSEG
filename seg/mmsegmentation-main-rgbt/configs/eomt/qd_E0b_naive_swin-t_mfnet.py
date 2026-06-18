# E0b: NAIVE degraded training. No quality mechanism, no distillation.
# = "the model has merely SEEN degraded data" baseline. Every other mechanism
# must beat THIS to claim an independent contribution.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    use_quality=False,            # no bias, no compensation, no quality loss
    teacher_cfg=None,             # no teacher
    teacher_ckpt=None,
    distill_loss_weight=0.0,      # no feature distillation
    output_distill_weight=0.0,    # no output distillation
)
