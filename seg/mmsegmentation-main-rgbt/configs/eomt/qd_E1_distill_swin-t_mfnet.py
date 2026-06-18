# E1: DISTILLATION ONLY. Degraded training + distillation toward the clean
# teacher, but NO quality mechanism (no bias, no compensation, no quality loss).
# E1 vs E0b  -> isolates the gain from distillation alone.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

teacher_ckpt = None  # fill with trained teacher checkpoint

model = dict(
    use_quality=False,            # mechanism OFF
    teacher_ckpt=teacher_ckpt,
    distill_loss_weight=1.0,      # feature distillation ON
    output_distill_weight=1.0,    # output distillation ON
)
