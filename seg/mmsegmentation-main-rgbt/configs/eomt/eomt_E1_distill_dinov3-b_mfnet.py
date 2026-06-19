# EoMT RGB-T E1: DISTILLATION ONLY. Degraded training + distillation toward the
# clean teacher, but NO quality mechanism (no bias, no compensation, no quality
# loss). DINOv3 mirror of qd_E1. E1 vs E0b isolates the gain from distillation
# alone; E2 vs E1 isolates the gain from the quality mechanism.
#
# teacher_cfg is inherited from the student base (it captured the clean
# EoMTRGBTFusion model dict). Provide the trained teacher checkpoint via
# --cfg-options teacher_ckpt=<best.pth>.
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

teacher_ckpt = None  # fill via --cfg-options teacher_ckpt=<best.pth>

model = dict(
    use_quality=False,            # mechanism OFF
    teacher_ckpt=teacher_ckpt,
    distill_loss_weight=1.0,      # feature distillation ON
    output_distill_weight=1.0,    # output distillation ON
    quality_loss_weight=0.0,
)
