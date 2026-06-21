# Swin RGB-T F2: FROZEN backbone + FULL quality mechanism (bias + compensation),
# still NO distillation. F2 - F1 isolates the cross-modal compensation increment
# on a frozen backbone. F0/F1/F2 form the frozen-backbone ablation of the quality
# mechanism's parts. DINOv3 mirror: eomt_F2_frozenfull.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_compensation=True,        # bias + compensation ON
    teacher_cfg=None,             # no distillation (isolate quality from distill)
    teacher_ckpt=None,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
