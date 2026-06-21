# Swin RGB-T F0: FROZEN-backbone baseline. Original Swin backbone frozen; only
# fusion + decode head trained (no quality, no distillation). Lower bound for
# the frozen-backbone increment study. DINOv3 mirror: eomt_F0_frozen.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze original Swin backbone
    use_quality=False,            # no quality mechanism
    use_compensation=False,
    teacher_cfg=None,             # no distillation
    teacher_ckpt=None,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
)
