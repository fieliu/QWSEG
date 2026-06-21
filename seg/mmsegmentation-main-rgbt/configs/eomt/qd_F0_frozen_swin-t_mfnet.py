# Swin RGB-T F0: FROZEN-backbone baseline, started from the CLEAN model.
# Adapter paradigm: warm-start from the trained clean teacher (qd_teacher) as
# the STARTING POINT, freeze its backbone, train only fusion + decode head
# (no quality, no distillation). Lower bound for the frozen-backbone study.
# Provide clean ckpt: --cfg-options model.teacher_ckpt=<clean qd_teacher best.pth>
# (teacher only seeds the start; distill weights 0 -> no distillation.)
# DINOv3 mirror: eomt_F0_frozen.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze the (clean-initialized) Swin backbone
    use_quality=False,            # no quality mechanism
    use_compensation=False,
    distill_loss_weight=0.0,      # NO distillation (teacher only seeds the start)
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
)
