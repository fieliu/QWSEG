# EoMT RGB-T F0: FROZEN-backbone baseline, started from the CLEAN model.
# Adapter paradigm: warm-start from the trained clean fusion baseline (teacher
# weights as the STARTING POINT), freeze its backbone, train only fusion +
# decode head (no quality, no distillation). Lower bound for the frozen-backbone
# increment study. Provide the clean checkpoint via:
#   --cfg-options model.teacher_ckpt=<clean fusion best.pth>
# (teacher is built only to warm-start the student start point; distill weights
# are 0 so it does not distill. It stays resident but contributes no loss.)
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze the (clean-initialized) DINOv3 backbone
    use_quality=False,            # no quality mechanism
    use_compensation=False,
    distill_loss_weight=0.0,      # NO distillation (teacher only seeds the start)
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
)
