# EoMT RGB-T F0: FROZEN-backbone baseline. Original ViT backbone frozen; only
# fusion + decode head trained (no quality, no distillation). Lower bound for
# the frozen-backbone increment study. F1/F2 must beat THIS to prove the
# quality mechanism adds value WITHOUT touching the backbone.
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze original DINOv3 ViT
    use_quality=False,            # no quality mechanism
    teacher_cfg=None,             # no distillation
    teacher_ckpt=None,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
)
