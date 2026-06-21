# Swin RGB-T F1: FROZEN backbone + ATTENTION-BIAS ONLY quality mechanism.
# Backbone frozen; quality predictor + attention bias trained, NO compensation,
# NO distillation. F1 - F0 isolates the pure increment of the attention-bias
# mechanism on a frozen backbone. DINOv3 mirror: eomt_F1_frozenbias.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_compensation=False,       # compensation OFF, bias ON
    teacher_cfg=None,             # no distillation
    teacher_ckpt=None,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,      # keep quality supervision (rank + clean floor)
)
