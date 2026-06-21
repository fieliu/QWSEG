# EoMT RGB-T F1: FROZEN backbone + ATTENTION-BIAS ONLY quality mechanism.
# Backbone frozen; quality predictor + attention bias trained, but NO cross-modal
# compensation and NO distillation. F1 - F0 isolates the pure increment of the
# attention-bias mechanism (does suppressing low-quality key tokens alone improve
# missing-modality robustness, without touching the backbone?).
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    use_quality=True,
    use_self_attn_bias=True,      # attention bias ON
    use_compensation=False,       # cross-modal compensation OFF
    teacher_cfg=None,             # no distillation
    teacher_ckpt=None,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,      # keep quality supervision (rank + clean floor)
)
