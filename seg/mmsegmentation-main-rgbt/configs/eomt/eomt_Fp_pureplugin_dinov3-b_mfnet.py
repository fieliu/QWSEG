# EoMT RGB-T Fp: PURE-PLUGIN. Host model (backbone + fusion + decode head)
# FULLY frozen from the clean checkpoint; ONLY the quality predictor +
# compensation + attention bias are trained. True LoRA-style robustness plugin.
# Fp - clean-baseline = the quality mechanism's pure increment, with the
# "fusion/head re-trained on degraded data" factor (which F0/F1 mix in)
# completely removed -> cleanly isolates whether the quality mechanism has
# independent value. No distillation (decode head is frozen -> distill is moot).
# Provide clean ckpt: --cfg-options model.teacher_ckpt=<clean fusion best.pth>
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    freeze_neck_head=True,        # also freeze fusion + decode head -> pure plugin
    use_quality=True,
    use_self_attn_bias=True,
    use_compensation=True,
    distill_loss_weight=0.0,      # decode head frozen -> distillation moot
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
