# EoMT RGB-T F0: FROZEN-backbone + FROZEN-fusion baseline.
# Adapter paradigm: warm-start from the trained clean fusion baseline (teacher
# weights as the STARTING POINT), freeze backbone + cross-attn fusion modules,
# train ONLY the decode head (no quality, no distillation, no fusion training).
# Lower bound for the frozen-backbone increment study:
#   F0 = freeze backbone + freeze fusion + disable quality -> only decode head
#   F1 = freeze backbone + train fusion + train quality   -> decode head + increments
#   F1-F0 gain = fusion + quality mechanism contribution (pure increment value)
# Provide the clean checkpoint via:
#   --cfg-options model.teacher_ckpt=<clean fusion best.pth>
# (teacher is built only to warm-start the student; distill weights are 0 so
# teacher is released immediately after warm-start to save GPU memory.)
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze DINOv3 backbone
    freeze_fusion=True,           # freeze cross-attn fusion modules (increment part)
    use_quality=False,            # disable quality mechanism (s=1, D=1 everywhere)
    use_compensation=False,
    use_self_attn_bias=False,     # no quality -> no self-attn bias needed
    distill_loss_weight=0.0,      # NO distillation -> teacher released after warm-start
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
    deg_ceiling_weight=0.0,       # no quality supervision at all
)
