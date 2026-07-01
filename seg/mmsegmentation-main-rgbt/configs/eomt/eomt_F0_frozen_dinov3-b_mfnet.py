# EoMT RGB-T F0: FROZEN-host + FROZEN-quality (pure data-aug baseline).
# Adapter paradigm: warm-start from clean fusion teacher, freeze ENTIRE host
# (ViT backbone + teacher baseline cross-attn fusions), freeze ALL quality
# modules, train ONLY the decode head. F0 = degraded data as augmentation.
#   F0 = freeze_host + freeze_quality + disable_quality -> only decode head
#   F1 = freeze_host + train_quality_increments         -> head + increments
#   F1-F0 = quality_predictors + quality_fusions + quality-attn (pure increment)
# Provide the clean checkpoint via:
#   --cfg-options model.teacher_ckpt=<clean fusion best.pth>
# (teacher is built only to warm-start the student; distill weights are 0 so
# teacher is released immediately after warm-start to save GPU memory.)
_base_ = ['./eomt_student_dinov3-b_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,         # freeze ViT + teacher baseline fusions
    freeze_fusion=True,           # additionally freeze quality_fusions + quality_predictors
    use_quality=False,            # disable quality mechanism (s=1, D=1 everywhere)
    use_compensation=False,
    use_self_attn_bias=False,     # no quality -> no self-attn bias needed
    distill_loss_weight=0.0,      # NO distillation -> teacher released after warm-start
    output_distill_weight=0.0,
    quality_loss_weight=0.0,
    clean_floor_weight=0.0,
    deg_ceiling_weight=0.0,       # no quality supervision at all
)
