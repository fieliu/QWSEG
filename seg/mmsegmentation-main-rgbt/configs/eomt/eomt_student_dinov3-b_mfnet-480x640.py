# EoMT RGB-T E2 (FULL): degraded training + distillation + quality mechanism.
# DINOv3 mirror of qd_student (A-set E2). Requires a trained clean teacher
# checkpoint (the eomt_rgbt_fusion baseline); set teacher_ckpt below or via
# --cfg-options teacher_ckpt=<best.pth>.
_base_ = ['./eomt_rgbt_fusion_dinov3-b_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.eomt_rgbt_quality',
             'mmseg.models.segmentors.eomt_rgbt_fusion',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

# Path to the trained clean teacher checkpoint (fill after training the
# eomt_rgbt_fusion baseline).
teacher_ckpt = None

# Teacher sub-model: the _base_ model dict is the clean EoMTRGBTFusion baseline.
# Capture it verbatim (full backbone / img_size / data_preprocessor / fusion
# config) as the frozen distillation teacher BEFORE the model dict below swaps
# the segmentor type to EoMTRGBTQuality.
teacher_cfg = {{_base_.model}}

model = dict(
    type='EoMTRGBTQuality',
    num_fusion_points=3,
    fusion_heads=8,
    teacher_cfg=teacher_cfg,
    teacher_ckpt=teacher_ckpt,
    quality_loss_weight=1.0,
    distill_loss_weight=1.0,      # feature distillation (merged token seq)
    output_distill_weight=1.0,    # output distillation (gated per-pixel KL)
    fuse_tau=0.5,
    bias_alpha=4.0,
    bias_gamma=3.0,
    use_quality=True,             # full mechanism ON
    use_self_attn_bias=True,
    use_compensation=True,
    degradation=dict(
        # MISSING-only: clean 20% / whole-modality missing 40% / local missing 40%
        kinds=('missing', 'local_missing'),
        kind_probs=(0.5, 0.5),
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)

custom_hooks = [
    dict(type='EoMTRGBTVisHook', interval=10, num_samples=1),
    dict(type='MissingModalityEvalHook', interval=5),
]
