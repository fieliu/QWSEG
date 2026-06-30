# DINOv3-Adapter + M2F Quality-aware RGB-T on MFNet (FULL version).
# Inherits clean baseline (Mask2FormerRGBTCrossAttn), adds quality modules
# as INCREMENTAL components BEFORE the baseline fusion.
#
# Architecture flow:
#   Backbone -> [Quality Repair: predict + quality-gated cross-attn] -> [Baseline Fusion: CrossAttn + ChannelSpatialAttn] -> M2F
#
# Quality increment = quality_predictors + quality_cross_fusions (NEW parameters)
# Baseline fusion = cross_fusions + cs_fusions (from clean checkpoint)
#
# Ablation: this is the FULL quality model; E0b/F0/F1a/F1b/F1 override switches.
_base_ = ['./adapter_m2f_rgbt_mfnet.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.dinov3_adapter_m2f_quality',
             'mmseg.models.backbones.dinov3_adapter',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

# Swap segmentor type to quality-aware version
model = dict(
    type='DINOv3AdapterM2FQuality',
    use_quality_gate=True,        # quality-gated cross-attention (suppress low-q keys)
    use_quality_merge=True,       # quality-weighted residual merge
    quality_loss_weight=1.0,
    quality_tau=0.5,
    mask_temperature=0.1,
    deg_ceiling_weight=0.1,
    deg_ceiling=0.2,
    freeze_backbone=False,
    freeze_fusion=False,
    degradation=dict(
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
    teacher_ckpt=None,  # set via --cfg-options model.teacher_ckpt=<clean best.pth>
)

# Quality visualization hook
custom_hooks = [dict(type='AdapterM2FQualityVisHook', interval=10, num_samples=1)]
