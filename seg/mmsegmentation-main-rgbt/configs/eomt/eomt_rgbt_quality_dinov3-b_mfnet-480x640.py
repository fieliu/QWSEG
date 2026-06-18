# EoMT RGB-T quality-aware (Paradigm One: internal degradation + masked-BCE).
# Deliverable 3 on MFNet.
_base_ = ['./eomt_dinov3-b_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.eomt_rgbt_quality',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

model = dict(
    type='EoMTRGBTQuality',
    num_fusion_points=3,
    fusion_heads=8,
    quality_loss_weight=1.0,
    fuse_tau=0.5,
    bias_alpha=4.0,              # convex attention-bias magnitude
    bias_gamma=3.0,             # convexity: concentrate penalty on low quality
    use_self_attn_bias=True,    # inject quality bias into modality-internal self-attn
    use_compensation=True,      # cross-modal compensation before merge
    degradation=dict(
        # MISSING-only: clean 20% / whole-modality missing 40% / local missing 40%
        kinds=('missing', 'local_missing'),
        kind_probs=(0.5, 0.5),
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)

# EoMTRGBTVisHook dumps per-segment quality maps + merged feature each epoch.
# (current_epoch is also set by it, needed if curriculum=True)
custom_hooks = [
    dict(type='EoMTRGBTVisHook', interval=10, num_samples=1),
    # RGB-missing / T-missing mIoU each val epoch (whole-modality zeroed)
    dict(type='MissingModalityEvalHook', interval=5),
]

