# EoMT RGB-T fusion baseline (shared ViT + cross-attention, NO quality).
# Deliverable 2: clean-scene multimodal baseline on MFNet.
_base_ = ['./eomt_dinov3-b_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.eomt_rgbt_fusion',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

model = dict(
    type='EoMTRGBTFusion',
    num_fusion_points=3,
    fusion_heads=8,
)

# RGB-missing / T-missing mIoU each val epoch (whole-modality zeroed)
custom_hooks = [dict(type='MissingModalityEvalHook', interval=5)]
