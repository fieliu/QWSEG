_base_ = ['./eomt_F1_frozenquality_dinov3-b_mfnet.py', '../_smoke_test_.py']
model = dict(freeze_backbone=True, freeze_fusion=False,
    teacher_cfg=None, teacher_ckpt=None, init_from_teacher=False,
    distill_loss_weight=0.0, output_distill_weight=0.0, quality_loss_weight=1.0)

# Remove heavy hooks for smoke
custom_hooks = []
