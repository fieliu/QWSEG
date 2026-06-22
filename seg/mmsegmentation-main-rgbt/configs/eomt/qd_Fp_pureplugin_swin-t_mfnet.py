# Swin RGB-T Fp: PURE-PLUGIN. Host model (backbone + fuse_convs + neck +
# decode head) FULLY frozen from the clean qd_teacher; ONLY the quality
# predictor + compensation trained. True LoRA-style robustness plugin.
# Fp - clean-baseline = quality mechanism's pure increment (the "fusion/head
# re-trained on degraded data" factor removed). No distillation (head frozen).
# Provide clean ckpt: --cfg-options model.teacher_ckpt=<clean qd_teacher best.pth>
# DINOv3 mirror: eomt_Fp_pureplugin.
_base_ = ['./qd_student_swin-t_mfnet-480x640.py']

model = dict(
    freeze_backbone=True,
    freeze_neck_head=True,
    use_quality=True,
    use_compensation=True,
    distill_loss_weight=0.0,
    output_distill_weight=0.0,
    quality_loss_weight=1.0,
)
