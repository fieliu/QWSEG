# Student: same arch + quality/compensation, trained on DEGRADED input with
# quality-gated distillation toward the FROZEN clean teacher.
# Requires a trained teacher checkpoint (set teacher_ckpt below).
_base_ = ['./qd_teacher_swin-t_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.quality_distill_student',
             'mmseg.models.segmentors.quality_distill_teacher',
             'mmseg.models.segmentors.swinmul_v6_mask2former',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

# Path to the trained teacher checkpoint (fill after training the teacher).
teacher_ckpt = None

# Teacher sub-model: the inherited model dict is ALREADY the QualityDistillTeacher
# (from qd_teacher config). Reuse it verbatim as the frozen teacher.
teacher_cfg = {{_base_.model}}

model = dict(
    type='QualityDistillStudent',
    fusion_dims=(96, 192, 384, 768),
    teacher_cfg=teacher_cfg,
    teacher_ckpt=teacher_ckpt,
    quality_loss_weight=1.0,
    distill_loss_weight=1.0,      # feature distillation (per-stage fused feats)
    output_distill_weight=1.0,    # output distillation (gated per-pixel KL)
    bias_alpha=4.0,               # max attention-bias magnitude (convex)
    bias_gamma=3.0,               # convexity: concentrate penalty on low quality
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
    dict(type='TrainVisHook', interval=5, num_samples=2),
    # RGB-missing / T-missing mIoU at every validation (whole-modality zeroed)
    dict(type='MissingModalityEvalHook', interval=1),
]
