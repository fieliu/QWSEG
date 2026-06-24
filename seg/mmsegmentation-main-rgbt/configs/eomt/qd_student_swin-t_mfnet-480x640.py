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
    distill_loss_weight=0.0,      # 蒸馏禁用:师生同骨干同warm-start无落差+干净teacher对缺失目标不可达(实测E1<E0b)
    output_distill_weight=0.0,    # 主线=退化+质量机制。teacher_cfg留作warm-start起点,将来大teacher可重开
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

# Distillation adds dense output-level gradients that dominate the total
# gradient norm under the standard Mask2Former clip_grad(max_norm=0.01),
# starving the sparse seg gradients and collapsing query specialization.
# Override to max_norm=10 (same as DINOv3 / standard Mask2Former large-backbone
# configs) so both seg and distill gradients can flow through effectively.
optim_wrapper = dict(clip_grad=dict(max_norm=10, norm_type=2))

# ---- Epoch-based schedule: 200 epochs (F0/F1 frozen-backbone training) ----
# MFNet train = 588 iters/epoch (batch 2), 200 epochs = 117600 iters.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=117600,
         by_epoch=False),
]
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=200, val_interval=5)

custom_hooks = [
    dict(type='TrainVisHook', interval=5, num_samples=2),
    # RGB-missing / T-missing mIoU at every validation (whole-modality zeroed)
    dict(type='MissingModalityEvalHook', interval=1),
]
