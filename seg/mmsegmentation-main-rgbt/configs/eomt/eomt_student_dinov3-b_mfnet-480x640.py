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
    allow_failed_imports=True)

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
    # Non-uniform fusion points: cover deeper layers for better cross-modal
    # semantic alignment. DINOv3-B has 12 blocks, decode_start=8 (blocks 0-7
    # dual-stream, 8-11 decode). [2,5,7] places fusion at shallow/mid/deep:
    #   block 2: low-level feature alignment (edges/textures differ across modalities)
    #   block 5: mid-level part/shape alignment
    #   block 7: high-level semantic consensus right before merge (no gap to decode)
    # Unlike the old even spacing [2,4,6] which left block 7 unfused, this
    # ensures the last dual-stream block is cross-aligned before merge.
    fusion_points=[2, 5, 7],
    teacher_cfg=teacher_cfg,
    teacher_ckpt=teacher_ckpt,
    quality_loss_weight=1.0,
    distill_loss_weight=0.0,      # 蒸馏禁用:师生同骨干同warm-start起点,无知识落差可蒸,
    output_distill_weight=0.0,    # 且干净teacher对缺失场景目标不可达(实测E1<E0b)。主线=退化+质量机制。
                                  # teacher_cfg保留仅用于warm-start起点;将来上更大teacher可重开蒸馏。
    fuse_tau=0.5,
    mask_temperature=0.1,         # soft mask sharpness: sigmoid((s-0.5)/0.1)
                                  # s=0.99→D=0.99(barely suppressed),
                                  # s=0.01→D=0.01(near-full suppress)
                                  # Multiplicative key-gate: D=0.8 -> weight x0.8
                                  # (true proportional suppression, unlike
                                  # additive bias which softmax amplifies)
    # Endpoint anchoring: pin s to [deg_ceiling, clean_floor] = [0.2, 0.9]
    # so D=sigmoid((s-0.5)/0.1) ranges [0.001, 0.999] (gate effective).
    # Without anchoring, s drifts upward (all ~0.8) and the gate is ineffective.
    clean_floor=0.9,              # upper anchor: clean token s >= 0.9
    clean_floor_weight=0.1,
    deg_ceiling=0.2,              # lower anchor: heavy-degraded token s <= 0.2
    deg_ceiling_weight=0.1,       # re-enabled: anchors lower end for soft mask
    use_quality=True,             # full mechanism ON
    use_self_attn_bias=True,
    use_compensation=True,
    degradation=dict(
        # RGBT-C 标准退化: 直接调用 rgbt_c 库 (13 种退化, 0-5 级).
        # 0 级=干净(权重10), 1-5 级逐渐增加(权重 10/15/20/25/30).
        # 详见 degradation.py make_paired.
        degrade_prob=0.8,
        curriculum=False,
        total_epochs=200,
    ),
)

# ---- Epoch-based schedule: 200 epochs (F0/F1 frozen-backbone training) ----
# MFNet train = 588 iters/epoch (batch 2), 200 epochs = 117600 iters.
# Override the base 100-epoch schedule for the degraded-training student.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=117600,
         by_epoch=False),
]
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=200, val_interval=5)

custom_hooks = [
    dict(type='EoMTRGBTVisHook', interval=10, num_samples=1),
    dict(type='MissingModalityEvalHook', interval=5),
    dict(type='PartialDegradeEvalHook', interval=5, num_samples=50),
]
