# EoMT RGB-T fusion baseline + 跨模态对比学习 (CLIP 风格双向 InfoNCE).
# 在最后一个融合点 (block 7) 融合前的双流特征上做对比, 让 T 向 RGB 对齐,
# 缓解 RGB 缺失时性能下降. 用于验证对比学习是否提升 teacher 质量.
_base_ = ['./eomt_rgbt_fusion_dinov3-b_mfnet-480x640.py']

model = dict(
    # 融合点与 student (eomt_F1_frozenquality) 一致, 保证 warm-start 对齐
    fusion_points=[2, 5, 7],
    # ---- 跨模态对比学习 ----
    use_contrast=True,
    contrast_weight=0.1,        # 辅助损失, 不主导分割损失
    contrast_layer=7,           # 最后一个融合点 (深层, 语义最强)
    contrast_samples=15,        # 每类采样 15 个 anchor
    contrast_tau=0.1,           # 温度
)
