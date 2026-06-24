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

# 对比学习需要较大 batch 以提供足够跨图负样本 (默认 batch=2 太小)
train_dataloader = dict(batch_size=6)

# batch=6 -> 196 iters/epoch; 200 epoch -> 39200 iters.
# warmup 1500 iters (~7.6 epoch, 对 ViT 微调更稳); PolyLR 衰减到 39200.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=39200,
         by_epoch=False),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)

