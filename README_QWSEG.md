# RGB-T 语义分割模型版本说明

## 项目概述

本项目基于 MMSegmentation 框架，实现多版本 RGB-Thermal 双模态语义分割模型。所有版本共享 M-SpecGene (MAE ViT-B) 预训练编码器，通过不同的融合策略、训练增强和辅助模块逐步提升模型鲁棒性。

## 版本总览

| 版本 | 模型类 | 核心特点 | 训练数据 | 训练Epoch | 关键创新 |
|------|--------|---------|---------|----------|---------|
| V1 Baseline | `RGBTv1Baseline` | 共享编码器 + LoRA + 通道拼接 | 干净数据 | 200 | LoRA 微调 + Focal+Dice |
| V1 Baseline (FMB-Init) | `RGBTv1Baseline` | FMB权重完全初始化 + LoRA | 干净数据 | 200 | FMB权重初始化backbone+neck+decoder |
| V2 Disentangle | `RGBTv2Disentangle` | 通用/私有特征解耦 + 跨模态注意力融合 | 干净数据 | 200 | HSIC解耦 + InfoNCE模态对齐 |
| V3 Degradation | `RGBTv3Degradation` | V2架构 + 在线退化增强 | 退化增强数据 | 300 | RGBTModalDegradation管道 |
| V4 QualityPruning | `RGBTv4QualityPruning` | V3架构 + 质量网络(冻结) | 退化增强数据 | 300 | 质量感知融合 + 质量锚定损失 |
| V5 QualityJoint | `RGBTv5QualityJoint` | V3架构 + 质量网络V2双分支(联合微调) | 退化增强数据 | 300 | 渐进式剪枝 + ListMLE + 质量引导对齐 |
| 原始版 | `EncoderDecoder` | 标准UPerNet + 全参数微调 | 干净数据 | 320k iter | LayerDecay + ClassBalance |

## 配置文件矩阵

| 版本 | FMB 768×768 | FMB 480×480 | MFNet 768×768 | MFNet 480×480 |
|------|-------------|-------------|---------------|---------------|
| V1 | `*_v1_*_200e_fmb-768x768` | `*_v1_*_200e_fmb-480x480` | `*_v1_*_200e_mfnet-768x768` | `*_v1_*_200e_mfnet-480x480` |
| V1 FMB-Init | — | `*_v1_baseline_fmb-init*_fmb-480x480` | — | — |
| V2 | `*_v2_*_200e_fmb-768x768` | `*_v2_*_200e_fmb-480x480` | `*_v2_*_200e_mfnet-768x768` | `*_v2_*_200e_mfnet-480x480` |
| V3 | `*_v3_*_300e_fmb-768x768` | `*_v3_*_300e_fmb-480x480` | `*_v3_*_300e_mfnet-768x768` | `*_v3_*_300e_mfnet-480x480` |
| V4 | `*_v4_*_300e_fmb-768x768` | `*_v4_*_300e_fmb-480x480` | `*_v4_*_300e_mfnet-768x768` | `*_v4_*_300e_mfnet-480x480` |
| V5 | `*_v5_*_300e_fmb-768x768` | `*_v5_*_300e_fmb-480x480` | `*_v5_*_300e_mfnet-768x768` | `*_v5_*_300e_mfnet-480x480` |

> 配置文件位于 `configs/mae/` 目录下，完整文件名前缀为 `mae-base_upernet_rgbt_`

---

## 常用训练与测试参数

### GPU 选择

通过环境变量 `CUDA_VISIBLE_DEVICES` 指定使用的 GPU：

```bash
# 使用单块 GPU（如 GPU 0）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/xxx.py

# 使用 GPU 2
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/mae/xxx.py

# 使用多块 GPU（如 GPU 0 和 GPU 1），需配合 --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mae/xxx.py 2

# 测试时指定 GPU
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/xxx.py work_dirs/xxx/best_mIoU.pth
```

> 注意：配置文件中的 `8xb2` 表示 8 GPU × batch_size=2。若实际 GPU 数量不同，需通过 `--cfg-options` 调整，如单卡训练时 `--cfg-options train_dataloader.batch_size=16` 以保持等效总 batch size。

### 训练参数 (`tools/train.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | 位置参数 | — | 配置文件路径 |
| `--work-dir` | str | 配置文件中指定 | 日志和模型保存目录 |
| `--resume` | flag | False | 从 work_dir 中最新 checkpoint 自动恢复训练 |
| `--amp` | flag | False | 启用自动混合精度训练（配置已默认开启则无需指定） |
| `--visualize` | flag | False | 启用训练可视化 Hook（TrainVisHook） |
| `--vis-interval` | int | 20 | 可视化触发间隔（epoch 数） |
| `--vis-num-samples` | int | 2 | 每次可视化采样的样本数 |
| `--vis-version` | str | auto | 可视化目录版本名（默认从模型类型自动推断） |
| `--cfg-options` | key=value | — | 覆盖配置项，如 `--cfg-options train_dataloader.batch_size=4` |
| `--launcher` | {none,pytorch,slurm,mpi} | none | 分布式启动方式 |

**训练示例**：

```bash
# 单卡训练 + 可视化
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/xxx.py \
    --visualize --vis-interval 10 --vis-num-samples 4

# 单卡训练 + 恢复训练
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mae/xxx.py --resume

# 单卡训练 + 覆盖配置
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/xxx.py \
    --cfg-options train_dataloader.batch_size=4 \
                  train_cfg.max_epochs=400

# 多卡训练（2 GPU）
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mae/xxx.py 2
```

### 测试参数 (`tools/test.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | 位置参数 | — | 配置文件路径 |
| `checkpoint` | 位置参数 | — | 模型权重文件路径 |
| `--work-dir` | str | — | 评估结果保存目录 |
| `--out` | str | — | 预测结果保存目录（离线评估） |
| `--show` | flag | False | 显示预测结果 |
| `--show-dir` | str | — | 可视化结果保存目录 |
| `--tta` | flag | False | 启用测试时增强 |
| `--cfg-options` | key=value | — | 覆盖配置项 |

### 鲁棒性测试参数 (`tools/test_robustness.py`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config` | 位置参数 | — | 配置文件路径 |
| `checkpoint` | 位置参数 | — | 模型权重文件路径 |
| `--work-dir` | str | eval/robustness/\<timestamp\> | 结果保存目录 |
| `--degradation` | str | CleanDegradation | 退化类名（单场景测试时使用） |
| `--deg-kwargs` | key=value | — | 退化类参数，如 `missing_ratio=0.5` |
| `--batch-all` | flag | False | 批量运行全部 9 种鲁棒性场景 |
| `--visualize` | flag | False | 启用 TensorBoard 可视化 |
| `--num-vis-samples` | int | 4 | 可视化采样数 |
| `--version` | str | original | 模型版本名（用于输出目录） |
| `--seed` | int | 42 | 随机种子 |

**鲁棒性测试示例**：

```bash
# 批量测试全部场景（推荐）
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/xxx.py \
    work_dirs/xxx/best_mIoU.pth --batch-all --version v4

# 单场景测试（如仅测试高斯噪声中度退化）
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/xxx.py \+
    work_dirs/xxx/best_mIoU.pth \
    --degradation GlobalDegradation \
    --deg-kwargs deg_type=gaussian_noise intensity=medium
```

### 训练可视化

训练可视化通过 `--visualize` 参数启用，使用 `TrainVisHook` 在训练过程中定期保存分割结果对比图：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/xxx.py \
    --visualize --vis-interval 10 --vis-num-samples 4 --vis-version v4
```

- 可视化结果保存在 `work_dirs/<config_name>/vis_data/vis_image/` 目录下
- `--vis-interval`：每隔多少 epoch 保存一次可视化
- `--vis-num-samples`：每次保存多少个样本的可视化
- `--vis-version`：可视化子目录名称，默认从模型类名自动推断

### 退化效果可视化 (Jupyter Notebook)

使用 `tools/robustness_visualization.ipynb` 可视化各种退化效果：

```bash
# 在 Jupyter 中打开
jupyter notebook tools/robustness_visualization.ipynb
```

Notebook 包含以下可视化内容：
1. **全局退化可视化**：运动模糊、高斯噪声、椒盐噪声、条纹噪声、热对比度降低（低/中/高三种强度）
2. **局部退化可视化**：局部过曝、局部过暗、水滴、污渍、传感器坏块、局部高斯噪声、热晕现象（低/中/高三种强度）
3. **4 张图像综合退化效果**：随机全局/局部退化
4. **模态缺失可视化**：RGB 缺失 / T 缺失
5. **训练增强策略可视化**：展示 RGBTModalDegradation 的 30%/30%/20%/20% 概率分布效果

> 修改 Notebook 第一个代码单元格中的 `DATASET` 变量可切换 FMB / MFNet 数据集。

---

## V1 Baseline

### 模型架构

```
RGB (3ch) ──┐
             ├── [共享 MAE ViT-B 编码器 (LoRA)] ──→ 通道拼接 ──→ UPerNet ──→ 分割输出
Thermal (3ch)┘
```

- **编码器**：M-SpecGene (MAE ViT-B)，RGB 和 Thermal 共享权重，分别前向传播
- **融合方式**：通道拼接 — RGB 特征 (768d) + Thermal 特征 (768d) → 拼接为 1536d
- **Neck**：Feature2Pyramid，将 1536d 特征缩放为多尺度金字塔
- **分割头**：UPerNet (主) + FCN (辅助)
- **微调方式**：LoRA (rank=4, alpha=4.0)，冻结 backbone，仅训练 LoRA 参数 + 解码器

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| FocalLoss (decode) | 1.0 | 主分割头，γ=2.0, α=0.25 |
| DiceLoss (decode) | 1.0 | 主分割头，处理类别不平衡 |
| FocalLoss (aux) | 0.4 | 辅助分割头 |
| DiceLoss (aux) | 0.4 | 辅助分割头 |

### 与原版区别

- 使用 LoRA 微调而非全参数微调，大幅减少可训练参数
- 使用 FocalLoss + DiceLoss 替代 CrossEntropyLoss
- 双模态通过共享编码器 + 通道拼接融合，而非单模态输入
- 支持 modality_dropout（训练时随机丢弃一个模态）

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py` | FMB 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-480x480.py` | FMB 480×480 配置 |
| `configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_mfnet-768x768.py` | MFNet 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_mfnet-480x480.py` | MFNet 480×480 配置 |
| `configs/_base_/datasets/fmb_768x768.py` | FMB 数据集配置 (768) |
| `configs/_base_/datasets/fmb_480x480.py` | FMB 数据集配置 (480) |
| `configs/_base_/datasets/mfnet_768x768.py` | MFNet 数据集配置 (768) |
| `configs/_base_/datasets/mfnet_480x480.py` | MFNet 数据集配置 (480) |
| `mmseg/models/segmentors/rgbt_v1_baseline.py` | 模型实现 |
| `mmseg/models/utils/lora.py` | LoRA 实现 |
| `mmseg/models/backbones/mae.py` | MAE 编码器 |

### 训练与评估

```bash
# 训练 (FMB 768x768)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py

# 训练 (FMB 480x480)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-480x480.py

# 训练 (MFNet 768x768)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_mfnet-768x768.py

# 训练 (MFNet 480x480)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_mfnet-480x480.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v1_baseline/best_mIoU_iter_xxx.pth

# 鲁棒性测试
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v1_baseline/best_mIoU_iter_xxx.pth --batch-all
```

---

## V1 Baseline (FMB-Init)

### 概述

V1 Baseline (FMB-Init) 使用 FMB 训练权重 (`FMB_save_iter_224000.pth`) 完全初始化 backbone、neck、decode_head 和 auxiliary_head，而非仅使用 MAE 预训练权重初始化 backbone。这提供了更好的初始化起点，有望加速收敛并提升性能。

### 与 V1 Baseline 区别

| 特性 | V1 Baseline | V1 Baseline (FMB-Init) |
|------|-------------|----------------------|
| Backbone 初始化 | MAE 预训练权重 | FMB 训练权重 |
| Neck 初始化 | 默认随机初始化 | FMB 训练权重 |
| Decode Head 初始化 | 默认随机初始化 | FMB 训练权重 |
| Auxiliary Head 初始化 | 默认随机初始化 | FMB 训练权重 |
| 微调方式 | LoRA (rank=4) | LoRA (rank=16) |
| LoRA 目标层 | qkv, proj | qkv, proj, fc1, fc2 |

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v1_baseline_fmb-init_8xb2-amp-200e_fmb-480x480.py` | FMB 480×480 配置 |
| `mmseg/models/segmentors/rgbt_v1_baseline.py` | 模型实现 (含自定义 init_weights) |
| `pretrain/FMB_save_iter_224000.pth` | FMB 训练权重 |

### 训练与评估

```bash
# 训练 (FMB 480x480)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_fmb-init_8xb2-amp-200e_fmb-480x480.py --amp

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_rgbt_v1_baseline_fmb-init_8xb2-amp-200e_fmb-480x480.py \
    work_dirs/mae-base_upernet_rgbt_v1_baseline_fmb-init/<timestamp>/weight/best_mIoU_*.pth
```

---

## V2 Disentangle

### 模型架构

```
RGB (3ch) ──┬── [共享 MAE ViT-B (通用编码器)] ──→ zc_rgb ──┐
             │                                              │
             └── [LightweightMAEBranch (私有编码器)] ──→ zp_rgb ─┤
                                                            ├── 跨模态注意力融合 ──→ UPerNet ──→ 分割输出
Thermal (3ch)──┬── [共享 MAE ViT-B (通用编码器)] ──→ zc_t ──┤
               │                                           │
               └── [LightweightMAEBranch (私有编码器)] ──→ zp_t ─┘
```

- **通用编码器**：M-SpecGene (MAE ViT-B)，RGB 和 Thermal 共享权重，提取通用特征 zc
- **私有编码器**：LightweightMAEBranch (ViT-Tiny 结构, embed_dims=192)，分别提取 RGB/Thermal 私有特征 zp
- **融合方式**：跨模态注意力融合 (CrossAttentionFusion)
  1. zc_rgb + zc_t → zc_sum (逐元素相加)
  2. CrossAttn(q=zc_sum, k=zp_rgb, v=zp_rgb) → fused_rgb
  3. CrossAttn(q=fused_rgb, k=zp_t, v=zp_t) → fused_final
- **zc 辅助分割头**：FCNHead 对通用特征 zc 进行辅助监督
- **优化器**：AdamW，universal_backbone 学习率 ×0.1

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| FocalLoss (decode) | 1.0 | 主分割头 |
| DiceLoss (decode) | 1.0 | 主分割头 |
| FocalLoss (aux) | 0.4 | 辅助分割头 |
| DiceLoss (aux) | 0.4 | 辅助分割头 |
| loss_seg_zc | 0.3 | zc 通用特征辅助分割 (CE Loss) |
| loss_disentangle_s0~s3 | 0.1/0.2/0.3/0.4 | HSIC 解耦损失 (zc ⊥ zp) |
| loss_modal_s0~s3 | 0.2 | InfoNCE 模态对齐损失 |
| loss_invariance | 0.1 | MSE 不变性损失 (zc_rgb ↔ zc_t) |

### 与 V1 区别

- 新增私有编码器 LightweightMAEBranch，分离通用/私有特征
- 融合方式从通道拼接改为跨模态注意力融合
- 新增 HSIC 解耦损失确保通用/私有特征正交
- 新增 InfoNCE 模态对齐损失确保通用特征的跨模态一致性
- 新增 MSE 不变性损失
- 新增 zc 辅助分割头监督通用特征
- 通用编码器使用低学习率 (×0.1) 而非 LoRA

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-768x768.py` | FMB 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-480x480.py` | FMB 480×480 配置 |
| `configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_mfnet-768x768.py` | MFNet 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_mfnet-480x480.py` | MFNet 480×480 配置 |
| `mmseg/models/segmentors/rgbt_v2_disentangle.py` | 模型实现 (含 HSIC, InfoNCE, CrossAttentionFusion) |
| `mmseg/models/backbones/lightweight_mae_branch.py` | 私有编码器实现 |
| `mmseg/models/backbones/mae.py` | 通用编码器 |

### 训练与评估

```bash
# 训练 (FMB 768x768)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-768x768.py

# 训练 (MFNet 768x768)
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_mfnet-768x768.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v2_disentangle/best_mIoU_iter_xxx.pth

# 鲁棒性测试
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v2_disentangle/best_mIoU_iter_xxx.pth --batch-all
```

---

## V3 Degradation

### 模型架构

与 V2 完全相同的模型架构，区别仅在于训练数据增强管道和训练 epoch 数。

### 在线退化增强 (RGBTModalDegradation)

训练时对每个样本随机应用以下策略之一，退化类型与鲁棒性测试完全一致：

| 策略 | 概率 | 说明 |
|------|------|------|
| 干净数据 | 30% | 不施加任何退化 |
| 单模态缺失 | 30% | RGB 或 Thermal 完全置零 |
| 全局退化 | 20% | 随机选择退化类型 + 随机强度 |
| 局部退化 | 20% | 随机选择退化类型 + 随机强度 |

**全局退化类型**（随机选择一种，随机选择低/中/高强度）：

| 退化类型 | 适用模态 | 同步性 |
|---------|---------|--------|
| 运动模糊 (`motion_blur`) | RGB + T | 必须同步（同一模糊核） |
| 高斯噪声 (`gaussian_noise`) | RGB / T | 独立应用 |
| 椒盐噪声 (`salt_pepper`) | RGB / T | 独立应用 |
| 条纹噪声 (`stripe_noise`) | 仅 T | 仅热红外 |
| 热对比度降低 (`thermal_contrast`) | 仅 T | 仅热红外 |

**局部退化类型**（随机选择一种，随机选择低/中/高强度）：

| 退化类型 | 适用模态 | 区域形状 | 同步性 |
|---------|---------|---------|--------|
| 局部过曝 (`local_overexposure`) | 仅 RGB | 随机矩形 | 仅 RGB |
| 局部过暗 (`local_lowlight`) | 仅 RGB | 随机矩形 | 仅 RGB |
| 水滴 (`water_stain`) | RGB + T | Perlin噪声 | 必须同步 |
| 污渍 (`stain`) | RGB / T | Perlin噪声 | 独立应用 |
| 传感器坏块 (`local_bad_block`) | RGB / T | 随机矩形 | 独立应用 |
| 局部高斯噪声 (`local_gaussian_noise`) | RGB / T | 随机矩形 | 独立应用 |
| 热晕现象 (`thermal_halo`) | 仅 T | 径向高斯 | 仅热红外 |

**模态选择规则**：
- 必须同步的类型：RGB 和 T 同时退化
- 仅特定模态的类型：只退化对应模态
- 双模态类型：0.9 概率随机选择单模态退化，0.1 概率双模态同时退化

> 详细的退化参数（强度等级、具体数值等）请参见 `tools/robustness_README.md`

### 损失函数

与 V2 完全相同。

### 与 V2 区别

- 训练时使用 RGBTModalDegradation 在线退化增强（与鲁棒性测试一致的退化类型）
- 训练 epoch 从 200 增加到 300
- 模型架构和损失函数不变
- 模型类为 `RGBTv3Degradation`（继承 V2 的特征提取逻辑）

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_fmb-768x768.py` | FMB 768×768 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_fmb-480x480.py` | FMB 480×480 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_mfnet-768x768.py` | MFNet 768×768 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_mfnet-480x480.py` | MFNet 480×480 配置 (继承 V2) |
| `mmseg/models/segmentors/rgbt_v3_degradation.py` | 模型实现 |
| `mmseg/datasets/transforms/rgbt_augmentation.py` | 退化增强管道 (RGBTModalDegradation) |
| `mmseg/datasets/transforms/robustness_degradation.py` | 鲁棒性退化参数和函数 (被 RGBTModalDegradation 复用) |

### 训练与评估

```bash
# 训练 (FMB 768x768)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_fmb-768x768.py

# 训练 (MFNet 480x480)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_mfnet-480x480.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v3_degradation/best_mIoU_iter_xxx.pth

# 鲁棒性测试
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v3_degradation/best_mIoU_iter_xxx.pth --batch-all
```

---

## V4 QualityPruning

### 模型架构

在 V3 基础上加入质量网络 (QualityNetwork)。

```
RGB (3ch) ──┬── [通用编码器] ──→ zc_rgb ──→ QualityNetwork ──→ q_rgb (质量分数)
             │                                              │
             └── [私有编码器] ──→ zp_rgb ──────────────────┤
                                                        │
                    质量感知融合 ←────────────────────────┘
                        │
Thermal (3ch)──┬── [通用编码器] ──→ zc_t ──→ QualityNetwork ──→ q_t (质量分数)
               │                                           │
               └── [私有编码器] ──→ zp_t ──────────────────┘
```

- **质量网络** (QualityNetwork)：双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
  - Token 投影：Linear(768→128)
  - 多尺度局部差异感知：LocalDifferenceModule (3个尺度的平均池化差异)
  - 评分头：LN(256)→Linear(256,128)→GELU→Linear(128,1)→Sigmoid
  - RGB和Thermal各有独立分支参数
- **质量感知融合**：使用质量分数加权模态对齐损失，高质量模态获得更大对齐权重

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| V2 全部损失 | 同 V2 | 分割 + 解耦 + 模态对齐 + 不变性 |
| loss_anchor | 0.5 | 质量锚定损失 (clean→0.8, deg→0.15, MSE) |
| loss_quality_ceiling | - | 高退化天花板损失 (最高退化质量<0.15) |
| loss_spatial_div | - | 空间多样性损失 (鼓励非均匀质量分布) |
| loss_quality_reg | 0.1 | 质量正则化 (质量分数均值约束) |

**注意**：V4 中 `loss_modal` 使用质量加权对齐替代 V2/V3 的 InfoNCE。

**边界情况处理**：当 batch 中仅包含干净样本或仅包含退化样本时，损失计算会自动调整：
- 仅有干净样本：只计算 `loss_anchor` 的 clean 部分，跳过 `loss_quality_ceiling`
- 仅有退化样本：只计算 `loss_quality_ceiling`，跳过 `loss_anchor` 的 clean 部分
- 两者都有：计算全部损失

### 与 V3 区别

- 新增 QualityNetwork 为每个 token 预测质量分数
- 模态对齐损失改为质量加权版本
- 新增质量锚定损失 (loss_anchor)、高退化天花板损失 (loss_quality_ceiling)、空间多样性损失 (loss_spatial_div) 和质量正则化 (loss_quality_reg)
- 训练时同样使用 RGBTModalDegradation 退化增强
- PackSegInputs 需要传递 `rgb_degradation` 和 `thermal_degradation` meta_keys

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_fmb-768x768.py` | FMB 768×768 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_fmb-480x480.py` | FMB 480×480 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_mfnet-768x768.py` | MFNet 768×768 配置 (继承 V2) |
| `configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_mfnet-480x480.py` | MFNet 480×480 配置 (继承 V2) |
| `mmseg/models/segmentors/rgbt_v4_quality_pruning.py` | 模型实现 |
| `mmseg/models/utils/quality_network.py` | 质量网络实现 (含 QualityAnchorLoss, LevelMarginLoss, SpatialDiversityLoss, HighDegCeilingLoss, LocalDifferenceModule) |
| `mmseg/datasets/transforms/rgbt_augmentation.py` | 退化增强管道 (RGBTModalDegradation) |

### 训练与评估

```bash
# 训练 (FMB 768x768)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_fmb-768x768.py

# 训练 (MFNet 480x480)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_mfnet-480x480.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v4_quality_pruning/best_mIoU_iter_xxx.pth

# 鲁棒性测试
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py configs/mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v4_quality_pruning/best_mIoU_iter_xxx.pth --batch-all
```

---

## 原始版 (EncoderDecoder)

### 模型架构

标准 MMSegmentation EncoderDecoder 结构，使用全参数微调。

- **编码器**：MAE ViT-B (单模态输入，in_channels=3)
- **Neck**：Feature2Pyramid (embed_dim=768)
- **分割头**：UPerNet + FCN (辅助)
- **优化器**：AdamW + LayerDecayOptimizerConstructor (layer_decay=0.65)
- **测试模式**：slide inference (crop=768, stride=341)

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| CrossEntropyLoss (decode) | 1.0 | 主分割头 |
| CrossEntropyLoss (aux) | 0.4 | 辅助分割头 |

### 数据集变体

| 配置文件 | 数据集 | 类别数 |
|---------|--------|-------|
| `mae-base_upernet_8xb2-amp-320k_fmb-768x768.py` | FMB | 15 |
| `mae-base_upernet_8xb2-amp-320k_mfnet-768x768.py` | MFNet | 9 |
| `mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py` | ADE20K | 26 |

### 与 V1 区别

- 使用标准 EncoderDecoder 而非 RGBT 专用架构
- 单模态输入 (in_channels=3)，非双模态
- 全参数微调 + LayerDecay，而非 LoRA
- CrossEntropyLoss 而非 FocalLoss + DiceLoss
- slide inference 测试模式

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/_base_/models/upernet_mae_classbalance.py` | 基础模型配置 |
| `configs/_base_/datasets/fmb_768x768.py` | FMB 数据集 |
| `configs/_base_/datasets/mfnet_768x768.py` | MFNet 数据集 |
| `configs/_base_/datasets/ade20k_768x768.py` | ADE20K 数据集 |
| `configs/_base_/schedules/schedule_320k.py` | 训练策略 (320k iterations) |

### 训练与评估

```bash
# FMB 训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_8xb2-amp-320k_fmb-768x768.py

# MFNet 训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_8xb2-amp-320k_mfnet-768x768.py

# ADE20K 训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mae/mae-base_upernet_8xb2-amp-320k_ade20k-768x768.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mae/mae-base_upernet_8xb2-amp-320k_fmb-768x768.py \
    work_dirs/mae-base_upernet_8xb2-amp-320k_fmb-768x768/best_mIoU_iter_xxx.pth
```

---

## 版本演进对比

```
原始版 (EncoderDecoder)
  │  单模态, 全参数微调, CE Loss
  │
  ▼
V1 Baseline (RGBTv1Baseline)
  │  双模态共享编码器, LoRA微调, Focal+Dice Loss
  │  + 通道拼接融合
  │  + modality_dropout
  │
  ▼
V2 Disentangle (RGBTv2Disentangle)
  │  通用/私有特征解耦
  │  + LightweightMAEBranch 私有编码器
  │  + 跨模态注意力融合
  │  + HSIC 解耦损失
  │  + InfoNCE 模态对齐损失
  │  + MSE 不变性损失
  │  + zc 辅助分割头
  │
  ▼
V3 Degradation (RGBTv3Degradation)
  │  V2 架构 + 在线退化增强 (300 epoch)
  │  + RGBTModalDegradation 训练管道
  │  + 30%干净 / 30%缺失 / 20%全局退化 / 20%局部退化
  │  + 与鲁棒性测试一致的退化类型
  │
  ▼
V4 QualityPruning (RGBTv4QualityPruning)
     V3 架构 + 质量网络 (300 epoch)
     + QualityNetwork 双分支质量评分 (RGB/Thermal独立)
     + 多尺度局部差异感知 (LocalDifferenceModule)
     + 质量加权模态对齐
     + 质量锚定损失 (loss_anchor, clean→0.8, deg→0.15)
     + 高退化天花板损失 (loss_quality_ceiling)
     + 空间多样性损失 (loss_spatial_div)
     + 质量正则化 (loss_quality_reg)
```

## 损失函数对比

| 损失项 | 原始版 | V1 | V2 | V3 | V4 | V5 |
|-------|--------|----|----|----|-----|-----|
| CrossEntropyLoss (decode) | ✓ (1.0) | | | | | |
| CrossEntropyLoss (aux) | ✓ (0.4) | | | | | |
| FocalLoss (decode) | | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) |
| DiceLoss (decode) | | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) | ✓ (1.0) |
| FocalLoss (aux) | | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) |
| DiceLoss (aux) | | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) | ✓ (0.4) |
| loss_seg_zc (CE) | | | ✓ (0.3) | ✓ (0.3) | ✓ (0.3) | ✓ (0.3) |
| loss_disentangle (HSIC) | | | ✓ (0.1~0.4) | ✓ (0.1~0.4) | ✓ (0.1~0.4) | ✓ (0.1~0.4) |
| loss_modal (InfoNCE) | | | ✓ (0.2) | ✓ (0.2) | 质量加权版 (0.2) | 质量引导对齐 (0.2) |
| loss_invariance (MSE) | | | ✓ (0.1) | ✓ (0.1) | ✓ (0.1) | Huber+余弦+质量加权 (0.001) |
| loss_anchor | | | | | ✓ (0.5) | ✗ |
| loss_quality_ceiling | | | | | ✓ | ✓ |
| loss_spatial_div | | | | | ✓ | ✓ |
| loss_quality_reg | | | | | ✓ (0.1) | ✓ (0.05) |

## 训练策略对比

| 项目 | 原始版 | V1 | V2 | V3 | V4 | V5 |
|------|--------|----|----|----|-----|-----|
| 优化器 | AdamW + LayerDecay | AdamW + LoRA | AdamW (通用×0.1) | 同 V2 | 同 V2 | 同 V2 |
| 学习率 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| 训练方式 | 320k iter | 200 epoch | 200 epoch | 300 epoch | 300 epoch | 300 epoch |
| Warmup | 1500 iter Linear | 4 epoch Linear | 4 epoch Linear | 4 epoch Linear | 4 epoch Linear | 4 epoch Linear |
| Scheduler | PolyLR | PolyLR | PolyLR | PolyLR | PolyLR | PolyLR |
| 混合精度 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 退化增强 | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 测试模式 | slide | whole | whole | whole | whole | whole |

## 公共文件

### V5 QualityJoint

### 模型架构

在 V3 基础上加入质量网络V2 (QualityNetworkV2)，实现真正的联合微调。

```
RGB (3ch) ──┬── [通用编码器] ──→ zc_rgb ──→ QualityNetworkV2 ──→ q_rgb (质量分数)
             │                                              │
             └── [私有编码器] ──→ zp_rgb ──→ 渐进式剪枝 ──┤
                                                        │
                    质量感知融合 ←────────────────────────┘
                        │
Thermal (3ch)──┬── [通用编码器] ──→ zc_t ──→ QualityNetworkV2 ──→ q_t (质量分数)
               │                                           │
               └── [私有编码器] ──→ zp_t ──→ 渐进式剪枝 ──┘
```

- **质量网络V2** (QualityNetworkV2)：双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
  - Token投影：Linear(768→128)
  - 多尺度局部差异感知：LocalDifferenceModule (3个尺度的平均池化差异)
  - 评分头：LN(256)→Linear(256,128)→GELU→Linear(128,1)→Sigmoid
  - RGB和Thermal各有独立分支参数
- **渐进式token剪枝**：
  - 早期(epoch 1-30)：软掩码，所有token参与但低质量被抑制
  - 中期(epoch 30-60)：Gumbel-Softmax，从软选择过渡到硬选择
  - 后期(epoch 60+)：掩码注意力，低质量token完全被忽略
- **推理阶段**：硬删除低于阈值的token，节省计算

### 质量网络V2自监督预训练

质量网络V2独立于分割网络预训练，使用自监督排序学习：

```bash
# 预训练质量网络
CUDA_VISIBLE_DEVICES=0 python tools/train_quality_network_v2.py \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_pretrain \
    --img-size 480 \
    --num-degradations 4 \
    --epochs 50 \
    --amp
```

预训练核心思路：
- 对同一图像生成K个递增退化版本，质量分数应满足 q_0 > q_1 > ... > q_K
- 使用质量锚定损失(QualityAnchorLoss)将clean分数拉向0.8，最高退化分数拉向0.15
- 使用级别间距损失(LevelMarginLoss)确保相邻退化级别间有足够间距
- 使用高退化天花板损失(HighDegCeilingLoss)限制最高退化级别的质量分数
- 使用空间多样性损失(SpatialDiversityLoss)鼓励同一图内不同token有不同质量分数

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| V2 全部分割损失 | 同 V2 | Focal+Dice + zc辅助 |
| loss_disentangle | 0.1~0.4 | HSIC 解耦损失 (zc ⊥ zp) |
| loss_modal | 0.2 | 质量引导模态对齐（替代InfoNCE） |
| loss_invariance | 0.001 | 质量感知加权Huber损失 + 余弦距离 + 分stage递增权重 |
| loss_quality_reg | 0.05 | 质量正则化（目标均值0.5，目标标准差0.2） |

### 与 V4 区别

- 质量网络从冻结改为联合微调
- 质量网络架构从单分支改为双分支(RGB/Thermal独立)，新增多尺度局部差异感知
- 模态对齐从InfoNCE改为质量引导对齐
- 不变性损失从MSE改为Huber+余弦距离+质量加权+分stage递增权重
- 新增渐进式token剪枝策略（软掩码→Gumbel-Softmax→掩码注意力）
- 保留质量锚定损失、高退化天花板损失、空间多样性损失，移除排序损失

### 相关文件

| 文件 | 说明 |
|------|------|
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-768x768.py` | FMB 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-480x480.py` | FMB 480×480 配置 |
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_mfnet-768x768.py` | MFNet 768×768 配置 |
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_mfnet-480x480.py` | MFNet 480×480 配置 |
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_pst900-480x480.py` | PST900 480×480 配置 |
| `configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_semanticrt-480x480.py` | SemanticRT 480×480 配置 |
| `mmseg/models/segmentors/rgbt_v5_quality_joint.py` | V5 模型实现 |
| `mmseg/models/utils/quality_network_v2.py` | QualityNetworkV2 (双分支，多尺度局部差异感知) |
| `tools/train_quality_network_v2.py` | 质量网络V2自监督预训练脚本 |

### 训练与评估

```bash
# Step 1: 预训练质量网络
CUDA_VISIBLE_DEVICES=0 python tools/train_quality_network_v2.py \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_pretrain \
    --img-size 480 --amp

# Step 2: 联合微调（使用预训练质量网络）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-768x768.py \
    --cfg-options model.quality_pretrained=work_dirs/quality_v2_pretrain/xxx/best_quality_net_v2.pth

# Step 2 (不使用预训练，从头联合训练)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-768x768.py

# 评估
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/xxx/best_mIoU.pth

# 鲁棒性测试
CUDA_VISIBLE_DEVICES=0 python tools/test_robustness.py \
    configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-768x768.py \
    work_dirs/xxx/best_mIoU.pth --batch-all --version v5
```

## 公共文件

| 文件 | 说明 |
|------|------|
| `mmseg/models/backbones/mae.py` | MAE ViT-B 编码器 |
| `mmseg/datasets/fmb.py` | FMB 数据集定义 |
| `mmseg/datasets/mfnet.py` | MFNet 数据集定义 |
| `mmseg/datasets/transforms/loading.py` | LoadRGBTImageFromFile / LoadRGBTImageFrom4Channel |
| `mmseg/datasets/transforms/rgbt_augmentation.py` | RGBT 数据增强 (含 RGBTModalDegradation) |
| `mmseg/datasets/transforms/robustness_degradation.py` | 鲁棒性测试退化变换 (退化参数和函数) |
| `tools/train.py` | 训练脚本 |
| `tools/test.py` | 测试脚本 |
| `tools/test_robustness.py` | 鲁棒性测试脚本 |
| `tools/robustness_README.md` | 鲁棒性测试说明 |
| `tools/robustness_visualization.ipynb` | 退化效果可视化 (支持 FMB/MFNet + 训练增强可视化) |
| `pretrain/M-SpecGene_VIT-B_seg_transform.pth` | 预训练权重 |
