# QWSEG: Quality-Aware RGB-T Semantic Segmentation

## 项目简介

QWSEG 是一个面向退化场景的 **RGB-T（可见光-热红外）鲁棒语义分割** 框架。核心创新在于提出了 **质量感知 Token 剪枝机制**：通过轻量级质量网络评估每个 ViT token 的质量分数，在退化场景下自动丢弃低质量 token，从而提升模型对光照不足、过曝、运动模糊、热噪声等退化因素的鲁棒性。

## 研究动机

RGB-T 语义分割利用可见光和热红外两种模态的互补性来提升分割性能。然而，在实际场景中，两种模态都可能遭受不同程度的退化：

- **RGB 退化**：低光照、过曝、运动模糊
- **热红外退化**：对比度下降、条纹噪声、热噪声、饱和失真

传统方法对所有 token 一视同仁，导致退化区域的噪声 token 干扰分割结果。本项目提出 **质量驱动的 token 剪枝** 策略，让模型学会"忽略"低质量信息，聚焦于可靠特征。

## 系统架构

项目采用渐进式设计，从 V1 到 V5 逐步引入新模块：

```
┌─────────────────────────────────────────────────────────────────┐
│                    QWSEG 整体架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: RGB (3ch) + Thermal (3ch)                                │
│         │              │                                        │
│         ▼              ▼                                        │
│  ┌──────────────────────────────┐                               │
│  │   Universal ViT Backbone     │  MAE / SAM (共享参数)          │
│  │   (参数共享处理双模态)         │  LoRA 微调                    │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             │  zc_rgb, zc_t  (共享特征)                          │
│             ▼                                                   │
│  ┌──────────────────────────────┐                               │
│  │   Private Branch (RGB/T)     │  模态私有特征提取               │
│  │   Lightweight MAE Branch     │  轻量级私有分支                 │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             │  zp_rgb, zp_t  (私有特征)                          │
│             ▼                                                   │
│  ┌──────────────────────────────┐                               │
│  │   Cross-Attention Fusion     │  跨模态交叉注意力               │
│  │   (4 stages)                 │  zc↔zp 互补融合                │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             │  融合特征                                          │
│             ▼                                                   │
│  ┌──────────────────────────────┐                               │
│  │   QualityNetworkV2           │  ★ 核心创新                    │
│  │   Token → Quality Score      │  每个 token 质量评估            │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             │  q_rgb, q_t ∈ [0,1]                               │
│             ▼                                                   │
│  ┌──────────────────────────────┐                               │
│  │   Quality-Guided Pruning     │  Gumbel-Softmax 可微剪枝       │
│  │   (GumbelSoftmaxMask)        │  低质量 token → 0              │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             │  剪枝后特征                                        │
│             ▼                                                   │
│  ┌──────────────────────────────┐                               │
│  │   UPerNet Decoder            │  多尺度特征金字塔解码           │
│  │   + Auxiliary Head           │  深度监督                      │
│  └──────────┬───────────────────┘                               │
│             │                                                   │
│             ▼                                                   │
│  输出: 语义分割结果 (15类)                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 版本演进

| 版本 | 模型 | 核心特性 |
|------|------|---------|
| **V1 Baseline** | RGBTv1Baseline | LoRA 微调 + 模态 dropout + 参数共享双模态处理 |
| **V1 Baseline (FMB-Init)** | RGBTv1Baseline | FMB权重完全初始化 (backbone+neck+decoder) + LoRA 微调 |
| **V2 Disentangle** | RGBTv2Disentangle | 共享/私有特征解耦 + HSIC 解耦损失 + InfoNCE 跨模态对齐 |
| **V3 Degradation** | RGBTv3Degradation | V2 + 退化感知训练 + 退化增强数据管道 |
| **V3 Degradation (MiT)** | MiTMulV3Degradation | MiT-B2+B0 + 质量网络V3 (GlobalNorm+LocalChannelAttn) + 空间退化训练 + 排序损失 |
| **V4 Quality Pruning** | RGBTv4QualityPruning | V2 + 质量网络 V1 + Token 硬阈值剪枝 + 排序损失 |
| **V5 Quality Joint** | RGBTv5QualityJoint | V2 + 质量网络 V2 (双分支) + Gumbel-Softmax 渐进式可微剪枝 + 质量引导对齐 |

## 核心模块详解

### 1. 质量网络 (QualityNetworkV2)

轻量级双分支 token 质量评估网络（~100K 参数），RGB 和 Thermal 各有独立分支，为每个 ViT token 输出 [0,1] 的质量分数：

```
ViT Token [B, N, 768]
    → Token Projection (768→64) + GELU
    → Local Statistics (3×3 DepthwiseConv + MaxPool)
    → Concatenate [token_feat, local_pool_feat] → [B, N, 128]
    → MLP (128→128→1) + Sigmoid
    → Quality Score [B, N] ∈ [0,1]

RGB分支: rgb_token_proj → rgb_local_conv → rgb_max_pool → rgb_mlp → q_rgb
T分支:   t_token_proj   → t_local_conv  → t_max_pool  → t_mlp  → q_t
```

**自监督训练**：利用多级退化生成排序监督信号，无需人工标注质量分数。

**5 项损失函数**：

| 损失 | 作用 |
|------|------|
| ListMLE 排序损失 | 保证质量分数排序正确：原图 > 低级退化 > 高级退化 |
| 高级退化质量上界损失 | 保证最高级退化质量 < 0.1 |
| 低级退化质量下界损失 | 保证非最高级退化质量 > 0.1 |
| 级别间距损失 | 相邻退化级别质量差 ≥ 0.2，避免分数聚集 |
| 空间方差损失 | 鼓励同一图像不同位置的质量差异 |

### 1.5 质量网络 V3 (QualityNetworkV3)

**核心设计原则**：保留 token 独立性，拒绝空间混合污染。

V3 版本重新设计了质量评估架构，使用两个核心模块替代 V2 的 Transformer 结构：

```
ViT Token [B, N, D]
    → GlobalNorm: 全局统计归一化
        ├─ 全图 token 取平均 → 全局描述子 [B, D]
        ├─ MLP → γ (通道缩放) [B, 1, D]
        ├─ MLP → β (通道偏置) [B, 1, D]
        └─ 逐通道仿射: tokens * (1 + γ) + β
    → 2D Reshape: [B, D, H, W]
    → LocalChannelAttention: 局部通道注意力
        ├─ 3×3 Conv → mid_feat [B, mid, H, W]
        ├─ 1×1 Conv → channel_attn [B, D, H, W]
        ├─ 自身特征 × 通道注意力 (不混合邻域特征!)
        └─ 1×1 Conv → score_map [B, 1, H, W]
    → Flatten → Sigmoid
    → Quality Score [B, N] ∈ [0,1]

RGB分支: rgb_global_norm → rgb_local_attn → q_rgb
T分支:   t_global_norm   → t_local_attn  → q_t
```

**关键特性**：
- **GlobalNorm**：校准全图亮度/对比度基线，但不混合不同 token 的像素
- **LocalChannelAttn**：3×3 卷积利用邻域上下文生成调制信号，但绝不将邻域特征直接加到自身
- **在线空间退化训练**：训练时随机生成 2-6 个退化区域（3×3~5×5 token 范围），使用五级退化

**训练约束**：
- 相邻退化级别质量差 ≥ 0.1（RankingMarginLoss）
- 五级退化质量 < 0.1（LevelConsistencyLoss）
- 原图质量锚定 0.9（CleanAnchorLoss）
- 空间多样性鼓励（SpatialDiversityLossV3）

### 2. 质量引导 Token 剪枝

**V4 硬阈值剪枝**：推理时直接根据质量阈值过滤低质量 token：

```python
keep_mask = (quality_score >= threshold).float()
pruned_tokens = tokens * keep_mask.unsqueeze(-1)
# 质量 < 阈值的 token 直接置零
```

**V5 Gumbel-Softmax 渐进式剪枝**：训练时使用三阶段策略，推理时使用硬阈值：

```python
# 训练阶段策略（根据 epoch 自动切换）：
# 1. soft_mask (epoch < 30): 质量分数直接作为软权重
# 2. gumbel (30 ≤ epoch < 60): Gumbel-Softmax 可微采样
# 3. mask_attention (epoch ≥ 60): 硬阈值 mask + 梯度传播

keep_logits = log(q),  drop_logits = log(1-q)
mask = GumbelSoftmax([keep_logits, drop_logits], tau)
# tau 退火从 1.0 → 0.1，逐渐从软决策过渡到硬决策
```

### 3. 特征解耦与跨模态融合

- **共享特征 (zc)**: Universal ViT Backbone 提取，两种模态共享参数
- **私有特征 (zp)**: Lightweight Private Branch 提取，保留模态独有信息
- **HSIC 解耦**: 最小化共享/私有特征的 Hilbert-Schmidt 独立性准则，确保解耦
- **InfoNCE 对齐**: 跨模态对比学习，对齐 RGB 和热红外的共享特征
- **交叉注意力融合**: 4 级 Cross-Attention，zc↔zp 互补融合

### 4. LoRA 参数高效微调

在冻结的 ViT Backbone 上应用 LoRA（Low-Rank Adaptation），仅微调 QKV 和投影层的低秩矩阵：

```python
# 原始线性层 + LoRA 旁路
output = W·x + (B·A·x) × (α/r)
# A: [in, r], B: [r, out], r=4 (rank)
# 可训练参数仅占总参数的 ~0.5%
```

### 5. 退化增强数据管道

支持 7 种退化类型，每种 4 个强度级别（Level 2~5），支持全局和局部退化：

| 模态 | 退化类型 | 参数 |
|------|---------|------|
| RGB | 低光照 (low_light) | 亮度缩放 + 噪声 |
| RGB | 过曝 (overexposure) | 增益系数 |
| RGB | 运动模糊 (motion_blur) | 卷积核大小 |
| 热红外 | 热对比度 (thermal_contrast) | 动态范围缩放 |
| 热红外 | 条纹噪声 (stripe_noise) | 噪声强度 |
| 热红外 | 热噪声 (thermal_noise) | 高斯噪声 σ |
| 热红外 | 热饱和 (thermal_saturation) | 饱和像素比例 |

## 支持的 Backbone

| Backbone | 预训练权重 | 特点 |
|----------|-----------|------|
| MAE ViT-B | M-SpecGene_VIT-B | 多光谱预训练，参数共享处理 RGBT |
| SAM ViT-B | sam_vit_b_01ec64 | Segment Anything 预训练，窗口注意力 |

## 数据集

| 数据集 | 分辨率 | 类别数 | 场景 |
|--------|--------|--------|------|
| FMB | 480×480 | 15 | 自动驾驶 |
| MFNet | 480×480 / 768×768 | 9 | 自动驾驶 |
| SemanticRT | 480×480 | 9 | 道路场景 |
| PST900 | 480×480 | 5 | 搜救场景 |

## 鲁棒性评估

`test_robustness.py` 支持 9 种退化场景的系统评估：

| 场景 | 描述 |
|------|------|
| Clean | 无退化基线 |
| RGB Missing | RGB 模态缺失 |
| Thermal Missing | 热红外模态缺失 |
| RGB Low Light | RGB 低光照退化 |
| RGB Overexposure | RGB 过曝退化 |
| RGB Motion Blur | RGB 运动模糊 |
| Thermal Contrast | 热红外对比度下降 |
| Thermal Noise | 热红外噪声 |
| Thermal Saturation | 热红外饱和失真 |

每个场景支持多强度级别测试，输出 mIoU、mAcc、aAcc 等指标。

## 项目结构

```
QWSEG/
├── configs/                          # 模型配置
│   ├── mae/                          # MAE backbone 配置
│   │   ├── *v1_baseline*             # V1 基线 (LoRA)
│   │   ├── *v1_baseline_fmb-init*    # V1 基线 (FMB权重完全初始化)
│   │   ├── *v2_disentangle*          # V2 解耦
│   │   ├── *v3_degradation*          # V3 退化感知
│   │   ├── *v4_quality_pruning*      # V4 质量剪枝
│   │   ├── *v5_quality_joint*        # V5 质量联合
│   │   └── *freeze_backbone*         # 冻结 backbone 测试
│   └── sam/                          # SAM backbone 配置
│       ├── *v1_sam_baseline*         # SAM + LoRA 基线
│       └── *freeze_backbone*         # SAM 冻结测试
├── mmseg/
│   └── models/
│       ├── backbones/
│       │   ├── mae.py                # MAE ViT Backbone
│       │   ├── sam_vit.py            # SAM ViT Backbone
│       │   └── lightweight_mae_branch.py  # 轻量级私有分支
│       ├── segmentors/
│       │   ├── rgbt_v1_baseline.py   # V1: LoRA 基线
│       │   ├── rgbt_v2_disentangle.py # V2: 特征解耦
│       │   ├── rgbt_v3_degradation.py # V3: 退化感知
│       │   ├── rgbt_v4_quality_pruning.py # V4: 质量剪枝
│       │   ├── rgbt_v5_quality_joint.py   # V5: 质量联合
│       │   ├── mitmul_v1_baseline.py  # MiT V1: MiT-B2 基线
│       │   ├── mitmul_v2_disentangle.py # MiT V2: 解耦
│       │   ├── mitmul_v3_degradation.py # MiT V3: 退化感知 + QualityNetworkV3
│       │   ├── mitmul_v4_quality_pruning.py # MiT V4: 质量剪枝
│       │   └── mitmul_v5_quality_joint.py   # MiT V5: 质量联合
│       └── utils/
│           ├── quality_network_v3.py  # 质量网络V3 (GlobalNorm+LocalChannelAttn) + 损失函数
│           ├── quality_network_v2.py  # 质量网络V2 (双分支) + 损失函数
│           ├── quality_network.py     # 质量网络 V1
│           ├── spatial_degradation_generator.py # 空间退化生成器
│           └── lora.py               # LoRA 实现
├── tools/
│   ├── train.py                      # 主训练脚本 (含目录结构整理)
│   ├── train_quality_network_v2.py   # 质量网络自监督预训练
│   ├── eval_quality_network.py       # 质量网络评估
│   ├── test_robustness.py            # 鲁棒性测试
│   ├── quality_network_README.md     # 质量网络详细文档
│   └── robustness_README.md          # 鲁棒性测试详细文档
└── pretrain/                         # 预训练权重
    ├── M-SpecGene_VIT-B_seg_transform.pth
    ├── FMB_save_iter_224000.pth      # FMB训练权重 (用于完全初始化)
    └── sam_vit_b_01ec64.pth
```

## 快速开始

### 环境配置

```bash
conda create -n mmseg python=3.10
conda activate mmseg
pip install torch torchvision  # CUDA 11.8+
pip install mmengine mmcv mmdet
pip install -e .
```

### 训练

```bash
# V1 基线 (MAE + LoRA)
python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-480x480.py --amp

# V1 基线 (FMB权重完全初始化)
python tools/train.py configs/mae/mae-base_upernet_rgbt_v1_baseline_fmb-init_8xb2-amp-200e_fmb-480x480.py --amp

# V5 质量联合 (MAE + 质量网络 + Gumbel 剪枝)
python tools/train.py configs/mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_fmb-480x480.py --amp

# SAM + LoRA 基线
python tools/train.py configs/sam/sam-base_upernet_rgbt_v1_sam_baseline_8xb2-amp-200e_fmb-480x480.py --amp

# MiT V3 退化感知 (MiT-B2+B0 + QualityNetworkV3)
python tools/train.py configs/segformer/mitmul_v3_mit-b2-b0_1xb2-40K_fmb-480x480.py --amp

# 冻结 backbone (测试特征提取能力)
python tools/train.py configs/mae/mae-base_upernet_freeze_backbone_8xb2-amp-40e_fmb-480x480.py --amp
```

### 质量网络训练

```bash
python tools/train_quality_network_v2.py \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_pretrain \
    --img-size 480 --batch-size 4 --epochs 50 --amp
```

### 质量网络评估

```bash
python tools/eval_quality_network.py \
    --checkpoint work_dirs/quality_v2_pretrain/<timestamp>/weights/best_quality_net_v2.pth \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_eval --amp
```

### 鲁棒性测试

```bash
# 全场景批量测试
python tools/test_robustness.py <config> <checkpoint> --batch-all --visualize

# 单场景测试
python tools/test_robustness.py <config> <checkpoint> \
    --degradation GlobalDegradation --deg-kwargs modality=rgb intensity=low
```

### TensorBoard 可视化

```bash
tensorboard --logdir=work_dirs/
```

## 输出目录结构

所有训练脚本统一使用以下目录结构：

```
work_dirs/
├── <version_name>/                    # 顶级目录：版本名称
│   ├── <YYYYMMDD_HHMMSS>/            # 运行时间戳（区分不同运行）
│   │   ├── config.txt                 # 本次运行的配置信息
│   │   ├── log/                       # 日志目录
│   │   │   ├── <timestamp>.log        # 终端训练日志
│   │   │   ├── <timestamp>.log.json   # JSON格式日志
│   │   │   └── events.out.tfevents.*  # TensorBoard事件文件
│   │   ├── weight/                    # 权重目录
│   │   │   ├── best_mIoU_*.pth        # 最优模型权重
│   │   │   ├── epoch_*.pth            # 定期保存的检查点
│   │   │   └── checkpoint_info.txt    # 最优/最终权重路径记录
│   │   └── vis_data/                  # 可视化目录
│   │       └── vis_image/             # 训练可视化图片
│   ├── <YYYYMMDD_HHMMSS>/            # 另一次运行
│   │   └── ...
│   └── ...
├── quality_v2_pretrain/               # 质量网络训练输出
│   └── <YYYYMMDD_HHMMSS>/
│       ├── config.txt
│       ├── log/
│       │   ├── train.log
│       │   └── events.out.tfevents.*
│       ├── weight/
│       │   ├── best_quality_net_v2.pth
│       │   ├── best_val_quality_net_v2.pth
│       │   ├── quality_net_v2_epoch*.pth
│       │   ├── final_quality_net_v2.pth
│       │   └── checkpoint_info.txt
│       └── vis_data/
│           ├── train_vis_*.png
│           └── val_vis_*.png
└── quality_v2_eval/                   # 质量网络评估输出
    └── <YYYYMMDD_HHMMSS>/
        └── ...
```

## 技术栈

- **框架**: PyTorch + MMSegmentation (MMEngine)
- **Backbone**: MAE ViT-B / SAM ViT-B
- **参数高效微调**: LoRA (rank=4)
- **训练**: AMP 混合精度 + 滑动推理
- **可视化**: TensorBoard + Matplotlib
- **评估**: mIoU / mAcc / aAcc + 多退化场景鲁棒性测试

## 关键技术亮点

1. **质量感知 Token 剪枝**: 首次在 RGB-T 分割中引入 token 级质量评估，实现退化自适应的动态剪枝
2. **自监督质量网络训练**: 利用多级退化排序作为监督信号，无需额外质量标注
3. **Gumbel-Softmax 可微剪枝**: 训练时软剪枝保留梯度流，推理时硬剪枝实现零开销
4. **共享-私有特征解耦**: HSIC 约束 + InfoNCE 对齐，有效分离模态共享/独有特征
5. **多 Backbone 对比**: MAE vs SAM 特征提取能力对比，LoRA vs 冻结微调策略对比
6. **系统化鲁棒性评估**: 9 种退化场景 × 多强度级别，全面验证模型退化鲁棒性

## 消融实验设计

消融实验基于最优版本 `QualityGatedSwinMask2Former`（分支 `ablation-baseline`，对应提交 `317db11`）。所有消融通过模型的开关参数控制，**单类共享同一前向主干**，保证"除被消融的那一个变量外，其余完全相同"——这是消融可信度的前提。

### 设计原则

1. **统一基线**：所有消融继承同一基线配置 `swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py`，每个消融配置只覆盖一个开关。
2. **连锁一致性由代码强制**：质量机制内部强耦合（质量分是 attention bias、硬掩码、cascade、质量加权融合、多个 loss 的共同输入）。关闭主开关时，依赖它的组件/损失由代码**自动同步关闭**，无需在配置里手动设零，避免漏配导致的"假消融"。
3. **统一实验条件**：同 seed、200 epoch、MFNet、Swin-T backbone，三栏报告 `clean / rgb_missing / thermal_missing` 三种 mIoU。
4. **消融用 Swin-T**（快、省显存，定架构足够）；主结果表才用 Swin-L。

### 组件依赖关系

```
质量预测器 → s_r/s_t (common), spr/spt (private)
   ├─→ attention bias (f_attn, 注入每个 Swin block)
   ├─→ 硬掩码 (f_hard_mask, tau_hard=0.2)
   ├─→ cascade_quality_suppress (跨阶段累乘抑制)
   ├─→ retention_loss (约束保留率)
   ├─→ common 融合 (质量加权平均) → zf
   ├─→ 私有加权 (f_fuse) → re/te
   ├─→ final_fusion(re, te, zf) → ff (送主解码头)
   ├─→ loss_align (质量门控对比采样)
   └─→ loss_invariant (质量门控特征不变性)
退化训练 → 第二次 forward → deg 损失 + loss_distill + loss_invariant
```

### 主消融表（进正文，leave-one-out）

每行从完整模型移除**一套机制**（非单个代码组件），对应论文的一个核心贡献点。

| ID | 配置文件 | 主开关 | 移除的内容 | 回答的问题 |
|----|----------|--------|------------|------------|
| M0 | （基线本身） | — | 无 | 完整模型，上界参考 |
| M1 | `ablation/M1_no_quality.py` | `use_quality=False` | 质量预测器 + attention bias + 硬掩码 + 质量加权融合 + retention/align/invariant 损失 | 质量感知机制整体的贡献 |
| M2 | `ablation/M2_no_degradation.py` | `use_degradation=False` | 退化第二次前向 + 全部退化损失 + distill + invariant | 退化鲁棒训练的贡献 |
| M3 | `ablation/M3_no_private_branch.py` | `use_private_branch=False` | 双私有 Swin 分支 + 私有预测器 + 私有辅助头 + final_fusion | 三分支架构的贡献 |
| M4 | `ablation/M4_no_cascade_retention.py` | `use_cascade=False` + `retention_loss_weight=0` | 跨阶段级联抑制 + 保留率约束 | 跨阶段一致性/稳定机制的贡献 |
| M5 | `ablation/M5_no_curriculum.py` | `skip_phases=True` | 三阶段训练课程（warmup） | 渐进式课程的必要性 |

### 每个消融的详细改动与连锁

**M1 — 去掉质量感知（连锁最复杂）**
- 主开关 `use_quality=False`，代码内部强制连锁关闭：① attention bias 传 None；② 硬掩码跳过；③ cascade 跳过；④ `retention_loss_weight`/`loss_align_weight`/`loss_invariant_weight` 自动归零（这三个损失都消费质量分，留着会用无意义的 s 算出错误损失）。
- **融合变化**：common 融合的质量加权 `(w_r·zc_r + w_t·zc_t)/(w_r+w_t)` → 简单平均 `(zc_r+zc_t)/2`；私有加权 `f_fuse(spr)` → 权重恒 1，即直接相加。`final_fusion`（参数化 DualGate 模块，不依赖质量分）结构保留。
- **意义**：结果等价于"同参数三分支 + 朴素融合"，同时回答审稿人"提升是否只来自参数量"。

**M2 — 去掉退化鲁棒训练（融合不变）**
- `use_degradation=False`：`loss()` 中 `if self.training and self.use_degradation` 跳过整个退化分支。代码连锁关闭 `loss_distill`（clean→deg 的 KL，无 deg 分支则无意义）和 `loss_invariant`（clean vs deg 特征）。
- **融合变化**：无。clean 前向与融合一字不变，是干净的单变量消融。

**M3 — 去掉私有双分支（架构级）**
- `use_private_branch=False`：不构建两个私有 Swin 分支、私有预测器；私有辅助头自动设 None。
- **融合变化**：`re/te` 不存在，**旁路 `final_fusion`**，直接用质量加权 common 融合 `zf` 作为最终特征 `ff` 送主解码头。即从"common + 双私有三路融合"退化为"纯 common 双模态融合"。common 分支的质量加权保留。

**M4 — 去掉跨阶段一致性（连锁最少）**
- `use_cascade=False` + `retention_loss_weight=0`（这两个是一对稳定机制，一起关）。
- **融合变化**：公式不变，但喂给下游的质量分 s 从"级联累乘抑制后的值"变为"每层独立原始值"。这会影响硬掩码的跨层稳定性（缺失区域可能逐层抖动地丢/留）。

**M5 — 去掉三阶段课程（调度级）**
- `skip_phases=True`：`_get_training_phase` 直接返回 phase 3，`force_all_keep=False` 从 epoch 0 生效，质量预测器与退化训练从头全开，无 warmup。
- **融合变化**：无。仅改训练调度，不改结构。

### 附录消融（设计选择验证）

确定主架构后再跑，证明"机制内部的设计选择合理"，放正文小表或附录。

| ID | 配置文件 | 改动 | 回答 |
|----|----------|------|------|
| D1a | `ablation/D1a_no_hard_mask.py` | `use_hard_mask=False`（保留软门控） | 硬掩码是否必要 |
| D1b | （调 `tau_hard` 0.1/0.2/0.3） | 硬掩码阈值 | 阈值敏感性 |
| D2 | （切换 common 融合方式） | 质量加权 vs add vs softmax | 为何选质量加权 |
| D3 | （调 `tau` 0.2/0.3/0.4） | 质量阈值 | 超参鲁棒性 |

**D1a 预测**：clean 持平或略升，缺失场景掉点——若如此，说明"硬掩码对干净输入冗余，但对模态缺失鲁棒性关键"，把可删组件变成有明确作用边界的设计；若哪儿都不掉，则直接删掉，更契合"用 gating 不用 pruning"的论文叙事。

### 边角损失删减探针（不进消融表）

用于决定是否从最终模型删除边际损失项。各跑一次 leave-one-out，掉点 < 0.3 mIoU 即从最终模型删除，论文一句话带过。三者权重小（0.1/0.3/0.03）且功能疑似重叠（distill 与 invariant 都做 deg→clean 对齐）。

| ID | 配置文件 | 改动 |
|----|----------|------|
| E1 | `ablation/E1_no_align.py` | `loss_align_weight=0` |
| E2 | `ablation/E2_no_distill.py` | `loss_distill_weight=0` |
| E3 | `ablation/E3_no_invariant.py` | `loss_invariant_weight=0` |

### 运行方式

```bash
# 单个消融训练（示例：M1）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/mask2former/ablation/M1_no_quality.py

# 显存紧张时加梯度检查点
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/mask2former/ablation/M1_no_quality.py \
    --cfg-options model.backbone.with_cp=True \
        model.private_branch_rgb.with_cp=True \
        model.private_branch_t.with_cp=True

# 鲁棒性评测（clean / rgb_missing / thermal_missing 三栏）
python tools/test_robustness.py \
    configs/mask2former/ablation/M1_no_quality.py \
    <checkpoint.pth>
```

> 注意：当前基线配置使用 fp32 + 梯度检查点。Swin + Mask2Former 在 AMP（`--amp`）下易出 NaN（deformable attention 与 dice/mask loss 在半精度下数值不稳），消融实验**不要加 `--amp`**，以免引入与消融无关的训练不稳定变量。

