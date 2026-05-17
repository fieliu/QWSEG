# QualityNetworkV2 质量网络

## 概述

QualityNetworkV2 是一个轻量级的**双分支** token 质量评估网络，用于 RGB-T 语义分割框架中。RGB 和 Thermal 各有独立的分支参数，分别接收 ViT backbone 输出的 token 特征，为每个 token 生成一个 [0, 1] 的质量分数，用于指导后续的 token 剪枝——质量低于阈值的 token 将被丢弃，从而在退化场景下减少噪声 token 对分割性能的影响。

## 网络架构

```
Input: ViT tokens [B, N, 768]
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│RGB分支 │ │T分支   │  (独立参数)
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
  Token Proj  Token Proj   Linear(768→128)
    │          │
    ▼          ▼
  LocalDiff   LocalDiff    多尺度局部差异感知 (3个尺度)
    │          │
    ▼          ▼
  Concat      Concat       [token_feat, local_diff_feat] → [B, N, 256]
    │          │
    ▼          ▼
  Score MLP   Score MLP    LN(256)→Linear(256,128)→GELU→Linear(128,1)→Sigmoid
    │          │
    ▼          ▼
  q_rgb       q_t          Quality scores [B, N] ∈ [0, 1]
```

### 设计要点

- **双分支独立参数**: RGB 和 Thermal 使用完全独立的投影、局部差异和评分参数，因为两种模态的退化类型不同，质量感知不能共享权重
- **Token Projection**: 将 768 维 ViT token 降维到 128 维，减少计算量
- **多尺度局部差异感知 (LocalDifferenceModule)**: 替代了之前的 DepthwiseConv + MaxPool 方案，通过多尺度平均池化差异提取每个 token 的局部上下文信息，能更好地捕捉空间质量变化
- **Sigmoid 输出**: 保证质量分数在 [0, 1] 范围内，1 表示高质量，0 表示低质量
- **初始化策略**: 最后一层 Linear 使用 mean=1.0 初始化，确保初始 sigmoid 输出接近 0.73，避免分数坍缩到过低值

### LocalDifferenceModule 详解

```
Input: projected tokens [B, N, 128] → reshape [B, 128, H, W]
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
  AvgPool  AvgPool  AvgPool   kernel_size=3,5,9 (stride=1, same padding)
    │    │    │
    ▼    ▼    ▼
  x-avg  x-avg  x-avg     计算与原始特征的差异
    │    │    │
    └────┼────┘
         │
         ▼
      Concat → [B, 384]     3个尺度的差异特征拼接
         │
         ▼
    Linear(384→128) + LayerNorm
         │
         ▼
    local_diff [B, N, 128]
```

**核心思想**: 每个尺度计算 token 特征与局部平均的差异，差异越大说明该区域越异常（可能退化），从而让质量评分网络感知到空间质量变化。

## 损失函数

训练采用多损失组合策略，总损失为：

```
L_total = λ₁·L_anchor + λ₂·L_ceiling + λ₃·L_margin + λ₄·L_spatial_div
```

### 1. 质量锚定损失 (QualityAnchorLoss, L_anchor)

**目的**: 将 clean 图像的质量分数拉向 0.8，最高退化级别的质量分数拉向 0.15

```
L_anchor_clean = MSE(q_clean, 0.8)
L_anchor_deg   = MSE(q_deg_highest, 0.15)
```

返回 `(L_anchor_clean, L_anchor_deg)` 两个损失项，可分别加权。

**参数**: `clean_target=0.8`, `deg_target=0.15`

### 2. 高退化天花板损失 (HighDegCeilingLoss, L_ceiling)

**目的**: 保证最高级退化的 token 质量分数不超过阈值（0.15），确保最严重退化的 token 被剪枝

```
L_ceiling = mean(ReLU(q_highest_deg - 0.15)²)
```

**参数**: `ceiling=0.15`

### 3. 级别间距损失 (LevelMarginLoss, L_margin)

**目的**: 防止质量分数在两端聚集，强制相邻退化级别之间至少有 0.2 的质量差值

```
L_margin = 1/(K-1) · Σᵢ mean(ReLU(0.2 - (qᵢ - qᵢ₊₁)))
```

对于质量序列 [q_clean, q_deg1, q_deg2, ..., q_degK]，相邻级别的质量差 qᵢ - qᵢ₊₁ 应 ≥ 0.2。

**参数**: `margin=0.2`

### 4. 空间多样性损失 (SpatialDiversityLoss, L_spatial_div)

**目的**: 鼓励同一图像不同位置的质量分数有差异，因为不同区域的退化程度天然不同

```
L_spatial_div = mean(ReLU(0.08 - std(q, dim=-1))²)
```

对每个样本计算质量分数的空间标准差，如果标准差低于 0.08 则产生惩罚。

**参数**: `min_std=0.08`

## 退化类型

训练时对每张图像生成 5 个退化级别（1=原图，2-5=递增退化），退化强度递增。

### RGB 退化类型

| 退化类型 | Level 2 | Level 3 | Level 4 | Level 5 |
|---------|---------|---------|---------|---------|
| low_light | brightness=0.5 | brightness=0.3 | brightness=0.15, noise=0.02 | brightness=0.06, noise=0.05 |
| overexposure | gain=1.5 | gain=2.5 | gain=4.0 | gain=8.0 |
| motion_blur | kernel=5 | kernel=9 | kernel=15 | kernel=23 |

### 热红外退化类型

| 退化类型 | Level 2 | Level 3 | Level 4 | Level 5 |
|---------|---------|---------|---------|---------|
| thermal_contrast | range=0.7 | range=0.4 | range=0.15 | range=0.05 |
| stripe_noise | intensity=0.03 | intensity=0.06 | intensity=0.12 | intensity=0.25 |
| thermal_noise | sigma=0.03 | sigma=0.06 | sigma=0.12 | sigma=0.20 |
| thermal_saturation | ratio=0.01 | ratio=0.05 | ratio=0.15 | ratio=0.30 |

## 评估场景

评估脚本 (`eval_quality_network.py`) 在 3 个场景下测试质量网络：

### 场景 1: 原图 (Clean)

直接输入原始 RGB-T 图像，期望质量分数接近 0.8。

### 场景 2: 全局退化 (Global Degradation)

- 随机选择退化类型和强度
- 对整张图像应用退化

### 场景 3: 局部退化 (Local Degradation)

- 随机选择退化类型和强度
- 随机选择 1~3 个矩形区域
- 只在选定区域应用退化

### 可视化格式

每个场景生成 3 组大图，每组包含 3 对 (RGB + T) 小图。每组大图包含 3 行：

1. **图像行**: 原始/退化后的 RGB 和热红外图像
2. **质量热图行**: 每个图像的质量分数热图（红色=高质量，蓝色=低质量）
3. **剪枝行**: 低于阈值的 token 被置零后的图像

## 使用方法

### 训练 (V2 脚本，推荐)

```bash
cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt

python tools/train_quality_network_v2.py \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_pretrain \
    --img-size 480 \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-3 \
    --num-degradations 4 \
    --loss-anchor-weight 1.0 \
    --loss-ceiling-weight 2.0 \
    --loss-margin-weight 0.5 \
    --loss-spatial-div-weight 0.5 \
    --quality-threshold 0.3 \
    --amp
```

### 训练 (V1 脚本，依赖分割模型)

```bash
python tools/train_quality_network.py \
    --config configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py \
    --checkpoint work_dirs/v1_baseline/fmb/best.pth \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_pretrain \
    --img-size 480 \
    --amp
```

### 评估

```bash
python tools/eval_quality_network.py \
    --checkpoint work_dirs/quality_v2_pretrain/<timestamp>/weight/best_quality_net_v2.pth \
    --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
    --work-dir work_dirs/quality_v2_eval \
    --img-size 480 \
    --quality-threshold 0.1 \
    --amp
```

### TensorBoard 可视化

```bash
tensorboard --logdir=work_dirs/quality_v2_pretrain/<timestamp>/log/
```

训练日志包含：
- `train/loss_anchor` - 质量锚定损失
- `train/loss_ceiling` - 高退化天花板损失
- `train/loss_margin` - 级别间距损失
- `train/loss_spatial_div` - 空间多样性损失
- `train/loss_total` - 总损失
- `train/q_rgb_clean_mean` - RGB clean 质量均值
- `train/q_t_clean_mean` - Thermal clean 质量均值
- `train/q_rgb_clean_std` - RGB clean 质量标准差
- `train/q_t_clean_std` - Thermal clean 质量标准差

## 参数说明

### 训练参数 (V2 脚本)

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--vit-pretrained` | (必填) | MAE ViT-B 预训练权重路径 |
| `--data-root` | `/home/lh/code/data/FMB_ALL` | 数据集根目录 |
| `--work-dir` | `work_dirs/quality_v2_pretrain` | 工作目录 |
| `--img-size` | 480 | 图像尺寸 |
| `--batch-size` | 4 | 批大小 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--num-workers` | 4 | 数据加载线程数 |
| `--num-degradations` | 4 | 退化级别数 K |
| `--loss-anchor-weight` | 1.0 | 质量锚定损失权重 |
| `--loss-ceiling-weight` | 2.0 | 高退化天花板损失权重 |
| `--loss-margin-weight` | 0.5 | 级别间距损失权重 |
| `--loss-spatial-div-weight` | 0.5 | 空间多样性损失权重 |
| `--quality-threshold` | 0.3 | 可视化用的质量阈值 |
| `--save-interval` | 10 | 每 N 个 epoch 保存检查点 |
| `--resume` | None | 从检查点恢复训练 |
| `--amp` | False | 启用混合精度训练 |

## 文件结构

```
mmseg/models/utils/quality_network.py       # QualityNetwork (V4使用) + 损失函数
mmseg/models/utils/quality_network_v2.py    # QualityNetworkV2 (V5使用)
mmseg/datasets/transforms/quality_degradation.py  # 退化变换实现
tools/train_quality_network.py              # V1训练脚本 (依赖分割模型)
tools/train_quality_network_v2.py           # V2训练脚本 (推荐，自监督)
tools/eval_quality_network.py               # 评估脚本
```

## 输出目录结构

训练完成后，工作目录结构如下：

```
work_dirs/quality_v2_pretrain/
└── <YYYYMMDD_HHMMSS>/            # 运行时间戳
    ├── config.txt                 # 本次运行的配置信息
    ├── log/                       # 日志目录
    │   ├── train.log              # 终端训练日志
    │   └── events.out.tfevents.*  # TensorBoard事件文件
    ├── weight/                    # 权重目录
    │   ├── best_quality_net_v2.pth      # 最优训练损失权重
    │   ├── best_val_quality_net_v2.pth  # 最优验证损失权重
    │   ├── quality_net_v2_epoch*.pth    # 定期检查点
    │   ├── final_quality_net_v2.pth     # 最终模型
    │   └── checkpoint_info.txt          # 最优/最终权重路径记录
    └── vis_data/                  # 可视化目录
        ├── train_vis_*.png        # 训练可视化 (每 epoch)
        └── val_vis_*.png          # 验证可视化 (每 val_interval)
```
