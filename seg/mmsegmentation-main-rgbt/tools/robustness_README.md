# RGB-T 鲁棒性测试说明

## 1. 退化类型总览

### 1.1 全局退化

| 退化类型 | 标识名 | 适用模态 | 同步性 | 低强度 | 中强度 | 高强度 |
|---------|--------|---------|--------|--------|--------|--------|
| 运动模糊 | `motion_blur` | RGB + T | 必须同步（同一模糊核） | 核大小 7 | 核大小 20 | 核大小 38 |
| 高斯噪声 | `gaussian_noise` | RGB / T | 独立应用（可设不同σ） | RGB σ=25, T σ=30 | RGB σ=50, T σ=55 | RGB σ=80, T σ=85 |
| 椒盐噪声 | `salt_pepper` | RGB / T | 独立应用 | 密度 0.02 | 密度 0.06 | 密度 0.12 |
| 条纹噪声 | `stripe_noise` | 仅 T | 仅热红外 | 振幅 0.03 | 振幅 0.08 | 振幅 0.15 |
| 热对比度降低 | `thermal_contrast` | 仅 T | 仅热红外 | α=0.7 | α=0.5 | α=0.3 |

**模态选择规则：**
- 必须同步的类型（运动模糊）：RGB 和 T 同时退化
- 仅 T 的类型（条纹、热对比度）：只退化热红外
- 双模态类型（高斯噪声、椒盐噪声）：0.9 概率随机选择单模态退化，0.1 概率双模态同时退化

### 1.2 局部退化

| 退化类型 | 标识名 | 适用模态 | 同步性 | 区域形状 | 低强度 | 中强度 | 高强度 |
|---------|--------|---------|--------|---------|--------|--------|--------|
| 局部过曝 | `local_overexposure` | 仅 RGB | 仅 RGB | 随机矩形 | 1-2块, 总<5%, +60 | 2-5块, 总<20%, +120 | 5-7块, 总<40%, +200 |
| 局部过暗 | `local_lowlight` | 仅 RGB | 仅 RGB | 随机矩形 | 1-2块, 总<5%, γ=1.5 | 2-5块, 总<20%, γ=2.5 | 5-7块, 总<40%, γ=4.0 |
| 水滴 | `water_stain` | RGB + T | 必须同步（同一掩码） | Perlin噪声 | 1-3个, 总<10%, 核3 | 3-6个, 总<30%, 核5 | 5-10个, 总<40%, 核7 |
| 污渍 | `stain` | RGB / T | 独立应用 | Perlin噪声 | 1-3个, 总<5%, α=0.3-0.5 | 3-6个, 总<10%, α=0.5-0.8 | 5-10个, 总<20%, α=0.8-1.0 |
| 传感器坏块 | `local_bad_block` | RGB / T | 独立应用 | 随机矩形 | 1-2块, 总<5% | 3-5块, 总<20% | 5-7块, 总<40% |
| 局部高斯噪声 | `local_gaussian_noise` | RGB / T | 独立应用 | 随机矩形 | 1-2块, 总<5%, σ同全局 | 2-5块, 总<20%, σ同全局 | 5-7块, 总<40%, σ同全局 |
| 热晕现象 | `thermal_halo` | 仅 T | 仅热红外 | 径向高斯 | 1-2中心, σ=4-6, 增益30-50 | 2-4中心, σ=8-12, 增益60-90 | 3-6中心, σ=15-20, 增益100-140 |

**模态选择规则：**
- 必须同步的类型（水滴）：RGB 和 T 使用同一 Perlin 噪声掩码
- 仅 RGB 的类型（过曝、过暗）：只退化 RGB
- 仅 T 的类型（热晕）：只退化热红外
- 双模态类型（污渍、坏块、局部高斯噪声）：0.9 概率随机选择单模态，0.1 概率双模态

**局部退化面积说明：**
- 每个矩形区域面积：0.5%~5% 图像面积
- 区域大小分布：均值 2%、方差 2% 的截断正态分布（取值范围 0.5%~5%），中等大小区域最多
- 总退化面积满足级别约束，区域数量不强制满足约束

**水滴退化详细参数：**

| 强度等级 | 数量 | 总覆盖面积 | RGB 退化参数 | T 退化参数 |
|---------|------|-----------|-------------|-----------|
| 低 | 1~3 个 | <10% | 模糊核 3×3, 亮度衰减 0.85 | 衰减 0.5 |
| 中 | 3~6 个 | 10%~30% | 模糊核 5×5, 亮度衰减 0.7 | 衰减 0.3 |
| 高 | 5~10 个 | 30%~40% | 模糊核 7×7, 亮度衰减 0.55 | 衰减 0.15 |

**污渍退化详细参数：**

| 强度等级 | 总覆盖面积 | 不透明度 (α) | RGB 替换值 | T 替换值 |
|---------|-----------|-------------|-----------|---------|
| 低 | <5% | 0.3~0.5 | 暗灰色 [40,40,40] | 环境温度 ×0.7 |
| 中 | 5%~10% | 0.5~0.8 | 深灰色 [20,20,20] | 环境温度 ×0.4 |
| 高 | 10%~20% | 0.8~1.0 | 黑色 [0,0,0] | 环境温度 ×0.1 |

**热晕现象详细参数：**

热晕模拟高温物体在红外图像中产生的径向亮度扩散效应。通过检测高温区域（像素值>200）或随机生成中心坐标，以高斯衰减公式叠加亮度增益。

| 强度等级 | 中心数量 | 高斯 σ（像素） | 等效半径（3σ） | 峰值增益 A | 视觉表现 |
|---------|---------|---------------|--------------|-----------|---------|
| 低 | 1~2 个 | 4~6 | 12~18 px | 30~50 | 高温物体周围轻微亮边 |
| 中 | 2~4 个 | 8~12 | 24~36 px | 60~90 | 明显光晕，可能覆盖紧邻小物体 |
| 高 | 3~6 个 | 15~20 | 45~60 px | 100~140 | 大面积光晕，严重干扰周围区域 |

### 1.3 模态缺失

| 退化类型 | 标识名 | 说明 |
|---------|--------|------|
| RGB 缺失 | `RGBMissingDegradation` | RGB 三通道全部置零 |
| 热红外缺失 | `ThermalMissingDegradation` | 热红外三通道全部置零 |

## 2. 测试场景

鲁棒性测试共 **9 个场景**：

| 序号 | 场景名 | 退化类 | 参数 | 说明 |
|-----|--------|-------|------|------|
| 1 | `clean` | CleanDegradation | {} | 干净基线 |
| 2 | `emm_rgb_missing` | RGBMissingDegradation | {} | RGB 模态缺失 |
| 3 | `emm_thermal_missing` | ThermalMissingDegradation | {} | 热红外模态缺失 |
| 4 | `global_low` | GlobalDegradation | deg_type=random, intensity=low | 随机全局退化（低强度） |
| 5 | `global_medium` | GlobalDegradation | deg_type=random, intensity=medium | 随机全局退化（中强度） |
| 6 | `global_high` | GlobalDegradation | deg_type=random, intensity=high | 随机全局退化（高强度） |
| 7 | `local_low` | LocalDegradation | deg_type=random, intensity=low | 随机局部退化（低强度） |
| 8 | `local_medium` | LocalDegradation | deg_type=random, intensity=medium | 随机局部退化（中强度） |
| 9 | `local_high` | LocalDegradation | deg_type=random, intensity=high | 随机局部退化（高强度） |

全局/局部退化使用 `deg_type='random'`，每次测试时随机选择一种退化类型和模态。

## 3. 日志目录结构

```
eval/robustness/
└── <version>/                          # --version 参数指定，默认 original
    └── <YYYYMMDD_HHMMSS>/             # 测试时间戳
        ├── _model_load/                # 模型加载临时目录
        ├── robustness_results.json     # 所有场景汇总结果
        ├── clean/                      # 各场景子目录
        │   └── metrics.json
        ├── emm_rgb_missing/
        │   └── metrics.json
        ├── emm_thermal_missing/
        │   └── metrics.json
        ├── global_low/
        │   └── metrics.json
        ├── global_medium/
        │   └── metrics.json
        ├── global_high/
        │   └── metrics.json
        ├── local_low/
        │   └── metrics.json
        ├── local_medium/
        │   └── metrics.json
        └── local_high/
            └── metrics.json
```

当启用 `--visualize` 时，每个场景子目录还会包含 TensorBoard 日志和可视化图像。

## 4. 使用方式

### 4.1 全场景批量测试

运行所有 9 个鲁棒性场景：

```bash
python tools/test_robustness.py \
    configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py \
    work_dirs/mae-base_upernet_rgbt_v1_baseline/best_mIoU_iter_xxx.pth \
    --batch-all \
    --version original
```

### 4.2 单场景测试

测试指定退化类型：

```bash
# 干净基线
python tools/test_robustness.py <config> <checkpoint> \
    --degradation CleanDegradation

# RGB 模态缺失
python tools/test_robustness.py <config> <checkpoint> \
    --degradation RGBMissingDegradation

# 全局退化 - 运动模糊（高强度）
python tools/test_robustness.py <config> <checkpoint> \
    --degradation GlobalDegradation \
    --deg-kwargs deg_type=motion_blur intensity=high dual_prob=0.1

# 局部退化 - 水滴（中强度）
python tools/test_robustness.py <config> <checkpoint> \
    --degradation LocalDegradation \
    --deg-kwargs deg_type=water_stain intensity=medium dual_prob=0.1

# 局部退化 - 热晕（高强度）
python tools/test_robustness.py <config> <checkpoint> \
    --degradation LocalDegradation \
    --deg-kwargs deg_type=thermal_halo intensity=high dual_prob=0.1

# 局部退化 - 随机类型（低强度）
python tools/test_robustness.py <config> <checkpoint> \
    --degradation LocalDegradation \
    --deg-kwargs deg_type=random intensity=low dual_prob=0.1
```

### 4.3 带可视化测试

python tools/test_robustness.py ./configs/mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-480x480.py /public/home/nwpu_liyl/code/QWSEG/seg/mmsegmentation-main-rgbt/work_dirs/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_fmb-480x480/epoch_20.pth \
    --batch-all \
    --visualize \
    --num-vis-samples 4

CUDA_VISIBLE_DEVICES=1 python tools/test_robustness.py ./configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-480x480.py  /public/home/nwpu_liyl/code/QWSEG/seg/mmsegmentation-main-rgbt/work_dirs/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-480x480/epoch_80.pth \
    --batch-all \
    --visualize \
    --num-vis-samples 4

```bash
python tools/test_robustness.py <config> <checkpoint> \
    --batch-all \
    --visualize \
    --num-vis-samples 8 \
    --version lora_v1
```

### 4.4 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `config` | 训练配置文件路径 | 必填 |
| `checkpoint` | 模型权重文件路径 | 必填 |
| `--work-dir` | 结果保存目录 | `eval/robustness/` |
| `--degradation` | 退化类名（单场景模式） | `CleanDegradation` |
| `--deg-kwargs` | 退化类参数，如 `deg_type=motion_blur intensity=high` | 无 |
| `--batch-all` | 运行全部 9 个场景 | 关闭 |
| `--visualize` | 启用 TensorBoard 可视化 | 关闭 |
| `--num-vis-samples` | 可视化样本数 | 4 |
| `--version` | 模型版本名（用于目录命名） | `original` |
| `--seed` | 随机种子 | 42 |

## 5. 退化类参数速查

### GlobalDegradation

| 参数 | 类型 | 可选值 | 说明 |
|-----|------|-------|------|
| `deg_type` | str | `random`, `motion_blur`, `gaussian_noise`, `salt_pepper`, `stripe_noise`, `thermal_contrast` | 退化类型，`random` 随机选择 |
| `intensity` | str | `low`, `medium`, `high` | 强度等级 |
| `dual_prob` | float | 0.0~1.0 | 双模态同时退化概率（默认 0.1） |
| `seed` | int | 任意 | 随机种子（可选） |

### LocalDegradation

| 参数 | 类型 | 可选值 | 说明 |
|-----|------|-------|------|
| `deg_type` | str | `random`, `local_overexposure`, `local_lowlight`, `water_stain`, `stain`, `local_bad_block`, `local_gaussian_noise`, `thermal_halo` | 退化类型，`random` 随机选择 |
| `intensity` | str | `low`, `medium`, `high` | 强度等级 |
| `dual_prob` | float | 0.0~1.0 | 双模态同时退化概率（默认 0.1） |
| `seed` | int | 任意 | 随机种子（可选） |
