# QWSEG 环境配置指南

## 版本兼容性说明

本项目基于 MMSegmentation 二次开发，依赖 mmcv、mmengine 等 OpenMMLab 生态包。**mmcv 的预编译 CUDA 扩展对 PyTorch 版本非常敏感**，必须严格匹配，否则会出现 `undefined symbol` 错误。

### 推荐环境组合

| 组件 | 推荐版本 | 备注 |
|------|---------|------|
| Python | 3.10 | 稳定兼容 |
| PyTorch | 2.4.0 | mmcv 有预编译包 |
| CUDA Toolkit | 12.1 | 匹配 PyTorch 2.4 |
| mmcv | 2.1.0 | 预编译包，无需本地编译 |
| mmengine | 0.10.4 | 兼容 mmcv 2.1.0 |
| mmseg | 开发模式安装 | 本项目即为 mmseg |

### 其他可用组合

| PyTorch | CUDA | mmcv 安装源 |
|---------|------|-------------|
| 2.1.0 | 11.8 | `cu118/torch2.1` |
| 2.1.0 | 12.1 | `cu121/torch2.1` |
| 2.2.0 | 11.8 | `cu118/torch2.2` |
| 2.2.0 | 12.1 | `cu121/torch2.2` |
| 2.3.0 | 12.1 | `cu121/torch2.3` |
| 2.4.0 | 12.1 | `cu121/torch2.4` |

> **不推荐** PyTorch 2.5+ 或 CUDA 13.0，mmcv 目前没有对应的预编译包，需要从源码编译，容易出错。

---

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n qwseg python=3.10 -y
conda activate qwseg
```

### 2. 安装 PyTorch

```bash
# 推荐：PyTorch 2.4 + CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

如果服务器 CUDA 驱动不支持 12.1，使用 CUDA 11.8：

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 mmcv

**必须与 PyTorch + CUDA 版本严格匹配**：

```bash
# PyTorch 2.4 + CUDA 12.1
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# PyTorch 2.1 + CUDA 11.8
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

> 如果下载慢，可以先设置 pip 镜像：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
> 但 mmcv 预编译包必须从 OpenMMLab 官方源下载，镜像中没有。

### 4. 安装 mmengine

```bash
pip install mmengine==0.10.4
```

### 5. 安装项目依赖

```bash
pip install matplotlib numpy packaging prettytable scipy
pip install timm ftfy regex
pip install opencv-python-headless
```

### 6. 安装本项目（开发模式）

```bash
cd /path/to/QWSEG/seg/mmsegmentation-main-rgbt
pip install -e .
```

开发模式安装后，修改代码无需重新安装即可生效。

### 7. 验证安装

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())

import mmcv
print('mmcv:', mmcv.__version__)

import mmengine
print('mmengine:', mmengine.__version__)

import mmseg
print('mmseg:', mmseg.__version__)

from mmseg.models.segmentors.mitmul_v6_disentangle import (
    MiTMulV6Baseline, MiTMulV6Disentangle,
    MiTMulV7Degradation, MiTMulV7DegradationFull)
print('MiTMul models import OK')

from mmseg.models.segmentors.mitmul_v8_quality_pyramid import MiTMulV8QualityPyramid
print('MiTMulV8QualityPyramid import OK')

print('All checks passed!')
"
```

---

## 常见问题

### Q1: `undefined symbol` 错误

```
ImportError: mmcv/_ext.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops...
```

**原因**：mmcv 预编译包与当前 PyTorch 版本不匹配。

**解决**：卸载后重新安装匹配版本的 mmcv：

```bash
pip uninstall mmcv mmcv-full -y
# 然后按照上面的版本对应表重新安装
```

### Q2: `No module named 'pkg_resources'`

**原因**：缺少 setuptools。

**解决**：

```bash
pip install setuptools wheel
```

### Q3: `Failed to import mmseg.models.segmentors.mitmul_v6_disentangle`

**原因**：PYTHONPATH 未包含项目目录。

**解决**：

```bash
export PYTHONPATH="/path/to/QWSEG/seg/mmsegmentation-main-rgbt:$PYTHONPATH"
```

或写入 `.bashrc` 永久生效：

```bash
echo 'export PYTHONPATH="/path/to/QWSEG/seg/mmsegmentation-main-rgbt:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### Q4: 临时跳过 mmcv CUDA 扩展

如果暂时无法安装匹配的 mmcv，可以设置环境变量跳过 CUDA ops：

```bash
export MMCV_WITH_OPS=0
```

Segformer 架构不依赖 mmcv 的 CUDA 算子，此方案可以正常训练，但部分 ops 会回退到纯 PyTorch 实现，速度稍慢。

### Q5: 服务器 CUDA 驱动版本过低

查看驱动支持的 CUDA 版本：

```bash
nvidia-smi
# 右上角显示 "CUDA Version: XX.X"
```

nvidia-smi 显示的 CUDA 版本是驱动支持的最高版本，PyTorch 的 CUDA 版本不能超过它。例如驱动显示 CUDA 11.8，则只能安装 `cu118` 的 PyTorch。

---

## 数据集准备

### MFNet

```bash
# 数据目录结构
MFNet/
├── images/
│   ├── 00000.png    # 4通道 RGBT 图像 (RGB + Thermal)
│   ├── 00001.png
│   └── ...
├── labels/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

### FMB

```bash
# 数据目录结构
FMB_ALL/
├── FMB/                    # RGB 图像
│   └── images/
│       ├── training/
│       └── validation/
└── FMB_T/                  # Thermal 图像（路径替换）
    └── images/
        ├── training/
        └── validation/
```

### PST900

```bash
# 数据目录结构
PST900/
├── rgb/
│   ├── train/
│   └── test/
├── thermal/
│   ├── train/
│   └── test/
└── labels/
    ├── train/
    └── test/
```

---

## 预训练权重

下载 Segformer 预训练权重到 `pretrain/` 目录：

```bash
mkdir -p pretrain

# MIT-B2（通用 backbone）
wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_212103-521ea855.pth -O pretrain/segformer_mit-b2_512x512_160k_ade20k.pth

# MIT-B0（私有分支）
wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9602.pth -O pretrain/segformer_mit-b0_512x512_160k_ade20k.pth
```

---

## 训练命令示例

```bash
cd /path/to/QWSEG/seg/mmsegmentation-main-rgbt

# V6 基线 - MFNet
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/segformer/mitmul_v6_baseline_mit-b2_1xb2-40K_mfnet-480x640.py \
    --work-dir work_dirs/v6_baseline_mfnet_480x640

# V6 解耦 - FMB
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/segformer/mitmul_v6_disentangle_mit-b2-b0_1xb2-40K_fmb-480x640.py \
    --work-dir work_dirs/v6_disentangle_fmb_480x640

# V7 退化 - MFNet
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/segformer/mitmul_v7_degradation_mit-b2-b0_1xb2-40K_mfnet-480x640.py \
    --work-dir work_dirs/v7_degradation_mfnet_480x640

# V7 退化完整分割 - FMB
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/segformer/mitmul_v7_degradation_full_mit-b2-b0_1xb2-40K_fmb-480x640.py \
    --work-dir work_dirs/v7_degradation_full_fmb_480x640
```

### 多卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
    configs/segformer/mitmul_v6_baseline_mit-b2_1xb2-40K_mfnet-480x640.py \
    2 --work-dir work_dirs/v6_baseline_mfnet_480x640
```

### 恢复训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/segformer/mitmul_v6_baseline_mit-b2_1xb2-40K_mfnet-480x640.py \
    --work-dir work_dirs/v6_baseline_mfnet_480x640 \
    --resume work_dirs/v6_baseline_mfnet_480x640/latest.pth
```
