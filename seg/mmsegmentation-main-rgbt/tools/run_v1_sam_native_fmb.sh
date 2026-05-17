#!/bin/bash
# 一键启动脚本：下载权重 + 训练 V1 SAM Native FMB
# 使用方法：bash tools/run_v1_sam_native_fmb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PRETRAIN_DIR="$PROJECT_DIR/pretrain"

echo "=========================================="
echo "  V1 SAM Native - FMB Training Pipeline"
echo "=========================================="
echo "Project dir: $PROJECT_DIR"
echo "Pretrain dir: $PRETRAIN_DIR"

mkdir -p "$PRETRAIN_DIR"

# 1. 下载 SAM ViT-B 权重
SAM_CKPT="$PRETRAIN_DIR/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_CKPT" ]; then
    echo "[1/3] Downloading SAM ViT-B checkpoint..."
    wget -O "$SAM_CKPT" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    echo "SAM checkpoint downloaded."
else
    echo "[1/3] SAM checkpoint already exists."
fi

# 2. 下载 CLIP ViT-B/16 权重
CLIP_CKPT="$PRETRAIN_DIR/clip_vit_base_patch16_224.pth"
if [ ! -f "$CLIP_CKPT" ]; then
    echo "[2/3] Downloading CLIP ViT-B/16 checkpoint..."
    wget -O "$CLIP_CKPT" https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth
    echo "CLIP checkpoint downloaded."
else
    echo "[2/3] CLIP checkpoint already exists."
fi

# 3. 启动训练
echo "[3/3] Starting training..."
cd "$PROJECT_DIR"

CONFIG=configs/sam_native/sam-base_native_rgbt_v1_sam_native_8xb2-amp-200e_fmb-480x480.py
WORK_DIR=./work_dirs/sam_native_v1_fmb

python tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --amp

echo "Training complete!"
