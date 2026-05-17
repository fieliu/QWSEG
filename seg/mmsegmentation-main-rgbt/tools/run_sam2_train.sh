#!/bin/bash

# ================================================
# SAM2 + LoRA + MLP融合 + 文本解码器 训练脚本
# ================================================

# ---------------------- 基础配置 ----------------------
GPUS=1
BATCH_SIZE=2
CONFIG="configs/sam2/sam2-hiera-l_rgbt_v1_sam2_lora-text-decoder_8xb2-amp-200e_mfnet-480x640.py"
WORK_DIR="./work_dirs/sam2_hiera_l_lora_mfnet"

# ---------------------- CLIP特征提取 ----------------------
CLIP_FEATURES_DIR="/home/lh/code/data/MFNet/text_embeddings"
CLIP_FEATURES_PATH="${CLIP_FEATURES_DIR}/mf_class_embedding.pt"

if [ ! -f "$CLIP_FEATURES_PATH" ]; then
    echo "CLIP文本特征文件不存在，正在提取..."
    python tools/extract_clip_features.py --dataset mf --clip_model ViT-L/14 --output_dir "$CLIP_FEATURES_DIR"
fi

# ---------------------- SAM2预训练权重 ----------------------
SAM2_CKPT="/home/lh/code/SHIFNet-main/checkpoints/sam2.1_hiera_large.pt"

# ---------------------- 切换到项目目录 ----------------------
cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt

# ---------------------- 构建命令 ----------------------
if [ "$GPUS" -eq 1 ]; then
    CMD="python tools/train.py ${CONFIG} --work-dir ${WORK_DIR}"
else
    CMD="bash ./tools/dist_train.sh ${CONFIG} ${GPUS} --work-dir ${WORK_DIR}"
fi

# ---------------------- 打印信息 ----------------------
echo "============================================"
echo " SAM2 + LoRA + MLP + TextDecoder 训练配置"
echo "============================================"
echo "配置文件:      ${CONFIG}"
echo "工作目录:      ${WORK_DIR}"
echo "GPU数量:       ${GPUS}"
echo "批次大小:      ${BATCH_SIZE}"
echo "CLIP特征:      ${CLIP_FEATURES_PATH}"
echo "SAM2权重:      ${SAM2_CKPT:-无}"
echo "============================================"
echo ""
echo "执行命令:"
echo "$CMD"
echo ""

# ---------------------- 执行训练 ----------------------
$CMD
