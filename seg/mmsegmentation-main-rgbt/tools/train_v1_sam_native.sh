#!/bin/bash
# Train V1 SAM Native on FMB dataset
# SAM ViT-B + LoRA + SAM Native Decoder (LoRA on qv) + CLIP Text Encoder + Auto Point Prompts

CONFIG=sam-base_native_rgbt_v1_sam_native_8xb2-amp-200e_fmb-480x480
WORK_DIR=./work_dirs/sam_native_v1_fmb

python tools/train.py \
    configs/sam_native/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
