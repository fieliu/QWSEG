#!/bin/bash
# Train V2 SAM Disentangle on FMB dataset
# SAM ViT-B + LoRA + Universal/Private Disentangle + Cross-Attention Fusion

CONFIG=sam-base_upernet_rgbt_v2_sam_disentangle_8xb2-amp-200e_fmb-480x480
WORK_DIR=./work_dirs/sam_v2_disentangle_fmb

python tools/train.py \
    configs/sam/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
