#!/bin/bash
# Train V3 SAM Degradation on FMB dataset
# SAM ViT-B + LoRA + Disentangle + Online Degradation Augmentation

CONFIG=sam-base_upernet_rgbt_v3_sam_degradation_8xb2-amp-300e_fmb-480x480
WORK_DIR=./work_dirs/sam_v3_degradation_fmb

python tools/train.py \
    configs/sam/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
