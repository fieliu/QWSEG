#!/bin/bash
# Train V1 SAM Baseline on FMB dataset
# SAM ViT-B + LoRA + Feature Addition Fusion

CONFIG=sam-base_upernet_rgbt_v1_sam_baseline_8xb2-amp-200e_fmb-480x480
WORK_DIR=./work_dirs/sam_v1_baseline_fmb

python tools/train.py \
    configs/sam/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
