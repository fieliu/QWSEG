#!/bin/bash
# Train V4 SAM QualityPruning on FMB dataset
# SAM ViT-B + LoRA + Disentangle + Degradation + Quality Network

CONFIG=sam-base_upernet_rgbt_v4_sam_quality_pruning_8xb2-amp-300e_fmb-480x480
WORK_DIR=./work_dirs/sam_v4_quality_pruning_fmb

python tools/train.py \
    configs/sam/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
