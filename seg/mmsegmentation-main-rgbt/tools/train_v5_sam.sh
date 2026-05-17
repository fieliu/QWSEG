#!/bin/bash
# Train V5 SAM QualityJoint on FMB dataset
# SAM ViT-B + LoRA + Disentangle + Degradation + Quality Network Joint Fine-tuning

CONFIG=sam-base_upernet_rgbt_v5_sam_quality_joint_8xb2-amp-300e_fmb-480x480
WORK_DIR=./work_dirs/sam_v5_quality_joint_fmb

python tools/train.py \
    configs/sam/${CONFIG}.py \
    --work-dir ${WORK_DIR} \
    --amp
