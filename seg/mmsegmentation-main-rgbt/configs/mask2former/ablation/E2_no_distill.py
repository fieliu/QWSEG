# Edge-loss probe E2: remove distillation loss.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(loss_distill_weight=0.0)
