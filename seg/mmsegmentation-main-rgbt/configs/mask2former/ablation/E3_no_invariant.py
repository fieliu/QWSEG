# Edge-loss probe E3: remove invariant loss.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(loss_invariant_weight=0.0)
