# Ablation M2: remove degradation-robust training.
# Disables the second (degraded) forward pass and all degradation losses.
# Chained off in code: distill, invariant. Clean forward & fusion unchanged.
# => measures the contribution of degradation simulation training.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(use_degradation=False)
