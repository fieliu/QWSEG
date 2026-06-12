# Ablation D1a: remove hard mask only (keep soft gating: attn bias +
# quality-weighted fusion). Answers "is the hard mask necessary?".
# Prediction: clean ~unchanged, missing-modality may drop.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(use_hard_mask=False)
