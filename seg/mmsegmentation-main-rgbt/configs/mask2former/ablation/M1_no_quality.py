# Ablation M1: remove quality-aware mechanism.
# Disables quality predictors, attention bias, hard mask, quality-weighted
# fusion. Chained off in code: cascade, hard_mask, retention/align/invariant
# losses. Common fusion degrades to simple average; private to direct add.
# => same-parameter three-branch naive-fusion baseline (also answers the
#    "is it just more parameters?" reviewer question).
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(use_quality=False)
