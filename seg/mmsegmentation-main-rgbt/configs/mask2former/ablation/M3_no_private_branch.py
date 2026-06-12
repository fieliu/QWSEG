# Ablation M3: remove private dual-branch.
# Drops both private Swin branches + their quality predictors + their aux
# heads. final_fusion is bypassed; the quality-weighted common fusion (zf)
# becomes the final feature. => contribution of the three-branch design
# (also a parameter-fairness reference point).
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(use_private_branch=False)
