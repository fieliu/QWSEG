# FMB Ablation M3: remove private dual-branch. See MFNet M3.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(use_private_branch=False)
