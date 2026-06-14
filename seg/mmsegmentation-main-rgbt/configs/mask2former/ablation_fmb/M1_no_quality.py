# FMB Ablation M1: remove quality-aware mechanism. See MFNet M1 for details.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(use_quality=False)
