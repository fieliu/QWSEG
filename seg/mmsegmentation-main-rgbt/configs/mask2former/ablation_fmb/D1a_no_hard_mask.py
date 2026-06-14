# FMB Ablation D1a: remove hard mask (keep soft gating). See MFNet D1a.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(use_hard_mask=False)
