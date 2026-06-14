# FMB Ablation M2: remove degradation-robust training. See MFNet M2.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(use_degradation=False)
