# FMB Ablation M5: remove three-phase curriculum. See MFNet M5.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(skip_phases=True)
