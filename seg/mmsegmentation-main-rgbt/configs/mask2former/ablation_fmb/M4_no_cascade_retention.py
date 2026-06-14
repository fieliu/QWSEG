# FMB Ablation M4: remove cross-stage consistency (cascade + retention). See MFNet M4.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_fmb-480x640.py']
model = dict(use_cascade=False, retention_loss_weight=0.0)
