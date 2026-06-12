# Ablation M4: remove cross-stage consistency (cascade + retention).
# Disables cascade_quality_suppress and the retention loss together (they
# form one stability mechanism). Quality scores become per-stage raw values.
# Fusion formula unchanged; only the score values fed downstream differ.
# => contribution of the cross-stage stability mechanism.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(use_cascade=False, retention_loss_weight=0.0)
