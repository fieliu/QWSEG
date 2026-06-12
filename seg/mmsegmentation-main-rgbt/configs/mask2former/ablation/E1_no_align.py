# Edge-loss probe E1: remove cross-modal contrastive (align) loss.
# Not a main-table row; used to decide whether to delete this term from the
# final model. If clean/missing drop < 0.3 mIoU, drop it.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(loss_align_weight=0.0)
