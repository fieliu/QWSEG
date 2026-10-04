# Prepare disjoint train.txt / val.txt / test.txt before training.
# Keep the legacy MFNet configs unchanged for other experiments.
_base_ = ['../_base_/datasets/mfnet_480x640.py']

val_dataloader = dict(dataset=dict(ann_file='val.txt'))
