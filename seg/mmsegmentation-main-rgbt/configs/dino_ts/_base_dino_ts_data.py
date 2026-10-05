# Prepare disjoint train.txt / val.txt / test.txt before training.
# Keep the legacy MFNet configs unchanged for other experiments.
_base_ = ['../_base_/datasets/mfnet_480x640.py']

mfnet_root = '{{$MFNET_ROOT:/root/autodl-tmp/data/MFNet}}'
train_dataloader = dict(
    batch_size=1, dataset=dict(data_root=mfnet_root))
val_dataloader = dict(
    dataset=dict(data_root=mfnet_root, ann_file='val.txt'))
test_dataloader = dict(
    dataset=dict(data_root=mfnet_root, ann_file='test.txt'))
