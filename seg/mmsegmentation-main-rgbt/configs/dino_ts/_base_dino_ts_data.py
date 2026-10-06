# The final protocol trains on the official train+val samples and evaluates on
# the held-out official test split.  Generate trainval.txt with
# tools/prepare_mfnet_trainval.py before training.
# Keep the legacy MFNet configs unchanged for other experiments.
_base_ = ['../_base_/datasets/mfnet_480x640.py']

mfnet_root = '{{$MFNET_ROOT:/root/autodl-tmp/data/MFNet}}'
train_dataloader = dict(
    batch_size=1, dataset=dict(data_root=mfnet_root, ann_file='trainval.txt'))
val_dataloader = dict(
    dataset=dict(data_root=mfnet_root, ann_file='test.txt'))
test_dataloader = dict(
    dataset=dict(data_root=mfnet_root, ann_file='test.txt'))
