"""One-stream control for verifying DINOv3 + decoder before RGB-T training."""
_base_ = ['stage2a_dense_mfnet.py']

model = dict(forward_mode='rgb_only')

# Validate every epoch while diagnosing the pipeline.  This config is an
# ablation/control, not a final RGB-T result.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=1,
        max_keep_ckpts=1, save_last=True, save_best='mIoU', rule='greater'),
    visualization=dict(type='SegVisualizationHook', draw=False))
