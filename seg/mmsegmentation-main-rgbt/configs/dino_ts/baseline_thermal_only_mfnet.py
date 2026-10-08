# Thermal-only baseline (100 epochs, MFNet).
#
# Single-modality control at the same budget as the RGB-T runs: same data
# (MFNet trainval, 1176), same optimizer/scheduler, same seed, same 100-epoch
# schedule, so the only difference from the Dense RGB-T model is which modality
# the backbone sees.  Together these give the reference points the fusion
# results have to be read against:
#
#   rgb_only       -> what RGB alone reaches
#   thermal_only   -> what thermal alone reaches; the number rgb_missing must be
#                     compared against (comparing it to an RGB-only baseline
#                     would be comparing two different modalities)
#
# The best checkpoint of each is the baseline.
_base_ = ['stage2a_dense_mfnet.py']

crop_size = (480, 640)

model = dict(
    forward_mode='thermal_only',
    # Both adapters are disabled so the two single-modality baselines have an
    # identical structure and an identical starting point: the pretrained
    # DINOv3 weights, with the thermal branch seeded by copying the RGB
    # patch-embed convolution.  The only variable left is which modality the
    # backbone sees.  (No Stage-1 transfer is applied either -- that would give
    # thermal an advantage RGB does not get.)
    backbone=dict(rgb_adapter_identity=True, thermal_adapter_identity=True),
    # No fusion/Extra path is exercised in a single-stream run, so the degraded
    # and missing-modality objectives are irrelevant here.
    lambda_deg=0.0,
    lambda_missing=0.0)

randomness = dict(seed=471605737, deterministic=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=100,
         by_epoch=True, convert_to_iter_based=True),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best='mIoU', rule='greater'),
    visualization=dict(type='SegVisualizationHook', draw=False))
