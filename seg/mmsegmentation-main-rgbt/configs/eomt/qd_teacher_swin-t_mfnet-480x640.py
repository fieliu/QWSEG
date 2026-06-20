# Teacher: clean dual-branch shared-Swin Mask2Former, concat+reduce fusion.
# Train this FIRST on clean RGB-T; its checkpoint feeds the student.
_base_ = ['../mask2former/swinmul_v6_mask2former_swin-t_1xb2-80K_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.quality_distill_teacher',
             'mmseg.models.segmentors.swinmul_v6_mask2former',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

# only swap the segmentor type + add concat-reduce fusion dims; everything
# else (backbone / decode_head / schedule / dataset) inherited from V6.
model = dict(
    type='QualityDistillTeacher',
    fusion_dims=(96, 192, 384, 768),
)

# ---- Epoch-based schedule (aligned with the DINOv3 EoMT configs) ----
# The V6 base is iter-based (80k iters) which logs a broken "Epoch [1][N/588]"
# counter. Switch to a clean epoch-based loop. MFNet train = 588 iters/epoch
# (batch 2), so 100 epochs = 58800 iters. The LR scheduler stays in ITER
# units (epoch-loop + iter-scheduler is what the DINOv3 configs use too), so
# the warmup/decay curve is identical to the original.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=1500, end=58800,
         by_epoch=False),
]
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=100, val_interval=5)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,
                    save_best='mIoU'))

# teacher is the clean-trained, no-quality baseline -> also report its
# missing-modality mIoU as the contrast the student is measured against.
# (base V6 only ships TrainVisHook; append the eval hook, keep vis.)
custom_hooks = [
    dict(type='TrainVisHook', interval=5, num_samples=2),
    dict(type='MissingModalityEvalHook', interval=5),
]
