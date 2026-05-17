norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        480,
        640,
    ),
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
model = dict(
    type='EncoderDecoder_mult',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            123.675, 116.28, 103.53, 0, 0, 0
        ],
        std=[58.395, 57.12, 57.375, 1, 1, 1],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(480,640)),
    pretrained='/root/RXDistill/pretrain/bit_seg_mit_b2.pth',
    backbone=dict(
        type='BIMixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head =dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=14,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=[
            dict(loss_weight=0.5, type='CrossEntropyLoss', use_sigmoid=False),
        ],
        loss_decode_modal=[
            dict(loss_weight=1.0, type='M_CrossEntropyLoss', use_sigmoid=False),
        ],
        loss_decode_akd=[
            dict(loss_weight=0.5, type='AKDLoss')
        ],
        loss_decode_head=[
            dict(type='RegionL1', loss_weight=1.0, tau=1.0, N_cls=14),
        ],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'FMB_ADE20KDataset'
data_root = '/root/autodl-tmp/ade_fmb/'

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='FMB_ADE20KDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='rgbx_np/training', seg_map_path='annotations/training'),
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(
                type='RandomResize',
                scale=(480,640),
                ratio_range=(0.5, 2),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(480,640), cat_max_ratio=0.75),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='FMB_ADE20KDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='rgbx_np/validation',
            seg_map_path='annotations/validation'),
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(type='Resize', scale=(480, 640), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='PackSegInputs')
        ]))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

_test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    format_only=True,
    output_dir='work_dirs/format_results')

_test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='FMB_ADE20KDataset',
        data_root=data_root,
        data_prefix=dict(img_path='rgbx_np/validation',
                         seg_map_path='annotations/validation'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(480, 640), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=True)

log_level = 'INFO'
resume = False
tta_model = dict(type='SegTTAModel')


optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))),
    loss_scale='dynamic')

param_scheduler = [
    dict(
        T_max=5,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=200,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            180,
            1,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='multi_IoUMetric')


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(name='run_eaef_b2', project='segformer-eaef'),
        type='swanlabVisBackend'),
]

visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=vis_backends)

default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=10,
        rule='greater',
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, interval=1, type='SegVisualizationHook'))

launcher = 'none'
work_dir = './work_dirs\\FMB'
