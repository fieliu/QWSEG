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
data_root = '/root/autodl-tmp/ade_mf/'
dataset_type = 'MSRS_ADE20KDataset'

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75
]
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            4,
            6,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='BIMixVisionTransformer'),
    data_preprocessor=dict(
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
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
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
            dict(type='RegionL1', loss_weight=1.0, tau=1.0, N_cls=9),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=9,
        type='SegformerHead'),
    pretrained='/root/RXDistill/pretrain/bit_seg_mit_b2.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder_mult')



norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')

param_scheduler = [
    dict(
        T_max=100,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='rgbx_np/validation',
            seg_map_path='annotations/validation'),
        data_root='/root/autodl-tmp/ade_mf/',
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(keep_ratio=True, scale=(
                480,
                640,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MSRS_ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='multi_IoUMetric')

test_pipeline = [
    dict(type='LoadImageFromFile_rgbx'),
    dict(keep_ratio=True, scale=(
        480,
        640,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_cfg = dict(
    dynamic_intervals=[
        (
            80,
            2,
        ),
    ],
    max_epochs=300,
    type='EpochBasedTrainLoop',
    val_interval=5)

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='rgbx_np/training', seg_map_path='annotations/training'),
        data_root='/root/autodl-tmp/ade_mf/',
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    1.75,
                ),
                scale=(
                    480,
                    640,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    480,
                    640,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='MSRS_ADE20KDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile_rgbx'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='rgbx_np/validation',
            seg_map_path='annotations/validation'),
        data_root='/root/autodl-tmp/ade_mf/',
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(keep_ratio=True, scale=(
                480,
                640,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='MSRS_ADE20KDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

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


work_dir = './work_dirs\\MF'
