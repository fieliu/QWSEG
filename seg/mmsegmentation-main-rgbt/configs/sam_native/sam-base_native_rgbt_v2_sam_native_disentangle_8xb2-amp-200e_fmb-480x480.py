"""
RGB-T语义分割 - V2 SAM Native Disentangle (SAM ViT-B + LoRA + 解耦 + SAM原生解码器 + CLIP文本引导) - FMB 480x480

架构说明：
- 通用编码器：SAM ViT-B (ImageEncoderViT)，RGB和Thermal分别输入
- 私有分支：LightweightMAEBranch (ViT-Tiny, 192d)，分别提取RGB/T私有特征
- 微调方式：LoRA (rank=4, alpha=4.0) - backbone qkv,proj; decoder q_proj,v_proj
- 融合：通用特征相加 + 交叉注意力融合私有特征
- 解码器：SAM原生TwoWayTransformer解码器，融合自动生成的点提示
- 文本引导：CLIP文本编码器编码类名
- 损失：CE + Dice + 解耦损失(HSIC) + 模态对比损失(InfoNCE) + 不变损失
- 数据集：FMB_ALL (15类)
- 输入尺寸：480x480
"""

_base_ = [
    '../_base_/datasets/fmb_480x480.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.rgbt_v2_sam_native',
             'mmseg.models.backbones.sam_vit',
             'mmseg.models.backbones.lightweight_mae_branch',
             'mmseg.models.text_encoder.clip_text_encoder',
             'mmseg.datasets.fmb',
             'mmseg.datasets.transforms.loading'],
    allow_failed_imports=False)

crop_size = (480, 480)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

fmb_class_names = [
    'Background', 'Road', 'Sidewalk', 'Building',
    'Traffic Light', 'Traffic Sign', 'Vegetation', 'Sky',
    'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pole']

model = dict(
    type='RGBTv2SAMNative',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SAMViT',
        img_size=480,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
        out_indices=(3, 5, 7, 11),
        pretrained='./pretrain/sam_vit_b_01ec64.pth'),
    private_branch_rgb=dict(
        type='LightweightMAEBranch',
        img_size=480,
        patch_size=16,
        in_channels=3,
        embed_dims=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        out_indices=(3, 5, 7, 11)),
    private_branch_t=dict(
        type='LightweightMAEBranch',
        img_size=480,
        patch_size=16,
        in_channels=3,
        embed_dims=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        out_indices=(3, 5, 7, 11)),
    num_classes=15,
    image_size=480,
    prompt_embed_dim=256,
    decoder_depth=2,
    decoder_num_heads=8,
    decoder_mlp_dim=2048,
    num_multimask_outputs=3,
    clip_embed_dim=768,
    clip_checkpoint='/home/lh/code/ViT-L-14.pt',
    class_names=fmb_class_names,
    point_grid_size=32,
    use_lora_backbone=True,
    lora_rank=4,
    lora_alpha=4.0,
    lora_dropout=0.1,
    lora_target_modules=['q_proj', 'v_proj', 'proj'],
    use_lora_decoder=True,
    decoder_lora_rank=4,
    decoder_lora_alpha=4.0,
    decoder_lora_dropout=0.0,
    freeze_backbone_non_lora=True,
    freeze_decoder_non_lora=True,
    freeze_prompt_encoder=True,
    freeze_clip=True,
    train_patch_embed=True,
    loss_seg_zc_weight=0.3,
    loss_modal_weight=0.2,
    loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
    loss_invariance_weight=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'lora_A': dict(lr_mult=1.0),
            'lora_B': dict(lr_mult=1.0),
            'backbone.patch_embed': dict(lr_mult=0.1),
            'mask_decoder': dict(lr_mult=1.0),
            'channel_proj': dict(lr_mult=1.0),
            'private_branch_rgb': dict(lr_mult=1.0),
            'private_branch_t': dict(lr_mult=1.0),
        }))

warmup_epochs = 4
max_epochs = 200

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=warmup_epochs),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_epochs,
        end=max_epochs,
        by_epoch=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))

fp16 = dict(loss_scale='dynamic')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=True)

train_dataloader = dict(batch_size=2, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

test_pipeline = [
    dict(type='LoadRGBTImageFromFile', ir_replace_src='FMB_ALL/FMB', ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='Resize', scale=(1600, 480), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
