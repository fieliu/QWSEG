# DINOv3 + ViT-Adapter + Mask2Former: RGB-T dual-modal on MFNet.
# Following official ViT-Adapter config pattern.
# Two adapter branches share DINOv3 backbone, fused via CrossAttn, decoded by M2F.
# Uses Mask2FormerRGBTCrossAttn (cross-attention fusion, no quality gating).
# This is the fair Clean baseline: cross-attention already present,
# so the quality version's increment is PURELY from quality-gating.
_base_ = [
    '../_base_/datasets/mfnet_480x640.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.mask2former_rgbt_crossattn',
             'mmseg.models.backbones.dinov3_adapter',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

crop_size = (480, 640)
num_classes = 9
embed_dim = 768  # ViT-B

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='Mask2FormerRGBTCrossAttn',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RGBTDINOv3Adapter',
        backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        backbone_ckpt='pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        img_size=crop_size,
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        depth=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        init_values=0.,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        freeze_vit=False,
        local_files_only=True,
        with_cp=False,
        fusion_type='ADD',
        thr_in_channels=3,
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            _scope_='mmdet',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(

				self_attn_cfg=dict(
				    embed_dims=256,
				    num_heads=8,
				    attn_drop=0.0,
				    proj_drop=0.0,
				    dropout_layer=None,
				    batch_first=True),
				cross_attn_cfg=dict(
				    embed_dims=256,
				    num_heads=8,
				    attn_drop=0.0,
				    proj_drop=0.0,
				    dropout_layer=None,
				    batch_first=True),
				ffn_cfg=dict(
				    embed_dims=256,
				    feedforward_channels=2048,
				    num_fcs=2,
				    act_cfg=dict(type='ReLU', inplace=True)),
				norm_cfg=dict(type='LN')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0,
                    use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'))),
    fusion_type='crossattn',
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

# Official optimizer: AdamW with LayerDecayOptimizerConstructor
# ViT layers get layer-wise decay; adapter modules get highest lr
optimizer = dict(
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.95))

# LR schedule: poly with linear warmup, 200 epochs = 117600 iters
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=117600,
         by_epoch=False),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

custom_hooks = [
    dict(type='PartialDegradeEvalHook', interval=5, num_samples=50),
]
