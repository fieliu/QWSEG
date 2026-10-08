# FMB variant of _base_dino_ts_m2f.py: 13 foreground classes and native
# 800x600 input.
#
# It exists as a separate file rather than as a set of overrides in the stage
# config because a child cannot change these values from outside: mmengine
# evaluates this file's ``data_preprocessor = dict(size=dino_crop_size, ...)``
# and ``decode_head = dict(num_classes=num_classes, ...)`` when this file is
# parsed, so redefining ``dino_crop_size`` / ``num_classes`` in the child has
# no effect on them, and the ``{{_base_.X}}`` placeholder resolves to the
# pre-merge value.
#
# ``dino_crop_size`` is (608, 800): FMB images are 800x600 and 600 is not a
# multiple of the patch size 16, so the data preprocessor pads height to 608
# with seg_pad_val=255 instead of resizing (which would distort by 1.3%) or
# feeding 600 (which would make the patch-embed convolution silently discard
# the bottom 8 rows).  ``size`` and ``size_divisor`` are mutually exclusive
# in stack_batch, hence size=None.
custom_imports = dict(
    imports=[
        'mmseg.models.backbones.dino_shared_vit',
        'mmseg.models.segmentors.dino_ts',
        'mmseg.engine',
        'mmseg.evaluation.metrics.stage1_alignment_metric',
        'mmseg.datasets.mfnet',
        'mmseg.datasets.fmb',
        'mmseg.datasets.transforms.loading',
        'mmdet.models',
    ],
    allow_failed_imports=False)

dino_crop_size = (608, 800)
num_classes = 13
embed_dim = 768  # DINOv3 ViT-B
dinov3_checkpoint = '{{$DINOV3_CHECKPOINT:/root/autodl-tmp/pretrain/dinov3-vitb16}}'

# Shared training policy; fog, stripe_noise and t_quantization are held out.
# Changing this list changes what can legitimately be called unseen at test.
degradation_policy = dict(
    corruptions=['seen'], degrade_prob=0.8,
    modality_probs=(0.45, 0.45, 0.10),
    scope_probs=(0.5, 0.25, 0.25),
    severity_range=(1, 5), area_range=(0.1, 0.6),
    missing_prob=0.1, weak_severity=0)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=None,
    size_divisor=16,
    test_cfg=dict(size_divisor=16))

# backbone: one shared DINOv3 ViT, per-modality patch embeds + shallow adapters,
# anchor/extra fusion at block R=3, utility router. Deep blocks run the joint
# anchor+extra sequence.
backbone = dict(
    type='DinoSharedViT',
    backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    backbone_ckpt=dinov3_checkpoint,
    img_size=dino_crop_size,
    patch_size=16,
    embed_dims=embed_dim,
    depth=12,
    fusion_block=3,       # R: shallow independent blocks before fusion
    d_adapter=64,
    # Align only a compact common subspace.  The original 768-D modality
    # tokens remain available to Anchor/Extra fusion, preserving thermal-only
    # information instead of forcing the whole representation to mimic RGB.
    align_out_dim=256,
    thr_in_channels=3,
    freeze_vit=False,
    # SDPA dispatches to FlashAttention on supported Ampere+ CUDA inputs and
    # safely falls back on older GPUs/CPU. Checkpointing trades compute for a
    # substantial reduction in the 9 deep joint blocks' saved activations.
    attention_backend='sdpa',
    with_cp=True,
    local_files_only=True)

neck = dict(
    type='Feature2Pyramid',
    embed_dim=embed_dim,
    rescales=[4, 2, 1, 0.5],
    norm_cfg=dict(type='BN', requires_grad=True))

decode_head = dict(
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
        type='mmdet.MSDeformAttnPixelDecoder',
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='ReLU'),
        encoder=dict(
            num_layers=6,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, num_levels=3, num_points=4,
                    im2col_step=64, dropout=0.0, batch_first=True,
                    norm_cfg=None, init_cfg=None),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=1024, num_fcs=2,
                    ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        positional_encoding=dict(num_feats=128, normalize=True),
        init_cfg=None),
    enforce_decoder_input_project=False,
    positional_encoding=dict(num_feats=128, normalize=True),
    transformer_decoder=dict(
        return_intermediate=True,
        num_layers=9,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                dropout_layer=None, batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0,
                dropout_layer=None, batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN')),
        init_cfg=None),
    loss_cls=dict(
        type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
        reduction='mean', class_weight=[1.0] * num_classes + [0.1]),
    loss_mask=dict(
        type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean',
        loss_weight=5.0),
    loss_dice=dict(
        type='mmdet.DiceLoss', use_sigmoid=True, activate=True,
        reduction='mean', naive_dice=True, eps=1.0, loss_weight=5.0),
    train_cfg=dict(
        num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
        assigner=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.ClassificationCost', weight=2.0),
                dict(type='mmdet.CrossEntropyLossCost', weight=5.0,
                     use_sigmoid=True),
                dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0),
            ]),
        sampler=dict(type='mmdet.MaskPseudoSampler')))
