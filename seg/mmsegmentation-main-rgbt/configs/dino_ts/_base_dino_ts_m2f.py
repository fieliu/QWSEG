# Shared model fragment: DINO shared-ViT (anchor/extra) + Feature2Pyramid +
# Mask2Former decode head. Imported by the per-stage configs via _base_.
# The segmentor `type` and any stage-specific losses are set in each stage
# config (this fragment only fixes the backbone / neck / head / preprocessor).
custom_imports = dict(
    imports=[
        'mmseg.models.backbones.dino_shared_vit',
        'mmseg.models.segmentors.dino_ts',
        'mmseg.engine',
        'mmseg.datasets.mfnet',
        'mmseg.datasets.transforms.loading',
        'mmdet.models',
    ],
    allow_failed_imports=False)

crop_size = (480, 640)
num_classes = 9
embed_dim = 768  # DINOv3 ViT-B

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

# backbone: one shared DINOv3 ViT, per-modality patch embeds + shallow adapters,
# anchor/extra fusion at block R=3, utility router. Deep blocks run the joint
# anchor+extra sequence.
backbone = dict(
    type='DinoSharedViT',
    backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    backbone_ckpt='pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    img_size=crop_size,
    patch_size=16,
    embed_dims=embed_dim,
    depth=12,
    fusion_block=3,       # R: shallow independent blocks before fusion
    d_adapter=64,
    thr_in_channels=3,
    freeze_vit=False,
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
