#!/usr/bin/env python
"""Test SAM ViT backbone forward pass and verify output shapes."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmseg.models.backbones.sam_vit import SAMViT


def test_sam_vit_forward():
    print("=" * 60)
    print("Testing SAM ViT Backbone Forward Pass")
    print("=" * 60)

    img_size = 480
    batch_size = 2

    model = SAMViT(
        img_size=img_size,
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
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    x = torch.randn(batch_size, 3, img_size, img_size)
    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        outs = model(x)

    print(f"Number of output stages: {len(outs)}")
    h, w = img_size // 16, img_size // 16
    for i, out in enumerate(outs):
        expected_shape = (batch_size, 768, h, w)
        assert out.shape == expected_shape, \
            f"Stage {i}: expected {expected_shape}, got {out.shape}"
        print(f"  Stage {i} output shape: {out.shape} ✓")

    print("\n✓ All forward pass tests passed!")
    return model


def test_sam_vit_with_lora():
    print("\n" + "=" * 60)
    print("Testing SAM ViT with LoRA")
    print("=" * 60)

    from mmseg.models.utils.lora import apply_lora_to_model, freeze_non_lora_params

    model = SAMViT(
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
        window_size=14,
        global_attn_indexes=(2, 5, 8, 11),
        out_indices=(3, 5, 7, 11),
    )

    apply_lora_to_model(
        model,
        rank=4,
        alpha=4.0,
        dropout=0.1,
        target_modules=['qkv', 'proj'])
    freeze_non_lora_params(model)

    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'lora_A' in n or 'lora_B' in n)
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"LoRA parameters: {lora_params / 1e6:.4f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.4f}M")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"LoRA ratio: {lora_params / total_params * 100:.2f}%")

    x = torch.randn(2, 3, 480, 480)
    with torch.no_grad():
        outs = model(x)

    for i, out in enumerate(outs):
        print(f"  Stage {i} output shape: {out.shape} ✓")

    print("\n✓ LoRA test passed!")
    return model


def test_sam_vit_v1_baseline_compat():
    print("\n" + "=" * 60)
    print("Testing V1-SAM Baseline Compatibility")
    print("=" * 60)

    from mmseg.models.segmentors.rgbt_v1_baseline import RGBTv1Baseline

    data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(480, 480))

    model = RGBTv1Baseline(
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
            window_size=14,
            global_attn_indexes=(2, 5, 8, 11),
            out_indices=(3, 5, 7, 11)),
        neck=dict(
            type='Feature2Pyramid',
            embed_dim=768 * 2,
            rescales=[4, 2, 1, 0.5]),
        decode_head=dict(
            type='UPerHead',
            in_channels=[768 * 2, 768 * 2, 768 * 2, 768 * 2],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=768,
            dropout_ratio=0.1,
            num_classes=15,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=[
                dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                     alpha=0.25, loss_weight=1.0, loss_name='loss_focal'),
                dict(type='DiceLoss', use_sigmoid=True, activate=True,
                     naive_dice=False, loss_weight=1.0,
                     loss_name='loss_dice'),
            ]),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=768 * 2,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=15,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=[
                dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                     alpha=0.25, loss_weight=0.4, loss_name='loss_focal'),
                dict(type='DiceLoss', use_sigmoid=True, activate=True,
                     naive_dice=False, loss_weight=0.4,
                     loss_name='loss_dice'),
            ]),
        use_lora=True,
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_target_modules=['qkv', 'proj'],
        freeze_backbone=True,
        test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(213, 213)),
    )

    lora_info = model.get_lora_info()
    print(f"LoRA params: {lora_info['lora_params'] / 1e6:.4f}M")
    print(f"Trainable params: {lora_info['trainable_params'] / 1e6:.4f}M")
    print(f"Total params: {lora_info['total_params'] / 1e6:.2f}M")
    print(f"LoRA ratio: {lora_info['lora_ratio']:.2f}%")
    print(f"Trainable ratio: {lora_info['trainable_ratio']:.2f}%")

    x = torch.randn(2, 6, 480, 480)
    with torch.no_grad():
        seg_logits = model.whole_inference(x, [
            dict(ori_shape=(480, 480), img_shape=(480, 480),
                 pad_shape=(480, 480), padding_size=[0, 0, 0, 0])
        ])

    print(f"Seg logits shape: {seg_logits.shape}")
    assert seg_logits.shape == (2, 15, 480, 480), \
        f"Expected (2, 15, 480, 480), got {seg_logits.shape}"
    print("✓ V1-SAM Baseline compatibility test passed!")


if __name__ == '__main__':
    test_sam_vit_forward()
    test_sam_vit_with_lora()
    test_sam_vit_v1_baseline_compat()
    print("\n" + "=" * 60)
    print("All tests passed! V1-SAM Baseline is ready for training.")
    print("=" * 60)
