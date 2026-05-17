import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify v1 baseline pretrained weight loading')
    parser.add_argument('config', help='v1 baseline config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = './work_dirs/verify_pretrained'

    pretrained_path = cfg.model.get('pretrained', None)
    print(f"\nConfig pretrained: {pretrained_path}")

    runner = Runner.from_cfg(cfg)

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module

    backbone = model.backbone

    print(f"\nBackbone type: {type(backbone).__name__}")
    print(f"Backbone img_size: {backbone.img_size}")
    print(f"Backbone patch_shape: {backbone.patch_shape}")

    print(f"\nChecking backbone weights...")

    cls_token = backbone.cls_token.data
    pos_embed = backbone.pos_embed.data
    print(f"  cls_token: shape={cls_token.shape}, norm={cls_token.norm():.6f}")

    if cls_token.norm() < 1e-6:
        print(f"  ⚠️  cls_token is near zero - likely NOT loaded from pretrained!")
    else:
        print(f"  ✅ cls_token has non-trivial values - likely loaded from pretrained")

    print(f"  pos_embed: shape={pos_embed.shape}, norm={pos_embed.norm():.6f}")
    if pos_embed.norm() < 1e-6:
        print(f"  ⚠️  pos_embed is near zero - likely NOT loaded from pretrained!")
    else:
        print(f"  ✅ pos_embed has non-trivial values - likely loaded from pretrained")

    patch_weight = backbone.patch_embed.projection.weight.data
    print(f"  patch_embed.projection.weight: shape={patch_weight.shape}, norm={patch_weight.norm():.6f}")
    if patch_weight.norm() < 1e-6:
        print(f"  ⚠️  patch_embed weight is near zero - likely NOT loaded from pretrained!")
    else:
        print(f"  ✅ patch_embed weight has non-trivial values - likely loaded from pretrained")

    first_layer_weight = backbone.layers[0].attn.qkv.weight.data
    print(f"  layers.0.attn.qkv.weight: shape={first_layer_weight.shape}, norm={first_layer_weight.norm():.6f}")
    if first_layer_weight.norm() < 1e-6:
        print(f"  ⚠️  First layer qkv weight is near zero - likely NOT loaded from pretrained!")
    else:
        print(f"  ✅ First layer qkv weight has non-trivial values - likely loaded from pretrained")

    has_lora = hasattr(backbone.layers[0].attn.qkv, 'lora_A')
    if has_lora:
        lora_A = backbone.layers[0].attn.qkv.lora_A.data
        lora_B = backbone.layers[0].attn.qkv.lora_B.data
        print(f"  lora_A: shape={lora_A.shape}, norm={lora_A.norm():.6f}")
        print(f"  lora_B: shape={lora_B.shape}, norm={lora_B.norm():.6f} (should be ~0)")
        print(f"  ✅ LoRA layers present")

    print(f"\nComparing with pretrained file directly...")
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt_sd = ckpt['state_dict']
        else:
            ckpt_sd = ckpt

        ckpt_pos_embed = ckpt_sd.get('pos_embed', None)
        if ckpt_pos_embed is not None:
            print(f"  Checkpoint pos_embed: shape={ckpt_pos_embed.shape}")
            if ckpt_pos_embed.shape == pos_embed.shape:
                diff = (ckpt_pos_embed - pos_embed).abs().max().item()
                print(f"  Max diff from checkpoint: {diff:.8f}")
                if diff < 1e-4:
                    print(f"  ✅ pos_embed matches checkpoint exactly")
                else:
                    print(f"  ℹ️  pos_embed differs (expected if resolution changed, interpolation applied)")
            else:
                print(f"  ℹ️  Shape differs (768→480 interpolation was applied)")

        ckpt_cls_token = ckpt_sd.get('cls_token', None)
        if ckpt_cls_token is not None:
            print(f"  Checkpoint cls_token: shape={ckpt_cls_token.shape}")
            if ckpt_cls_token.shape == cls_token.shape:
                diff = (ckpt_cls_token - cls_token).abs().max().item()
                print(f"  Max diff from checkpoint: {diff:.8f}")
                if diff < 1e-4:
                    print(f"  ✅ cls_token matches checkpoint exactly")
                else:
                    print(f"  ⚠️  cls_token differs from checkpoint!")

        ckpt_qkv = ckpt_sd.get('layers.0.attn.qkv.weight', None)
        if ckpt_qkv is not None:
            diff = (ckpt_qkv - first_layer_weight).abs().max().item()
            print(f"  layers.0.attn.qkv.weight max diff: {diff:.8f}")
            if diff < 1e-4:
                print(f"  ✅ qkv weight matches checkpoint exactly")
            else:
                print(f"  ⚠️  qkv weight differs from checkpoint!")
    elif pretrained_path:
        print(f"  ⚠️  Pretrained file not found: {pretrained_path}")

    if hasattr(model, 'get_lora_info'):
        info = model.get_lora_info()
        print(f"\nLoRA Info:")
        print(f"  LoRA params:      {info['lora_params']:,}")
        print(f"  Frozen params:    {info['frozen_params']:,}")
        print(f"  Trainable params: {info['trainable_params']:,}")
        print(f"  Total params:     {info['total_params']:,}")

    print(f"\n{'='*50}")
    print(f"Verification complete!")


if __name__ == '__main__':
    main()
