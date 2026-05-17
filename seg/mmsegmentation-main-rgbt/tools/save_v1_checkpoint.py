import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load pretrained weights into v1 model and save as checkpoint')
    parser.add_argument('config', help='v1 baseline config file path')
    parser.add_argument(
        '--backbone-pretrained',
        default=None,
        help='backbone pretrained weight path (e.g. M-SpecGene_VIT-B_seg_transform.pth). '
             'Uses MAE.init_weights() flow with auto interpolation.')
    parser.add_argument(
        '--full-checkpoint',
        default=None,
        help='full model checkpoint path (e.g. FMB_save_iter_224000_480x480.pth). '
             'Loads non-backbone parts (neck, decode_head, auxiliary_head). '
             'If --backbone-pretrained is not set, also loads backbone from this.')
    parser.add_argument(
        '--output',
        default=None,
        help='output checkpoint path. If not specified, save to '
             'work_dirs/<config_name>/v1_pretrained_init.pth')
    args = parser.parse_args()
    return args


def load_backbone_from_pretrained(model, pretrained_path):
    ckpt = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in ckpt:
        backbone_ckpt = ckpt['state_dict']
    else:
        backbone_ckpt = ckpt

    if any(k.startswith('backbone.') for k in backbone_ckpt.keys()):
        backbone_ckpt = {
            k[len('backbone.'):]: v
            for k, v in backbone_ckpt.items() if k.startswith('backbone.')
        }

    if hasattr(model.backbone, 'resize_rel_pos_embed'):
        backbone_checkpoint = {'state_dict': backbone_ckpt}
        backbone_ckpt = model.backbone.resize_rel_pos_embed(backbone_checkpoint)
        if isinstance(backbone_ckpt, dict) and 'state_dict' in backbone_ckpt:
            backbone_ckpt = backbone_ckpt['state_dict']

    if hasattr(model.backbone, 'resize_abs_pos_embed'):
        backbone_ckpt = model.backbone.resize_abs_pos_embed(backbone_ckpt)

    missing_b, unexpected_b = model.backbone.load_state_dict(
        backbone_ckpt, strict=False)
    print(f"\nBackbone loading (from pretrained: {os.path.basename(pretrained_path)}):")
    print(f"  Loaded keys: {len(backbone_ckpt)}")
    print(f"  Missing keys: {len(missing_b)}")
    if missing_b:
        lora_missing = [k for k in missing_b if 'lora_' in k]
        other_missing = [k for k in missing_b if 'lora_' not in k]
        if lora_missing:
            print(f"  Missing LoRA keys (expected): {len(lora_missing)}")
        if other_missing:
            print(f"  Missing other keys: {other_missing[:10]}")
    print(f"  Unexpected keys: {len(unexpected_b)}")
    if unexpected_b:
        print(f"  Unexpected keys (first 10): {unexpected_b[:10]}")


def load_non_backbone_from_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt_state_dict = ckpt['state_dict']
    else:
        ckpt_state_dict = ckpt

    non_backbone_ckpt = {
        k: v for k, v in ckpt_state_dict.items()
        if not k.startswith('backbone.')
    }

    if not non_backbone_ckpt:
        print(f"\nNo non-backbone keys found in {os.path.basename(ckpt_path)}")
        return

    missing_nb, unexpected_nb = model.load_state_dict(
        non_backbone_ckpt, strict=False)
    print(f"\nNon-backbone loading (from: {os.path.basename(ckpt_path)}):")
    print(f"  Loaded keys: {len(non_backbone_ckpt)}")
    print(f"  Missing keys: {len(missing_nb)}")
    if missing_nb:
        backbone_missing = [k for k in missing_nb if k.startswith('backbone.')]
        other_missing = [k for k in missing_nb if not k.startswith('backbone.')]
        if backbone_missing:
            print(f"  Missing backbone keys (expected, loaded separately): {len(backbone_missing)}")
        if other_missing:
            print(f"  Missing other keys: {other_missing[:10]}")
    print(f"  Unexpected keys: {len(unexpected_nb)}")
    if unexpected_nb:
        print(f"  Unexpected keys (first 10): {unexpected_nb[:10]}")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.output is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        output_dir = os.path.join('./work_dirs', config_name)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, 'v1_pretrained_init.pth')

    cfg.work_dir = os.path.dirname(args.output) or './work_dirs'
    cfg.model.pretrained = None
    cfg.model.init_cfg = None

    runner = Runner.from_cfg(cfg)

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module

    if args.backbone_pretrained:
        load_backbone_from_pretrained(model, args.backbone_pretrained)

    if args.full_checkpoint:
        if args.backbone_pretrained:
            load_non_backbone_from_checkpoint(model, args.full_checkpoint)
        else:
            ckpt = torch.load(args.full_checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                ckpt_state_dict = ckpt['state_dict']
            else:
                ckpt_state_dict = ckpt

            backbone_ckpt = {}
            non_backbone_ckpt = {}
            for k, v in ckpt_state_dict.items():
                if k.startswith('backbone.'):
                    backbone_ckpt[k[len('backbone.'):]] = v
                else:
                    non_backbone_ckpt[k] = v

            if backbone_ckpt and hasattr(model.backbone, 'resize_rel_pos_embed'):
                backbone_checkpoint = {'state_dict': backbone_ckpt}
                backbone_ckpt = model.backbone.resize_rel_pos_embed(backbone_checkpoint)
                if isinstance(backbone_ckpt, dict) and 'state_dict' in backbone_ckpt:
                    backbone_ckpt = backbone_ckpt['state_dict']

            if backbone_ckpt and hasattr(model.backbone, 'resize_abs_pos_embed'):
                backbone_ckpt = model.backbone.resize_abs_pos_embed(backbone_ckpt)

            if backbone_ckpt:
                missing_b, unexpected_b = model.backbone.load_state_dict(
                    backbone_ckpt, strict=False)
                print(f"\nBackbone loading (from: {os.path.basename(args.full_checkpoint)}):")
                print(f"  Loaded keys: {len(backbone_ckpt)}")
                print(f"  Missing keys: {len(missing_b)}")
                if missing_b:
                    lora_missing = [k for k in missing_b if 'lora_' in k]
                    other_missing = [k for k in missing_b if 'lora_' not in k]
                    if lora_missing:
                        print(f"  Missing LoRA keys (expected): {len(lora_missing)}")
                    if other_missing:
                        print(f"  Missing other keys: {other_missing[:10]}")
                print(f"  Unexpected keys: {len(unexpected_b)}")

            if non_backbone_ckpt:
                missing_nb, unexpected_nb = model.load_state_dict(
                    non_backbone_ckpt, strict=False)
                print(f"\nNon-backbone loading (from: {os.path.basename(args.full_checkpoint)}):")
                print(f"  Loaded keys: {len(non_backbone_ckpt)}")
                print(f"  Missing keys: {len(missing_nb)}")
                print(f"  Unexpected keys: {len(unexpected_nb)}")

    state_dict = model.state_dict()

    save_ckpt = {
        'meta': {},
        'state_dict': state_dict,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(save_ckpt, args.output)
    print(f'\nCheckpoint saved to: {args.output}')

    if hasattr(model, 'get_lora_info'):
        info = model.get_lora_info()
        print(f"\nLoRA Info:")
        print(f"  LoRA params:      {info['lora_params']:,}")
        print(f"  Frozen params:    {info['frozen_params']:,}")
        print(f"  Trainable params: {info['trainable_params']:,}")
        print(f"  Total params:     {info['total_params']:,}")
        print(f"  LoRA ratio:       {info['lora_ratio']:.4f}%")
        print(f"  Trainable ratio:  {info['trainable_ratio']:.4f}%")

    backbone_loaded = sum(
        1 for k in state_dict
        if k.startswith('backbone.') and state_dict[k].abs().sum() > 0)
    backbone_total = sum(1 for k in state_dict if k.startswith('backbone.'))
    print(f"\nBackbone params: {backbone_loaded}/{backbone_total} non-zero")


if __name__ == '__main__':
    main()
