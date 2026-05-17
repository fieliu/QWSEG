"""
QualityNetworkV2 评估脚本

3个场景:
  1. 原图 (clean)
  2. 全局退化 (global degradation): 随机退化类型 + 随机模态 + 随机强度
  3. 局部退化 (local degradation): 随机退化类型 + 随机模态 + 随机强度 + 随机区域

可视化:
  每个场景随机选3张图组成1大图, 选3次 = 3大图
  每大图3行: 图像 / 质量热图 / 剪枝后图像
  输出到本地文件和TensorBoard

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/eval_quality_network.py \
      --checkpoint work_dirs/quality_v2_pretrain/best_quality_net_v2.pth \
      --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
      --work-dir work_dirs/quality_v2_eval \
      --img-size 480 \
      --amp
"""

import os
import sys
import argparse
import logging
import random

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmseg.models.backbones.mae import MAE
from mmseg.models.utils.quality_network_v2 import QualityNetworkV2
from mmseg.datasets.transforms.degradation import (
    apply_degradation,
    apply_multi_region_degradation,
    _RGB_LEVEL_CONFIGS,
    _THERMAL_LEVEL_CONFIGS,
)
from train_quality_network_v2 import ViTFeatureExtractor, FMBQualityDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_RGB_DEG_TYPES = list(_RGB_LEVEL_CONFIGS.keys())
_THERMAL_DEG_TYPES = list(_THERMAL_LEVEL_CONFIGS.keys())
_RGB_LEVELS = sorted(_RGB_LEVEL_CONFIGS[_RGB_DEG_TYPES[0]].keys())
_THERMAL_LEVELS = sorted(_THERMAL_LEVEL_CONFIGS[_THERMAL_DEG_TYPES[0]].keys())


def apply_random_global_degradation(img_tensor, modality='rgb'):
    if modality == 'rgb':
        deg_type = random.choice(_RGB_DEG_TYPES)
        level = random.choice(_RGB_LEVELS)
    else:
        deg_type = random.choice(_THERMAL_DEG_TYPES)
        level = random.choice(_THERMAL_LEVELS)

    deg_img = apply_degradation(img_tensor, deg_type, level, modality)
    return deg_img, deg_type, level


def apply_random_local_degradation(img_tensor, modality='rgb'):
    B, C, H, W = img_tensor.shape
    num_regions = random.randint(1, 3)

    if modality == 'rgb':
        deg_type = random.choice(_RGB_DEG_TYPES)
        level = random.choice(_RGB_LEVELS)
    else:
        deg_type = random.choice(_THERMAL_DEG_TYPES)
        level = random.choice(_THERMAL_LEVELS)

    region_configs = []
    min_size = 64
    for _ in range(num_regions):
        rh = random.randint(min_size, H // 2)
        rw = random.randint(min_size, W // 2)
        sh = random.randint(0, H - rh)
        sw = random.randint(0, W - rw)
        region_configs.append(((sh, sw, rh, rw), deg_type, level))

    deg_img = apply_multi_region_degradation(
        img_tensor, region_configs, modality)
    return deg_img, deg_type, level, region_configs


def visualize_eval_scene(
    scene_name, images_list, q_list, titles_list,
    quality_threshold, save_dir, writer=None, patch_size=16,
):
    num_groups = len(images_list)
    imgs_per_group = len(images_list[0])

    fig, axes = plt.subplots(
        3 * num_groups, imgs_per_group,
        figsize=(3.5 * imgs_per_group, 3 * 3 * num_groups),
    )
    if num_groups == 1 and imgs_per_group == 1:
        axes = np.array([axes])
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(3 * num_groups, imgs_per_group)

    for g in range(num_groups):
        row_offset = g * 3
        for col in range(imgs_per_group):
            ax_img = axes[row_offset, col]
            ax_qmap = axes[row_offset + 1, col]
            ax_prune = axes[row_offset + 2, col]

            img = images_list[g][col].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            img_H, img_W = img.shape[0], img.shape[1]

            q = q_list[g][col].detach().cpu().numpy()
            patch_H = img_H // patch_size
            patch_W = img_W // patch_size
            q_2d = q.reshape(patch_H, patch_W)
            q_up = F.interpolate(
                torch.from_numpy(q_2d).float().unsqueeze(0).unsqueeze(0),
                size=(img_H, img_W), mode='bilinear', align_corners=False,
            ).squeeze().numpy()

            mask = (q >= quality_threshold).astype(np.float32)
            mask_2d = mask.reshape(patch_H, patch_W)
            mask_up = F.interpolate(
                torch.from_numpy(mask_2d).float().unsqueeze(0).unsqueeze(0),
                size=(img_H, img_W), mode='nearest',
            ).squeeze().numpy()

            pruned = img * mask_up[:, :, np.newaxis]

            ax_img.imshow(img)
            ax_img.set_title(titles_list[g][col], fontsize=8)
            ax_img.axis('off')

            ax_qmap.imshow(q_up, vmin=0, vmax=1, cmap='RdBu_r')
            q_mean = q.mean()
            ax_qmap.set_title(f'q={q_mean:.2f}', fontsize=7)
            ax_qmap.axis('off')

            ax_prune.imshow(pruned)
            keep_ratio = mask.mean() * 100
            ax_prune.set_title(f'keep={keep_ratio:.0f}%', fontsize=7)
            ax_prune.axis('off')

        axes[row_offset, 0].set_ylabel('Image', fontsize=9, fontweight='bold')
        axes[row_offset + 1, 0].set_ylabel(
            'Quality Map', fontsize=9, fontweight='bold')
        axes[row_offset + 2, 0].set_ylabel(
            'Pruned', fontsize=9, fontweight='bold')

    fig.suptitle(
        f'Scene: {scene_name} | threshold={quality_threshold}',
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{scene_name}.png')
    if writer is not None:
        writer.add_figure(f'Eval/{scene_name}', fig, global_step=0)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Eval visualization saved: {save_path}')


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.work_dir, 'logs'))

    img_size = args.img_size

    logger.info('Building MAE backbone...')
    mae_backbone = MAE(
        img_size=(img_size, img_size),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
    )

    if args.vit_pretrained and os.path.exists(args.vit_pretrained):
        ckpt = torch.load(args.vit_pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith('universal_backbone.'):
                backbone_state[k.replace('universal_backbone.', '')] = v
            elif not k.startswith(('decode_head', 'auxiliary_head', 'neck',
                                   'private_branch', 'cross_attn',
                                   'zc_seg_head', 'quality_network',
                                   'token_classifier')):
                backbone_state[k] = v
        if backbone_state:
            pos_embed_key = 'pos_embed'
            if pos_embed_key in backbone_state:
                pos_embed_pretrained = backbone_state[pos_embed_key]
                pretrain_size = int(
                    (pos_embed_pretrained.shape[1] - 1) ** 0.5)
                target_size = int(
                    (mae_backbone.pos_embed.shape[1] - 1) ** 0.5)
                if pretrain_size != target_size:
                    cls_token = pos_embed_pretrained[:, :1, :]
                    pos_tokens = pos_embed_pretrained[:, 1:, :].reshape(
                        1, pretrain_size, pretrain_size, 768)
                    pos_tokens = pos_tokens.permute(0, 3, 1, 2)
                    pos_tokens = F.interpolate(
                        pos_tokens, size=(target_size, target_size),
                        mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(
                        0, 2, 3, 1).flatten(1, 2)
                    backbone_state[pos_embed_key] = torch.cat(
                        (cls_token, pos_tokens), dim=1)
            mae_backbone.load_state_dict(backbone_state, strict=False)
            logger.info(f'Loaded backbone weights from {args.vit_pretrained}')

    mae_backbone.eval()
    mae_backbone.requires_grad_(False)
    mae_backbone.to(device)

    vit_encoder = ViTFeatureExtractor(mae_backbone, num_layers=4)

    logger.info('Building QualityNetworkV2...')
    quality_net = QualityNetworkV2(
        embed_dim=768,
        proj_dim=64,
        local_window_size=3,
        mlp_hidden_dim=128,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f'Loaded quality net from {args.checkpoint}')
    else:
        logger.warning('No checkpoint provided, using random weights!')

    quality_net.eval()

    val_dataset = FMBQualityDataset(
        data_root=args.data_root,
        split='testing',
        img_size=img_size,
    )
    logger.info(f'Val set size: {len(val_dataset)}')

    num_groups = 3
    imgs_per_group = 3
    total_needed = num_groups * imgs_per_group
    indices = random.sample(range(len(val_dataset)), min(total_needed, len(val_dataset)))

    vis_dir = os.path.join(args.work_dir, 'eval_vis')
    os.makedirs(vis_dir, exist_ok=True)

    all_scene_data = {}

    with torch.no_grad():
        for scene_name, apply_fn in [
            ('clean', None),
            ('global_degradation', 'global'),
            ('local_degradation', 'local'),
        ]:
            images_list = []
            q_list = []
            titles_list = []

            for g in range(num_groups):
                group_images = []
                group_q = []
                group_titles = []

                for i in range(imgs_per_group):
                    idx = indices[g * imgs_per_group + i]
                    img_rgb, img_t = val_dataset[idx]
                    img_rgb = img_rgb.unsqueeze(0).to(device)
                    img_t = img_t.unsqueeze(0).to(device)

                    if apply_fn is None:
                        deg_rgb = img_rgb
                        deg_t = img_t
                        title_rgb = 'RGB clean'
                        title_t = 'T clean'
                    elif apply_fn == 'global':
                        modality_rgb = 'rgb'
                        modality_t = 'ir'
                        deg_rgb, deg_type_rgb, level_rgb = \
                            apply_random_global_degradation(
                                img_rgb, modality_rgb)
                        deg_t, deg_type_t, level_t = \
                            apply_random_global_degradation(
                                img_t, modality_t)
                        title_rgb = f'RGB {deg_type_rgb}(L{level_rgb})'
                        title_t = f'T {deg_type_t}(L{level_t})'
                    else:
                        modality_rgb = 'rgb'
                        modality_t = 'ir'
                        deg_rgb, deg_type_rgb, level_rgb, regions_rgb = \
                            apply_random_local_degradation(
                                img_rgb, modality_rgb)
                        deg_t, deg_type_t, level_t, regions_t = \
                            apply_random_local_degradation(
                                img_t, modality_t)
                        title_rgb = f'RGB {deg_type_rgb}(L{level_rgb}) local'
                        title_t = f'T {deg_type_t}(L{level_t}) local'

                    with autocast(enabled=args.amp):
                        tok_rgb, _ = vit_encoder(deg_rgb)
                        tok_t, _ = vit_encoder(deg_t)
                        q_rgb = quality_net.forward_rgb(tok_rgb)
                        q_t = quality_net.forward_thermal(tok_t)

                    group_images.extend([deg_rgb[0], deg_t[0]])
                    group_q.extend([q_rgb[0], q_t[0]])
                    group_titles.extend([title_rgb, title_t])

                images_list.append(group_images)
                q_list.append(group_q)
                titles_list.append(group_titles)

            visualize_eval_scene(
                scene_name, images_list, q_list, titles_list,
                args.quality_threshold, vis_dir, writer=writer,
            )

            all_scene_data[scene_name] = {
                'images_list': images_list,
                'q_list': q_list,
                'titles_list': titles_list,
            }

    logger.info('=== Quality Statistics ===')
    for scene_name, data in all_scene_data.items():
        all_q = []
        for group_q in data['q_list']:
            for q in group_q:
                all_q.append(q.detach().cpu().numpy())
        all_q = np.concatenate(all_q)
        logger.info(
            f'{scene_name}: mean={all_q.mean():.4f}, '
            f'std={all_q.std():.4f}, '
            f'min={all_q.min():.4f}, '
            f'max={all_q.max():.4f}, '
            f'ratio>0.1={(all_q > 0.1).mean() * 100:.1f}%, '
            f'ratio>0.5={(all_q > 0.5).mean() * 100:.1f}%')

        writer.add_scalar(f'eval/{scene_name}_q_mean', all_q.mean(), 0)
        writer.add_scalar(f'eval/{scene_name}_q_std', all_q.std(), 0)
        writer.add_scalar(
            f'eval/{scene_name}_ratio_above_0.1',
            (all_q > 0.1).mean(), 0)
        writer.add_scalar(
            f'eval/{scene_name}_ratio_above_0.5',
            (all_q > 0.5).mean(), 0)

    writer.close()
    logger.info(f'Evaluation complete. Results saved to {args.work_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate QualityNetworkV2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to quality network checkpoint')
    parser.add_argument('--vit-pretrained', type=str,
                        default='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
                        help='Path to MAE pretrained weights')
    parser.add_argument('--data-root', type=str,
                        default='/home/lh/code/QWSEG/dataset',
                        help='Root directory of dataset')
    parser.add_argument('--work-dir', type=str,
                        default='work_dirs/quality_v2_eval',
                        help='Working directory')
    parser.add_argument('--img-size', type=int, default=480,
                        help='Image size')
    parser.add_argument('--quality-threshold', type=float, default=0.1,
                        help='Quality threshold for pruning')
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
