"""
QualityAwarePyramidNet 测试可视化脚本

测试集分布:
  - 30% 单模态缺失 (RGB或T全零)
  - 30% 全局退化
  - 40% 局部退化

可视化布局:
  第1行: 输入图像 (RGB, T)
  第2-5行: Stage 0-3 的质量热图

预处理与语义分割网络完全一致:
  LoadRGBTImageFrom4Channel → Resize(640,480) → SegDataPreProcessor归一化

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/test_quality_pyramid_net.py \
      --dataset mfnet \
      --checkpoint work_dirs/quality_pyramid_pretrain/<timestamp>/weight/best_quality_pyramid_net.pth \
      --num-samples 20 \
      --save-dir work_dirs/quality_test_vis
"""

import os
import sys
import argparse
import random
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmseg.registry import DATASETS, MODELS
from mmseg.datasets.transforms.loading import LoadRGBTImageFrom4Channel
from mmseg.datasets.transforms import Resize
from mmseg.models.utils.quality_aware_pyramid_net import QualityAwarePyramidNet
from mmseg.datasets.transforms.quality_degradation import (
    _QUALITY_RGB_DEG_TYPES, _QUALITY_T_DEG_TYPES,
    apply_quality_degradation_rgb, apply_quality_degradation_t)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    'mfnet': {
        'dataset_type': 'MFNetDataset',
        'data_root': '/home/lh/code/data/MFNet',
        'ann_file': 'test.txt',
        'data_prefix': dict(img_path='images', seg_map_path='labels'),
        'model_config': 'configs/segformer/mitmul_v10_quality_embed_mit-b2-b0_1xb2-40K_mfnet-480x640.py',
    },
    'fmb': {
        'dataset_type': 'FMBDataset',
        'data_root': '/home/lh/code/data/FMB_ALL/FMB',
        'ann_file': 'test.txt',
        'data_prefix': dict(img_path='images', seg_map_path='labels'),
        'model_config': 'configs/segformer/mitmul_v10_quality_embed_mit-b2-b0_1xb2-40K_fmb-480x640.py',
    },
}

MEAN_STD = {
    'mfnet': {
        'rgb_mean': torch.tensor([123.675, 116.28, 103.53]),
        'rgb_std': torch.tensor([58.395, 57.12, 57.375]),
        't_mean': torch.tensor([123.675, 116.28, 103.53]),
        't_std': torch.tensor([58.395, 57.12, 57.375]),
    },
    'fmb': {
        'rgb_mean': torch.tensor([123.675, 116.28, 103.53]),
        'rgb_std': torch.tensor([58.395, 57.12, 57.375]),
        't_mean': torch.tensor([123.675, 116.28, 103.53]),
        't_std': torch.tensor([58.395, 57.12, 57.375]),
    },
}


def denorm_to_01(norm_tensor, mean, std):
    raw = norm_tensor * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    return (raw / 255.0).clamp(0, 1)


def renorm_from_01(tensor_01, mean, std):
    raw = tensor_01 * 255.0
    return (raw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def apply_degradation(img_norm, modality, mean, std,
                      deg_type=None, level=None,
                      is_local=False, local_mask=None):
    B, C, H, W = img_norm.shape
    img_01 = denorm_to_01(img_norm, mean, std)

    if deg_type is None:
        if modality == 'rgb':
            deg_type = random.choice(_QUALITY_RGB_DEG_TYPES)
        else:
            deg_type = random.choice(_QUALITY_T_DEG_TYPES)

    if level is None:
        level = random.randint(2, 5)

    if modality == 'rgb':
        deg_img_01 = apply_quality_degradation_rgb(img_01, deg_type, level)
    else:
        deg_img_01 = apply_quality_degradation_t(img_01, deg_type, level)

    if is_local and local_mask is not None:
        if local_mask.shape[2:] != (H, W):
            local_mask = F.interpolate(local_mask.float(), size=(H, W), mode='nearest')
        local_mask_01 = local_mask.expand(B, C, H, W)
        deg_img_01 = img_01 * (1 - local_mask_01) + deg_img_01 * local_mask_01

    result = renorm_from_01(deg_img_01, mean, std)
    return result, deg_type, level


def generate_local_mask(B, H, W, num_regions=3, device='cpu'):
    mask = torch.zeros(B, 1, H, W, device=device)
    region_h = H // 4
    region_w = W // 4
    for b in range(B):
        for _ in range(num_regions):
            rh = random.randint(region_h, H // 2)
            rw = random.randint(region_w, W // 2)
            y1 = random.randint(0, H - rh)
            x1 = random.randint(0, W - rw)
            mask[b, 0, y1:y1 + rh, x1:x1 + rw] = 1.0
    return mask


def build_test_dataset(dataset_type, data_root=None):
    cfg = DATASET_CONFIGS[dataset_type]
    ds_type = cfg['dataset_type']
    ds_root = data_root or cfg['data_root']

    load_transform = LoadRGBTImageFrom4Channel(to_float32=True)
    resize_transform = Resize(scale=(640, 480), keep_ratio=False)
    pipeline = [load_transform, resize_transform]

    dataset_cls = DATASETS.get(ds_type)
    kwargs = dict(
        data_root=ds_root,
        data_prefix=cfg['data_prefix'],
        pipeline=pipeline,
    )
    if cfg.get('ann_file'):
        kwargs['ann_file'] = cfg['ann_file']
    dataset = dataset_cls(**kwargs)
    return dataset


def collate_fn(batch):
    img_list = []
    for item in batch:
        img = item['img']
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] in (3, 4, 6):
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img.copy())
        while img.dim() > 3:
            img = img.squeeze(0)
        img_list.append(img.float())
    return torch.stack(img_list, dim=0)


def visualize_sample(
    rgb_input_01, t_input_01,
    q_rgb_maps, q_t_maps,
    deg_type_rgb, deg_type_t,
    deg_level_rgb, deg_level_t,
    sample_idx, save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    num_stages = len(q_rgb_maps)
    img_H, img_W = rgb_input_01.shape[2], rgb_input_01.shape[3]

    num_cols = 2
    num_rows = 1 + num_stages

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(6 * num_cols, 3 * num_rows),
    )

    rgb_disp = rgb_input_01[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
    t_disp = t_input_01[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)

    if t_disp.shape[2] == 3:
        t_gray = 0.299 * t_disp[:, :, 0] + 0.587 * t_disp[:, :, 1] + 0.114 * t_disp[:, :, 2]
        t_disp_vis = np.stack([t_gray] * 3, axis=-1)
    else:
        t_disp_vis = t_disp

    axes[0, 0].imshow(rgb_disp)
    deg_info_rgb = f'{deg_type_rgb}(L{deg_level_rgb})' if deg_type_rgb != 'clean' else 'clean'
    axes[0, 0].set_title(f'RGB Input [{deg_info_rgb}]', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(t_disp_vis)
    deg_info_t = f'{deg_type_t}(L{deg_level_t})' if deg_type_t != 'clean' else 'clean'
    axes[0, 1].set_title(f'T Input [{deg_info_t}]', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    for s in range(num_stages):
        q_rgb = q_rgb_maps[s][0, 0].detach().cpu().numpy()
        q_t = q_t_maps[s][0, 0].detach().cpu().cpu().numpy()

        q_rgb_up = F.interpolate(
            torch.from_numpy(q_rgb).float().unsqueeze(0).unsqueeze(0),
            size=(img_H, img_W), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        q_t_up = F.interpolate(
            torch.from_numpy(q_t).float().unsqueeze(0).unsqueeze(0),
            size=(img_H, img_W), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        ax_rgb = axes[1 + s, 0]
        im_rgb = ax_rgb.imshow(q_rgb_up, vmin=0, vmax=1, cmap='RdBu_r')
        ax_rgb.set_title(
            f'Stage {s} RGB  mean={q_rgb.mean():.3f}  min={q_rgb.min():.3f}  max={q_rgb.max():.3f}',
            fontsize=9)
        ax_rgb.axis('off')
        plt.colorbar(im_rgb, ax=ax_rgb, fraction=0.046, pad=0.04)

        ax_t = axes[1 + s, 1]
        im_t = ax_t.imshow(q_t_up, vmin=0, vmax=1, cmap='RdBu_r')
        ax_t.set_title(
            f'Stage {s} T  mean={q_t.mean():.3f}  min={q_t.min():.3f}  max={q_t.max():.3f}',
            fontsize=9)
        ax_t.axis('off')
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

    row_labels = ['Input'] + [f'Stage {s} (RF={4*(2**s)}×{4*(2**s)})' for s in range(num_stages)]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Sample {sample_idx} | RGB: {deg_info_rgb} | T: {deg_info_t}',
        fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'sample_{sample_idx:03d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


@torch.no_grad()
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    ms = MEAN_STD[args.dataset]
    rgb_mean = ms['rgb_mean'].to(device)
    rgb_std = ms['rgb_std'].to(device)
    t_mean = ms['t_mean'].to(device)
    t_std = ms['t_std'].to(device)

    logger.info('Building QualityAwarePyramidNet...')
    quality_net = QualityAwarePyramidNet(
        in_channels=3,
        mid_channels=64,
        num_stages=4,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f'Loaded checkpoint: {args.checkpoint}')
    else:
        logger.info('No checkpoint provided, using random init')

    quality_net.eval()

    logger.info(f'Building {args.dataset} test dataset...')
    dataset = build_test_dataset(args.dataset, args.data_root)
    logger.info(f'Dataset size: {len(dataset)}')

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    missing_ratio = 0.3
    global_deg_ratio = 0.3

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    stats = {
        'missing': 0, 'global_deg': 0, 'local_deg': 0,
        'q_rgb_mean': [], 'q_t_mean': [],
        'q_rgb_missing_mean': [], 'q_t_missing_mean': [],
    }

    for idx, sample_idx in enumerate(indices):
        data = dataset[sample_idx]
        raw_img = data['img']
        if isinstance(raw_img, np.ndarray):
            if raw_img.ndim == 3 and raw_img.shape[2] in (3, 4, 6):
                raw_img = raw_img.transpose(2, 0, 1)
            raw_img = torch.from_numpy(raw_img.copy())
        while raw_img.dim() > 3:
            raw_img = raw_img.squeeze(0)
        raw_img = raw_img.float().unsqueeze(0).to(device)

        rgb_raw = raw_img[:, :3]
        t_raw = raw_img[:, 3:]

        rgb_norm = renorm_from_01(rgb_raw / 255.0, rgb_mean, rgb_std)
        t_norm = renorm_from_01(t_raw / 255.0, t_mean, t_std)

        r = random.random()
        deg_type_rgb = 'clean'
        deg_type_t = 'clean'
        deg_level_rgb = 0
        deg_level_t = 0

        if r < missing_ratio:
            if random.random() < 0.5:
                rgb_norm = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand_as(rgb_raw).clone()
                deg_type_rgb = 'missing'
                deg_level_rgb = 5
            else:
                t_norm = (-t_mean / t_std).view(1, 3, 1, 1).expand_as(t_raw).clone()
                deg_type_t = 'missing'
                deg_level_t = 5
            stats['missing'] += 1

        elif r < missing_ratio + global_deg_ratio:
            if random.random() < 0.5:
                rgb_norm, deg_type_rgb, deg_level_rgb = apply_degradation(
                    rgb_norm, 'rgb', rgb_mean, rgb_std)
            else:
                t_norm, deg_type_t, deg_level_t = apply_degradation(
                    t_norm, 't', t_mean, t_std)
            stats['global_deg'] += 1

        else:
            B, C, H, W = rgb_norm.shape
            local_mask = generate_local_mask(B, H, W, num_regions=3, device=device)
            if random.random() < 0.5:
                rgb_norm, deg_type_rgb, deg_level_rgb = apply_degradation(
                    rgb_norm, 'rgb', rgb_mean, rgb_std,
                    is_local=True, local_mask=local_mask)
            else:
                t_norm, deg_type_t, deg_level_t = apply_degradation(
                    t_norm, 't', t_mean, t_std,
                    is_local=True, local_mask=local_mask)
            stats['local_deg'] += 1

        rgb_input_01 = denorm_to_01(rgb_norm, rgb_mean, rgb_std)
        t_input_01 = denorm_to_01(t_norm, t_mean, t_std)

        q_rgb_maps = quality_net.forward_rgb(rgb_norm)
        q_t_maps = quality_net.forward_thermal(t_norm)

        for s in range(len(q_rgb_maps)):
            stats['q_rgb_mean'].append(q_rgb_maps[s][0, 0].mean().item())
            stats['q_t_mean'].append(q_t_maps[s][0, 0].mean().item())

        if deg_type_rgb == 'missing':
            for s in range(len(q_rgb_maps)):
                stats['q_rgb_missing_mean'].append(q_rgb_maps[s][0, 0].mean().item())
        if deg_type_t == 'missing':
            for s in range(len(q_t_maps)):
                stats['q_t_missing_mean'].append(q_t_maps[s][0, 0].mean().item())

        save_path = visualize_sample(
            rgb_input_01, t_input_01,
            q_rgb_maps, q_t_maps,
            deg_type_rgb, deg_type_t,
            deg_level_rgb, deg_level_t,
            idx, save_dir)

        logger.info(
            f'[{idx+1}/{num_samples}] Sample {sample_idx} | '
            f'RGB: {deg_type_rgb}(L{deg_level_rgb}) | '
            f'T: {deg_type_t}(L{deg_level_t}) | '
            f'Saved: {save_path}')

    logger.info('=' * 60)
    logger.info('Test Summary:')
    logger.info(f'  Total samples: {num_samples}')
    logger.info(f'  Missing: {stats["missing"]} ({stats["missing"]/num_samples*100:.1f}%)')
    logger.info(f'  Global deg: {stats["global_deg"]} ({stats["global_deg"]/num_samples*100:.1f}%)')
    logger.info(f'  Local deg: {stats["local_deg"]} ({stats["local_deg"]/num_samples*100:.1f}%)')
    if stats['q_rgb_mean']:
        logger.info(f'  RGB quality mean: {np.mean(stats["q_rgb_mean"]):.4f}')
        logger.info(f'  T quality mean: {np.mean(stats["q_t_mean"]):.4f}')
    if stats['q_rgb_missing_mean']:
        logger.info(f'  RGB missing quality mean: {np.mean(stats["q_rgb_missing_mean"]):.4f} (should be ~0)')
    if stats['q_t_missing_mean']:
        logger.info(f'  T missing quality mean: {np.mean(stats["q_t_missing_mean"]):.4f} (should be ~0)')
    logger.info(f'  Results saved to: {save_dir}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='QualityAwarePyramidNet Test Visualization')
    parser.add_argument('--dataset', type=str, default='mfnet',
                        choices=['fmb', 'mfnet'])
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to quality net checkpoint')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--save-dir', type=str,
                        default='work_dirs/quality_test_vis')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)
