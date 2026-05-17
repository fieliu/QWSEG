"""
QualityAwarePyramidNet v2 自监督预训练脚本

核心改进（相比v1）：
  - 缺失场景训练直接在归一化空间生成 (-mean/std)，与语义分割阶段完全一致
  - 增加缺失场景训练频率和多样性（全缺失 + 空间部分缺失）
  - 更强的缺失质量约束（排序损失 + q² 损失 + 对比损失）
  - 训练数据流与语义分割完全一致：归一化数据 → 质量网络

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/train_quality_pyramid_net_v2.py \
      --dataset mfnet \
      --work-dir work_dirs/quality_pyramid_pretrain_v2 \
      --amp

  python tools/train_quality_pyramid_net_v2.py \
      --dataset fmb \
      --work-dir work_dirs/quality_pyramid_pretrain_v2 \
      --amp
"""

import os
import sys
import argparse
import logging
import datetime
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmengine.config import Config
from mmseg.registry import DATASETS, TRANSFORMS, MODELS
from mmseg.datasets.transforms.loading import (
    LoadRGBTImageFromFile, LoadRGBTImageFrom4Channel)
from mmseg.models.utils.quality_aware_pyramid_net import (
    QualityAwarePyramidNet, PyramidRankingLoss)
from mmseg.models.utils.spatial_degradation_generator import SpatialDegradationGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    'mfnet': {
        'config': 'configs/_base_/datasets/mfnet_480x640.py',
        'model_config': 'configs/segformer/mitmul_v10_quality_embed_mit-b2-b0_1xb2-40K_mfnet-480x640.py',
        'dataset_type': 'MFNetDataset',
    },
    'fmb': {
        'config': 'configs/_base_/datasets/fmb_480x640.py',
        'model_config': 'configs/segformer/mitmul_v10_quality_embed_mit-b2-b0_1xb2-40K_fmb-480x640.py',
        'dataset_type': 'FMBDataset',
    },
}


def build_quality_pipeline(dataset_type, img_size, split='train'):
    if dataset_type == 'mfnet':
        load_transform = LoadRGBTImageFrom4Channel(to_float32=True)
    elif dataset_type == 'fmb':
        load_transform = LoadRGBTImageFromFile(
            to_float32=True,
            ir_replace_src='FMB_ALL/FMB',
            ir_replace_dst='FMB_ALL/FMB_T')
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')

    from mmseg.datasets.transforms import Resize, RandomFlip
    if split == 'train':
        pipeline = [load_transform, Resize(scale=(640, img_size), keep_ratio=False), RandomFlip(prob=0.5)]
    else:
        pipeline = [load_transform, Resize(scale=(640, img_size), keep_ratio=False)]
    return pipeline


def build_dataset_from_config(dataset_type, data_root, img_size, split='train'):
    cfg = Config.fromfile(DATASET_CONFIGS[dataset_type]['config'])

    if split == 'train':
        dl_cfg = cfg.train_dataloader
    else:
        dl_cfg = cfg.val_dataloader

    ds_cfg = dl_cfg.dataset.copy()

    if data_root is not None:
        ds_cfg['data_root'] = data_root

    pipeline = build_quality_pipeline(dataset_type, img_size, split)

    if 'ann_file' in ds_cfg:
        if split == 'train':
            ds_cfg['ann_file'] = 'train.txt'
        elif split == 'val':
            ds_cfg['ann_file'] = 'val.txt'

    ds_cfg.pop('pipeline', None)

    dataset_cls = DATASETS.get(DATASET_CONFIGS[dataset_type]['dataset_type'])
    kwargs = dict(
        data_root=ds_cfg.get('data_root'),
        data_prefix=ds_cfg.get('data_prefix'),
        pipeline=pipeline,
    )
    if ds_cfg.get('ann_file') is not None:
        kwargs['ann_file'] = ds_cfg['ann_file']
    dataset = dataset_cls(**kwargs)
    logger.info(f'Built {dataset_type} {split} dataset: {len(dataset)} samples')
    return dataset


def build_preprocessor(dataset_type):
    model_config_path = DATASET_CONFIGS[dataset_type]['model_config']
    cfg = Config.fromfile(model_config_path)
    preprocessor_cfg = cfg.model.data_preprocessor
    preprocessor = MODELS.build(preprocessor_cfg)
    logger.info(
        f'Built SegDataPreProcessor from {model_config_path}: '
        f'mean={preprocessor.mean.squeeze().tolist()[:3]}..., '
        f'std={preprocessor.std.squeeze().tolist()[:3]}...')
    return preprocessor


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


def _denorm_to_01(norm_tensor, mean, std):
    raw = norm_tensor * std + mean
    return (raw / 255.0).clamp(0, 1)


def _renorm_from_01(tensor_01, mean, std):
    raw = tensor_01 * 255.0
    return (raw - mean) / std


def _generate_spatial_missing_mask(B, C, H, W, num_regions_range=(3, 8),
                                   region_size_range=(32, 96), device='cpu'):
    mask = torch.zeros(B, 1, H, W, device=device)
    num_regions = random.randint(*num_regions_range)
    for b in range(B):
        for _ in range(num_regions):
            rh = random.randint(region_size_range[0], min(region_size_range[1], H // 2))
            rw = random.randint(region_size_range[0], min(region_size_range[1], W // 2))
            y0 = random.randint(0, max(0, H - rh))
            x0 = random.randint(0, max(0, W - rw))
            mask[b, 0, y0:y0 + rh, x0:x0 + rw] = 1.0
    return mask


def _compute_missing_loss(quality_net, input_norm, mean, std, modality,
                          missing_ratio=0.5):
    B, C, H, W = input_norm.shape
    device = input_norm.device

    loss_missing = torch.tensor(0.0, device=device)
    count = 0

    if random.random() < missing_ratio:
        full_missing = (-mean / std).view(1, C, 1, 1).expand(B, C, H, W).clone()
        if modality == 'rgb':
            q_missing = quality_net.forward_rgb(full_missing)
        else:
            q_missing = quality_net.forward_thermal(full_missing)

        for q in q_missing:
            loss_missing = loss_missing + (q ** 2).mean()
        count += 1
        del q_missing

    if random.random() < 0.3:
        spatial_mask = _generate_spatial_missing_mask(
            B, C, H, W, device=device)
        spatial_mask_full = spatial_mask.expand(B, C, H, W)

        missing_val = (-mean / std).view(1, C, 1, 1)
        spatial_missing = input_norm.clone()
        spatial_missing = spatial_missing * (1 - spatial_mask_full) + missing_val * spatial_mask_full

        if modality == 'rgb':
            q_spatial = quality_net.forward_rgb(spatial_missing)
        else:
            q_spatial = quality_net.forward_thermal(spatial_missing)

        for q in q_spatial:
            q_up = F.interpolate(q, size=(H, W), mode='bilinear', align_corners=False)
            loss_missing = loss_missing + (q_up * spatial_mask).mean()
        count += 1
        del q_spatial, spatial_missing, spatial_mask, spatial_mask_full

    if count > 0:
        loss_missing = loss_missing / count

    return loss_missing


def _compute_missing_ranking_loss(quality_net, input_norm, mean, std, modality):
    B, C, H, W = input_norm.shape
    device = input_norm.device

    full_missing = (-mean / std).view(1, C, 1, 1).expand(B, C, H, W).clone()

    if modality == 'rgb':
        q_clean = quality_net.forward_rgb(input_norm)
        q_missing = quality_net.forward_rgb(full_missing)
    else:
        q_clean = quality_net.forward_thermal(input_norm)
        q_missing = quality_net.forward_thermal(full_missing)

    loss = torch.tensor(0.0, device=device)
    for q_c, q_m in zip(q_clean, q_missing):
        q_c_mean = q_c.mean()
        q_m_mean = q_m.mean()
        violation = 0.3 + q_m_mean - q_c_mean
        loss = loss + F.relu(violation)

    loss = loss / len(q_clean)
    del q_clean, q_missing
    return loss


def visualize_pyramid_quality(
    rgb_clean, t_clean,
    rgb_deg_list, t_deg_list,
    q_rgb_maps_list, q_t_maps_list,
    token_masks_rgb, token_masks_t,
    deg_type_rgb, deg_type_t,
    epoch, save_dir, writer=None,
    tag='Quality/Visualization', prefix='',
):
    os.makedirs(save_dir, exist_ok=True)

    B = rgb_clean.shape[0]
    K = len(rgb_deg_list)
    num_stages = len(q_rgb_maps_list[0])
    img_H, img_W = rgb_clean.shape[2], rgb_clean.shape[3]

    num_cols = 2 + 2 * K
    num_rows = 1 + num_stages

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(2.5 * num_cols, 2.5 * num_rows),
    )
    if num_cols == 1:
        axes = axes.reshape(num_rows, 1)

    for b in range(min(B, 2)):
        all_images = [rgb_clean[b], t_clean[b]]
        all_q_stages = [q_rgb_maps_list[0], q_t_maps_list[0]]
        col_titles = ['RGB clean', 'T clean']

        for k in range(K):
            all_images.append(rgb_deg_list[k][b])
            all_q_stages.append(q_rgb_maps_list[k + 1])
            col_titles.append(f'RGB deg{k+2}')

        for k in range(K):
            all_images.append(t_deg_list[k][b])
            all_q_stages.append(q_t_maps_list[k + 1])
            col_titles.append(f'T deg{k+2}')

        for col_idx in range(num_cols):
            ax_img = axes[0, col_idx]
            img = all_images[col_idx].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax_img.imshow(img)
            ax_img.set_title(col_titles[col_idx], fontsize=7)
            ax_img.axis('off')

            q_maps = all_q_stages[col_idx]
            for s in range(num_stages):
                ax_q = axes[1 + s, col_idx]
                q = q_maps[s][b, 0].detach().cpu().numpy()
                q_up = F.interpolate(
                    torch.from_numpy(q).float().unsqueeze(0).unsqueeze(0),
                    size=(img_H, img_W), mode='bilinear',
                    align_corners=False).squeeze().numpy()
                ax_q.imshow(q_up, vmin=0, vmax=1, cmap='RdBu_r')
                ax_q.set_title(f'S{s} q={q.mean():.2f}', fontsize=6)
                ax_q.axis('off')

    row_labels = ['Image'] + [f'Stage {s}' for s in range(num_stages)]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=9, fontweight='bold')

    fig.suptitle(
        f'Epoch {epoch+1} | RGB: {deg_type_rgb} | T: {deg_type_t}',
        fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(
        save_dir,
        f'{prefix}_epoch_{epoch+1:03d}.png' if prefix
        else f'epoch_{epoch+1:03d}.png')
    if writer is not None:
        writer.add_figure(tag, fig, global_step=epoch)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def validate(quality_net, spatial_deg_gen, loss_fn,
             val_loader, preprocessor, device, epoch, vis_dir, writer, args):
    quality_net.eval()

    rgb_mean = preprocessor.mean[:3]
    rgb_std = preprocessor.std[:3]
    ir_mean = preprocessor.mean[3:]
    ir_std = preprocessor.std[3:]

    val_loss_total = 0.0
    val_loss_rank = 0.0
    val_loss_deg5 = 0.0
    val_loss_consist = 0.0
    val_loss_clean = 0.0
    val_loss_cross_scale = 0.0
    val_loss_missing = 0.0
    num_batches = 0

    for batch_idx, raw_batch in enumerate(val_loader):
        raw_batch = raw_batch.to(device, non_blocking=True)

        with torch.no_grad():
            norm_batch = (raw_batch - preprocessor.mean) / preprocessor.std
            input_rgb = norm_batch[:, :3]
            input_ir = norm_batch[:, 3:]

            rgb_raw = _denorm_to_01(input_rgb, rgb_mean, rgb_std)
            t_raw = _denorm_to_01(input_ir, ir_mean, ir_std)

            spatial_deg_gen.prepare_degradation(rgb_raw, modality='rgb')

            rgb_batch, B_rgb = spatial_deg_gen.apply_all_levels_batched(
                rgb_raw, mean=rgb_mean, std=rgb_std)
            q_rgb_all = quality_net.forward_rgb(rgb_batch)
            del rgb_batch

            q_rgb_maps_list = []
            for i in range(5):
                s, e = i * B_rgb, (i + 1) * B_rgb
                q_rgb_maps_list.append([q[s:e] for q in q_rgb_all])
            del q_rgb_all

            rgb_mask_list = spatial_deg_gen.get_level_mask_list()

            _, loss_dict_rgb = loss_fn(q_rgb_maps_list, rgb_mask_list)

            loss_missing_rgb = _compute_missing_loss(
                quality_net, input_rgb, rgb_mean, rgb_std, 'rgb',
                missing_ratio=1.0)

            if batch_idx == 0:
                q_rgb_vis = [[q_s.detach() for q_s in q_level] for q_level in q_rgb_maps_list]
                rgb_mask_vis = [m.clone() for m in rgb_mask_list]
                rgb_deg_imgs = []
                for k in range(1, 5):
                    rgb_deg_imgs.append(spatial_deg_gen.apply_level(rgb_raw, k).clone())
                deg_type_rgb = spatial_deg_gen._deg_type

            del q_rgb_maps_list, rgb_mask_list

            spatial_deg_gen.prepare_degradation(t_raw, modality='t')

            t_batch, B_t = spatial_deg_gen.apply_all_levels_batched(
                t_raw, mean=ir_mean, std=ir_std)
            q_t_all = quality_net.forward_thermal(t_batch)
            del t_batch

            q_t_maps_list = []
            for i in range(5):
                s, e = i * B_t, (i + 1) * B_t
                q_t_maps_list.append([q[s:e] for q in q_t_all])
            del q_t_all

            t_mask_list = spatial_deg_gen.get_level_mask_list()

            _, loss_dict_t = loss_fn(q_t_maps_list, t_mask_list)

            loss_missing_t = _compute_missing_loss(
                quality_net, input_ir, ir_mean, ir_std, 't',
                missing_ratio=1.0)

            if batch_idx == 0:
                q_t_vis = [[q_s.detach() for q_s in q_level] for q_level in q_t_maps_list]
                t_mask_vis = [m.clone() for m in t_mask_list]
                t_deg_imgs = []
                for k in range(1, 5):
                    t_deg_imgs.append(spatial_deg_gen.apply_level(t_raw, k).clone())
                deg_type_t = spatial_deg_gen._deg_type
                visualize_pyramid_quality(
                    rgb_raw, t_raw,
                    rgb_deg_imgs, t_deg_imgs,
                    q_rgb_vis, q_t_vis,
                    rgb_mask_vis, t_mask_vis,
                    deg_type_rgb, deg_type_t,
                    epoch, vis_dir, writer=writer,
                    tag='Quality/Val_Visualization',
                    prefix='val',
                )
                del q_rgb_vis, q_t_vis, rgb_mask_vis, t_mask_vis
                del rgb_deg_imgs, t_deg_imgs

            del q_t_maps_list, t_mask_list
            del rgb_raw, t_raw, norm_batch, input_rgb, input_ir, raw_batch

        loss = (loss_dict_rgb['total_loss'] +
                loss_dict_t['total_loss']) / 2.0
        loss_miss = (loss_missing_rgb + loss_missing_t) / 2.0

        val_loss_total += loss.item()
        val_loss_rank += (loss_dict_rgb['loss_rank'].item() +
                          loss_dict_t['loss_rank'].item()) / 2.0
        val_loss_deg5 += (loss_dict_rgb['loss_deg5'].item() +
                          loss_dict_t['loss_deg5'].item()) / 2.0
        val_loss_consist += (loss_dict_rgb['loss_consist'].item() +
                             loss_dict_t['loss_consist'].item()) / 2.0
        val_loss_clean += (loss_dict_rgb['loss_clean'].item() +
                           loss_dict_t['loss_clean'].item()) / 2.0
        val_loss_cross_scale += (loss_dict_rgb['loss_cross_scale'].item() +
                                 loss_dict_t['loss_cross_scale'].item()) / 2.0
        val_loss_missing += loss_miss.item()
        num_batches += 1

    avg_loss = val_loss_total / max(num_batches, 1)
    avg_rank = val_loss_rank / max(num_batches, 1)
    avg_deg5 = val_loss_deg5 / max(num_batches, 1)
    avg_consist = val_loss_consist / max(num_batches, 1)
    avg_clean = val_loss_clean / max(num_batches, 1)
    avg_cross_scale = val_loss_cross_scale / max(num_batches, 1)
    avg_missing = val_loss_missing / max(num_batches, 1)

    logger.info(
        f'[Val] Epoch {epoch+1} | '
        f'Loss: {avg_loss:.4f} | '
        f'Rank: {avg_rank:.4f} | '
        f'Deg5: {avg_deg5:.4f} | '
        f'Consist: {avg_consist:.4f} | '
        f'Clean: {avg_clean:.4f} | '
        f'CrossS: {avg_cross_scale:.4f} | '
        f'Missing: {avg_missing:.4f}')

    writer.add_scalar('val/loss_total', avg_loss, epoch)
    writer.add_scalar('val/loss_rank', avg_rank, epoch)
    writer.add_scalar('val/loss_deg5', avg_deg5, epoch)
    writer.add_scalar('val/loss_consist', avg_consist, epoch)
    writer.add_scalar('val/loss_clean', avg_clean, epoch)
    writer.add_scalar('val/loss_cross_scale', avg_cross_scale, epoch)
    writer.add_scalar('val/loss_missing', avg_missing, epoch)

    quality_net.train()
    return avg_loss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(args.work_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)

    log_dir = os.path.join(work_dir, 'log')
    weight_dir = os.path.join(work_dir, 'weight')
    vis_dir = os.path.join(work_dir, 'vis_data')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    config_path = os.path.join(work_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f'Model: QualityAwarePyramidNet v2\n')
        f.write(f'Image size: {args.img_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Learning rate: {args.lr}\n')
        f.write(f'AMP: {args.amp}\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Data root: {args.data_root}\n')
        f.write(f'Losses:\n')
        f.write(f'  Ranking margin weight: {args.w_rank}\n')
        f.write(f'  Deg5 quality weight: {args.w_deg5}\n')
        f.write(f'  Consistency weight: {args.w_consist}\n')
        f.write(f'  Contrast weight: {args.w_contrast}\n')
        f.write(f'  Missing weight: {args.w_missing}\n')
        f.write(f'  Missing ranking weight: {args.w_missing_rank}\n')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f'Work dir: {work_dir}')
    logger.info(f'Args: {vars(args)}')

    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    best_val_loss = float('inf')

    scaler = GradScaler(enabled=args.amp)

    logger.info('Building QualityAwarePyramidNet...')
    quality_net = QualityAwarePyramidNet(
        in_channels=3,
        mid_channels=64,
        num_stages=4,
    ).to(device)

    trainable_params = sum(
        p.numel() for p in quality_net.parameters() if p.requires_grad)
    logger.info(f'QualityAwarePyramidNet params: {trainable_params:,}')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f'Resumed from {args.resume}')

    loss_fn = PyramidRankingLoss(
        margin=args.margin_rank,
        max_deg5_quality=args.max_deg5_quality,
        consistency_margin=args.consistency_margin,
        contrast_margin=args.contrast_margin,
        contrast_topk=args.contrast_topk,
        min_clean_quality=args.min_clean_quality,
        w_rank=args.w_rank,
        w_deg5=args.w_deg5,
        w_consist=args.w_consist,
        w_contrast=args.w_contrast,
        w_clean=args.w_clean,
        w_cross_scale=1.0,
    )

    spatial_deg_gen = SpatialDegradationGenerator(
        num_regions_range=(5, 10),
        region_size_range=(32, 80),
        num_levels=5,
        num_stages=4,
    )

    optimizer = optim.AdamW(
        quality_net.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    logger.info(f'Building {args.dataset} dataset via mmseg pipeline...')
    train_dataset = build_dataset_from_config(
        args.dataset, args.data_root, args.img_size, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_dataset = build_dataset_from_config(
        args.dataset, args.data_root, args.img_size, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info('Building SegDataPreProcessor from segmentation config...')
    preprocessor = build_preprocessor(args.dataset)
    preprocessor.to(device)

    rgb_mean = preprocessor.mean[:3]
    rgb_std = preprocessor.std[:3]
    ir_mean = preprocessor.mean[3:]
    ir_std = preprocessor.std[3:]
    logger.info(
        f'  RGB mean: {rgb_mean.squeeze().tolist()}, '
        f'std: {rgb_std.squeeze().tolist()}')
    logger.info(
        f'  IR  mean: {ir_mean.squeeze().tolist()}, '
        f'std: {ir_std.squeeze().tolist()}')
    logger.info(
        f'  RGB missing value (norm): {(-rgb_mean / rgb_std).squeeze().tolist()}')
    logger.info(
        f'  IR  missing value (norm): {(-ir_mean / ir_std).squeeze().tolist()}')

    global_step = 0

    for epoch in range(args.epochs):
        quality_net.train()
        epoch_loss = 0.0
        epoch_loss_rank = 0.0
        epoch_loss_deg5 = 0.0
        epoch_loss_consist = 0.0
        epoch_loss_contrast = 0.0
        epoch_loss_clean = 0.0
        epoch_loss_cross_scale = 0.0
        epoch_loss_missing = 0.0
        epoch_loss_missing_rank = 0.0
        num_batches = 0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, raw_batch in enumerate(pbar):
            raw_batch = raw_batch.to(device, non_blocking=True)

            norm_batch = (raw_batch - preprocessor.mean) / preprocessor.std
            input_rgb = norm_batch[:, :3]
            input_ir = norm_batch[:, 3:]

            rgb_raw = _denorm_to_01(input_rgb, rgb_mean, rgb_std)
            t_raw = _denorm_to_01(input_ir, ir_mean, ir_std)

            with autocast(enabled=args.amp):
                spatial_deg_gen.prepare_degradation(rgb_raw, modality='rgb')

                rgb_batch, B_rgb = spatial_deg_gen.apply_all_levels_batched(
                    rgb_raw, mean=rgb_mean, std=rgb_std)
                q_rgb_all = quality_net.forward_rgb(rgb_batch)
                del rgb_batch

                q_rgb_maps_list = []
                for i in range(5):
                    s, e = i * B_rgb, (i + 1) * B_rgb
                    q_rgb_maps_list.append([q[s:e] for q in q_rgb_all])
                del q_rgb_all

                rgb_mask_list = spatial_deg_gen.get_level_mask_list()

                _, loss_dict_rgb = loss_fn(q_rgb_maps_list, rgb_mask_list)
                loss_rgb = loss_dict_rgb['total_loss'] / args.grad_accum_steps

                loss_missing_rgb = _compute_missing_loss(
                    quality_net, input_rgb, rgb_mean, rgb_std, 'rgb',
                    missing_ratio=args.missing_ratio)

                loss_missing_rank_rgb = _compute_missing_ranking_loss(
                    quality_net, input_rgb, rgb_mean, rgb_std, 'rgb')

                loss_rgb = loss_rgb + (
                    args.w_missing * loss_missing_rgb +
                    args.w_missing_rank * loss_missing_rank_rgb
                ) / args.grad_accum_steps

                del q_rgb_maps_list, rgb_mask_list
                del loss_missing_rgb, loss_missing_rank_rgb

                spatial_deg_gen.prepare_degradation(t_raw, modality='t')

                t_batch, B_t = spatial_deg_gen.apply_all_levels_batched(
                    t_raw, mean=ir_mean, std=ir_std)
                q_t_all = quality_net.forward_thermal(t_batch)
                del t_batch

                q_t_maps_list = []
                for i in range(5):
                    s, e = i * B_t, (i + 1) * B_t
                    q_t_maps_list.append([q[s:e] for q in q_t_all])
                del q_t_all

                t_mask_list = spatial_deg_gen.get_level_mask_list()

                _, loss_dict_t = loss_fn(q_t_maps_list, t_mask_list)
                loss_t = loss_dict_t['total_loss'] / args.grad_accum_steps

                loss_missing_t = _compute_missing_loss(
                    quality_net, input_ir, ir_mean, ir_std, 't',
                    missing_ratio=args.missing_ratio)

                loss_missing_rank_t = _compute_missing_ranking_loss(
                    quality_net, input_ir, ir_mean, ir_std, 't')

                loss_t = loss_t + (
                    args.w_missing * loss_missing_t +
                    args.w_missing_rank * loss_missing_rank_t
                ) / args.grad_accum_steps

                del q_t_maps_list, t_mask_list
                del loss_missing_t, loss_missing_rank_t

                total_loss = loss_rgb + loss_t

            scaler.scale(total_loss).backward()
            del loss_rgb, loss_t, total_loss
            del rgb_raw, t_raw, norm_batch, raw_batch

            if (batch_idx + 1) % args.grad_accum_steps == 0 or \
               (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss_val = (loss_dict_rgb['total_loss'].item() +
                              loss_dict_t['total_loss'].item()) / 2.0
            epoch_loss += total_loss_val
            epoch_loss_rank += (loss_dict_rgb['loss_rank'].item() +
                                loss_dict_t['loss_rank'].item()) / 2.0
            epoch_loss_deg5 += (loss_dict_rgb['loss_deg5'].item() +
                                loss_dict_t['loss_deg5'].item()) / 2.0
            epoch_loss_consist += (loss_dict_rgb['loss_consist'].item() +
                                   loss_dict_t['loss_consist'].item()) / 2.0
            epoch_loss_contrast += (loss_dict_rgb['loss_contrast'].item() +
                                    loss_dict_t['loss_contrast'].item()) / 2.0
            epoch_loss_clean += (loss_dict_rgb['loss_clean'].item() +
                                 loss_dict_t['loss_clean'].item()) / 2.0
            epoch_loss_cross_scale += (loss_dict_rgb['loss_cross_scale'].item() +
                                       loss_dict_t['loss_cross_scale'].item()) / 2.0

            writer.add_scalar('train/loss_step', total_loss_val, global_step)
            global_step += 1

            pbar.set_postfix(
                loss=f'{total_loss_val:.4f}',
                lr=f'{optimizer.param_groups[0]["lr"]:.6f}')

            if batch_idx == 0:
                with torch.no_grad():
                    rgb_raw_v = _denorm_to_01(input_rgb, rgb_mean, rgb_std)
                    t_raw_v = _denorm_to_01(input_ir, ir_mean, ir_std)

                    spatial_deg_gen.prepare_degradation(rgb_raw_v, modality='rgb')
                    rgb_batch_v, B_v = spatial_deg_gen.apply_all_levels_batched(
                        rgb_raw_v, mean=rgb_mean, std=rgb_std)
                    q_rgb_all_v = quality_net.forward_rgb(rgb_batch_v)
                    del rgb_batch_v
                    q_rgb_v = []
                    for i in range(5):
                        s, e = i * B_v, (i + 1) * B_v
                        q_rgb_v.append([q[s:e] for q in q_rgb_all_v])
                    del q_rgb_all_v
                    rgb_mask_v = spatial_deg_gen.get_level_mask_list()
                    rgb_deg_imgs = []
                    for k in range(1, 5):
                        rgb_deg_imgs.append(spatial_deg_gen.apply_level(rgb_raw_v, k))
                    deg_type_rgb_v = spatial_deg_gen._deg_type

                    spatial_deg_gen.prepare_degradation(t_raw_v, modality='t')
                    t_batch_v, B_v = spatial_deg_gen.apply_all_levels_batched(
                        t_raw_v, mean=ir_mean, std=ir_std)
                    q_t_all_v = quality_net.forward_thermal(t_batch_v)
                    del t_batch_v
                    q_t_v = []
                    for i in range(5):
                        s, e = i * B_v, (i + 1) * B_v
                        q_t_v.append([q[s:e] for q in q_t_all_v])
                    del q_t_all_v
                    t_mask_v = spatial_deg_gen.get_level_mask_list()
                    t_deg_imgs = []
                    for k in range(1, 5):
                        t_deg_imgs.append(spatial_deg_gen.apply_level(t_raw_v, k))
                    deg_type_t_v = spatial_deg_gen._deg_type

                    visualize_pyramid_quality(
                        rgb_raw_v, t_raw_v,
                        rgb_deg_imgs, t_deg_imgs,
                        q_rgb_v, q_t_v,
                        rgb_mask_v, t_mask_v,
                        deg_type_rgb_v, deg_type_t_v,
                        epoch, vis_dir, writer=writer,
                        tag='Quality/Train_Visualization',
                        prefix='train',
                    )
                    del q_rgb_v, q_t_v
                    del rgb_mask_v, t_mask_v, rgb_deg_imgs, t_deg_imgs
                    del rgb_raw_v, t_raw_v

            del input_rgb, input_ir

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_rank = epoch_loss_rank / max(num_batches, 1)
        avg_deg5 = epoch_loss_deg5 / max(num_batches, 1)
        avg_consist = epoch_loss_consist / max(num_batches, 1)
        avg_contrast = epoch_loss_contrast / max(num_batches, 1)
        avg_clean = epoch_loss_clean / max(num_batches, 1)
        avg_cross_scale = epoch_loss_cross_scale / max(num_batches, 1)

        logger.info(
            f'[Train] Epoch {epoch+1} | '
            f'Loss: {avg_loss:.4f} | '
            f'Rank: {avg_rank:.4f} | '
            f'Deg5: {avg_deg5:.4f} | '
            f'Consist: {avg_consist:.4f} | '
            f'Contrast: {avg_contrast:.4f} | '
            f'Clean: {avg_clean:.4f} | '
            f'CrossS: {avg_cross_scale:.4f}')

        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('train/loss_rank', avg_rank, epoch)
        writer.add_scalar('train/loss_deg5', avg_deg5, epoch)
        writer.add_scalar('train/loss_consist', avg_consist, epoch)
        writer.add_scalar('train/loss_contrast', avg_contrast, epoch)
        writer.add_scalar('train/loss_clean', avg_clean, epoch)
        writer.add_scalar('train/loss_cross_scale', avg_cross_scale, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(
                weight_dir, 'best_quality_pyramid_net.pth')
            torch.save(quality_net.state_dict(), save_path)
            logger.info(f'Best model saved: {save_path} (loss={best_loss:.4f})')

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                weight_dir,
                f'quality_pyramid_net_epoch_{epoch+1}.pth')
            torch.save(quality_net.state_dict(), save_path)

        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(
                quality_net, spatial_deg_gen, loss_fn,
                val_loader, preprocessor, device, epoch, vis_dir, writer, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    weight_dir, 'best_val_quality_pyramid_net.pth')
                torch.save(quality_net.state_dict(), save_path)
                logger.info(
                    f'Best val model saved: {save_path} '
                    f'(val_loss={best_val_loss:.4f})')

    final_path = os.path.join(weight_dir, 'final_quality_pyramid_net.pth')
    torch.save(quality_net.state_dict(), final_path)
    logger.info(f'Final model saved: {final_path}')

    info_path = os.path.join(weight_dir, 'checkpoint_info.txt')
    with open(info_path, 'w') as f:
        f.write(f'best_train_loss: {best_loss:.6f}\n')
        f.write(f'best_val_loss: {best_val_loss:.6f}\n')
        f.write(f'best_train_path: {os.path.join(weight_dir, "best_quality_pyramid_net.pth")}\n')
        f.write(f'best_val_path: {os.path.join(weight_dir, "best_val_quality_pyramid_net.pth")}\n')
        f.write(f'final_path: {final_path}\n')

    writer.close()
    logger.info('Training complete!')


def parse_args():
    parser = argparse.ArgumentParser(
        description='QualityAwarePyramidNet v2 Pretraining')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root from config')
    parser.add_argument('--dataset', type=str, default='mfnet',
                        choices=['fmb', 'mfnet'])
    parser.add_argument('--work-dir', type=str,
                        default='work_dirs/quality_pyramid_pretrain_v2')
    parser.add_argument('--img-size', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum-steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--margin-rank', type=float, default=0.1)
    parser.add_argument('--max-deg5-quality', type=float, default=0.1)
    parser.add_argument('--consistency-margin', type=float, default=0.05)
    parser.add_argument('--w-rank', type=float, default=1.0)
    parser.add_argument('--w-deg5', type=float, default=0.1)
    parser.add_argument('--w-consist', type=float, default=0.5)
    parser.add_argument('--contrast-margin', type=float, default=0.3)
    parser.add_argument('--contrast-topk', type=float, default=0.2)
    parser.add_argument('--w-contrast', type=float, default=0.3)
    parser.add_argument('--min-clean-quality', type=float, default=0.7)
    parser.add_argument('--w-clean', type=float, default=0.3)
    parser.add_argument('--missing-ratio', type=float, default=0.5,
                        help='Probability of training missing scenario per batch')
    parser.add_argument('--w-missing', type=float, default=2.0,
                        help='Weight for missing quality loss (q^2)')
    parser.add_argument('--w-missing-rank', type=float, default=1.0,
                        help='Weight for missing vs clean ranking loss')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
