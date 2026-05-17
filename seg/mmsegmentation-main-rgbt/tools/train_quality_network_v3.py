"""
QualityNetworkV3 自监督预训练脚本

核心思路:
  - 输入: MAE ViT-B 最后阶段输出的token特征 (768维)
  - 架构: 双分支(RGB/Thermal独立)，GlobalNorm + LocalChannelAttention，逐token质量评分
  - 训练: 对同一图像生成5级递增空间退化版本 (随机区域, 非全局退化)
  - 退化: 随机2-6个区域, 每区域3x3-5x5 token范围, 5级别 (1=原图, 5=最强)
  - 损失: 级别一致性损失 + 相邻级别排序损失
  - 不使用伪标签、SpatialDiversityLoss和CleanAnchorLoss

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/train_quality_network_v3.py \
      --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
      --work-dir work_dirs/quality_v3_pretrain \
      --img-size 480 \
      --amp
"""

import os
import sys
import argparse
import logging
import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmseg.models.backbones.mae import MAE
from mmseg.models.utils.quality_network_v3 import (
    QualityNetworkV3,
    RankingMarginLoss,
    LevelConsistencyLoss,
    ImagePseudoLabelGenerator,
    ThermalPseudoLabelGenerator,
    IntraImageRankingLoss,
)
from mmseg.models.utils.spatial_degradation_generator import SpatialDegradationGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FMBQualityDataset(Dataset):

    def __init__(self, data_root, split='training', img_size=480):
        self.data_root = data_root
        self.img_size = img_size

        split_map = {
            'training': 'training',
            'validation': 'validation',
            'testing': 'validation',
            'val': 'validation',
            'train': 'training'
        }
        actual_split = split_map.get(split, split)

        self.rgb_dir = os.path.join(data_root, 'FMB', 'images', actual_split)
        self.thermal_dir = os.path.join(data_root, 'FMB_T', 'images', actual_split)

        if not os.path.exists(self.rgb_dir):
            self.rgb_dir = os.path.join(data_root, 'images', actual_split)
            self.thermal_dir = os.path.join(data_root, 'images', actual_split)

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB directory not found. Tried:\n"
                                    f"  1. {os.path.join(data_root, 'FMB', 'images', actual_split)}\n"
                                    f"  2. {os.path.join(data_root, 'images', actual_split)}")
        if not os.path.exists(self.thermal_dir):
            raise FileNotFoundError(f"Thermal directory not found. Tried:\n"
                                    f"  1. {os.path.join(data_root, 'FMB_T', 'images', actual_split)}\n"
                                    f"  2. {os.path.join(data_root, 'images', actual_split)}")

        self.filenames = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.rgb_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

        valid = []
        for name in self.filenames:
            rgb_ok = os.path.exists(
                os.path.join(self.rgb_dir, name + '.png'))
            t_ok = os.path.exists(
                os.path.join(self.thermal_dir, name + '.png'))
            if rgb_ok and t_ok:
                valid.append(name)
        self.filenames = valid

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        rgb_bgr = cv2.imread(os.path.join(self.rgb_dir, name + '.png'))
        t_bgr = cv2.imread(os.path.join(self.thermal_dir, name + '.png'))

        if rgb_bgr.shape[0] != self.img_size or \
           rgb_bgr.shape[1] != self.img_size:
            rgb_bgr = cv2.resize(rgb_bgr, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
            t_bgr = cv2.resize(t_bgr, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)

        rgb_float = rgb_bgr.astype(np.float32) / 255.0
        t_float = t_bgr.astype(np.float32) / 255.0

        img_rgb = torch.from_numpy(rgb_float).permute(2, 0, 1)
        img_t = torch.from_numpy(t_float).permute(2, 0, 1)

        return img_rgb, img_t


def visualize_quality_epoch(
    rgb_clean, t_clean,
    rgb_deg_list, t_deg_list,
    q_rgb_list, q_t_list,
    token_masks_rgb, token_masks_t,
    deg_type_rgb, deg_type_t,
    epoch, save_dir, writer=None,
    tag='Quality/Visualization', prefix='',
    prune_threshold=0.3,
):
    os.makedirs(save_dir, exist_ok=True)

    B = rgb_clean.shape[0]
    K = len(rgb_deg_list)
    img_H, img_W = rgb_clean.shape[2], rgb_clean.shape[3]

    N_q = q_rgb_list[0].shape[-1]
    q_H = q_W = int(N_q ** 0.5)

    num_cols = 2 + 2 * K

    fig, axes = plt.subplots(
        3, num_cols * B,
        figsize=(2.5 * num_cols * B, 3 * 3),
    )
    if num_cols * B == 1:
        axes = axes.reshape(3, 1)

    for b in range(B):
        col_offset = b * num_cols

        all_images = [rgb_clean[b], t_clean[b]]
        all_q = [q_rgb_list[0][b], q_t_list[0][b]]
        col_titles = ['RGB clean', 'T clean']

        for k in range(K):
            all_images.append(rgb_deg_list[k][b])
            all_q.append(q_rgb_list[k + 1][b])
            col_titles.append(f'RGB deg{k+2}')

        for k in range(K):
            all_images.append(t_deg_list[k][b])
            all_q.append(q_t_list[k + 1][b])
            col_titles.append(f'T deg{k+2}')

        for col_idx in range(num_cols):
            ax_img = axes[0, col_offset + col_idx]
            ax_qmap = axes[1, col_offset + col_idx]
            ax_prune = axes[2, col_offset + col_idx]

            img = all_images[col_idx].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            ax_img.imshow(img)
            ax_img.set_title(col_titles[col_idx], fontsize=8)
            ax_img.axis('off')

            q = all_q[col_idx].detach().cpu().numpy()
            q_2d = q.reshape(q_H, q_W)
            q_up = F.interpolate(
                torch.from_numpy(q_2d).float().unsqueeze(0).unsqueeze(0),
                size=(img_H, img_W), mode='bilinear', align_corners=False,
            ).squeeze().numpy()

            ax_qmap.imshow(q_up, vmin=0, vmax=1, cmap='RdBu_r')
            q_mean = q.mean()
            ax_qmap.set_title(f'q={q_mean:.2f}', fontsize=7)
            ax_qmap.axis('off')

            prune_mask = (q < prune_threshold).astype(np.float32)
            prune_mask_2d = prune_mask.reshape(q_H, q_W)
            prune_mask_up = F.interpolate(
                torch.from_numpy(prune_mask_2d).float().unsqueeze(0).unsqueeze(0),
                size=(img_H, img_W), mode='nearest',
            ).squeeze().numpy()
            pruned_img = img.copy()
            pruned_img[prune_mask_up > 0.5] = 0.0
            prune_ratio = prune_mask.mean() * 100
            ax_prune.imshow(pruned_img)
            ax_prune.set_title(f'pruned {prune_ratio:.0f}%', fontsize=7)
            ax_prune.axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Quality Map', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel(f'Pruned (<{prune_threshold})', fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Epoch {epoch+1} | RGB deg: {deg_type_rgb} | T deg: {deg_type_t}',
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_epoch_{epoch+1:03d}.png') if prefix else os.path.join(save_dir, f'epoch_{epoch+1:03d}.png')
    if writer is not None:
        writer.add_figure(tag, fig, global_step=epoch)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Quality visualization saved: {save_path}')


@torch.no_grad()
def validate(quality_net, mae_backbone, spatial_deg_gen,
             pseudo_label_gen, thermal_pseudo_label_gen, intra_rank_fn,
             val_loader, device,
             epoch, vis_dir, writer, args):
    quality_net.eval()

    ranking_fn = RankingMarginLoss(margin=args.level_margin)
    level_fn = LevelConsistencyLoss(
        num_levels=5, level_margin=args.level_margin,
        min_low_quality=args.min_low_quality,
        min_clean_quality=args.min_clean_quality)

    val_loss_total = 0.0
    val_loss_level = 0.0
    val_loss_ranking = 0.0
    val_loss_intra = 0.0
    num_batches = 0

    q_rgb_clean_all = []
    q_t_clean_all = []
    q_rgb_deg_last_all = []
    q_t_deg_last_all = []

    for batch_idx, (rgb_clean, t_clean) in enumerate(val_loader):
        rgb_clean = rgb_clean.to(device, non_blocking=True)
        t_clean = t_clean.to(device, non_blocking=True)

        rgb_multi, deg_type_rgb = spatial_deg_gen.generate_multi_level(
            rgb_clean, modality='rgb')
        t_multi, deg_type_t = spatial_deg_gen.generate_multi_level(
            t_clean, modality='t')

        q_rgb_list = []
        q_t_list = []
        for level in range(5):
            feat_rgb = mae_backbone(rgb_multi[level][0])
            if isinstance(feat_rgb, (tuple, list)):
                feat_rgb = feat_rgb[-1]
            tok_rgb = feat_rgb.flatten(2).transpose(1, 2)

            feat_t = mae_backbone(t_multi[level][0])
            if isinstance(feat_t, (tuple, list)):
                feat_t = feat_t[-1]
            tok_t = feat_t.flatten(2).transpose(1, 2)

            q_rgb_list.append(quality_net(tok_rgb, modality='rgb'))
            q_t_list.append(quality_net(tok_t, modality='t'))

        tok_H = tok_rgb.shape[1] ** 0.5
        tok_H = int(tok_H) if tok_H == int(tok_H) else int(tok_H ** 0.5)
        tok_size = (tok_H, tok_H)
        rgb_token_mask = F.adaptive_avg_pool2d(
            rgb_multi[1][1], tok_size).reshape(rgb_multi[1][1].shape[0], -1)
        t_token_mask = F.adaptive_avg_pool2d(
            t_multi[1][1], tok_size).reshape(t_multi[1][1].shape[0], -1)

        if batch_idx == 0:
            rgb_deg_imgs = [rgb_multi[k][0] for k in range(1, 5)]
            t_deg_imgs = [t_multi[k][0] for k in range(1, 5)]
            rgb_masks = [rgb_multi[k][1] for k in range(1, 5)]
            t_masks = [t_multi[k][1] for k in range(1, 5)]
            visualize_quality_epoch(
                rgb_clean, t_clean,
                rgb_deg_imgs, t_deg_imgs,
                q_rgb_list, q_t_list,
                rgb_masks, t_masks,
                deg_type_rgb, deg_type_t,
                epoch, vis_dir, writer=writer,
                tag='Quality/Val_Visualization',
                prefix='val',
                prune_threshold=args.prune_threshold,
            )

        loss_level_rgb = level_fn(q_rgb_list, mask=rgb_token_mask)
        loss_level_t = level_fn(q_t_list, mask=t_token_mask)
        loss_level = (loss_level_rgb + loss_level_t) / 2.0

        loss_ranking = torch.tensor(0.0, device=device)
        for i in range(len(q_rgb_list) - 1):
            loss_ranking = loss_ranking + ranking_fn(q_rgb_list[i], q_rgb_list[i + 1], mask=rgb_token_mask)
            loss_ranking = loss_ranking + ranking_fn(q_t_list[i], q_t_list[i + 1], mask=t_token_mask)
        loss_ranking = loss_ranking / (2 * (len(q_rgb_list) - 1))

        rgb_pseudo = pseudo_label_gen(rgb_clean * 255.0)
        t_pseudo = thermal_pseudo_label_gen(t_clean * 255.0)
        loss_intra_rgb = intra_rank_fn(q_rgb_list[0], rgb_pseudo)
        loss_intra_t = intra_rank_fn(q_t_list[0], t_pseudo)
        loss_intra = (loss_intra_rgb + loss_intra_t) / 2.0

        loss = (loss_level * args.loss_level_weight
                + loss_ranking * args.loss_ranking_weight
                + loss_intra * args.loss_intra_weight)

        val_loss_total += loss.item()
        val_loss_level += loss_level.item()
        val_loss_ranking += loss_ranking.item()
        val_loss_intra += loss_intra.item()
        num_batches += 1

        q_rgb_clean_all.append(q_rgb_list[0].mean().item())
        q_t_clean_all.append(q_t_list[0].mean().item())
        q_rgb_deg_last_all.append(q_rgb_list[-1].mean().item())
        q_t_deg_last_all.append(q_t_list[-1].mean().item())

    avg_loss = val_loss_total / max(num_batches, 1)
    avg_level = val_loss_level / max(num_batches, 1)
    avg_ranking = val_loss_ranking / max(num_batches, 1)
    avg_intra = val_loss_intra / max(num_batches, 1)

    avg_q_rgb_clean = np.mean(q_rgb_clean_all) if q_rgb_clean_all else 0.0
    avg_q_t_clean = np.mean(q_t_clean_all) if q_t_clean_all else 0.0
    avg_q_rgb_deg_last = np.mean(q_rgb_deg_last_all) if q_rgb_deg_last_all else 0.0
    avg_q_t_deg_last = np.mean(q_t_deg_last_all) if q_t_deg_last_all else 0.0

    logger.info(
        f'[Val] Epoch {epoch+1} | '
        f'Loss: {avg_loss:.4f} | '
        f'Level: {avg_level:.4f} | '
        f'Ranking: {avg_ranking:.4f} | '
        f'Intra: {avg_intra:.4f} | '
        f'q_rgb_clean: {avg_q_rgb_clean:.3f} | '
        f'q_t_clean: {avg_q_t_clean:.3f} | '
        f'q_rgb_deg_last: {avg_q_rgb_deg_last:.3f} | '
        f'q_t_deg_last: {avg_q_t_deg_last:.3f}')

    writer.add_scalar('val/loss_total', avg_loss, epoch)
    writer.add_scalar('val/loss_level', avg_level, epoch)
    writer.add_scalar('val/loss_ranking', avg_ranking, epoch)
    writer.add_scalar('val/loss_intra', avg_intra, epoch)
    writer.add_scalar('val/q_rgb_clean_mean', avg_q_rgb_clean, epoch)
    writer.add_scalar('val/q_t_clean_mean', avg_q_t_clean, epoch)
    writer.add_scalar('val/q_rgb_deg_last_mean', avg_q_rgb_deg_last, epoch)
    writer.add_scalar('val/q_t_deg_last_mean', avg_q_t_deg_last, epoch)

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
        f.write(f'Model: QualityNetworkV3\n')
        f.write(f'Backbone: MAE ViT-B (frozen, last stage only)\n')
        f.write(f'Degradation: Spatial (random 2-6 regions, 3x3-5x5 tokens)\n')
        f.write(f'Image size: {args.img_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Learning rate: {args.lr}\n')
        f.write(f'AMP: {args.amp}\n')
        f.write(f'Losses:\n')
        f.write(f'  Level consistency weight: {args.loss_level_weight}\n')
        f.write(f'  Ranking margin weight: {args.loss_ranking_weight}\n')
        f.write(f'  Intra-image ranking weight: {args.loss_intra_weight}\n')
        f.write(f'  Level margin: {args.level_margin}\n')
        f.write(f'  Intra rank margin: {args.intra_rank_margin}\n')
        f.write(f'  Min low quality: {args.min_low_quality}\n')
        f.write(f'  Min clean quality: {args.min_clean_quality}\n')
        f.write(f'ViT pretrained: {args.vit_pretrained}\n')
        f.write(f'Data root: {args.data_root}\n')
        for k, v in sorted(vars(args).items()):
            f.write(f'  {k}: {v}\n')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f'Work dir: {work_dir}')
    logger.info(f'Args: {vars(args)}')

    writer = SummaryWriter(log_dir=log_dir)
    img_size = args.img_size
    global_step = 0
    best_loss = float('inf')
    best_val_loss = float('inf')

    scaler = GradScaler(enabled=args.amp)

    logger.info('Building MAE backbone (last stage only, out_indices=-1)...')
    mae_backbone = MAE(
        img_size=(img_size, img_size),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=-1,
        final_norm=True,
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
                    logger.info(
                        f'Interpolated pos_embed from '
                        f'{pretrain_size}x{pretrain_size} to '
                        f'{target_size}x{target_size}')
            mae_backbone.load_state_dict(backbone_state, strict=False)
            logger.info(f'Loaded backbone weights from {args.vit_pretrained}')

    mae_backbone.eval()
    mae_backbone.requires_grad_(False)
    mae_backbone.to(device)
    logger.info('MAE backbone frozen')

    logger.info('Building SpatialDegradationGenerator...')
    spatial_deg_gen = SpatialDegradationGenerator(
        num_regions_range=(2, 6),
        region_size_range=(32, 80),
        num_levels=5,
        num_stages=4,
    )
    logger.info('SpatialDegradationGenerator: 2-6 regions, 32-80px, 5 levels')

    logger.info('Building QualityNetworkV3 (InputProj + GlobalNorm + LocalChannelAttention)...')
    quality_net = QualityNetworkV3(
        embed_dim=768,
        proj_dim=128,
        global_norm_hidden=32,
        local_attn_mid=32,
    ).to(device)

    trainable_params = sum(p.numel()
                           for p in quality_net.parameters()
                           if p.requires_grad)
    logger.info(f'QualityNetworkV3 trainable params: {trainable_params:,}')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f'Resumed from {args.resume}')

    ranking_fn = RankingMarginLoss(margin=args.level_margin)
    level_fn = LevelConsistencyLoss(
        num_levels=5,
        level_margin=args.level_margin,
        min_low_quality=args.min_low_quality,
        min_clean_quality=args.min_clean_quality)

    pseudo_label_gen = ImagePseudoLabelGenerator(patch_size=16).to(device)
    pseudo_label_gen.eval()
    thermal_pseudo_label_gen = ThermalPseudoLabelGenerator(patch_size=16).to(device)
    thermal_pseudo_label_gen.eval()
    intra_rank_fn = IntraImageRankingLoss(
        margin=0.0, top_k_ratio=args.intra_top_k_ratio)

    trainable_params_list = [p for p in quality_net.parameters()
                             if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params_list, lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    train_dataset = FMBQualityDataset(
        data_root=args.data_root,
        split='training',
        img_size=img_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f'Train set size: {len(train_dataset)}')

    val_dataset = FMBQualityDataset(
        data_root=args.data_root,
        split='testing',
        img_size=img_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(f'Val set size: {len(val_dataset)}')

    logger.info(f'Start training: epochs={args.epochs}, levels=5, '
                f'val_interval={args.val_interval}')

    for epoch in range(args.epochs):
        quality_net.train()

        epoch_loss_level = 0.0
        epoch_loss_ranking = 0.0
        epoch_loss_intra = 0.0
        epoch_loss_total = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (rgb_clean, t_clean) in enumerate(pbar):
            rgb_clean = rgb_clean.to(device, non_blocking=True)
            t_clean = t_clean.to(device, non_blocking=True)

            with torch.no_grad():
                rgb_multi, deg_type_rgb = spatial_deg_gen.generate_multi_level(
                    rgb_clean, modality='rgb')
                t_multi, deg_type_t = spatial_deg_gen.generate_multi_level(
                    t_clean, modality='t')

                tok_rgb_list = []
                tok_t_list = []
                for level in range(5):
                    feat_rgb = mae_backbone(rgb_multi[level][0])
                    if isinstance(feat_rgb, (tuple, list)):
                        feat_rgb = feat_rgb[-1]
                    tok_rgb = feat_rgb.flatten(2).transpose(1, 2)
                    tok_rgb_list.append(tok_rgb)

                    feat_t = mae_backbone(t_multi[level][0])
                    if isinstance(feat_t, (tuple, list)):
                        feat_t = feat_t[-1]
                    tok_t = feat_t.flatten(2).transpose(1, 2)
                    tok_t_list.append(tok_t)

                tok_H = tok_rgb.shape[1] ** 0.5
                tok_H = int(tok_H) if tok_H == int(tok_H) else int(tok_H ** 0.5)
                tok_size = (tok_H, tok_H)
                rgb_token_mask = F.adaptive_avg_pool2d(
                    rgb_multi[1][1], tok_size).reshape(rgb_multi[1][1].shape[0], -1)
                t_token_mask = F.adaptive_avg_pool2d(
                    t_multi[1][1], tok_size).reshape(t_multi[1][1].shape[0], -1)

            with autocast(enabled=args.amp):
                q_rgb_list = []
                q_t_list = []
                for level in range(5):
                    q_rgb_list.append(quality_net(tok_rgb_list[level], modality='rgb'))
                    q_t_list.append(quality_net(tok_t_list[level], modality='t'))

                if batch_idx == 0:
                    rgb_deg_imgs = [rgb_multi[k][0] for k in range(1, 5)]
                    t_deg_imgs = [t_multi[k][0] for k in range(1, 5)]
                    rgb_masks = [rgb_multi[k][1] for k in range(1, 5)]
                    t_masks = [t_multi[k][1] for k in range(1, 5)]
                    visualize_quality_epoch(
                        rgb_clean, t_clean,
                        rgb_deg_imgs, t_deg_imgs,
                        q_rgb_list, q_t_list,
                        rgb_masks, t_masks,
                        deg_type_rgb, deg_type_t,
                        epoch, vis_dir, writer=writer,
                        tag='Quality/Train_Visualization',
                        prefix='train',
                        prune_threshold=args.prune_threshold,
                    )

                loss_level_rgb = level_fn(q_rgb_list, mask=rgb_token_mask)
                loss_level_t = level_fn(q_t_list, mask=t_token_mask)
                loss_level = (loss_level_rgb + loss_level_t) / 2.0

                loss_ranking = torch.tensor(0.0, device=device)
                for i in range(len(q_rgb_list) - 1):
                    loss_ranking = loss_ranking + ranking_fn(q_rgb_list[i], q_rgb_list[i + 1], mask=rgb_token_mask)
                    loss_ranking = loss_ranking + ranking_fn(q_t_list[i], q_t_list[i + 1], mask=t_token_mask)
                loss_ranking = loss_ranking / (2 * (len(q_rgb_list) - 1))

                rgb_pseudo = pseudo_label_gen(rgb_clean * 255.0)
                t_pseudo = thermal_pseudo_label_gen(t_clean * 255.0)
                loss_intra_rgb = intra_rank_fn(q_rgb_list[0], rgb_pseudo)
                loss_intra_t = intra_rank_fn(q_t_list[0], t_pseudo)
                loss_intra = (loss_intra_rgb + loss_intra_t) / 2.0

                loss = (loss_level * args.loss_level_weight
                        + loss_ranking * args.loss_ranking_weight
                        + loss_intra * args.loss_intra_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss_level += loss_level.item()
            epoch_loss_ranking += loss_ranking.item()
            epoch_loss_intra += loss_intra.item()
            epoch_loss_total += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                writer.add_scalar('train/loss_level',
                                  loss_level.item(), global_step)
                writer.add_scalar('train/loss_ranking',
                                  loss_ranking.item(), global_step)
                writer.add_scalar('train/loss_intra',
                                  loss_intra.item(), global_step)
                writer.add_scalar('train/loss_total',
                                  loss.item(), global_step)
                writer.add_scalar('train/q_rgb_clean_mean',
                                  q_rgb_list[0].mean().item(), global_step)
                writer.add_scalar('train/q_t_clean_mean',
                                  q_t_list[0].mean().item(), global_step)
                writer.add_scalar('train/q_rgb_clean_std',
                                  q_rgb_list[0].std().item(), global_step)
                writer.add_scalar('train/q_t_clean_std',
                                  q_t_list[0].std().item(), global_step)
                writer.add_scalar('train/q_rgb_deg5_mean',
                                  q_rgb_list[-1].mean().item(), global_step)
                writer.add_scalar('train/q_t_deg5_mean',
                                  q_t_list[-1].mean().item(), global_step)

            global_step += 1

            pbar.set_postfix({
                'llvl': f'{loss_level.item():.4f}',
                'lrnk': f'{loss_ranking.item():.4f}',
                'lintra': f'{loss_intra.item():.4f}',
                'q_rgb': f'{q_rgb_list[0].mean().item():.3f}',
                'q_t': f'{q_t_list[0].mean().item():.3f}',
            })

        scheduler.step()

        avg_loss = epoch_loss_total / max(num_batches, 1)
        avg_level = epoch_loss_level / max(num_batches, 1)
        avg_ranking = epoch_loss_ranking / max(num_batches, 1)
        avg_intra = epoch_loss_intra / max(num_batches, 1)

        logger.info(
            f'Epoch {epoch+1}/{args.epochs} | '
            f'Loss: {avg_loss:.4f} | '
            f'Level: {avg_level:.4f} | '
            f'Ranking: {avg_ranking:.4f} | '
            f'Intra: {avg_intra:.4f}')

        writer.add_scalar('epoch/avg_loss', avg_loss, epoch)
        writer.add_scalar('epoch/avg_level', avg_level, epoch)
        writer.add_scalar('epoch/avg_intra', avg_intra, epoch)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(weight_dir, 'best_quality_net_v3.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': quality_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f'Best model saved: {save_path}')

        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(
                quality_net, mae_backbone, spatial_deg_gen,
                pseudo_label_gen, thermal_pseudo_label_gen, intra_rank_fn,
                val_loader, device,
                epoch, vis_dir, writer, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(weight_dir, 'best_val_quality_net_v3.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': quality_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                logger.info(f'Best val model saved: {save_path}')

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                weight_dir, f'quality_net_v3_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': quality_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f'Checkpoint saved: {save_path}')

    save_path = os.path.join(weight_dir, 'final_quality_net_v3.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': quality_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    logger.info(f'Final model saved: {save_path}')

    best_weight_path = os.path.join(weight_dir, 'best_quality_net_v3.pth')
    final_weight_path = save_path
    info_path = os.path.join(weight_dir, 'checkpoint_info.txt')
    with open(info_path, 'w') as f:
        f.write(f'best: {best_weight_path}\n')
        f.write(f'final: {final_weight_path}\n')
    logger.info(f'Checkpoint info saved: {info_path}')

    writer.close()
    logger.info('Training complete!')


def parse_args():
    parser = argparse.ArgumentParser(
        description='QualityNetworkV3 self-supervised pre-training')
    parser.add_argument('--vit-pretrained', type=str, required=True,
                        help='Path to MAE ViT-B pretrained weights')
    parser.add_argument('--data-root', type=str,
                        default='/home/lh/code/data/FMB_ALL',
                        help='Dataset root directory')
    parser.add_argument('--work-dir', type=str,
                        default='work_dirs/quality_v3_pretrain',
                        help='Working directory')
    parser.add_argument('--img-size', type=int, default=480,
                        help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--val-interval', type=int, default=10,
                        help='Validate every N epochs')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision')
    parser.add_argument('--loss-level-weight', type=float, default=0.3,
                        help='Weight for level consistency loss')
    parser.add_argument('--loss-ranking-weight', type=float, default=0.5,
                        help='Weight for ranking margin loss')
    parser.add_argument('--loss-intra-weight', type=float, default=0.2,
                        help='Weight for intra-image ranking loss')
    parser.add_argument('--level-margin', type=float, default=0.1,
                        help='Margin between adjacent degradation levels')
    parser.add_argument('--intra-rank-margin', type=float, default=0.1,
                        help='Margin for intra-image pairwise ranking loss (unused, kept for compat)')
    parser.add_argument('--intra-top-k-ratio', type=float, default=0.3,
                        help='Only use top-K fraction of pairs ranked by pseudo-label difference')
    parser.add_argument('--min-low-quality', type=float, default=0.1,
                        help='Minimum quality for non-highest degradation levels, maximum for highest')
    parser.add_argument('--min-clean-quality', type=float, default=0.5,
                        help='Minimum quality for clean (original) image tokens')
    parser.add_argument('--prune-threshold', type=float, default=0.3,
                        help='Quality threshold for token pruning visualization')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
