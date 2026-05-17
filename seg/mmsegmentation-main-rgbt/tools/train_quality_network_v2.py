"""
QualityNetworkV2 自监督预训练脚本

核心思路:
  - 输入: 仅图像嵌入后的token特征 (16x16卷积得到)
  - 架构: 双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分
  - 训练: 对同一图像生成5级递增退化版本 (与鲁棒性测试退化类型对齐)
  - 退化: RGB/T独立选择退化类型, 全局应用, 5级别 (1=原图, 5=最强)
  - 损失: 质量锚定损失(MSE) + 级别间距损失 + 高退化天花板损失 + 空间多样性损失
  - 不依赖分割网络, 不与人类感知对齐, 无跨模态一致性

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/train_quality_network_v2.py \
      --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
      --work-dir work_dirs/quality_v2_pretrain \
      --img-size 480 \
      --amp
"""

import os
import sys
import argparse
import logging
import datetime
import random

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
from mmseg.models.utils.quality_network_v2 import QualityNetworkV2
from mmseg.models.utils.quality_network import (
    QualityAnchorLoss,
    LevelMarginLoss,
    SpatialDiversityLoss,
    HighDegCeilingLoss,
)
from mmseg.datasets.transforms.quality_degradation import (
    _QUALITY_RGB_DEG_TYPES,
    _QUALITY_T_DEG_TYPES,
    generate_quality_degradation_levels,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViTFeatureExtractor(nn.Module):

    def __init__(self, mae_model, num_layers=4):
        super().__init__()
        self.mae = mae_model
        self.num_layers = min(num_layers, len(mae_model.layers))

    @torch.no_grad()
    def forward(self, x):
        B = x.shape[0]
        tokens, hw_shape = self.mae.patch_embed(x)

        need_resize = (hw_shape[0] != self.mae.patch_shape[0] or
                       hw_shape[1] != self.mae.patch_shape[1])

        if need_resize:
            pos_embed = self.mae._get_interpolated_pos_embed(hw_shape)
            orig_state = self.mae._save_rel_pos_bias_state()
            self.mae._update_rel_pos_bias(hw_shape)
        else:
            pos_embed = self.mae.pos_embed

        cls_tokens = self.mae.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + pos_embed

        for i in range(self.num_layers):
            tokens = self.mae.layers[i](tokens)

        if need_resize:
            self.mae._restore_rel_pos_bias_state(orig_state)

        patch_tokens = tokens[:, 1:]
        return patch_tokens, hw_shape


class FMBQualityDataset(Dataset):

    def __init__(self, data_root, split='training', img_size=480):
        self.data_root = data_root
        self.img_size = img_size
        
        # 映射 split 到正确的目录名称
        split_map = {
            'training': 'training',
            'validation': 'validation',
            'testing': 'validation',  # testing 使用 validation 目录
            'val': 'validation',
            'train': 'training'
        }
        actual_split = split_map.get(split, split)
        
        # 正确的路径结构
        self.rgb_dir = os.path.join(data_root, 'FMB', 'images', actual_split)
        self.thermal_dir = os.path.join(data_root, 'FMB_T', 'images', actual_split)

        # 检查目录是否存在
        if not os.path.exists(self.rgb_dir):
            # 尝试备用路径结构
            self.rgb_dir = os.path.join(data_root, 'images', actual_split)
            self.thermal_dir = os.path.join(data_root, 'images', actual_split)

        # 如果仍然不存在，抛出详细的错误信息
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


def generate_quality_training_data(img_tensor, modality, num_levels=5):
    if modality == 'rgb':
        deg_types = _QUALITY_RGB_DEG_TYPES
    else:
        deg_types = _QUALITY_T_DEG_TYPES

    deg_type = random.choice(deg_types)

    deg_images = generate_quality_degradation_levels(
        img_tensor, deg_type, modality, num_levels=num_levels)

    return deg_images, deg_type


def visualize_quality_epoch(
    rgb_clean, t_clean,
    rgb_deg_images, t_deg_images,
    q_rgb_clean, q_t_clean,
    q_rgb_deg_list, q_t_deg_list,
    deg_type_rgb, deg_type_t,
    quality_threshold, epoch, save_dir, writer=None, patch_size=16,
    tag='Quality/Visualization', prefix='',
):
    os.makedirs(save_dir, exist_ok=True)

    B = rgb_clean.shape[0]
    K = len(rgb_deg_images)
    img_H, img_W = rgb_clean.shape[2], rgb_clean.shape[3]
    patch_H = img_H // patch_size
    patch_W = img_W // patch_size

    num_cols = 2 + 2 * K

    fig, axes = plt.subplots(
        3, num_cols * B,
        figsize=(2.5 * num_cols * B, 3 * 3),
    )
    if B == 1 and num_cols == 1:
        axes = axes.reshape(3, 1)

    for b in range(B):
        col_offset = b * num_cols

        all_images = [rgb_clean[b], t_clean[b]]
        all_q = [q_rgb_clean[b], q_t_clean[b]]
        col_titles = ['RGB', 'T']

        for k in range(K):
            all_images.append(rgb_deg_images[k][b])
            all_q.append(q_rgb_deg_list[k][b])
            col_titles.append(f'RGB deg{k+1}')

        for k in range(K):
            all_images.append(t_deg_images[k][b])
            all_q.append(q_t_deg_list[k][b])
            col_titles.append(f'T deg{k+1}')

        for col_idx in range(num_cols):
            ax_img = axes[0, col_offset + col_idx]
            ax_qmap = axes[1, col_offset + col_idx]
            ax_prune = axes[2, col_offset + col_idx]

            img = all_images[col_idx].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            q = all_q[col_idx].detach().cpu().numpy()
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
            ax_img.set_title(col_titles[col_idx], fontsize=8)
            ax_img.axis('off')

            ax_qmap.imshow(q_up, vmin=0, vmax=1, cmap='RdBu_r')
            q_mean = q.mean()
            ax_qmap.set_title(f'q={q_mean:.2f}', fontsize=7)
            ax_qmap.axis('off')

            ax_prune.imshow(pruned)
            keep_ratio = mask.mean() * 100
            ax_prune.set_title(f'keep={keep_ratio:.0f}%', fontsize=7)
            ax_prune.axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Quality Map', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Pruned', fontsize=10, fontweight='bold')

    fig.suptitle(
        f'Epoch {epoch+1} | RGB deg: {deg_type_rgb} | T deg: {deg_type_t} | '
        f'threshold={quality_threshold}',
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{prefix}_epoch_{epoch+1:03d}.png') if prefix else os.path.join(save_dir, f'epoch_{epoch+1:03d}.png')
    if writer is not None:
        writer.add_figure(tag, fig, global_step=epoch)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Quality visualization saved: {save_path}')


@torch.no_grad()
def validate(quality_net, vit_encoder, val_loader, device,
             quality_threshold, epoch, vis_dir, writer, args):
    quality_net.eval()

    quality_anchor_fn = QualityAnchorLoss(clean_target=0.8, deg_target=0.15)
    high_deg_ceiling_fn = HighDegCeilingLoss(ceiling=0.15)
    level_margin_fn = LevelMarginLoss(margin=0.2)
    spatial_div_fn = SpatialDiversityLoss(min_std=0.08)

    val_loss_total = 0.0
    val_loss_anchor = 0.0
    val_loss_ceiling = 0.0
    val_loss_margin = 0.0
    val_loss_spatial_div = 0.0
    num_batches = 0

    q_rgb_clean_all = []
    q_t_clean_all = []
    q_rgb_deg_last_all = []
    q_t_deg_last_all = []

    for batch_idx, (rgb_clean, t_clean) in enumerate(val_loader):
        rgb_clean = rgb_clean.to(device, non_blocking=True)
        t_clean = t_clean.to(device, non_blocking=True)

        rgb_deg_images, deg_type_rgb = generate_quality_training_data(
            rgb_clean, modality='rgb', num_levels=5)
        t_deg_images, deg_type_t = generate_quality_training_data(
            t_clean, modality='t', num_levels=5)

        tok_rgb_list = []
        tok_t_list = []
        for level in range(5):
            tok_rgb, _ = vit_encoder(rgb_deg_images[level])
            tok_t, _ = vit_encoder(t_deg_images[level])
            tok_rgb_list.append(tok_rgb)
            tok_t_list.append(tok_t)

        q_rgb_list = []
        q_t_list = []
        for level in range(5):
            q_rgb_list.append(quality_net(tok_rgb_list[level], modality='rgb'))
            q_t_list.append(quality_net(tok_t_list[level], modality='t'))

        q_rgb_clean = q_rgb_list[0]
        q_t_clean = q_t_list[0]
        q_rgb_deg_list = q_rgb_list[1:]
        q_t_deg_list = q_t_list[1:]

        if batch_idx == 0:
            visualize_quality_epoch(
                rgb_clean, t_clean,
                rgb_deg_images[1:], t_deg_images[1:],
                q_rgb_clean, q_t_clean,
                q_rgb_deg_list, q_t_deg_list,
                deg_type_rgb, deg_type_t,
                quality_threshold, epoch, vis_dir, writer=writer,
                tag='Quality/Val_Visualization',
                prefix='val',
            )

        loss_anchor_rgb_c, loss_anchor_rgb_d = quality_anchor_fn(
            q_rgb_clean, q_rgb_deg_list[-1])
        loss_anchor_t_c, loss_anchor_t_d = quality_anchor_fn(
            q_t_clean, q_t_deg_list[-1])
        loss_anchor = (loss_anchor_rgb_c + loss_anchor_rgb_d +
                       loss_anchor_t_c + loss_anchor_t_d) / 2.0

        loss_ceiling_rgb = high_deg_ceiling_fn(q_rgb_list[-1])
        loss_ceiling_t = high_deg_ceiling_fn(q_t_list[-1])
        loss_ceiling = (loss_ceiling_rgb + loss_ceiling_t) / 2.0

        loss_margin_rgb = level_margin_fn(q_rgb_list)
        loss_margin_t = level_margin_fn(q_t_list)
        loss_margin = (loss_margin_rgb + loss_margin_t) / 2.0

        loss_spatial_div = torch.tensor(0.0, device=device)
        for q_rgb_level in q_rgb_list:
            loss_spatial_div = loss_spatial_div + spatial_div_fn(q_rgb_level)
        for q_t_level in q_t_list:
            loss_spatial_div = loss_spatial_div + spatial_div_fn(q_t_level)
        loss_spatial_div = loss_spatial_div / (len(q_rgb_list) + len(q_t_list))

        loss = (loss_anchor * args.loss_anchor_weight +
                loss_ceiling * args.loss_ceiling_weight +
                loss_margin * args.loss_margin_weight +
                loss_spatial_div * args.loss_spatial_div_weight)

        val_loss_total += loss.item()
        val_loss_anchor += loss_anchor.item()
        val_loss_ceiling += loss_ceiling.item()
        val_loss_margin += loss_margin.item()
        val_loss_spatial_div += loss_spatial_div.item()
        num_batches += 1

        q_rgb_clean_all.append(q_rgb_clean.mean().item())
        q_t_clean_all.append(q_t_clean.mean().item())
        q_rgb_deg_last_all.append(q_rgb_deg_list[-1].mean().item())
        q_t_deg_last_all.append(q_t_deg_list[-1].mean().item())

    avg_loss = val_loss_total / max(num_batches, 1)
    avg_anchor = val_loss_anchor / max(num_batches, 1)
    avg_ceiling = val_loss_ceiling / max(num_batches, 1)
    avg_margin = val_loss_margin / max(num_batches, 1)
    avg_spatial_div = val_loss_spatial_div / max(num_batches, 1)

    avg_q_rgb_clean = np.mean(q_rgb_clean_all) if q_rgb_clean_all else 0.0
    avg_q_t_clean = np.mean(q_t_clean_all) if q_t_clean_all else 0.0
    avg_q_rgb_deg_last = np.mean(q_rgb_deg_last_all) if q_rgb_deg_last_all else 0.0
    avg_q_t_deg_last = np.mean(q_t_deg_last_all) if q_t_deg_last_all else 0.0

    logger.info(
        f'[Val] Epoch {epoch+1} | '
        f'Loss: {avg_loss:.4f} | '
        f'Anchor: {avg_anchor:.4f} | '
        f'Ceiling: {avg_ceiling:.4f} | '
        f'Margin: {avg_margin:.4f} | '
        f'SpatialDiv: {avg_spatial_div:.4f} | '
        f'q_rgb_clean: {avg_q_rgb_clean:.3f} | '
        f'q_t_clean: {avg_q_t_clean:.3f} | '
        f'q_rgb_deg_last: {avg_q_rgb_deg_last:.3f} | '
        f'q_t_deg_last: {avg_q_t_deg_last:.3f}')

    writer.add_scalar('val/loss_total', avg_loss, epoch)
    writer.add_scalar('val/loss_anchor', avg_anchor, epoch)
    writer.add_scalar('val/loss_ceiling', avg_ceiling, epoch)
    writer.add_scalar('val/loss_margin', avg_margin, epoch)
    writer.add_scalar('val/loss_spatial_div', avg_spatial_div, epoch)
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
        f.write(f'Model: QualityNetworkV2\n')
        f.write(f'Backbone: MAE ViT-B (frozen)\n')
        f.write(f'Image size: {args.img_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Learning rate: {args.lr}\n')
        f.write(f'AMP: {args.amp}\n')
        f.write(f'Losses:\n')
        f.write(f'  Anchor weight: {args.loss_anchor_weight}\n')
        f.write(f'  Ceiling weight: {args.loss_ceiling_weight}\n')
        f.write(f'  Margin weight: {args.loss_margin_weight}\n')
        f.write(f'  SpatialDiv weight: {args.loss_spatial_div_weight}\n')
        f.write(f'ViT pretrained: {args.vit_pretrained}\n')
        f.write(f'Data root: {args.data_root}\n')
        f.write(f'Val interval: {args.val_interval}\n')
        f.write(f'Save interval: {args.save_interval}\n')
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

    vit_encoder = ViTFeatureExtractor(mae_backbone, num_layers=4)

    logger.info('Building QualityNetworkV2 (3x3 conv + max pool + MLP)...')
    quality_net = QualityNetworkV2(
        embed_dim=768,
        proj_dim=64,
        mlp_hidden_dim=128,
    ).to(device)

    trainable_params = sum(p.numel()
                           for p in quality_net.parameters()
                           if p.requires_grad)
    logger.info(f'QualityNetworkV2 trainable params: {trainable_params:,}')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f'Resumed from {args.resume}')

    quality_anchor_fn = QualityAnchorLoss(clean_target=0.8, deg_target=0.15)
    high_deg_ceiling_fn = HighDegCeilingLoss(ceiling=0.15)
    level_margin_fn = LevelMarginLoss(margin=0.2)
    spatial_div_fn = SpatialDiversityLoss(min_std=0.08)

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

    logger.info(f'Start training: epochs={args.epochs}, levels=5, val_interval={args.val_interval}')

    for epoch in range(args.epochs):
        quality_net.train()

        epoch_loss_anchor = 0.0
        epoch_loss_ceiling = 0.0
        epoch_loss_margin = 0.0
        epoch_loss_spatial_div = 0.0
        epoch_loss_total = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (rgb_clean, t_clean) in enumerate(pbar):
            rgb_clean = rgb_clean.to(device, non_blocking=True)
            t_clean = t_clean.to(device, non_blocking=True)

            with torch.no_grad():
                rgb_deg_images, deg_type_rgb = generate_quality_training_data(
                    rgb_clean, modality='rgb', num_levels=5)
                t_deg_images, deg_type_t = generate_quality_training_data(
                    t_clean, modality='t', num_levels=5)

                tok_rgb_list = []
                tok_t_list = []
                for level in range(5):
                    tok_rgb, _ = vit_encoder(rgb_deg_images[level])
                    tok_t, _ = vit_encoder(t_deg_images[level])
                    tok_rgb_list.append(tok_rgb)
                    tok_t_list.append(tok_t)

            with autocast(enabled=args.amp):
                q_rgb_list = []
                q_t_list = []
                for level in range(5):
                    q_rgb_list.append(quality_net(tok_rgb_list[level], modality='rgb'))
                    q_t_list.append(quality_net(tok_t_list[level], modality='t'))

                q_rgb_clean = q_rgb_list[0]
                q_t_clean = q_t_list[0]
                q_rgb_deg_list = q_rgb_list[1:]
                q_t_deg_list = q_t_list[1:]

                if batch_idx == 0:
                    visualize_quality_epoch(
                        rgb_clean, t_clean,
                        rgb_deg_images[1:], t_deg_images[1:],
                        q_rgb_clean, q_t_clean,
                        q_rgb_deg_list, q_t_deg_list,
                        deg_type_rgb, deg_type_t,
                        args.quality_threshold, epoch, vis_dir, writer=writer,
                        tag='Quality/Train_Visualization',
                        prefix='train',
                    )

                loss_anchor_rgb_c, loss_anchor_rgb_d = quality_anchor_fn(
                    q_rgb_clean, q_rgb_deg_list[-1])
                loss_anchor_t_c, loss_anchor_t_d = quality_anchor_fn(
                    q_t_clean, q_t_deg_list[-1])
                loss_anchor = (loss_anchor_rgb_c + loss_anchor_rgb_d +
                               loss_anchor_t_c + loss_anchor_t_d) / 2.0

                loss_ceiling_rgb = high_deg_ceiling_fn(q_rgb_list[-1])
                loss_ceiling_t = high_deg_ceiling_fn(q_t_list[-1])
                loss_ceiling = (loss_ceiling_rgb + loss_ceiling_t) / 2.0

                loss_margin_rgb = level_margin_fn(q_rgb_list)
                loss_margin_t = level_margin_fn(q_t_list)
                loss_margin = (loss_margin_rgb + loss_margin_t) / 2.0

                loss_spatial_div = torch.tensor(0.0, device=device)
                for q_rgb_level in q_rgb_list:
                    loss_spatial_div = loss_spatial_div + spatial_div_fn(q_rgb_level)
                for q_t_level in q_t_list:
                    loss_spatial_div = loss_spatial_div + spatial_div_fn(q_t_level)
                loss_spatial_div = loss_spatial_div / (len(q_rgb_list) + len(q_t_list))

                loss = (loss_anchor * args.loss_anchor_weight +
                        loss_ceiling * args.loss_ceiling_weight +
                        loss_margin * args.loss_margin_weight +
                        loss_spatial_div * args.loss_spatial_div_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss_anchor += loss_anchor.item()
            epoch_loss_ceiling += loss_ceiling.item()
            epoch_loss_margin += loss_margin.item()
            epoch_loss_spatial_div += loss_spatial_div.item()
            epoch_loss_total += loss.item()
            num_batches += 1

            if global_step % 50 == 0:
                writer.add_scalar('train/loss_anchor',
                                  loss_anchor.item(), global_step)
                writer.add_scalar('train/loss_ceiling',
                                  loss_ceiling.item(), global_step)
                writer.add_scalar('train/loss_margin',
                                  loss_margin.item(), global_step)
                writer.add_scalar('train/loss_spatial_div',
                                  loss_spatial_div.item(), global_step)
                writer.add_scalar('train/loss_total',
                                  loss.item(), global_step)
                writer.add_scalar('train/q_rgb_clean_mean',
                                  q_rgb_clean.mean().item(), global_step)
                writer.add_scalar('train/q_t_clean_mean',
                                  q_t_clean.mean().item(), global_step)
                writer.add_scalar('train/q_rgb_clean_std',
                                  q_rgb_clean.std().item(), global_step)
                writer.add_scalar('train/q_t_clean_std',
                                  q_t_clean.std().item(), global_step)

                writer.add_scalar(
                    'train/q_rgb_deg0_mean',
                    q_rgb_deg_list[0].mean().item(), global_step)
                writer.add_scalar(
                    'train/q_rgb_deg_last_mean',
                    q_rgb_deg_list[-1].mean().item(), global_step)

            global_step += 1

            pbar.set_postfix({
                'lanc': f'{loss_anchor.item():.4f}',
                'lcl': f'{loss_ceiling.item():.4f}',
                'lmar': f'{loss_margin.item():.4f}',
                'lsd': f'{loss_spatial_div.item():.4f}',
                'q_rgb': f'{q_rgb_clean.mean().item():.3f}',
                'q_t': f'{q_t_clean.mean().item():.3f}',
            })

        scheduler.step()

        avg_loss = epoch_loss_total / max(num_batches, 1)
        avg_anchor = epoch_loss_anchor / max(num_batches, 1)
        avg_ceiling = epoch_loss_ceiling / max(num_batches, 1)
        avg_margin = epoch_loss_margin / max(num_batches, 1)
        avg_spatial_div = epoch_loss_spatial_div / max(num_batches, 1)

        logger.info(
            f'Epoch {epoch+1}/{args.epochs} | '
            f'Loss: {avg_loss:.4f} | '
            f'Anchor: {avg_anchor:.4f} | '
            f'Ceiling: {avg_ceiling:.4f} | '
            f'Margin: {avg_margin:.4f} | '
            f'SpatialDiv: {avg_spatial_div:.4f}')

        writer.add_scalar('epoch/avg_loss', avg_loss, epoch)
        writer.add_scalar('epoch/avg_anchor', avg_anchor, epoch)
        writer.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(weight_dir, 'best_quality_net_v2.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': quality_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f'Best model saved: {save_path}')

        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(
                quality_net, vit_encoder, val_loader, device,
                args.quality_threshold, epoch, vis_dir, writer, args)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(weight_dir, 'best_val_quality_net_v2.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': quality_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                logger.info(f'Best val model saved: {save_path}')

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                weight_dir, f'quality_net_v2_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': quality_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f'Checkpoint saved: {save_path}')

    save_path = os.path.join(weight_dir, 'final_quality_net_v2.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': quality_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    logger.info(f'Final model saved: {save_path}')

    best_weight_path = os.path.join(weight_dir, 'best_quality_net_v2.pth')
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
        description='QualityNetworkV2 self-supervised pre-training')
    parser.add_argument('--vit-pretrained', type=str, required=True,
                        help='Path to MAE ViT-B pretrained weights')
    parser.add_argument('--data-root', type=str,
                        default='/home/lh/code/data/FMB_ALL',
                        help='Dataset root directory')
    parser.add_argument('--work-dir', type=str,
                        default='work_dirs/quality_v2_pretrain',
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
    parser.add_argument('--loss-anchor-weight', type=float, default=1.0,
                        help='Weight for quality anchor loss')
    parser.add_argument('--loss-ceiling-weight', type=float, default=2.0,
                        help='Weight for high-degradation quality ceiling loss')
    parser.add_argument('--loss-margin-weight', type=float, default=0.5,
                        help='Weight for level margin loss')
    parser.add_argument('--loss-spatial-div-weight', type=float, default=0.5,
                        help='Weight for spatial diversity loss')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable automatic mixed precision')
    parser.add_argument('--quality-threshold', type=float, default=0.3,
                        help='Quality score threshold for pruning visualization')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
