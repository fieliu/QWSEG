"""
质量网络(QualityNetwork)独立预训练脚本

模型结构:
  - 冻结的MAE patch_embed + pos_embed + 前N层Transformer: 提取token特征
  - 冻结的分割模型: 从原始图像提取高质量锚点
  - 可训练的QualityNetwork: 双分支(RGB/Thermal独立)，多尺度局部差异感知，逐token质量评分

损失函数:
  - QualityAnchorLoss: 质量锚定损失(MSE)，将clean分数拉向0.8，退化分数拉向0.15
  - LevelMarginLoss: 级别间距损失，确保相邻退化级别间有足够间距
  - HighDegCeilingLoss: 高退化天花板损失，限制最高退化级别的质量分数
  - SpatialDiversityLoss: 空间多样性损失，鼓励同一图内不同token有不同质量分数

运行方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
  python tools/train_quality_network.py \
      --config configs/mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_fmb-768x768.py \
      --checkpoint work_dirs/v1_baseline/fmb/best.pth \
      --vit-pretrained ./pretrain/M-SpecGene_VIT-B_seg_transform.pth \
      --work-dir work_dirs/quality_pretrain \
      --img-size 480 \
      --amp
"""

import os
import sys
import random
import argparse
import logging
import datetime

import cv2
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
from mmseg.models.utils.quality_network import (
    QualityNetwork,
    QualityAnchorLoss, LevelMarginLoss,
    SpatialDiversityLoss, HighDegCeilingLoss)
from mmseg.datasets.transforms.degradation import (
    apply_degradation, apply_multi_region_degradation)
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.models import build_segmentor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dynamic_compute_token_entropy(seg_probs):
    pixel_log_probs = torch.log(seg_probs.clamp(min=1e-10))
    pixel_entropy = -(seg_probs * pixel_log_probs).sum(dim=1)

    B, C, H, W = seg_probs.shape
    target_h = (H // 16) * 16
    target_w = (W // 16) * 16
    target_size = (target_h, target_w)

    pixel_entropy = F.interpolate(
        pixel_entropy.unsqueeze(1),
        size=target_size,
        mode='bilinear',
        align_corners=False).squeeze(1)

    num_patches_h = target_h // 16
    num_patches_w = target_w // 16
    num_tokens = num_patches_h * num_patches_w

    pe_reshaped = pixel_entropy.reshape(
        B, num_patches_h, 16, num_patches_w, 16).permute(0, 1, 3, 2, 4)
    token_entropy = pe_reshaped.mean(dim=(3, 4)).reshape(B, num_tokens)

    return token_entropy, (num_patches_h, num_patches_w)


def dynamic_compute_token_accuracy(seg_probs, label):
    label = label.to(seg_probs.device)
    pred = seg_probs.argmax(dim=1)

    B, C, H, W = seg_probs.shape
    target_h = (H // 16) * 16
    target_w = (W // 16) * 16
    target_size = (target_h, target_w)

    if len(label.shape) == 2:
        label = label.unsqueeze(0).unsqueeze(1)
    elif len(label.shape) == 3:
        label = label.unsqueeze(1)

    pred_4d = pred.unsqueeze(1).float()
    pred_resized = F.interpolate(
        pred_4d, size=target_size, mode='nearest').squeeze(1).long()
    label_resized = F.interpolate(
        label.float(), size=target_size, mode='nearest').squeeze(1).long()

    num_patches_h = target_h // 16
    num_patches_w = target_w // 16
    num_tokens = num_patches_h * num_patches_w

    correct = (pred_resized == label_resized).float()
    correct_reshaped = correct.reshape(
        B, num_patches_h, 16, num_patches_w, 16).permute(0, 1, 3, 2, 4)
    token_accuracy = correct_reshaped.mean(dim=(3, 4)).reshape(B, num_tokens)

    return token_accuracy, (num_patches_h, num_patches_w)


def dynamic_normalize_and_flip_entropy(token_entropy):
    B, num_tokens = token_entropy.shape
    flipped_entropy = torch.zeros_like(token_entropy)

    for b in range(B):
        entropy = token_entropy[b]
        min_entropy = entropy.min()
        max_entropy = entropy.max()
        normalized_entropy = (entropy - min_entropy) / (
            max_entropy - min_entropy + 1e-10)
        flipped_entropy[b] = 1 - normalized_entropy

    return flipped_entropy


def dynamic_select_high_quality_mask(seg_probs, label, top_ratio=0.2):
    token_entropy, num_patches_hw = dynamic_compute_token_entropy(seg_probs)
    token_accuracy, _ = dynamic_compute_token_accuracy(seg_probs, label)

    flipped_entropy = dynamic_normalize_and_flip_entropy(token_entropy)
    score = token_accuracy * flipped_entropy

    B, num_tokens = score.shape
    high_quality_mask = torch.zeros_like(score, dtype=torch.bool)
    num_top = int(num_tokens * top_ratio)

    for b in range(B):
        s = score[b]
        _, top_indices = torch.topk(s, num_top, largest=True)
        high_quality_mask[b, top_indices] = True

    return high_quality_mask, score, num_patches_hw


def compute_token_deg_level_map(deg_regions, region_levels, img_size=480,
                                 patch_size=16):
    num_patches = img_size // patch_size
    token_levels = torch.zeros(num_patches * num_patches, dtype=torch.long)

    centers_h = torch.arange(num_patches) * patch_size + patch_size // 2
    centers_w = torch.arange(num_patches) * patch_size + patch_size // 2
    grid_h, grid_w = torch.meshgrid(centers_h, centers_w, indexing='ij')
    grid_h = grid_h.reshape(-1).float()
    grid_w = grid_w.reshape(-1).float()

    for (start_h, start_w, region_h, region_w), level in zip(
            deg_regions, region_levels):
        mask = (grid_h >= start_h) & (grid_h < start_h + region_h) & \
               (grid_w >= start_w) & (grid_w < start_w + region_w)
        token_levels[mask] = torch.maximum(
            token_levels[mask], torch.tensor(level))

    return token_levels


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
        self.rgb_dir = os.path.join(data_root, 'FMB', 'images', split)
        self.thermal_dir = os.path.join(data_root, 'FMB_T', 'images', split)
        self.label_dir = os.path.join(data_root, 'FMB', 'annotations', split)

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
            l_ok = os.path.exists(
                os.path.join(self.label_dir, name + '.png'))
            if rgb_ok and t_ok and l_ok:
                valid.append(name)
        self.filenames = valid

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        rgb_bgr = cv2.imread(os.path.join(self.rgb_dir, name + '.png'))
        t_bgr = cv2.imread(os.path.join(self.thermal_dir, name + '.png'))
        label_np = cv2.imread(
            os.path.join(self.label_dir, name + '.png'),
            cv2.IMREAD_GRAYSCALE)

        if rgb_bgr.shape[0] != self.img_size or \
           rgb_bgr.shape[1] != self.img_size:
            rgb_bgr = cv2.resize(rgb_bgr, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
            t_bgr = cv2.resize(t_bgr, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            label_np = cv2.resize(label_np,
                                  (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)

        rgb_float = rgb_bgr.astype(np.float32) / 255.0
        t_float = t_bgr.astype(np.float32) / 255.0
        label = torch.from_numpy(label_np).long()

        img_rgb = torch.from_numpy(rgb_float).permute(2, 0, 1)
        img_t = torch.from_numpy(t_float).permute(2, 0, 1)

        return img_rgb, img_t, label


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

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
        f.write(f'Model: QualityNetwork (v1)\n')
        f.write(f'Backbone: MAE ViT-B (frozen)\n')
        f.write(f'Image size: {args.img_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Learning rate: {args.lr}\n')
        f.write(f'AMP: {args.amp}\n')
        f.write(f'Config: {args.config}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n')
        f.write(f'ViT pretrained: {args.vit_pretrained}\n')
        f.write(f'Data root: {args.data_root}\n')
        for k, v in sorted(vars(args).items()):
            f.write(f'  {k}: {v}\n')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f'工作目录: {work_dir}')
    logger.info(f'参数: {vars(args)}')

    writer = SummaryWriter(log_dir=log_dir)
    img_size = args.img_size
    global_step = 0
    best_loss = float('inf')
    accumulation_steps = args.accumulation_steps
    K = args.num_degradations

    scaler = GradScaler(enabled=args.amp)

    logger.info('加载分割模型用于动态高质量锚点选择...')
    cfg = Config.fromfile(args.config)
    init_default_scope('mmseg')

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    seg_model = build_segmentor(cfg.model)
    seg_model = seg_model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    seg_model.load_state_dict(state_dict, strict=False)
    seg_model.eval()
    seg_model.requires_grad_(False)
    if seg_model.test_cfg is None:
        from mmengine.config import ConfigDict
        seg_model.test_cfg = ConfigDict(mode='whole')
    else:
        seg_model.test_cfg.mode = 'whole'
    logger.info(f'分割模型已加载并冻结: {args.checkpoint}')

    logger.info('构建MAE backbone...')
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
                        f"Interpolated pos_embed from "
                        f"{pretrain_size}x{pretrain_size} to "
                        f"{target_size}x{target_size}")
            mae_backbone.load_state_dict(backbone_state, strict=False)
            logger.info(f"Loaded backbone weights from {args.vit_pretrained}")

    mae_backbone.eval()
    mae_backbone.requires_grad_(False)
    mae_backbone.to(device)
    logger.info('MAE backbone已冻结')

    vit_encoder = ViTFeatureExtractor(mae_backbone, num_layers=4)

    logger.info('构建质量网络(Feature Pyramid + MLP)...')
    quality_net = QualityNetwork(
        embed_dim=768,
        proj_dim=64,
        mlp_hidden_dim=128,
        num_classes=15,
    ).to(device)

    trainable_params = sum(p.numel()
                           for p in quality_net.parameters()
                           if p.requires_grad)
    logger.info(f'QualityNetwork可训练参数: {trainable_params:,}')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'model_state_dict' in ckpt:
            quality_net.load_state_dict(
                ckpt['model_state_dict'], strict=False)
        else:
            quality_net.load_state_dict(ckpt, strict=False)
        logger.info(f"Resumed from {args.resume}")

    quality_anchor_fn = QualityAnchorLoss(clean_target=0.8, deg_target=0.15)
    level_margin_fn = LevelMarginLoss(margin=0.2)
    spatial_div_fn = SpatialDiversityLoss(min_std=0.08)
    high_deg_ceiling_fn = HighDegCeilingLoss(ceiling=0.15)

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
    logger.info(f'训练集大小: {len(train_dataset)}')

    rgb_deg_types = ['low_light', 'overexposure', 'motion_blur',
                     'modality_missing']
    rgb_deg_types_no_missing = ['low_light', 'overexposure', 'motion_blur']
    t_deg_types = ['thermal_contrast', 'stripe_noise', 'thermal_noise',
                   'thermal_saturation', 'modality_missing']
    t_deg_types_no_missing = ['thermal_contrast', 'stripe_noise',
                              'thermal_noise', 'thermal_saturation']

    logger.info(f'开始训练: epochs={args.epochs}, K={K}, '
                f'accumulation={accumulation_steps}')

    for epoch in range(args.epochs):
        quality_net.train()

        epoch_loss_margin = 0.0
        epoch_loss_anchor = 0.0
        epoch_loss_ceiling = 0.0
        epoch_loss_spatial_div = 0.0
        epoch_loss_total = 0.0
        num_batches = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (rgb_clean, t_clean, seg_label) in enumerate(pbar):
            rgb_clean = rgb_clean.to(device, non_blocking=True)
            t_clean = t_clean.to(device, non_blocking=True)
            seg_label = seg_label.to(device, non_blocking=True)

            with torch.no_grad():
                tok_rgb, _ = vit_encoder(rgb_clean)
                tok_t, _ = vit_encoder(t_clean)

                from mmseg.structures import SegDataSample

                rgb_clean_768 = F.interpolate(
                    rgb_clean, size=(768, 768), mode='bilinear',
                    align_corners=False)
                t_clean_768 = F.interpolate(
                    t_clean, size=(768, 768), mode='bilinear',
                    align_corners=False)

                rgb_clean_768_uint8 = (rgb_clean_768 * 255).to(torch.uint8)
                t_clean_768_uint8 = (t_clean_768 * 255).to(torch.uint8)

                data_samples = []
                for i in range(len(rgb_clean)):
                    sample = SegDataSample()
                    sample.set_metainfo({
                        'ori_shape': (768, 768),
                        'img_shape': (768, 768),
                        'pad_shape': (768, 768),
                        'scale_factor': (1.0, 1.0),
                    })
                    data_samples.append(sample)

                pseudo_data_batch = {
                    'inputs': torch.cat(
                        [rgb_clean_768_uint8, t_clean_768_uint8], dim=1),
                    'data_samples': data_samples
                }

                results = seg_model.test_step(pseudo_data_batch)

                seg_logits_list = [
                    res.seg_logits.data.unsqueeze(0) for res in results]
                seg_logits_list_480 = [
                    F.interpolate(sl, size=(img_size, img_size),
                                  mode='bilinear', align_corners=False)
                    for sl in seg_logits_list]
                seg_probs_list = [F.softmax(sl, dim=1)
                                  for sl in seg_logits_list_480]
                seg_probs = torch.cat(seg_probs_list, dim=0)

                _hq_mask, _hq_score, _num_patches_hw = \
                    dynamic_select_high_quality_mask(
                        seg_probs, seg_label, top_ratio=args.top_ratio)

            B, C, H, W = rgb_clean.shape
            strip_h = H // 3
            deg_regions = []
            for i in range(3):
                region_h = random.randint(
                    int(H * 0.1), min(int(H * 0.3), strip_h))
                region_w = random.randint(int(W * 0.1), int(W * 0.3))
                start_h = random.randint(
                    i * strip_h, (i + 1) * strip_h - region_h)
                start_w = random.randint(0, W - region_w)
                deg_regions.append((start_h, start_w, region_h, region_w))

            q_deg_list_rgb = []
            q_deg_list_t = []
            all_region_levels_rgb = []
            all_region_levels_t = []

            with autocast(enabled=args.amp):
                for deg_idx in range(K):
                    img_levels_rgb = []
                    img_levels_t = []
                    rgb_region_configs = []
                    t_region_configs = []

                    for region_idx in range(3):
                        if deg_idx == K - 1:
                            level_rgb = 5
                            level_t = 5
                            rgb_deg_type = random.choice(rgb_deg_types)
                            t_deg_type = random.choice(t_deg_types)
                        else:
                            level_rgb = random.randint(2, 5)
                            level_t = random.randint(2, 5)
                            rgb_deg_type = random.choice(
                                rgb_deg_types_no_missing)
                            t_deg_type = random.choice(
                                t_deg_types_no_missing)

                        img_levels_rgb.append(level_rgb)
                        img_levels_t.append(level_t)
                        rgb_region_configs.append(
                            (deg_regions[region_idx],
                             rgb_deg_type, level_rgb))
                        t_region_configs.append(
                            (deg_regions[region_idx],
                             t_deg_type, level_t))

                    deg_img_rgb = apply_multi_region_degradation(
                        rgb_clean, rgb_region_configs, modality='rgb')
                    deg_img_t = apply_multi_region_degradation(
                        t_clean, t_region_configs, modality='ir')

                    all_region_levels_rgb.append(img_levels_rgb)
                    all_region_levels_t.append(img_levels_t)

                    with torch.no_grad():
                        tok_deg_rgb, _ = vit_encoder(deg_img_rgb)
                        tok_deg_t, _ = vit_encoder(deg_img_t)
                    q_deg_rgb = quality_net.forward_rgb(tok_deg_rgb)
                    q_deg_t = quality_net.forward_thermal(tok_deg_t)
                    q_deg_list_rgb.append(q_deg_rgb)
                    q_deg_list_t.append(q_deg_t)

                q_clean_rgb, q_clean_t = quality_net(tok_rgb, tok_t)

                all_token_levels_rgb = []
                all_token_levels_t = []
                for deg_idx in range(K):
                    tl_rgb = compute_token_deg_level_map(
                        deg_regions, all_region_levels_rgb[deg_idx],
                        img_size)
                    tl_t = compute_token_deg_level_map(
                        deg_regions, all_region_levels_t[deg_idx],
                        img_size)
                    all_token_levels_rgb.append(tl_rgb.to(device))
                    all_token_levels_t.append(tl_t.to(device))

                quality_by_level_rgb = {l: [] for l in range(6)}
                quality_by_level_t = {l: [] for l in range(6)}

                quality_by_level_rgb[0].append(q_clean_rgb)
                quality_by_level_t[0].append(q_clean_t)

                for deg_idx in range(K):
                    for level in range(1, 6):
                        mask_rgb = (
                            all_token_levels_rgb[deg_idx] == level)
                        mask_t = (
                            all_token_levels_t[deg_idx] == level)
                        if mask_rgb.any():
                            quality_by_level_rgb[level].append(
                                q_deg_list_rgb[deg_idx][:, mask_rgb])
                        if mask_t.any():
                            quality_by_level_t[level].append(
                                q_deg_list_t[deg_idx][:, mask_t])

                q_mean_rgb = {}
                q_mean_t = {}
                for level in range(6):
                    if quality_by_level_rgb[level]:
                        q_mean_rgb[level] = torch.cat(
                            quality_by_level_rgb[level], dim=1).mean()
                    if quality_by_level_t[level]:
                        q_mean_t[level] = torch.cat(
                            quality_by_level_t[level], dim=1).mean()

                available_levels_rgb = sorted(q_mean_rgb.keys())
                available_levels_t = sorted(q_mean_t.keys())

                q_list_rgb = [q_mean_rgb[l]
                              for l in available_levels_rgb]
                q_list_t = [q_mean_t[l] for l in available_levels_t]

                loss_margin_rgb = level_margin_fn(
                    q_list_rgb) if len(q_list_rgb) >= 2 \
                    else torch.tensor(0.0, device=device)
                loss_margin_t = level_margin_fn(
                    q_list_t) if len(q_list_t) >= 2 \
                    else torch.tensor(0.0, device=device)
                loss_margin = (loss_margin_rgb + loss_margin_t) / 2.0

                q_deg_last_rgb = q_deg_list_rgb[-1] if q_deg_list_rgb else None
                q_deg_last_t = q_deg_list_t[-1] if q_deg_list_t else None

                loss_anchor_rgb_c, loss_anchor_rgb_d = quality_anchor_fn(
                    q_clean_rgb, q_deg_last_rgb)
                loss_anchor_t_c, loss_anchor_t_d = quality_anchor_fn(
                    q_clean_t, q_deg_last_t)
                loss_anchor = (loss_anchor_rgb_c + loss_anchor_rgb_d +
                               loss_anchor_t_c + loss_anchor_t_d) / 2.0

                loss_ceiling = torch.tensor(0.0, device=device)
                if q_deg_list_rgb:
                    loss_ceiling = loss_ceiling + high_deg_ceiling_fn(
                        q_deg_list_rgb[-1])
                if q_deg_list_t:
                    loss_ceiling = loss_ceiling + high_deg_ceiling_fn(
                        q_deg_list_t[-1])

                loss_spatial_div = (spatial_div_fn(q_clean_rgb) +
                                    spatial_div_fn(q_clean_t))

                loss_total = loss_margin + loss_anchor + loss_ceiling + loss_spatial_div

            loss_scaled = loss_total / accumulation_steps
            scaler.scale(loss_scaled).backward()

            epoch_loss_margin += loss_margin.item()
            epoch_loss_anchor += loss_anchor.item()
            epoch_loss_ceiling += loss_ceiling.item()
            epoch_loss_spatial_div += loss_spatial_div.item()
            epoch_loss_total += loss_total.item()
            num_batches += 1

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params_list, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                if (batch_idx + 1) % args.log_interval == 0:
                    avg_margin = epoch_loss_margin / num_batches
                    avg_anchor = epoch_loss_anchor / num_batches
                    avg_ceiling = epoch_loss_ceiling / num_batches
                    avg_spatial_div = epoch_loss_spatial_div / num_batches
                    avg_total = epoch_loss_total / num_batches

                    logger.info(
                        f"Epoch [{epoch + 1}/{args.epochs}], "
                        f"Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {avg_total:.4f}, "
                        f"Margin: {avg_margin:.4f}, "
                        f"Anchor: {avg_anchor:.4f}, "
                        f"Ceiling: {avg_ceiling:.4f}, "
                        f"SpatialDiv: {avg_spatial_div:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                    writer.add_scalar('Loss/total', avg_total, global_step)
                    writer.add_scalar('Loss/margin', avg_margin, global_step)
                    writer.add_scalar('Loss/anchor', avg_anchor, global_step)
                    writer.add_scalar('Loss/ceiling', avg_ceiling,
                                      global_step)
                    writer.add_scalar('Loss/spatial_div', avg_spatial_div,
                                      global_step)
                    writer.add_scalar('LR',
                                      optimizer.param_groups[0]['lr'],
                                      global_step)

                    with torch.no_grad():
                        q_clean_mean_rgb = q_clean_rgb.mean().item()
                        q_clean_mean_t = q_clean_t.mean().item()
                    writer.add_scalar('Quality/clean_mean_rgb',
                                      q_clean_mean_rgb, global_step)
                    writer.add_scalar('Quality/clean_mean_t',
                                      q_clean_mean_t, global_step)

            pbar.set_postfix({
                'margin': f'{loss_margin.item():.4f}',
                'anchor': f'{loss_anchor.item():.4f}',
                'total': f'{loss_total.item():.4f}',
            })

        scheduler.step()

        avg_epoch_loss = epoch_loss_total / max(num_batches, 1)
        logger.info(
            f'Epoch [{epoch + 1}/{args.epochs}] 完成, '
            f'平均损失: {avg_epoch_loss:.4f}')

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(weight_dir, 'best_quality_net.pth')
            torch.save(quality_net.state_dict(), save_path)
            logger.info(
                f'最佳模型已保存: {save_path} (loss={best_loss:.4f})')

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(
                weight_dir, f'quality_net_epoch{epoch + 1}.pth')
            torch.save(quality_net.state_dict(), save_path)

    final_path = os.path.join(weight_dir, 'final_quality_net.pth')
    torch.save(quality_net.state_dict(), final_path)
    logger.info(f'训练完成! 最终模型已保存: {final_path}')

    best_weight_path = os.path.join(weight_dir, 'best_quality_net.pth')
    info_path = os.path.join(weight_dir, 'checkpoint_info.txt')
    with open(info_path, 'w') as f:
        f.write(f'best: {best_weight_path}\n')
        f.write(f'final: {final_path}\n')

    logger.info('开始验证质量网络排序能力...')
    quality_net.eval()

    val_dataset = FMBQualityDataset(
        data_root=args.data_root,
        split='validation',
        img_size=img_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    sum_acc_rgb = 0.0
    sum_acc_t = 0.0
    num_batches_rgb = 0
    num_batches_t = 0

    with torch.no_grad():
        for rgb, thermal, seg_label in tqdm(val_loader, desc='验证中'):
            rgb = rgb.to(device)
            thermal = thermal.to(device)

            tok_rgb, _ = vit_encoder(rgb)
            q_clean_rgb = quality_net.forward_rgb(tok_rgb)

            for deg_type, level in [
                    ('low_light', 3), ('motion_blur', 3),
                    ('overexposure', 3)]:
                deg_img, _ = apply_degradation(
                    rgb, deg_type, level, 'rgb')
                tok_deg, _ = vit_encoder(deg_img)
                q_deg = quality_net.forward_rgb(tok_deg)
                correct = (q_clean_rgb > q_deg).float().mean().item()
                sum_acc_rgb += correct
                num_batches_rgb += 1

            tok_t, _ = vit_encoder(thermal)
            q_clean_t = quality_net.forward_thermal(tok_t)

            for deg_type, level in [
                    ('thermal_contrast', 3), ('stripe_noise', 3),
                    ('thermal_noise', 3), ('thermal_saturation', 3)]:
                deg_img, _ = apply_degradation(
                    thermal, deg_type, level, 'ir')
                tok_deg, _ = vit_encoder(deg_img)
                q_deg = quality_net.forward_thermal(tok_deg)
                correct = (q_clean_t > q_deg).float().mean().item()
                sum_acc_t += correct
                num_batches_t += 1

    ranking_acc_rgb = sum_acc_rgb / max(num_batches_rgb, 1)
    ranking_acc_t = sum_acc_t / max(num_batches_t, 1)
    ranking_acc_avg = (ranking_acc_rgb + ranking_acc_t) / 2.0

    logger.info(f'RGB验证排序准确率: {ranking_acc_rgb:.4f} (期望>0.85)')
    logger.info(f'Thermal验证排序准确率: {ranking_acc_t:.4f} (期望>0.85)')
    logger.info(f'平均验证排序准确率: {ranking_acc_avg:.4f}')

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Pretrain Quality Network')
    parser.add_argument('--config', type=str, required=True,
                        help='分割模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='分割模型checkpoint路径')
    parser.add_argument('--data-root', type=str,
                        default='/home/lh/code/data/FMB_ALL',
                        help='FMB数据集根目录')
    parser.add_argument('--vit-pretrained', type=str,
                        default='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
                        help='预训练MAE backbone权重')
    parser.add_argument('--resume', type=str, default=None,
                        help='继续训练的质量网络checkpoint')
    parser.add_argument('--work-dir', type=str,
                        default='./work_dirs/quality_pretrain',
                        help='输出目录')
    parser.add_argument('--img-size', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-degradations', type=int, default=4,
                        help='退化图像数量K')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--top-ratio', type=float, default=0.2,
                        help='高质量锚点选择比例')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--amp', action='store_true',
                        help='启用自动混合精度训练')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
