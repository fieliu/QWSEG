import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import nlc_to_nchw
from .base import BaseSegmentor


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class QualityAwareSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, sr_ratio=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, hw_shape=None, quality=None):
        B, N, D = x.shape
        residual = x

        q = self.q_proj(x).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1 and hw_shape is not None:
            x_kv = x.transpose(1, 2).reshape(B, D, hw_shape[0], hw_shape[1])
            x_kv = self.sr(x_kv)
            x_kv = x_kv.flatten(2).transpose(1, 2)
            x_kv = self.sr_norm(x_kv)
        else:
            x_kv = x

        k = self.k_proj(x_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if quality is not None:
            attn_bias = self._compute_quality_bias(quality, hw_shape, N)
            attn_weights = attn_weights + attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + residual)
        return out

    def _compute_quality_bias(self, quality, hw_shape, N):
        B = quality.shape[0]
        if hw_shape is not None:
            H, W = hw_shape
            if quality.shape[2:] != (H, W):
                quality = F.interpolate(
                    quality, size=(H, W), mode='nearest')
            q_flat = quality.squeeze(1).reshape(B, -1)
            if q_flat.shape[1] != N:
                q_flat = q_flat[:, :N]
        else:
            q_flat = quality.squeeze(1).reshape(B, -1)

        q_norm = q_flat.unsqueeze(1)
        q_pair = q_norm.transpose(-2, -1) + q_norm
        bias = torch.log(q_pair / 2.0 + 1e-6)
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        return bias


class QualityAwareCrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, sr_ratio=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v, hw_shape=None, quality=None):
        B, N, D = q.shape
        residual = q

        q_out = self.q_proj(q).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1 and hw_shape is not None:
            k_kv = k.transpose(1, 2).reshape(B, D, hw_shape[0], hw_shape[1])
            k_kv = self.sr(k_kv)
            k_kv = k_kv.flatten(2).transpose(1, 2)
            k_kv = self.sr_norm(k_kv)
            v_kv = v.transpose(1, 2).reshape(B, D, hw_shape[0], hw_shape[1])
            v_kv = self.sr(v_kv)
            v_kv = v_kv.flatten(2).transpose(1, 2)
        else:
            k_kv = k
            v_kv = v

        k_out = self.k_proj(k_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_out = self.v_proj(v_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(
            q_out, k_out.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if quality is not None:
            attn_bias = self._compute_quality_bias(quality, hw_shape, N)
            attn_weights = attn_weights + attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(
            attn_weights, v_out).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + residual)
        return out

    def _compute_quality_bias(self, quality, hw_shape, N_q):
        B = quality.shape[0]
        if hw_shape is not None:
            H, W = hw_shape
            if quality.shape[2:] != (H, W):
                quality = F.interpolate(
                    quality, size=(H, W), mode='nearest')
            q_flat = quality.squeeze(1).reshape(B, -1)
            if q_flat.shape[1] != N_q:
                q_flat = q_flat[:, :N_q]
        else:
            q_flat = quality.squeeze(1).reshape(B, -1)

        bias = torch.log(q_flat.unsqueeze(1) + 1e-6)
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        return bias


class SimpleMLPSegHead(nn.Module):

    def __init__(self, in_channels_list, num_classes, channels=256):
        super().__init__()
        num_stages = len(in_channels_list)
        self.convs = nn.ModuleList()
        for i in range(num_stages):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels_list[i], channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)))

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_stages, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        outs = []
        target_size = inputs[0].shape[2:]
        for idx in range(len(inputs)):
            x = self.convs[idx](inputs[idx])
            x = F.interpolate(
                x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        out = self.fusion(torch.cat(outs, dim=1))
        return self.cls_seg(out)


def compute_orthogonal_loss(zc, zp):
    B, C, N = zc.shape
    zc_norm = F.normalize(zc, dim=2)
    zp_norm = F.normalize(zp, dim=2)
    corr = torch.bmm(zc_norm, zp_norm.transpose(1, 2))
    loss = (corr ** 2).mean()
    return loss


def compute_infonce_loss(zc_rgb, zc_t, temperature=0.07):
    B = zc_rgb.shape[0]
    zc_rgb_gap = F.adaptive_avg_pool2d(zc_rgb, 1).flatten(1)
    zc_t_gap = F.adaptive_avg_pool2d(zc_t, 1).flatten(1)
    zc_rgb_norm = F.normalize(zc_rgb_gap, dim=1)
    zc_t_norm = F.normalize(zc_t_gap, dim=1)
    sim = torch.mm(zc_rgb_norm, zc_t_norm.t()) / temperature
    labels = torch.arange(B, device=zc_rgb.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    return loss


def compute_quality_aware_alignment_loss(zc_rgb, zc_t, q_rgb, q_t,
                                          threshold=0.1):
    B, C, H, W = zc_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)

    valid_mask = (q_rgb_s > threshold) & (q_t_s > threshold)

    q_diff = torch.abs(q_rgb_s - q_t_s)
    q_min = torch.min(q_rgb_s, q_t_s)

    similarity_weight = torch.exp(-q_diff * 5.0)
    quality_weight = q_min

    weight = similarity_weight * quality_weight * valid_mask.float()

    token_dist = ((zc_rgb - zc_t) ** 2).mean(dim=1)

    total_weight = weight.sum() + 1e-8
    loss = (weight * token_dist).sum() / total_weight

    return loss


def compute_cosine_similarity_loss(clean_feat, deg_feat):
    clean_norm = F.normalize(clean_feat, dim=1)
    deg_norm = F.normalize(deg_feat, dim=1)
    cos_sim = (clean_norm * deg_norm).sum(dim=1)
    loss = (1 - cos_sim).mean()
    return loss


class ModalityDegradation(nn.Module):

    def __init__(self, global_ratio=0.4, region_ratio=0.6,
                 region_min_ratio=0.1, region_max_ratio=0.5,
                 noise_std_range=(0.01, 0.1),
                 blur_kernel_sizes=None,
                 brightness_range=(0.5, 1.5),
                 contrast_range=(0.5, 1.5),
                 missing_ratio=0.3):
        super().__init__()
        if blur_kernel_sizes is None:
            blur_kernel_sizes = [3, 5, 7]
        self.global_ratio = global_ratio
        self.region_ratio = region_ratio
        self.region_min_ratio = region_min_ratio
        self.region_max_ratio = region_max_ratio
        self.noise_std_range = noise_std_range
        self.blur_kernel_sizes = blur_kernel_sizes
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.missing_ratio = missing_ratio

    def _apply_noise(self, x):
        std = torch.empty(1).uniform_(*self.noise_std_range).item()
        noise = torch.randn_like(x) * std
        return (x + noise).clamp(0, 1)

    def _apply_blur(self, x):
        k = self.blur_kernel_sizes[
            torch.randint(len(self.blur_kernel_sizes), (1,)).item()]
        padding = k // 2
        x_pad = F.pad(x, [padding] * 4, mode='reflect')
        return F.avg_pool2d(x_pad, k, stride=1)

    def _apply_brightness(self, x):
        factor = torch.empty(1).uniform_(*self.brightness_range).item()
        return (x * factor).clamp(0, 1)

    def _apply_contrast(self, x):
        factor = torch.empty(1).uniform_(*self.contrast_range).item()
        mean = x.mean(dim=[2, 3], keepdim=True)
        return ((x - mean) * factor + mean).clamp(0, 1)

    def _apply_missing(self, x):
        return torch.zeros_like(x)

    def _get_random_degradation(self):
        deg_types = ['noise', 'blur', 'brightness', 'contrast', 'missing']
        weights = [0.25, 0.25, 0.2, 0.2, 0.1]
        idx = torch.multinomial(torch.tensor(weights), 1).item()
        return deg_types[idx]

    def _apply_global_degradation(self, x):
        deg = self._get_random_degradation()
        if deg == 'noise':
            return self._apply_noise(x)
        elif deg == 'blur':
            return self._apply_blur(x)
        elif deg == 'brightness':
            return self._apply_brightness(x)
        elif deg == 'contrast':
            return self._apply_contrast(x)
        else:
            return self._apply_missing(x)

    def _apply_region_degradation(self, x):
        B, C, H, W = x.shape
        deg = self._get_random_degradation()
        if deg == 'missing':
            return self._apply_missing(x)

        ratio = torch.empty(1).uniform_(
            self.region_min_ratio, self.region_max_ratio).item()
        h_size = max(int(H * ratio), 1)
        w_size = max(int(W * ratio), 1)
        h_start = torch.randint(0, max(H - h_size, 1), (1,)).item()
        w_start = torch.randint(0, max(W - w_size, 1), (1,)).item()

        region = x[:, :, h_start:h_start + h_size,
                    w_start:w_start + w_size]
        if deg == 'noise':
            region = self._apply_noise(region)
        elif deg == 'blur':
            region = self._apply_blur(region)
        elif deg == 'brightness':
            region = self._apply_brightness(region)
        elif deg == 'contrast':
            region = self._apply_contrast(region)

        x = x.clone()
        x[:, :, h_start:h_start + h_size,
          w_start:w_start + w_size] = region
        return x

    def forward(self, rgb, thermal):
        B = rgb.shape[0]
        deg_rgb = rgb.clone()
        deg_t = thermal.clone()

        for b in range(B):
            r = torch.rand(1).item()
            if r < self.global_ratio:
                if torch.rand(1).item() < 0.5:
                    deg_rgb[b] = self._apply_global_degradation(
                        rgb[b:b + 1]).squeeze(0)
                if torch.rand(1).item() < 0.5:
                    deg_t[b] = self._apply_global_degradation(
                        thermal[b:b + 1]).squeeze(0)
            else:
                if torch.rand(1).item() < 0.5:
                    deg_rgb[b] = self._apply_region_degradation(
                        rgb[b:b + 1]).squeeze(0)
                if torch.rand(1).item() < 0.5:
                    deg_t[b] = self._apply_region_degradation(
                        thermal[b:b + 1]).squeeze(0)

        return deg_rgb, deg_t


@MODELS.register_module()
class MiTMulV7QualityAdaptive(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 universal_embed_dims=None,
                 private_embed_dims=None,
                 loss_seg_zc_weight=0.3,
                 loss_seg_zp_residual_weight=0.5,
                 loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
                 loss_invariant_weight=0.5,
                 loss_modality_weight=0.3,
                 loss_variance_weight=0.1,
                 loss_degradation_consistency_weight=1.0,
                 loss_deg_seg_weight=0.5,
                 loss_align_weight=0.5,
                 quality_threshold=0.1,
                 deg_loss_warmup_iters=3000,
                 degradation_cfg=None,
                 quality_pretrained=None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        if universal_embed_dims is None:
            universal_embed_dims = [64, 128, 320, 512]
        if private_embed_dims is None:
            private_embed_dims = [32, 64, 128, 256]
        self.universal_embed_dims = universal_embed_dims
        self.private_embed_dims = private_embed_dims

        num_stages = len(universal_embed_dims)

        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)

        if quality_pretrained and os.path.exists(quality_pretrained):
            ckpt = torch.load(quality_pretrained, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.quality_pyramid_net.load_state_dict(
                    ckpt['model_state_dict'], strict=False)
            else:
                self.quality_pyramid_net.load_state_dict(
                    ckpt, strict=False)
            print_log(
                f'Loaded quality pyramid net from {quality_pretrained}',
                logger='current')

        self.channel_attn = nn.ModuleList()
        self.spatial_attn = nn.ModuleList()
        self.fusion_mlps = nn.ModuleList()
        for i in range(num_stages):
            self.channel_attn.append(
                ChannelAttention(universal_embed_dims[i] * 2))
            self.spatial_attn.append(
                SpatialAttention())
            self.fusion_mlps.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i] * 2, universal_embed_dims[i],
                          1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))

        self.self_attn_universal = nn.ModuleList()
        self.self_attn_rgb = nn.ModuleList()
        self.self_attn_t = nn.ModuleList()
        self.private_proj_rgb = nn.ModuleList()
        self.private_proj_t = nn.ModuleList()
        attn_sr_ratios = [8, 4, 2, 1]
        for i in range(num_stages):
            sr = attn_sr_ratios[i] if i < len(attn_sr_ratios) else 1
            self.self_attn_universal.append(
                QualityAwareSelfAttention(
                    embed_dim=universal_embed_dims[i], sr_ratio=sr))
            self.self_attn_rgb.append(
                QualityAwareSelfAttention(
                    embed_dim=universal_embed_dims[i], sr_ratio=sr))
            self.self_attn_t.append(
                QualityAwareSelfAttention(
                    embed_dim=universal_embed_dims[i], sr_ratio=sr))
            self.private_proj_rgb.append(nn.Sequential(
                nn.Conv2d(private_embed_dims[i], universal_embed_dims[i],
                          1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))
            self.private_proj_t.append(nn.Sequential(
                nn.Conv2d(private_embed_dims[i], universal_embed_dims[i],
                          1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))

        self.cross_attn_rgb = nn.ModuleList()
        self.cross_attn_t = nn.ModuleList()
        self.cross_proj_rgb = nn.ModuleList()
        self.cross_proj_t = nn.ModuleList()
        for i in range(num_stages):
            sr = attn_sr_ratios[i] if i < len(attn_sr_ratios) else 1
            self.cross_attn_rgb.append(
                QualityAwareCrossAttention(
                    embed_dim=universal_embed_dims[i], sr_ratio=sr))
            self.cross_attn_t.append(
                QualityAwareCrossAttention(
                    embed_dim=universal_embed_dims[i], sr_ratio=sr))
            self.cross_proj_rgb.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          1, bias=True),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))
            self.cross_proj_t.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          1, bias=True),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))

        self.zc_seg_head_mlp = SimpleMLPSegHead(
            in_channels_list=universal_embed_dims,
            num_classes=decode_head.get('num_classes', 15),
            channels=256)

        zp_in_channels = [d * 2 for d in private_embed_dims]
        self.zp_residual_head = SimpleMLPSegHead(
            in_channels_list=zp_in_channels,
            num_classes=decode_head.get('num_classes', 15),
            channels=256)

        self.modality_classifier = nn.ModuleList()
        for i in range(num_stages):
            self.modality_classifier.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(private_embed_dims[i], 2)))

        self.loss_seg_zc_weight = loss_seg_zc_weight
        self.loss_seg_zp_residual_weight = loss_seg_zp_residual_weight
        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_modality_weight = loss_modality_weight
        self.loss_variance_weight = loss_variance_weight
        self.loss_degradation_consistency_weight = \
            loss_degradation_consistency_weight
        self.loss_deg_seg_weight = loss_deg_seg_weight
        self.loss_align_weight = loss_align_weight
        self.quality_threshold = quality_threshold
        self.deg_loss_warmup_iters = deg_loss_warmup_iters
        self._train_iter_count = 0

        if degradation_cfg is None:
            degradation_cfg = dict(
                global_ratio=0.4,
                region_ratio=0.6,
                region_min_ratio=0.1,
                region_max_ratio=0.5,
                noise_std_range=(0.01, 0.1),
                blur_kernel_sizes=[3, 5, 7],
                brightness_range=(0.5, 1.5),
                contrast_range=(0.5, 1.5),
                missing_ratio=0.3)
        self.degradation = ModalityDegradation(**degradation_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def init_weights(self):
        if self.init_cfg is not None and 'checkpoint' in self.init_cfg:
            checkpoint_path = self.init_cfg['checkpoint']
            print_log(
                f'Loading checkpoint from: {checkpoint_path}',
                logger='current')

            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt

                new_state_dict = {}
                loaded_keys = []
                skipped_keys = []

                skip_prefixes = (
                    'quality_pyramid_net.',
                    'self_attn_universal.', 'self_attn_rgb.',
                    'self_attn_t.', 'cross_attn_rgb.', 'cross_attn_t.',
                    'decode_head.', 'auxiliary_head.',
                    'zc_seg_head_mlp.', 'zp_residual_head.',
                    'modality_classifier.')

                for k, v in state_dict.items():
                    if any(k.startswith(p) for p in skip_prefixes):
                        continue

                    if k in self.state_dict().keys():
                        if self.state_dict()[k].shape == v.shape:
                            new_state_dict[k] = v
                            loaded_keys.append(k)
                        else:
                            skipped_keys.append(
                                (k, v.shape, self.state_dict()[k].shape))

                print_log(
                    f'Loaded {len(loaded_keys)} keys from checkpoint',
                    logger='current')
                if skipped_keys:
                    print_log(
                        f'Skipped {len(skipped_keys)} keys due to shape '
                        f'mismatch:', logger='current')
                    for key, src_shape, dst_shape in skipped_keys[:10]:
                        print_log(
                            f'  {key}: {src_shape} -> {dst_shape}',
                            logger='current')

                self.load_state_dict(new_state_dict, strict=False)
                print_log(
                    'Successfully initialized from checkpoint',
                    logger='current')

            except Exception as e:
                print_log(
                    f'Failed to load checkpoint: {e}', logger='current',
                    level=logging.WARNING)
                super().init_weights()
        else:
            super().init_weights()

    def _forward_backbone_with_quality(self, backbone, x, q_maps,
                                        threshold=0.1):
        outs = []
        for i, layer in enumerate(backbone.layers):
            x, hw_shape = layer[0](x)
            H, W = hw_shape
            if i < len(q_maps):
                q_i = q_maps[i]
                if q_i.shape[2:] != (H, W):
                    q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
                q_flat = q_i.squeeze(1).reshape(x.shape[0], -1)
                if q_flat.shape[1] == x.shape[1]:
                    q_token = q_flat.unsqueeze(2)
                    x = x * q_token
                    mask = (q_flat < threshold)
                    if mask.any():
                        x[mask] = 0.0
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in backbone.out_indices:
                outs.append(x)
        return outs

    def extract_feat(self, input_rgb: torch.Tensor, input_t: torch.Tensor):
        B = input_rgb.shape[0]

        q_rgb_maps = self.quality_pyramid_net.forward_rgb(input_rgb)
        q_t_maps = self.quality_pyramid_net.forward_thermal(input_t)

        zc_rgb_list = self._forward_backbone_with_quality(
            self.backbone, input_rgb, q_rgb_maps, self.quality_threshold)
        zc_t_list = self._forward_backbone_with_quality(
            self.backbone, input_t, q_t_maps, self.quality_threshold)

        zp_rgb_list = self._forward_backbone_with_quality(
            self.private_branch_rgb, input_rgb, q_rgb_maps,
            self.quality_threshold)
        zp_t_list = self._forward_backbone_with_quality(
            self.private_branch_t, input_t, q_t_maps,
            self.quality_threshold)

        num_stages = len(zc_rgb_list)

        zc_concat_feats = []
        zc_enhanced_feats = []
        for i in range(num_stages):
            concat_feat = torch.cat([zc_rgb_list[i], zc_t_list[i]], dim=1)
            concat_feat = self.channel_attn[i](concat_feat)
            concat_feat = self.spatial_attn[i](concat_feat)
            zc_concat_feats.append(concat_feat)

            fused_feat = self.fusion_mlps[i](concat_feat)
            zc_enhanced_feats.append(fused_feat)

        universal_tokens_list = []
        zp_rgb_aligned_list = []
        zp_t_aligned_list = []
        for i in range(num_stages):
            H_i, W_i = zc_enhanced_feats[i].shape[-2:]
            hw_shape_i = (H_i, W_i)
            universal_tokens = zc_enhanced_feats[i].flatten(2).transpose(1, 2)

            q_fused_i = torch.min(q_rgb_maps[i], q_t_maps[i])
            if q_fused_i.shape[2:] != (H_i, W_i):
                q_fused_i = F.interpolate(
                    q_fused_i, size=(H_i, W_i), mode='nearest')

            universal_tokens = self.self_attn_universal[i](
                universal_tokens, hw_shape_i, quality=q_fused_i)
            universal_tokens_list.append(universal_tokens)

            zp_rgb_aligned = self.private_proj_rgb[i](zp_rgb_list[i])
            zp_t_aligned = self.private_proj_t[i](zp_t_list[i])
            zp_rgb_tokens = zp_rgb_aligned.flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_aligned.flatten(2).transpose(1, 2)

            q_rgb_i = q_rgb_maps[i]
            if q_rgb_i.shape[2:] != (H_i, W_i):
                q_rgb_i = F.interpolate(
                    q_rgb_i, size=(H_i, W_i), mode='nearest')
            zp_rgb_tokens = self.self_attn_rgb[i](
                zp_rgb_tokens, hw_shape_i, quality=q_rgb_i)

            q_t_i = q_t_maps[i]
            if q_t_i.shape[2:] != (H_i, W_i):
                q_t_i = F.interpolate(
                    q_t_i, size=(H_i, W_i), mode='nearest')
            zp_t_tokens = self.self_attn_t[i](
                zp_t_tokens, hw_shape_i, quality=q_t_i)

            zp_rgb_aligned_list.append(zp_rgb_tokens)
            zp_t_aligned_list.append(zp_t_tokens)

        fused_feats = []
        for i in range(num_stages):
            H_i, W_i = zc_enhanced_feats[i].shape[-2:]
            hw_shape_i = (H_i, W_i)
            universal_tokens = universal_tokens_list[i]
            zp_rgb_tokens = zp_rgb_aligned_list[i]
            zp_t_tokens = zp_t_aligned_list[i]

            q_rgb_i = q_rgb_maps[i]
            if q_rgb_i.shape[2:] != (H_i, W_i):
                q_rgb_i = F.interpolate(
                    q_rgb_i, size=(H_i, W_i), mode='nearest')

            cross_rgb = self.cross_attn_rgb[i](
                universal_tokens, zp_rgb_tokens, zp_rgb_tokens,
                hw_shape_i, quality=q_rgb_i)
            rgb_enhanced = cross_rgb.permute(0, 2, 1).reshape(
                B, -1, H_i, W_i)
            rgb_enhanced = self.cross_proj_rgb[i](rgb_enhanced)
            rgb_enhanced_tokens = rgb_enhanced.flatten(2).transpose(1, 2)

            q_t_i = q_t_maps[i]
            if q_t_i.shape[2:] != (H_i, W_i):
                q_t_i = F.interpolate(
                    q_t_i, size=(H_i, W_i), mode='nearest')

            cross_t = self.cross_attn_t[i](
                rgb_enhanced_tokens, zp_t_tokens, zp_t_tokens,
                hw_shape_i, quality=q_t_i)
            t_enhanced = cross_t.permute(0, 2, 1).reshape(
                B, -1, H_i, W_i)
            t_enhanced = self.cross_proj_t[i](t_enhanced)
            fused_feats.append(t_enhanced)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
            zc_enhanced_feats_neck = self.neck(zc_enhanced_feats)
        else:
            zc_enhanced_feats_neck = zc_enhanced_feats

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_enhanced_feats_neck, fused_feats,
                q_rgb_maps, q_t_maps)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        fused_feats = results[5]
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _zc_mlp_seg_forward_train(self, zc_enhanced_feats, data_samples):
        seg_logits = self.zc_seg_head_mlp(zc_enhanced_feats)
        seg_label = torch.stack([
            ds.gt_sem_seg.data for ds in data_samples
        ]).squeeze(1).long()
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=seg_label.shape[-2:],
                mode='bilinear', align_corners=False)
        loss_ce = F.cross_entropy(
            seg_logits, seg_label, ignore_index=255)
        return loss_ce, seg_logits

    def _zp_residual_seg_forward_train(self, zp_rgb_list, zp_t_list,
                                       zc_logits, data_samples):
        num_stages = len(zp_rgb_list)
        zp_fused = []
        for i in range(num_stages):
            zp_cat = torch.cat([zp_rgb_list[i], zp_t_list[i]], dim=1)
            zp_fused.append(zp_cat)
        residual_logits = self.zp_residual_head(zp_fused)
        if residual_logits.shape[-2:] != zc_logits.shape[-2:]:
            residual_logits = F.interpolate(
                residual_logits, size=zc_logits.shape[-2:],
                mode='bilinear', align_corners=False)
        combined_logits = zc_logits.detach() + residual_logits
        seg_label = torch.stack([
            ds.gt_sem_seg.data for ds in data_samples
        ]).squeeze(1).long()
        if combined_logits.shape[-2:] != seg_label.shape[-2:]:
            combined_logits = F.interpolate(
                combined_logits, size=seg_label.shape[-2:],
                mode='bilinear', align_corners=False)
        loss_ce = F.cross_entropy(
            combined_logits, seg_label, ignore_index=255)
        return loss_ce

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(
                    inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(
                inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    class _DecodeHeadGradBlocker:
        def __init__(self, decode_head):
            self._head = decode_head
            self._saved_requires_grad = {}

        def __enter__(self):
            for name, param in self._head.named_parameters():
                self._saved_requires_grad[name] = param.requires_grad
                param.requires_grad = False
            return self

        def __exit__(self, *args):
            for name, param in self._head.named_parameters():
                param.requires_grad = self._saved_requires_grad.get(
                    name, param.requires_grad)

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_enhanced_feats, fused_feats,
         q_rgb_maps, q_t_maps) = self.extract_feat(input_rgb, input_ir)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        loss_zc_seg, zc_logits = self._zc_mlp_seg_forward_train(
            zc_enhanced_feats, data_samples)
        losses['loss_seg_zc'] = loss_zc_seg * self.loss_seg_zc_weight

        loss_zp_residual = self._zp_residual_seg_forward_train(
            zp_rgb_list, zp_t_list, zc_logits, data_samples)
        losses['loss_seg_zp_residual'] = loss_zp_residual * \
            self.loss_seg_zp_residual_weight

        num_stages = len(zc_rgb_list)

        loss_invariant_total = 0.0
        for i in range(num_stages):
            loss_invariant_total += compute_infonce_loss(
                zc_rgb_list[i], zc_t_list[i])
        losses['loss_invariant'] = loss_invariant_total / num_stages * \
            self.loss_invariant_weight

        for i in range(num_stages):
            zp_rgb_aligned = self.private_proj_rgb[i](zp_rgb_list[i])
            zp_t_aligned = self.private_proj_t[i](zp_t_list[i])

            zc_feat = zc_enhanced_feats[i].flatten(2)
            zp_rgb_feat = zp_rgb_aligned.flatten(2)
            zp_t_feat = zp_t_aligned.flatten(2)

            w = self.loss_disentangle_weights[i] \
                if i < len(self.loss_disentangle_weights) \
                else self.loss_disentangle_weights[-1]

            loss_ortho_rgb = compute_orthogonal_loss(zc_feat, zp_rgb_feat)
            loss_ortho_t = compute_orthogonal_loss(zc_feat, zp_t_feat)
            losses[f'loss_disentangle_s{i}'] = \
                (loss_ortho_rgb + loss_ortho_t) * w

        B = inputs.shape[0]
        modality_labels_rgb = torch.zeros(
            B, dtype=torch.long, device=inputs.device)
        modality_labels_t = torch.ones(
            B, dtype=torch.long, device=inputs.device)
        loss_modality_total = 0.0
        for i in range(num_stages):
            pred_rgb = self.modality_classifier[i](zp_rgb_list[i])
            pred_t = self.modality_classifier[i](zp_t_list[i])
            loss_modality_total += F.cross_entropy(
                pred_rgb, modality_labels_rgb)
            loss_modality_total += F.cross_entropy(
                pred_t, modality_labels_t)
        losses['loss_modality'] = loss_modality_total / num_stages * \
            self.loss_modality_weight

        loss_var_total = 0.0
        for i in range(num_stages):
            var_rgb = zp_rgb_list[i].var(dim=[2, 3]).mean()
            var_t = zp_t_list[i].var(dim=[2, 3]).mean()
            loss_var_total += -torch.log(var_rgb + 1e-6)
            loss_var_total += -torch.log(var_t + 1e-6)
        losses['loss_variance'] = loss_var_total / (num_stages * 2) * \
            self.loss_variance_weight

        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += compute_quality_aware_alignment_loss(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i])
        losses['loss_align'] = loss_align_total / num_stages * \
            self.loss_align_weight

        loss_q_reg = 0.0
        for i in range(num_stages):
            loss_q_reg += (1.0 - q_rgb_maps[i]).mean()
            loss_q_reg += (1.0 - q_t_maps[i]).mean()
        losses['loss_quality_reg'] = loss_q_reg / (num_stages * 2) * 0.1

        self._train_iter_count += 1
        warmup_scale = min(1.0, self._train_iter_count /
                           max(self.deg_loss_warmup_iters, 1))

        with torch.no_grad():
            rgb_norm = input_rgb * 0.5 + 0.5
            t_norm = input_ir * 0.5 + 0.5
            deg_rgb, deg_t = self.degradation(rgb_norm, t_norm)
            deg_rgb = (deg_rgb - 0.5) / 0.5
            deg_t = (deg_t - 0.5) / 0.5

        (deg_zc_rgb_list, deg_zc_t_list, deg_zp_rgb_list, deg_zp_t_list,
         deg_zc_enhanced, deg_fused_feats,
         _, _) = self.extract_feat(deg_rgb, deg_t)

        loss_consistency = 0.0
        for i in range(num_stages):
            loss_consistency += compute_cosine_similarity_loss(
                fused_feats[i].detach(), deg_fused_feats[i])
        losses['loss_deg_consist'] = loss_consistency / num_stages * \
            self.loss_degradation_consistency_weight * warmup_scale

        if warmup_scale > 0:
            with self._DecodeHeadGradBlocker(self.decode_head):
                deg_seg_logits = self.decode_head.forward(deg_fused_feats)
                deg_loss_dict = self.decode_head.loss_by_feat(
                    deg_seg_logits, data_samples)
                deg_seg_total = sum(v for k, v in deg_loss_dict.items()
                                   if k.startswith('loss_'))
                losses['loss_deg_seg'] = deg_seg_total * \
                    self.loss_deg_seg_weight * warmup_scale

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: torch.Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:],
                     img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:],
                     padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        fused_feats = results[5]
        return self.decode_head.forward(fused_feats)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros(
            (batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros(
            (batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(
                    crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2),
                                int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)
