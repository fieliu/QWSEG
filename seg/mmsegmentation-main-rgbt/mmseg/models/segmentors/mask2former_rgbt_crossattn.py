"""Mask2Former RGB-T with Channel-Spatial Attention Fusion.

Clean baseline for DINOv3+Adapter+M2F architecture.
The backbone (RGBTDINOv3Adapter) already performs inter-stage cross-modal
cross-attention, so each branch's features already "sense" the other modality.
The segmentor's job is simply:
  1. Concatenate [rgb_feat; thr_feat] (both already cross-modal-aware)
  2. Channel-Spatial Attention: CBAM-like module fuses and reduces dim back

This is the ORIGINAL model structure (no quality mechanism).
The quality version (DINOv3AdapterM2FQuality) adds quality-gated cross-attention
as NEW parameters inside the backbone, as a separate incremental module.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.logging import print_log
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from .mask2former_rgbt_add import Mask2FormerRGBTAdd


class ChannelSpatialAttention(nn.Module):
    """CBAM-like channel + spatial attention for fusion and dimensionality reduction.

    Input: concatenated features [B, 2C, H, W]
    Output: fused features [B, C, H, W]
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 1)
        self.fc1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # Dimensionality reduction: 2C -> C
        self.conv_reduce = nn.Conv2d(in_channels * 2, in_channels, 1, bias=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """x: [B, 2C, H, W] -> [B, C, H, W]"""
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        ch_attn = self.sigmoid(avg_out + max_out)  # [B, 2C, 1, 1]
        x = x * ch_attn

        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True)[0]
        sp_attn = self.sigmoid(self.conv_spatial(torch.cat([avg_s, max_s], dim=1)))
        x = x * sp_attn

        # Reduce dim: 2C -> C
        x = self.relu(self.bn(self.conv_reduce(x)))
        return x


@MODELS.register_module()
class Mask2FormerRGBTCrossAttn(Mask2FormerRGBTAdd):
    """Mask2Former RGB-T with Channel-Spatial Attention Fusion (no quality gating).

    The backbone (RGBTDINOv3Adapter) already performs inter-stage cross-modal
    attention, so rgb_feats[i] and thr_feats[i] are already cross-modal-aware.
    The segmentor simply concatenates them and uses CBAM-like attention to fuse
    and reduce dimensionality.

    Flow per scale:
      1. RGBTDINOv3Adapter returns (rgb_feats, thr_feats) with return_dual=True
         - Features are already cross-modal-aware (backbone does CrossAttn)
      2. Concatenate [rgb_feat; thr_feat] -> [B, 2C, H, W]
      3. ChannelSpatialAttention -> [B, C, H, W]
      4. Output to M2F head
    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 fusion_type: str = 'crossattn'):
        # Force backbone to return dual features
        if isinstance(backbone, dict):
            backbone = dict(backbone, return_dual=True)

        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            fusion_type=fusion_type)

        dim = self.backbone.embed_dims  # 768 for ViT-B

        # Channel-Spatial attention fusion: one per scale level.
        # Concat [rgb_f, thr_f] gives 2*dim channels → need 2*dim input.
        self.cs_fusions = nn.ModuleList(
            [ChannelSpatialAttention(2 * dim) for _ in range(4)]
        )

        print_log(
            f'Mask2FormerRGBTCrossAttn: backbone CrossAttn + '
            f'{len(self.cs_fusions)} ChannelSpatialAttention, dim={dim}',
            logger='current')

    def extract_feat(self, inputs):
        """Extract and fuse features.

        The backbone already performs cross-modal attention at each stage,
        so features are already cross-modal-aware. We just concatenate and
        apply channel-spatial attention to fuse and reduce dimensions.
        """
        # Dual-branch features (already cross-modal-aware from backbone)
        rgb_feats, thr_feats = self.backbone(inputs)

        # Fusion at each scale
        fused_feats = []

        for i in range(4):
            rgb_f = rgb_feats[i]  # [B, C, H_i, W_i]
            thr_f = thr_feats[i]  # [B, C, H_i, W_i]

            # Concatenate + Channel-Spatial attention fusion
            concat_feat = torch.cat([rgb_f, thr_f], dim=1)  # [B, 2C, H, W]
            fused = self.cs_fusions[i](concat_feat)  # [B, C, H, W]

            fused_feats.append(fused)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return fused_feats
