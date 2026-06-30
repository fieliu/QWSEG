"""Quality-aware DINOv3-Adapter + Mask2Former for RGB-T segmentation.

Architecture:
  RGBTDINOv3Adapter backbone (with quality repair hooks) ->
    ChannelSpatialAttention fusion -> M2F head

The backbone internally runs at each of 4 stages:
  1. RGB adapter interaction + T adapter interaction
  2. [INCREMENT] QualityPredictor + Quality-gated CrossAttn (BEFORE baseline)
  3. [BASELINE] CrossAttn (sense each other, propagate to next stage)
  4. Output features to next stage

The segmentor handles:
  - Setting quality modules onto the backbone
  - Concatenation + ChannelSpatialAttention fusion (after backbone)
  - Quality loss computation
  - Degradation generation
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.logging import print_log
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from .eomt_fusion_blocks import TokenQualityPredictor, CrossAttnFusion
from .degradation import DegradationGenerator
from .eomt_quality_attn import soft_keep_mask, complementary_fix_tokens
from .mask2former_rgbt_crossattn import Mask2FormerRGBTCrossAttn, ChannelSpatialAttention


@MODELS.register_module()
class DINOv3AdapterM2FQuality(Mask2FormerRGBTCrossAttn):
    """Quality-aware DINOv3-Adapter + Mask2Former segmentor.

    The backbone (RGBTDINOv3Adapter) handles all inter-stage cross-modal
    attention internally. This segmentor:
    1. Sets quality modules (predictors + quality_cross_fusions) onto the backbone
    2. The backbone then runs quality repair BEFORE baseline CrossAttn at each stage
    3. After backbone forward, concatenates + ChannelSpatialAttention fusion
    4. Computes quality supervision loss during training

    Ablation switches:
      use_quality_gate:  quality key-gating in quality_cross_fusions
      use_quality_merge: quality-weighted residual merge after quality repair
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
                 fusion_type: str = 'crossattn',
                 quality_loss_weight: float = 1.0,
                 quality_tau: float = 0.5,
                 mask_temperature: float = 0.1,
                 deg_ceiling_weight: float = 0.1,
                 deg_ceiling: float = 0.2,
                 use_quality_gate: bool = True,
                 use_quality_merge: bool = True,
                 freeze_backbone: bool = False,
                 freeze_fusion: bool = False,
                 degradation: Optional[dict] = None,
                 teacher_ckpt: Optional[str] = None):
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
        n_stages = len(self.backbone.interaction_indexes)  # 4

        # ===== INCREMENTAL modules (not in clean baseline) =====
        # These are set onto the backbone so they run INSIDE the backbone
        # forward, BEFORE the baseline CrossAttn at each stage.

        # Quality predictors
        self.quality_predictors = nn.ModuleList(
            [TokenQualityPredictor(dim) for _ in range(n_stages)]
        )

        # Quality-gated cross-attention: NEW parameters, separate from baseline
        self.quality_cross_fusions = nn.ModuleList(
            [CrossAttnFusion(dim, num_heads=8) for _ in range(n_stages)]
        )

        # Set quality modules onto the backbone
        self.backbone.quality_predictors = self.quality_predictors
        self.backbone.quality_cross_fusions = self.quality_cross_fusions

        # Quality params
        self.quality_loss_weight = quality_loss_weight
        self.quality_tau = quality_tau
        self.mask_temperature = mask_temperature
        self.deg_ceiling_weight = deg_ceiling_weight
        self.deg_ceiling = deg_ceiling

        # Sub-component switches
        self.use_quality_gate = use_quality_gate
        self.use_quality_merge = use_quality_merge
        self.use_quality = use_quality_gate or use_quality_merge

        # Propagate switches to backbone
        self.backbone.use_quality = self.use_quality
        self.backbone.use_quality_gate = self.use_quality_gate
        self.backbone.use_quality_merge = self.use_quality_merge
        self.backbone.quality_tau = self.quality_tau
        self.backbone.mask_temperature = self.mask_temperature

        # Degradation generator
        deg_cfg = degradation or {}
        self.degrader = DegradationGenerator(**deg_cfg) if deg_cfg else None

        # Warm-start from clean baseline checkpoint
        self._warm_start(teacher_ckpt)

        # Freeze options
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_fusion:
            self._freeze_fusion()

        # State for quality loss / visualization
        self._last_quality = None
        self._last_deg_masks = None

        # Visualization state
        self._vis_rgb_feats = None
        self._vis_thr_feats = None
        self._vis_fused_feats = None
        self._vis_quality_rgb = None
        self._vis_quality_thr = None

    def _warm_start(self, ckpt_path):
        """Load weights from a clean baseline (Mask2FormerRGBTCrossAttn) checkpoint."""
        if ckpt_path is None:
            return
        from mmengine.runner import CheckpointLoader
        ckpt = CheckpointLoader.load_checkpoint(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        student_only = [k for k in missing if not k.startswith('teacher.')]
        loaded = len(state_dict) - len(unexpected)
        print_log(
            f'DINOv3AdapterM2FQuality warm-start: '
            f'~{loaded} params loaded; '
            f'{len(student_only)} new quality params kept own init; '
            f'{len(unexpected)} unexpected.', logger='current')

    def _freeze_backbone(self):
        """Freeze ViT backbone + adapter modules (but not quality modules)."""
        n = 0
        for name, p in self.backbone.named_parameters():
            # Don't freeze quality modules (they belong to the segmentor)
            if 'quality_predictors' in name or 'quality_cross_fusions' in name:
                continue
            p.requires_grad = False
            n += 1
        print_log(
            f'DINOv3AdapterM2FQuality: freeze_backbone=True -> '
            f'froze {n} backbone param tensors; quality/fusion/head trainable.',
            logger='current')

    def _freeze_fusion(self):
        """Freeze baseline fusion modules (backbone.cross_fusions + cs_fusions)."""
        n = 0
        # Baseline cross_fusions are in the backbone
        for p in self.backbone.cross_fusions.parameters():
            p.requires_grad = False
            n += 1
        # ChannelSpatialAttention fusion is in the segmentor
        for p in self.cs_fusions.parameters():
            p.requires_grad = False
            n += 1
        print_log(
            f'DINOv3AdapterM2FQuality: freeze_fusion=True -> '
            f'froze {n} baseline fusion param tensors; '
            f'quality_cross_fusions + quality_predictors + head trainable.',
            logger='current')

    def _apply_degradation(self, inputs):
        """Apply on-the-fly degradation via DegradationGenerator."""
        if self.degrader is None or not self.training:
            return inputs, None, None

        rgb = inputs[:, :3, :, :]
        ir = inputs[:, 3:6, :, :]

        drgb, dir_, mask_rgb, mask_ir = self.degrader(
            rgb, ir, epoch=getattr(self, 'current_epoch', 0))

        degraded = torch.cat([drgb, dir_], dim=1)
        return degraded, mask_rgb, mask_ir

    def extract_feat(self, inputs):
        """Extract features with quality repair (in backbone) + fusion.

        Full flow:
          1. [Optional] Apply degradation during training
          2. Backbone forward (includes quality repair + baseline CrossAttn per stage)
             -> (rgb_feats, thr_feats) at 4 scales
          3. Concat + ChannelSpatialAttention -> fused features
          4. Neck (if any) -> Mask2FormerHead
        """
        # Optional on-the-fly degradation during training
        if self.training and self.degrader is not None:
            inputs, mask_rgb, mask_ir = self._apply_degradation(inputs)
        else:
            mask_rgb, mask_ir = None, None

        # Ensure backbone switches are up-to-date
        self.backbone.use_quality = self.use_quality
        self.backbone.use_quality_gate = self.use_quality_gate
        self.backbone.use_quality_merge = self.use_quality_merge
        self.backbone.quality_tau = self.quality_tau
        self.backbone.mask_temperature = self.mask_temperature

        # Dual-branch features (backbone does quality repair + CrossAttn internally)
        rgb_feats, thr_feats = self.backbone(inputs)

        # Get quality predictions from backbone
        self._last_quality = getattr(self.backbone, '_all_quality', [None] * 4)
        self._last_deg_masks = (mask_rgb, mask_ir)

        # Fusion at each scale: concat + ChannelSpatialAttention
        fused_feats = []
        for i in range(4):
            rgb_f = rgb_feats[i]  # [B, C, H_i, W_i]
            thr_f = thr_feats[i]  # [B, C, H_i, W_i]

            concat_feat = torch.cat([rgb_f, thr_f], dim=1)  # [B, 2C, H, W]
            fused = self.cs_fusions[i](concat_feat)  # [B, C, H, W]
            fused_feats.append(fused)

        # Visualization features (eval mode only)
        if not self.training:
            self._vis_rgb_feats = [f.detach() for f in rgb_feats]
            self._vis_thr_feats = [f.detach() for f in thr_feats]
            self._vis_fused_feats = [f.detach() for f in fused_feats]
            if self.use_quality and self._last_quality[-1] is not None:
                self._vis_quality_rgb = self._last_quality[-1][0].detach()
                self._vis_quality_thr = self._last_quality[-1][1].detach()
            else:
                self._vis_quality_rgb = None
                self._vis_quality_thr = None

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return fused_feats

    def loss(self, inputs, data_samples):
        """Compute segmentation + quality losses."""
        losses = super().loss(inputs, data_samples)

        # Quality supervision loss
        if self.use_quality and self.quality_loss_weight > 0 and self._last_quality:
            q_loss = self._compute_quality_loss()
            if q_loss is not None:
                losses['loss_quality'] = q_loss * self.quality_loss_weight

        return losses

    def _compute_quality_loss(self):
        """Quality supervision: degraded tokens -> target 0, clean -> 1 (BCE)."""
        mask_rgb, mask_ir = getattr(self, '_last_deg_masks', (None, None))
        if mask_rgb is None and mask_ir is None:
            return None

        total_loss = 0.0
        n_valid = 0

        for q in self._last_quality:
            if q is None:
                continue
            s_rgb, s_thr = q  # [B, N, 1]
            N = s_rgb.shape[1]
            B = s_rgb.shape[0]
            H_t = W_t = int(math.sqrt(N))
            if H_t * W_t != N:
                continue

            # Downsample masks to token grid
            if mask_rgb is not None:
                m_rgb_small = F.interpolate(
                    mask_rgb.float(), size=(H_t, W_t),
                    mode='bilinear', align_corners=False)
            else:
                m_rgb_small = torch.zeros(B, 1, H_t, W_t, device=s_rgb.device)
            if mask_ir is not None:
                m_ir_small = F.interpolate(
                    mask_ir.float(), size=(H_t, W_t),
                    mode='bilinear', align_corners=False)
            else:
                m_ir_small = torch.zeros(B, 1, H_t, W_t, device=s_rgb.device)

            target_rgb = 1.0 - m_rgb_small.view(B, -1)
            target_thr = 1.0 - m_ir_small.view(B, -1)

            pred_rgb = s_rgb.squeeze(-1)
            pred_thr = s_thr.squeeze(-1)

            loss_rgb = F.binary_cross_entropy(pred_rgb, target_rgb, reduction='mean')
            loss_thr = F.binary_cross_entropy(pred_thr, target_thr, reduction='mean')

            # deg_ceiling: push degraded tokens below deg_ceiling
            if self.deg_ceiling_weight > 0:
                deg_rgb_tokens = m_rgb_small.view(B, -1) > 0.5
                deg_thr_tokens = m_ir_small.view(B, -1) > 0.5
                ceiling_loss = torch.tensor(0.0, device=s_rgb.device)
                if deg_rgb_tokens.any():
                    ceiling_loss = ceiling_loss + F.relu(
                        pred_rgb[deg_rgb_tokens] - self.deg_ceiling).mean()
                if deg_thr_tokens.any():
                    ceiling_loss = ceiling_loss + F.relu(
                        pred_thr[deg_thr_tokens] - self.deg_ceiling).mean()
                total_loss += ceiling_loss * self.deg_ceiling_weight

            total_loss += loss_rgb + loss_thr
            n_valid += 1

        if n_valid == 0:
            return None
        return total_loss / n_valid

    def train(self, mode=True):
        super().train(mode)
        return self
