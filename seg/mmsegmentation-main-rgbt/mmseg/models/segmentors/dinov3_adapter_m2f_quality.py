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
                 rank_margin: float = 0.20,
                 deg_ceiling_weight: float = 0.1,
                 deg_ceiling: float = 0.2,
                 clean_floor_weight: float = 0.1,
                 clean_floor: float = 0.9,
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
        self.rank_margin = rank_margin
        self.deg_ceiling_weight = deg_ceiling_weight
        self.deg_ceiling = deg_ceiling
        self.clean_floor_weight = clean_floor_weight
        self.clean_floor = clean_floor

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
        # If quality is disabled (F0), also freeze quality modules to prevent drift
        if not self.use_quality:
            self._freeze_quality_modules()

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

    def _freeze_quality_modules(self):
        """Freeze quality modules when quality is disabled (F0 ablation)."""
        n = 0
        for p in self.backbone.quality_predictors.parameters():
            p.requires_grad = False
            n += 1
        for p in self.backbone.quality_cross_fusions.parameters():
            p.requires_grad = False
            n += 1
        print_log(
            f'DINOv3AdapterM2FQuality: quality disabled -> '
            f'froze {n} quality module param tensors.',
            logger='current')

    def extract_feat(self, inputs):
        """Extract features with quality repair (in backbone) + fusion.

        Full flow:
          1. Backbone forward (includes quality repair + baseline CrossAttn per stage)
             -> (rgb_feats, thr_feats) at 4 scales
          2. Concat + ChannelSpatialAttention -> fused features
          3. Neck (if any) -> Mask2FormerHead

        Note: degradation is applied in loss() via make_paired (paired path),
        not here. This method is called by both loss (with degraded 2B input)
        and predict/_forward (clean input).
        """
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

    # ---- paired light/heavy degradation + rank-only quality loss ----
    def loss(self, inputs, data_samples):
        """Compute segmentation + quality losses via paired degradation path.

        Aligns with EoMT's _loss_paired:
          1. make_paired -> light/heavy versions + level masks (0=clean, 1-5=degraded)
          2. cat light(0:B) + heavy(B:2B) -> one forward pass
          3. segmentation loss on BOTH versions (targets duplicated)
          4. quality supervision: rank + clean_floor + deg_ceiling (not BCE)
        """
        if not self.training or self.degrader is None:
            # eval / no degrader: standard single-forward loss
            return super().loss(inputs, data_samples)
        # F0 (use_quality=False) also uses paired degradation for fair
        # comparison with F1. With quality_loss_weight=0, no quality
        # supervision is applied; only segmentation loss on degraded data.
        # This ensures F1-F0 = pure quality mechanism increment (same
        # degradation, different modules). Aligns with EoMT's loss().

        rgb = inputs[:, :3, :, :]
        ir = inputs[:, 3:6, :, :]
        epoch = getattr(self, 'current_epoch', 0)
        mean = self.data_preprocessor.mean.flatten()
        std = self.data_preprocessor.std.flatten()
        (l_rgb, l_ir, h_rgb, h_ir,
         lvl_light_rgb, lvl_light_ir,
         lvl_heavy_rgb, lvl_heavy_ir,
         rank_mask, level_gap) = \
            self.degrader.make_paired(rgb, ir, mean, std, epoch=epoch)
        B = rgb.shape[0]

        # batch-cat light(0:B) + heavy(B:2B), one forward
        cat_inputs = torch.cat([
            torch.cat([l_rgb, l_ir], dim=1),   # [B, 6, H, W] light
            torch.cat([h_rgb, h_ir], dim=1),   # [B, 6, H, W] heavy
        ], dim=0)                               # [2B, 6, H, W]

        fused_feats = self.extract_feat(cat_inputs)
        quality_info = self._last_quality  # list of (s_rgb, s_t), each [2B,N,1]

        # segmentation loss on BOTH versions (targets duplicated light+heavy)
        data_samples2 = list(data_samples) + list(data_samples)
        losses = dict()
        loss_decode = self.decode_head.loss(
            fused_feats, data_samples2, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples2)
            losses.update(loss_aux)

        # quality supervision using LEVEL masks (L0=clean, L1-L5=degraded)
        if self.quality_loss_weight > 0 and quality_info:
            grid = getattr(self.backbone, '_token_grid', None)
            if grid is not None:
                lvl_light_rgb_tok = self._mask_to_token(lvl_light_rgb, grid)
                lvl_light_t_tok = self._mask_to_token(lvl_light_ir, grid)
                lvl_heavy_rgb_tok = self._mask_to_token(lvl_heavy_rgb, grid)
                lvl_heavy_t_tok = self._mask_to_token(lvl_heavy_ir, grid)

                losses['loss_quality'] = self.quality_loss_weight * \
                    self._rank_loss(
                        quality_info,
                        lvl_light_rgb_tok, lvl_light_t_tok,
                        lvl_heavy_rgb_tok, lvl_heavy_t_tok,
                        B, margin=self.rank_margin,
                        rank_mask=rank_mask, level_gap=level_gap)

                if self.clean_floor_weight > 0:
                    losses['loss_clean_floor'] = self.clean_floor_weight * \
                        self._clean_floor_loss(
                            quality_info,
                            (lvl_light_rgb_tok, lvl_light_t_tok),
                            (lvl_heavy_rgb_tok, lvl_heavy_t_tok), B)

                if self.deg_ceiling_weight > 0:
                    losses['loss_deg_ceiling'] = self.deg_ceiling_weight * \
                        self._deg_ceiling_loss(
                            quality_info,
                            (lvl_light_rgb_tok, lvl_light_t_tok),
                            (lvl_heavy_rgb_tok, lvl_heavy_t_tok), B)

        return losses

    def _mask_to_token(self, mask, grid):
        """Downsample a [B,1,H,W] level mask to the patch-token grid and
        flatten to [B, N]. Uses max-pool (conservative: take the HIGHEST level
        in the receptive field, so severe degradation is never missed).
        Aligns with EoMT's _mask_to_token."""
        gh, gw = grid
        m = F.adaptive_max_pool2d(mask.float(), (gh, gw))  # [B,1,gh,gw]
        return m.flatten(2).squeeze(1).long()              # [B, N]

    def _rank_loss(self, quality_info, lvl_light_rgb_tok, lvl_light_t_tok,
                   lvl_heavy_rgb_tok, lvl_heavy_t_tok, B,
                   margin=0.20, rank_mask=None, level_gap=None):
        """Rank quality loss using LEVEL masks (0=clean, 1-5=degraded).

        For each token in the degraded region, push s_light > s_heavy by a
        FIXED margin: L = mean max(0, margin - (s_light - s_heavy)).

        Aligns with EoMT's _rank_loss.
        """
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            if s_rgb is None:
                continue
            for s, lvl_l, lvl_h in (
                    (s_rgb, lvl_light_rgb_tok, lvl_heavy_rgb_tok),
                    (s_t, lvl_light_t_tok, lvl_heavy_t_tok)):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_l.shape[1]:
                    s = s[:, s.shape[1] - lvl_l.shape[1]:]
                s_light, s_heavy = s[:B], s[B:]         # [B, N] each
                gap = s_light - s_heavy                 # want > margin
                hinge = (margin - gap).clamp_min(0.0)   # [B, N]
                reg = ((lvl_h > 0) & (lvl_l < lvl_h)).float()  # [B, N]
                if rank_mask is not None:
                    hinge = hinge * rank_mask.view(B, 1)
                    reg = reg * rank_mask.view(B, 1)
                denom = reg.sum().clamp_min(1.0)
                total = total + (hinge * reg).sum() / denom
                z += 1
        return total / max(z, 1)

    def _clean_floor_loss(self, quality_info, lvl_light_tok, lvl_heavy_tok, B):
        """Anti-collapse soft floor on CLEAN tokens (level == 0).
        Soft lower bound relu(floor - s): only penalizes scores BELOW the
        floor, leaving [floor, 1] free.

        Aligns with EoMT's _clean_floor_loss.
        """
        floor = self.clean_floor
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            if s_rgb is None:
                continue
            for s, lvl_l, lvl_h in (
                    (s_rgb, lvl_light_tok[0], lvl_heavy_tok[0]),
                    (s_t, lvl_light_tok[1], lvl_heavy_tok[1])):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_l.shape[1]:
                    s = s[:, s.shape[1] - lvl_l.shape[1]:]
                lvl_all = torch.cat([lvl_l, lvl_h], dim=0)  # [2B, N]
                clean = (lvl_all == 0).float()              # [2B, N]
                below = (floor - s).clamp_min(0.0)          # penalize s<floor
                denom = clean.sum().clamp_min(1.0)
                total = total + (below * clean).sum() / denom
                z += 1
        return total / max(z, 1)

    def _deg_ceiling_loss(self, quality_info, lvl_light_tok, lvl_heavy_tok, B):
        """Lower-end anchor: on the HEAVY version's DEGRADED tokens (level >=
        1), push s_heavy BELOW deg_ceiling. The ceiling scales with level:
            ceiling(lvl) = deg_ceiling * (6 - lvl) / 5  # L1->base, L5->0.2*base

        Aligns with EoMT's _deg_ceiling_loss.
        """
        base_ceiling = self.deg_ceiling
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            if s_rgb is None:
                continue
            for s, lvl_h in (
                    (s_rgb, lvl_heavy_tok[0]),
                    (s_t, lvl_heavy_tok[1])):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_h.shape[1]:
                    s = s[:, s.shape[1] - lvl_h.shape[1]:]
                s_heavy = s[B:]                         # [B, N] heavy version
                lvl_f = lvl_h.float()
                ceiling = base_ceiling * (6.0 - lvl_f) / 5.0  # [B, N]
                above = (s_heavy - ceiling).clamp_min(0.0)
                reg = (lvl_h > 0).float()
                denom = reg.sum().clamp_min(1.0)
                total = total + (above * reg).sum() / denom
                z += 1
        return total / max(z, 1)

    def train(self, mode=True):
        super().train(mode)
        return self
