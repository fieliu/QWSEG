"""Quality-Gated Swin with Mask2Former Decoder.

Same three-branch architecture as QualityGatedMiTMamba, but with Swin
backbones instead of MiT.  Quality masks are converted to per-window
attention bias (cyclic-shift aligned for SW-MSA blocks).
Mask2Former decoder replaces SegformerHead.

Three-phase training curriculum:
  Phase 1 (epoch 0 ~ phase1_epochs):   Clean-only warmup. QP frozen,
        force_all_keep=True, no degradation branch, no distill/invariant loss.
  Phase 2 (phase1_epochs ~ phase1+phase2): Robustness warmup. QP frozen,
        force_all_keep=True, mild local+global degradation (no missing),
        distill + invariant loss activated (uniform quality weights).
  Phase 3 (phase1+phase2 ~ end):        Quality-aware. QP unfrozen,
        force_all_keep=False, missing degradation introduced progressively,
        distill + invariant loss with quality-weighted gating,
        retention loss activated.
"""

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.v9_utils import (
    QualityPredictor,
    f_attn,
    f_fuse,
    ste_hard_mask,
    cascade_quality_suppress,
    downsample_mask,
    get_missing_schedule,
    _apply_missing_degradation,
    _apply_local_missing,
    _generate_single_rect_mask,
    compute_cross_modal_contrastive_loss,
    quality_guided_loss,
)
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         SampleList, add_prefix)
from mmseg.models.backbones.swin import (
    ShiftWindowMSA, SwinBlock, SwinBlockSequence, WindowMSA,
)
from mmengine.utils import to_2tuple


# ---------------------------------------------------------------------------
# Quality-aware Swin components: inject quality_bias into attention
# ---------------------------------------------------------------------------

class _QualityWindowMSA(WindowMSA):
    """WindowMSA with an extra quality_bias injection point.

    After relative_position_bias is added to attention scores, quality_bias
    is added before the optional mask.  This allows quality information to
    influence attention without replacing the SW-MSA window mask.
    """

    def forward(self, x, mask=None, quality_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if quality_bias is not None:
            attn = attn + quality_bias

        if mask is not None:
            if mask.ndim == 4:
                attn = attn + mask
            else:
                nW = mask.shape[0]
                attn = attn.view(B // nW, nW, self.num_heads, N,
                                 N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn.float()).to(v.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _QualityShiftWindowMSA(ShiftWindowMSA):
    """ShiftWindowMSA that forwards quality_bias to the inner WindowMSA."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        qkv_bias = kwargs.get('qkv_bias', True)
        qk_scale = kwargs.get('qk_scale', None)
        attn_drop_rate = kwargs.get('attn_drop_rate', 0)
        proj_drop_rate = kwargs.get('proj_drop_rate', 0)
        self.w_msa = _QualityWindowMSA(
            embed_dims=kwargs['embed_dims'],
            num_heads=kwargs['num_heads'],
            window_size=to_2tuple(kwargs['window_size']),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

    def forward(self, query, hw_shape, extra_attn_mask=None,
                quality_bias=None):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)

        attn_windows = self.w_msa(query_windows, mask=attn_mask,
                                  quality_bias=quality_bias)

        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x


class QualitySwinBlock(SwinBlock):
    """SwinBlock that accepts quality_bias and passes it to the MSA module.

    quality_bias is added to attention scores *after* relative position bias
    and *before* the SW-MSA window mask, so both can coexist.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        attn_kwargs = dict(
            embed_dims=kwargs['embed_dims'],
            num_heads=kwargs['num_heads'],
            window_size=kwargs['window_size'],
            shift_size=kwargs['window_size'] // 2 if kwargs.get('shift', False) else 0,
            qkv_bias=kwargs.get('qkv_bias', True),
            qk_scale=kwargs.get('qk_scale', None),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
            proj_drop_rate=kwargs.get('drop_rate', 0.),
            dropout_layer=dict(type='DropPath', drop_prob=kwargs.get('drop_path_rate', 0.)),
            init_cfg=None)
        self.attn = _QualityShiftWindowMSA(**attn_kwargs)

    def forward(self, x, hw_shape, quality_bias=None):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape, quality_bias=quality_bias)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            from torch.utils.checkpoint import checkpoint as cp
            x = cp(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


def _replace_swin_blocks_with_quality(swin_model):
    """Replace SwinBlocks with QualitySwinBlocks that support quality_bias.

    All pretrained weights (norm, ffn, attn qkv/proj/rpb, drop) are preserved
    by direct attribute assignment or data copy.
    """
    for stage in swin_model.stages:
        new_blocks = nn.ModuleList()
        for block in stage.blocks:
            old_attn = block.attn
            new_block = QualitySwinBlock(
                embed_dims=old_attn.w_msa.embed_dims,
                num_heads=old_attn.w_msa.num_heads,
                feedforward_channels=block.ffn.feedforward_channels,
                window_size=old_attn.window_size,
                shift=old_attn.shift_size > 0,
                qkv_bias=old_attn.w_msa.qkv.bias is not None,
                with_cp=block.with_cp,
            )
            new_block.norm1 = block.norm1
            new_block.norm2 = block.norm2
            new_block.ffn = block.ffn
            new_block.attn.w_msa.qkv.weight.data.copy_(
                old_attn.w_msa.qkv.weight.data)
            if old_attn.w_msa.qkv.bias is not None:
                new_block.attn.w_msa.qkv.bias.data.copy_(
                    old_attn.w_msa.qkv.bias.data)
            new_block.attn.w_msa.relative_position_bias_table.data.copy_(
                old_attn.w_msa.relative_position_bias_table.data)
            new_block.attn.w_msa.relative_position_index.copy_(
                old_attn.w_msa.relative_position_index)
            new_block.attn.w_msa.proj.weight.data.copy_(
                old_attn.w_msa.proj.weight.data)
            if old_attn.w_msa.proj.bias is not None:
                new_block.attn.w_msa.proj.bias.data.copy_(
                    old_attn.w_msa.proj.bias.data)
            new_block.attn.drop = old_attn.drop
            new_blocks.append(new_block)
        stage.blocks = new_blocks


# ---------------------------------------------------------------------------
# Quality mask -> Swin window attention bias
# ---------------------------------------------------------------------------

def _quality_score_to_swin_bias(s_2d, window_size, shift_size,
                                 tau=0.3, alpha=10.0, is_precomputed_bias=False):
    """Convert continuous quality score to Swin window attention bias.

    Applies f_attn(s) piecewise function to generate bias, then performs
    the same padding, cyclic shift, and window partitioning as the original.

    Args:
        s_2d: [B, 1, H, W] continuous quality score in [0, 1]
        window_size: Swin window size
        shift_size: shift size for SW-MSA (0 for W-MSA)
        tau: quality threshold for f_attn
        alpha: max attention bias magnitude
        is_precomputed_bias: if True, s_2d is already f_attn output, skip f_attn
    Returns:
        quality_bias: [nW*B, 1, wh*wh, wh*wh] attention bias tensor
    """
    B, _, H, W = s_2d.shape
    if is_precomputed_bias:
        bias_map = s_2d
    else:
        bias_map = f_attn(s_2d, tau=tau, alpha=alpha)
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        bias_map = F.pad(bias_map, (0, pad_r, 0, pad_b), value=0.0)
    H_pad, W_pad = bias_map.shape[2], bias_map.shape[3]
    if shift_size > 0:
        bias_map = torch.roll(bias_map, shifts=(-shift_size, -shift_size), dims=(2, 3))
    wh = window_size
    nW_h, nW_w = H_pad // wh, W_pad // wh
    bias_windows = bias_map.reshape(B, nW_h, wh, nW_w, wh)
    bias_windows = bias_windows.permute(0, 1, 3, 2, 4).contiguous()
    bias_windows = bias_windows.reshape(-1, wh * wh)
    return bias_windows.view(-1, 1, 1, wh * wh).expand(-1, 1, wh * wh, wh * wh).contiguous()


# ---------------------------------------------------------------------------
# Swin quality-aware forward: common dual-modality
# ---------------------------------------------------------------------------

def _forward_swin_common_dual_pruned(swin_branch, input_rgbt, orig_B,
                                      predictors_rgb, predictors_t,
                                      training=True,
                                      tau=0.3, alpha=10.0):
    window_size = swin_branch.stages[0].blocks[0].attn.window_size
    outs, all_s_rgb, all_s_t = [], [], []
    stages = swin_branch.stages
    x, (H, W) = swin_branch.patch_embed(input_rgbt)
    if swin_branch.use_abs_pos_embed:
        x = x + swin_branch.absolute_pos_embed
    x = swin_branch.drop_after_pos(x)
    B_tok = x.shape[0]
    cum_rgb, cum_t = None, None

    for i, stage in enumerate(stages):
        blocks, downsample = stage.blocks, stage.downsample

        s_rgb = s_t = None
        if i < len(predictors_rgb) and predictors_rgb[i] is not None:
            x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
            s_rgb = predictors_rgb[i](x_2d[:orig_B],
                                       torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype))
            s_t = predictors_t[i](x_2d[orig_B:],
                                   torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype))
        if s_rgb is None:
            s_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if s_t is None:
            s_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        s_rgb, cum_rgb = cascade_quality_suppress(s_rgb, cum_rgb, H, W, clamp_min=0.1)
        s_t, cum_t = cascade_quality_suppress(s_t, cum_t, H, W, clamp_min=0.1)

        all_s_rgb.append(s_rgb)
        all_s_t.append(s_t)

        both_bad = (s_rgb < tau) & (s_t < tau)
        if both_bad.any():
            winner_r = s_rgb >= s_t
            hard_r = torch.where(both_bad & winner_r, torch.ones_like(s_rgb), s_rgb)
            hard_t = torch.where(both_bad & ~winner_r, torch.ones_like(s_t), s_t)
            s_rgb_attn = hard_r + s_rgb - s_rgb.detach()
            s_t_attn = hard_t + s_t - s_t.detach()
        else:
            s_rgb_attn = s_rgb
            s_t_attn = s_t

        bias_rgb = f_attn(s_rgb_attn, tau=tau, alpha=alpha)
        bias_t = f_attn(s_t_attn, tau=tau, alpha=alpha)
        bias_combined = torch.cat([bias_rgb, bias_t], dim=0)

        for j, block in enumerate(blocks):
            shift_size = block.attn.shift_size
            quality_bias = _quality_score_to_swin_bias(
                bias_combined, window_size, shift_size, tau=tau, alpha=alpha,
                is_precomputed_bias=True)
            x = block(x, (H, W), quality_bias=quality_bias)

        if i in swin_branch.out_indices:
            norm_layer = getattr(swin_branch, f'norm{i}', None)
            out = norm_layer(x) if norm_layer is not None else x
            stage_out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(stage_out)

        if downsample is not None:
            x, (H, W) = downsample(x, (H, W))

    return outs, all_s_rgb, all_s_t


# ---------------------------------------------------------------------------
# Swin quality-aware forward: single branch
# ---------------------------------------------------------------------------

def _forward_swin_branch_pruned(swin_branch, img, predictor_list,
                                 training=True,
                                 tau=0.3, alpha=10.0):
    """Quality-aware forward through a single Swin branch.

    Uses continuous quality score with piecewise modulation.
    Cross-stage cascading suppression ensures low-quality regions
    detected in shallow stages are never "recovered" by deeper stages.

    Returns:
        outs:        list of feature maps [B, C_i, H_i, W_i]
        all_s:       list of per-stage quality scores
    """
    window_size = swin_branch.stages[0].blocks[0].attn.window_size
    outs, all_s = [], []
    stages = swin_branch.stages
    x, (H, W) = swin_branch.patch_embed(img)
    if swin_branch.use_abs_pos_embed:
        x = x + swin_branch.absolute_pos_embed
    x = swin_branch.drop_after_pos(x)
    B_tok = x.shape[0]
    cum_s = None

    for i, stage in enumerate(stages):
        blocks, downsample = stage.blocks, stage.downsample
        s = None

        if i < len(predictor_list) and predictor_list[i] is not None:
            x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
            s = predictor_list[i](x_2d,
                                   torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype))

        if s is None:
            s = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        s, cum_s = cascade_quality_suppress(s, cum_s, H, W, clamp_min=0.1)

        all_s.append(s)

        for j, block in enumerate(blocks):
            shift_size = block.attn.shift_size
            quality_bias = _quality_score_to_swin_bias(
                s, window_size, shift_size, tau=tau, alpha=alpha)
            x = block(x, (H, W), quality_bias=quality_bias)

        if i in swin_branch.out_indices:
            norm_layer = getattr(swin_branch, f'norm{i}', None)
            out = norm_layer(x) if norm_layer is not None else x
            stage_out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(stage_out)

        if downsample is not None:
            x, (H, W) = downsample(x, (H, W))

    return outs, all_s


# ===================================================================
# Swin V9 Model  (3-branch, same structure as MiT V9)
# ===================================================================

@MODELS.register_module()
class QualityGatedSwinMask2Former(BaseSegmentor):
    """Quality-gated Swin with Mask2Former decoder.

    Three-branch architecture:
      - Common:  one Swin backbone processing RGB+T concatenated -> zc_rgb, zc_t
      - Private: two Swin branches (RGB, T) -> zp_rgb, zp_t
      - QualityPredictor x 16 (4 stages x 4 predictor sets)
      - Mask2FormerHead main decoder
      - SegformerHead auxiliary decoders
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 decode_head: ConfigType,
                 common_decode_head: OptConfigType = None,
                 rgb_private_decode_head: OptConfigType = None,
                 t_private_decode_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 phase1_epochs: int = 10,
                 phase2_epochs: int = 20,
                 total_epochs: int = 200,
                 phase_mode: str = 'absolute',
                 loss_align_weight: float = 0.1,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_distill_weight: float = 0.3,
                 distill_temperature: float = 4.0,
                 aux_loss_weight: float = 0.3,
                 loss_invariant_weight: float = 0.03,
                 loss_q_guide_weight: float = 0.1,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 tau: float = 0.3,
                 alpha: float = 10.0,
                 fuse_epsilon: float = 1e-6,
                 fuse_beta: float = 3.0,
                 skip_phases: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            backbone['pretrained'] = pretrained
        self.backbone = MODELS.build(backbone)
        _replace_swin_blocks_with_quality(self.backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        _replace_swin_blocks_with_quality(self.private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        _replace_swin_blocks_with_quality(self.private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg

        embed_dims = backbone.get('embed_dims', 128)
        depths = backbone.get('depths', [2, 2, 6, 2])
        self.embed_dims_list = [embed_dims * (2 ** i) for i in range(len(depths))]

        self.tau = tau
        self.alpha = alpha
        self.fuse_epsilon = fuse_epsilon
        self.fuse_beta = fuse_beta
        self._build_predictors()
        self.phase1_epochs, self.phase2_epochs = phase1_epochs, phase2_epochs
        self.total_epochs = total_epochs
        self.phase_mode = phase_mode
        self.loss_align_weight = loss_align_weight
        self.contrast_tau, self.contrast_num_samples = contrast_tau, contrast_num_samples
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_q_guide_weight = loss_q_guide_weight
        self.skip_phases = skip_phases

    def _build_predictors(self):
        self.predictors_common_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_common_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_priv_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_priv_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])

    # ---- BaseSegmentor overrides ----

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_aux_heads(self, common_cfg, rgb_cfg, t_cfg, aux_cfg):
        self.common_decode_head = MODELS.build(common_cfg) if common_cfg else None
        self.rgb_private_decode_head = MODELS.build(rgb_cfg) if rgb_cfg else None
        self.t_private_decode_head = MODELS.build(t_cfg) if t_cfg else None
        if aux_cfg:
            self.auxiliary_head = MODELS.build(aux_cfg) if not isinstance(aux_cfg, list) \
                else nn.ModuleList([MODELS.build(h) for h in aux_cfg])

    @staticmethod
    def _stack_batch_gt(data_samples):
        return torch.stack([ds.gt_sem_seg.data for ds in data_samples], dim=0)

    @staticmethod
    def _build_pad_mask(data_samples, h, w, device):
        valid = torch.zeros(len(data_samples), h, w, dtype=torch.bool, device=device)
        for i, ds in enumerate(data_samples):
            ps = ds.metainfo.get('padding_size', [0, 0, 0, 0])
            pl, pr, pt, pb = ps
            valid[i, pt:h - pb, pl:w - pr] = True
        return valid

    def _get_training_phase(self, epoch):
        if self.skip_phases:
            return 3
        if self.phase_mode == 'ratio':
            r = min(epoch / max(self.total_epochs, 1), 1.0)
            if r < 0.075: return 1
            elif r < 0.15: return 2
            return 3
        if epoch < self.phase1_epochs: return 1
        elif epoch < self.phase1_epochs + self.phase2_epochs: return 2
        return 3

    def _update_training_phase(self, epoch):
        return self._get_training_phase(epoch)

    @property
    def force_all_keep(self):
        ph = self._get_training_phase(getattr(self, 'current_epoch', 0))
        return ph < 2

    def _get_seg_logits(self, features, data_samples=None):
        if data_samples is not None:
            batch_data_samples = data_samples
        else:
            B = features[0].shape[0] if isinstance(features, (list, tuple)) else features.shape[0]
            batch_data_samples = [
                SegDataSample(metainfo=dict(
                    img_shape=features[0].shape[2:],
                    pad_shape=features[0].shape[2:],
                )) for _ in range(B)]
        all_cls_scores, all_mask_preds = self.decode_head(features, batch_data_samples)
        mask_cls_results = all_cls_scores[-1].float()
        mask_pred_results = all_mask_preds[-1].float()
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg: super().init_weights()

    def _extract_feat_single(self, rgb, t):
        B = rgb.shape[0]

        zc_outs, s_r, s_t = _forward_swin_common_dual_pruned(
            self.backbone, torch.cat([rgb, t], dim=0), orig_B=B,
            predictors_rgb=self.predictors_common_rgb,
            predictors_t=self.predictors_common_t,
            training=self.training,
            tau=self.tau, alpha=self.alpha)
        zc_r = [f[:B] for f in zc_outs]; zc_t = [f[B:] for f in zc_outs]

        zp_r, spr = _forward_swin_branch_pruned(
            self.private_branch_rgb, rgb, self.predictors_priv_rgb,
            training=self.training,
            tau=self.tau, alpha=self.alpha)

        zp_t, spt = _forward_swin_branch_pruned(
            self.private_branch_t, t, self.predictors_priv_t,
            training=self.training,
            tau=self.tau, alpha=self.alpha)

        zf, re, te, ff = [], [], [], []
        for i in range(len(self.embed_dims_list)):
            zc_ri = zc_r[i]; zc_ti = zc_t[i]; zp_ri = zp_r[i]; zp_ti = zp_t[i]
            dev = zc_ri.device
            B1 = B
            sri = s_r[i] if s_r[i] is not None else torch.ones(B1,1,zc_ri.shape[2],zc_ri.shape[3],device=dev)
            sti = s_t[i] if s_t[i] is not None else torch.ones(B1,1,zc_ti.shape[2],zc_ti.shape[3],device=dev)
            spr_i = spr[i] if spr[i] is not None else torch.ones(B1,1,zc_ri.shape[2],zc_ri.shape[3],device=dev)
            spt_i = spt[i] if spt[i] is not None else torch.ones(B1,1,zc_ti.shape[2],zc_ti.shape[3],device=dev)

            if self.force_all_keep:
                mask_r = torch.ones(B1, 1, zc_ri.shape[2], zc_ri.shape[3], device=dev)
                mask_t = torch.ones(B1, 1, zc_ti.shape[2], zc_ti.shape[3], device=dev)
                mask_pr = torch.ones(B1, 1, zc_ri.shape[2], zc_ri.shape[3], device=dev)
                mask_pt = torch.ones(B1, 1, zc_ti.shape[2], zc_ti.shape[3], device=dev)
            else:
                mask_r  = ste_hard_mask(sri,  tau=self.tau)
                mask_t  = ste_hard_mask(sti,  tau=self.tau)
                mask_pr = ste_hard_mask(spr_i, tau=self.tau)
                mask_pt = ste_hard_mask(spt_i, tau=self.tau)

                both_bad = (mask_r + mask_t) < 0.5
                keep_r = (sri >= sti).float()
                keep_t = (sti > sri).float()
                fix_r = (both_bad.float() * keep_r + (1 - both_bad.float()) * mask_r)
                fix_t = (both_bad.float() * keep_t + (1 - both_bad.float()) * mask_t)
                mask_r = fix_r + sri - sri.detach()
                mask_t = fix_t + sti - sti.detach()

            # Common base: equal-weight average
            zf_i = (zc_ri * mask_r + zc_ti * mask_t) / (mask_r + mask_t + 1e-6)
            zf_i = zf_i.permute(0,2,3,1).contiguous()
            zf_i = F.layer_norm(zf_i, [zf_i.size(-1)])
            zf_i = zf_i.permute(0,3,1,2).contiguous()
            zf.append(zf_i)

            # Private: STE hard-zero (no safeguard needed — zf always survives)
            priv_r_i = zp_ri * mask_pr
            priv_t_i = zp_ti * mask_pt

            # Aux head: zf + private, equal-weight by active count
            re_i = (zc_ri * mask_r + zc_ti * mask_t + zp_ri * mask_pr) / (mask_r + mask_t + mask_pr + 1e-6)
            re_i = re_i.permute(0,2,3,1).contiguous()
            re_i = F.layer_norm(re_i, [re_i.size(-1)])
            re_i = re_i.permute(0,3,1,2).contiguous()
            re.append(re_i)

            te_i = (zc_ri * mask_r + zc_ti * mask_t + zp_ti * mask_pt) / (mask_r + mask_t + mask_pt + 1e-6)
            te_i = te_i.permute(0,2,3,1).contiguous()
            te_i = F.layer_norm(te_i, [te_i.size(-1)])
            te_i = te_i.permute(0,3,1,2).contiguous()
            te.append(te_i)

            # Final fusion: all four sources equal-weight by active count
            n_total = mask_r + mask_t + mask_pr + mask_pt
            ff_i = (zc_ri * mask_r + zc_ti * mask_t + zp_ri * mask_pr + zp_ti * mask_pt) / (n_total + 1e-6)
            ff_i = ff_i.permute(0,2,3,1).contiguous()
            ff_i = F.layer_norm(ff_i, [ff_i.size(-1)])
            ff_i = ff_i.permute(0,3,1,2).contiguous()
            ff.append(ff_i)

        all_s = [s for sl in [s_r, s_t, spr, spt] for s in sl]
        return zc_r, zc_t, zp_r, zp_t, zf, re, te, ff, s_r, s_t, all_s, spr, spt

    def _train_with_degradation(self, rgb, ir):
        dr, di, _, _, deg_level_rgb, deg_level_t = self._generate_degraded_inputs(rgb, ir)
        feats = self._extract_feat_single(dr, di)
        return feats + (deg_level_rgb, deg_level_t)

    def _generate_degraded_inputs(self, rgb, ir):
        B, C, H, W = rgb.shape
        dev = rgb.device
        rm = self.data_preprocessor.mean[:3].to(dev); rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev); iss = self.data_preprocessor.std[3:].to(dev)
        dr, di = rgb.clone(), ir.clone()
        dtr, dtt = ['none']*B, ['none']*B
        deg_level_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        deg_level_t = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        ep = getattr(self,'current_epoch',0)
        sched = get_missing_schedule(min(ep/max(self.total_epochs,1),1.0))
        for b in range(B):
            modality = 'rgb' if torch.rand(1, device=dev).item() < 0.5 else 'thermal'
            r = torch.rand(1, device=dev).item()
            if r < sched['p_global']:
                if modality == 'rgb':
                    dr[b:b+1] = _apply_missing_degradation(rgb[b:b+1], rm, rs)
                    dtr[b] = 'global_missing'
                    deg_level_rgb[b:b+1] = 5
                else:
                    di[b:b+1] = _apply_missing_degradation(ir[b:b+1], im, iss)
                    dtt[b] = 'global_missing'
                    deg_level_t[b:b+1] = 5
            elif r < sched['p_global'] + sched['p_local']:
                area = sched['local_area']
                if area <= 0:
                    continue
                lm = _generate_single_rect_mask(1, H, W, area, device=dev)
                if modality == 'rgb':
                    dr[b:b+1] = _apply_local_missing(rgb[b:b+1], rm, rs, lm)
                    dtr[b] = f'local_missing_{area:.0%}'
                    deg_level_rgb[b:b+1] = (lm > 0).long() * 5
                else:
                    di[b:b+1] = _apply_local_missing(ir[b:b+1], im, iss, lm)
                    dtt[b] = f'local_missing_{area:.0%}'
                    deg_level_t[b:b+1] = (lm > 0).long() * 5
        return dr.to(rgb.dtype), di.to(ir.dtype), dtr, dtt, deg_level_rgb, deg_level_t

    # ---- loss / predict / inference ----

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:,:3], inputs[:,3:]
        B = rgb.shape[0]
        ep = getattr(self,'current_epoch',0)
        ph = self._get_training_phase(ep)
        self._update_training_phase(ep)
        (zc_r,zc_t,zp_r,zp_t,zf,re,te,ff,s_r,s_t,all_s,spr,spt) = self._extract_feat_single(rgb,ir)
        losses = {}
        sl = self._stack_batch_gt(data_samples)
        for ds in data_samples:
            if hasattr(ds,'gt_sem_seg'): ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)
        losses.update(add_prefix(self.decode_head.loss(ff,data_samples,self.train_cfg),'decode'))
        if self.common_decode_head and zf: losses.update(add_prefix(self.common_decode_head.loss(zf,data_samples,self.train_cfg),'common_decode'))
        if self.rgb_private_decode_head and re: losses.update(add_prefix(self.rgb_private_decode_head.loss(re,data_samples,self.train_cfg),'rgb_private_decode'))
        if self.t_private_decode_head and te: losses.update(add_prefix(self.t_private_decode_head.loss(te,data_samples,self.train_cfg),'t_private_decode'))
        if self.loss_align_weight > 0:
            gt = sl.squeeze(1).long(); lc, cnt = 0., 0
            pm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            for i in range(len(zc_r)):
                if zc_r[i] is not None and zc_t[i] is not None:
                    Dr = (s_r[i] > self.tau).float().detach() if s_r[i] is not None else torch.ones(B,1,zc_r[i].shape[2],zc_r[i].shape[3],device=zc_r[i].device)
                    Dt = (s_t[i] > self.tau).float().detach() if s_t[i] is not None else torch.ones(B,1,zc_t[i].shape[2],zc_t[i].shape[3],device=zc_t[i].device)
                    sr_i = s_r[i].detach() if s_r[i] is not None else None
                    st_i = s_t[i].detach() if s_t[i] is not None else None
                    lc += compute_cross_modal_contrastive_loss(
                        zc_r[i],zc_t[i],gt,Dr,Dt,sr_i,st_i,
                        tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                        ignore_label=255, pad_mask=pm)
                    cnt += 1
            if cnt: losses['loss_align'] = (lc/cnt)*self.loss_align_weight
        if self.training:
            (dzcr,dzct,dzpr,dzpt,dzf,drl,dtl,df,ds_r,ds_t,dall_s,dspr,dspt,deg_level_rgb,deg_level_t) = self._train_with_degradation(rgb,ir)
            losses.update(add_prefix(self.decode_head.loss(df,data_samples,self.train_cfg),'deg_decode'))
            for head, feats, pfx in [
                (self.common_decode_head, dzf, 'deg_common_decode'),
                (self.rgb_private_decode_head, drl, 'deg_rgb_private_decode'),
                (self.t_private_decode_head, dtl, 'deg_t_private_decode'),
            ]:
                if head and feats:
                    ld = {k: v*self.aux_loss_weight for k,v in head.loss(feats,data_samples,self.train_cfg).items()}
                    losses.update(add_prefix(ld, pfx))
            if self.loss_align_weight > 0 and dzcr is not None and dzct is not None:
                dlc, dcnt = 0., 0
                for i in range(len(dzcr)):
                    if dzcr[i] is not None and dzct[i] is not None:
                        dDr_i = (ds_r[i] > self.tau).float().detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones(B,1,dzcr[i].shape[2],dzcr[i].shape[3],device=dzcr[i].device)
                        dDt_i = (ds_t[i] > self.tau).float().detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones(B,1,dzct[i].shape[2],dzct[i].shape[3],device=dzct[i].device)
                        dsr_i = ds_r[i].detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else None
                        dst_i = ds_t[i].detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else None
                        dlc += compute_cross_modal_contrastive_loss(
                            dzcr[i],dzct[i],gt,dDr_i,dDt_i,dsr_i,dst_i,
                            tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                            ignore_label=255, pad_mask=pm)
                        dcnt += 1
                if dcnt: losses['loss_align_deg'] = (dlc/dcnt)*self.loss_align_weight
            if self.loss_distill_weight > 0:
                T = self.distill_temperature
                cl = self._get_seg_logits(ff, data_samples).float()
                dl_ = self._get_seg_logits(df, data_samples).float()
                tp = F.softmax(cl.detach()/T,dim=1); sp = F.log_softmax(dl_/T,dim=1)
                kl = F.kl_div(sp,tp,reduction='none').sum(dim=1)
                losses['loss_distill'] = self.loss_distill_weight*(T*T)*kl.mean()

            if self.loss_invariant_weight > 0:
                inv_loss = torch.tensor(0.0, device=ff[0].device)
                cnt = 0
                for i in range(len(zf)):
                    if zf[i] is not None and dzf is not None and i < len(dzf) and dzf[i] is not None:
                        if zf[i].shape == dzf[i].shape:
                            Dc = torch.max(
                                (s_r[i] > self.tau).float().detach() if s_r[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                (s_t[i] > self.tau).float().detach() if s_t[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            Dd = torch.max(
                                (ds_r[i] > self.tau).float().detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                (ds_t[i] > self.tau).float().detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            D_gate = Dc * Dd
                            qc = torch.max(
                                s_r[i].detach() if s_r[i] is not None else torch.ones_like(D_gate),
                                s_t[i].detach() if s_t[i] is not None else torch.ones_like(D_gate))
                            qd = torch.max(
                                ds_r[i].detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones_like(D_gate),
                                ds_t[i].detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones_like(D_gate))
                            D_gate = F.interpolate(D_gate, size=zf[i].shape[2:], mode='nearest') if D_gate.shape[2:] != zf[i].shape[2:] else D_gate
                            qc = F.interpolate(qc, size=zf[i].shape[2:], mode='nearest') if qc.shape[2:] != zf[i].shape[2:] else qc
                            qd = F.interpolate(qd, size=zf[i].shape[2:], mode='nearest') if qd.shape[2:] != zf[i].shape[2:] else qd
                            q_distill = qc * qd * D_gate
                            diff = F.smooth_l1_loss(zf[i], dzf[i], reduction='none')
                            denom = q_distill.sum() + 1e-6
                            inv_loss += (q_distill * diff).sum() / denom
                            cnt += 1
                if cnt:
                    losses['loss_inv'] = self.loss_invariant_weight * inv_loss / cnt

            if self.loss_q_guide_weight > 0:
                num_stages = len(s_r)
                lqg = torch.tensor(0.0, device=ff[0].device)
                lqg_cnt = 0
                for s_clean_l, s_deg_l, deg_lv in [
                    (s_r, ds_r, deg_level_rgb),
                    (s_t, ds_t, deg_level_t),
                    (spr, dspr, deg_level_rgb),
                    (spt, dspt, deg_level_t),
                ]:
                    lqg = lqg + quality_guided_loss(s_clean_l, s_deg_l, deg_lv, num_stages=num_stages)
                    lqg_cnt += 1
                if lqg_cnt > 0:
                    losses['loss_q_guide'] = self.loss_q_guide_weight * lqg / lqg_cnt

        for key in list(losses.keys()):
            if not torch.isfinite(losses[key]):
                losses[key] = torch.tensor(0.0, device=losses[key].device)
            else:
                losses[key] = torch.clamp(losses[key], max=100.0)
        return losses

    def encode_decode(self, inputs, bm):
        rgb, ir = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,ir)[7]
        if self.with_neck: ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def extract_feat(self, inputs):
        rgb, t = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,t)[7]
        return self.neck(ff) if self.with_neck else ff

    def extract_feat_vis(self, inputs):
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]
        with torch.no_grad():
            (zc_r,zc_t,zp_r,zp_t,zf,re,te,ff,s_r,s_t,all_s,spr,spt) = self._extract_feat_single(rgb,t)
            fused = self.neck(ff) if self.with_neck else ff

            deg_rgb, deg_t, deg_type_rgb, deg_type_t = self._generate_degraded_vis_inputs(rgb, t)
            (zc_r_d,zc_t_d,zp_r_d,zp_t_d,zf_d,re_d,te_d,ff_d,
             ds_r_d,ds_t_d,dall_s_d,dspr_d,dspt_d) = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(ff_d) if self.with_neck else ff_d

        for i in range(len(zf)):
            if zc_r_d[i].shape[-2:] != zc_r[i].shape[-2:]:
                zc_r_d[i] = F.interpolate(zc_r_d[i], size=zc_r[i].shape[-2:], mode='bilinear')
                zc_t_d[i] = F.interpolate(zc_t_d[i], size=zc_t[i].shape[-2:], mode='bilinear')
                zf_d[i] = F.interpolate(zf_d[i], size=zf[i].shape[-2:], mode='bilinear')
                zp_r_d[i] = F.interpolate(zp_r_d[i], size=zp_r[i].shape[-2:], mode='bilinear')
                zp_t_d[i] = F.interpolate(zp_t_d[i], size=zp_t[i].shape[-2:], mode='bilinear')
                re_d[i] = F.interpolate(re_d[i], size=re[i].shape[-2:], mode='bilinear')
                te_d[i] = F.interpolate(te_d[i], size=te[i].shape[-2:], mode='bilinear')
                ff_d[i] = F.interpolate(ff_d[i], size=ff[i].shape[-2:], mode='bilinear')

        return dict(
            zc_rgb=zc_r, zc_t=zc_t,
            zc_fused=zf,
            zp_rgb=zp_r, zp_t=zp_t,
            rgb_pf=re, t_pf=te,
            final_fused=fused,
            s_rgb=s_r, s_t=s_t,
            s_rgb_priv=spr, s_t_priv=spt,
            q_rgb_maps=s_r, q_t_maps=s_t,
            q_rgb_priv=spr, q_t_priv=spt,
            clean_rgb_img=rgb, clean_t_img=t,
            deg_rgb_img=deg_rgb, deg_t_img=deg_t,
            deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
            deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
            zc_rgb_deg=zc_r_d, zc_t_deg=zc_t_d,
            zc_fused_deg=zf_d,
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d,
            final_fused_deg=fused_d,
            s_rgb_deg=ds_r_d, s_t_deg=ds_t_d,
            s_rgb_priv_deg=dspr_d, s_t_priv_deg=dspt_d,
            q_rgb_deg=ds_r_d, q_t_deg=ds_t_d,
            q_rgb_priv_deg=dspr_d, q_t_priv_deg=dspt_d,
        )

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        if hasattr(head, 'pixel_decoder'):
            batch_img_metas = [
                dict(ori_shape=feats[0].shape[2:],
                     img_shape=feats[0].shape[2:])
            ] * feats[0].shape[0]
            seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        else:
            seg_logits = head(feats)
        return seg_logits

    def _generate_degraded_vis_inputs(self, rgb, ir):
        """Visualization degradation — follows the same schedule as training."""
        dr, di, dtr, dtt, _, _ = self._generate_degraded_inputs(rgb, ir)
        return dr, di, dtr, dtt

    def _forward(self, inputs, data_samples=None):
        rgb, t = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,t)[7]
        feats = self.neck(ff) if self.with_neck else ff
        return self._get_seg_logits(feats, data_samples)

    def predict(self, inputs, data_samples=None):
        if data_samples:
            bm = [ds.metainfo for ds in data_samples]
        else:
            bm = [dict(ori_shape=inputs.shape[2:],img_shape=inputs.shape[2:],pad_shape=inputs.shape[2:],padding_size=[0,0,0,0])]*inputs.shape[0]
        sl = self.encode_decode(inputs, bm)
        return self.postprocess_result(sl, data_samples)

    @torch.no_grad()
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        results = self._run_forward(data, mode='predict')

        if not hasattr(self, '_val_deg_accum'):
            self._val_deg_accum = {}
        if data_samples is None:
            return results

        if not hasattr(self, '_class_names') and data_samples:
            metainfo = data_samples[0].metainfo
            if 'classes' in metainfo:
                self._class_names = list(metainfo['classes'])

        dev = inputs.device
        rm = self.data_preprocessor.mean[:3].to(dev)
        rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev)
        iss = self.data_preprocessor.std[3:].to(dev)
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        num_classes = self.num_classes
        ignore_index = 255

        for deg_name, deg_rgb, deg_ir in [
            ('rgb_missing',
             _apply_missing_degradation(rgb, rm, rs), ir),
            ('thermal_missing',
             rgb, _apply_missing_degradation(ir, im, iss)),
        ]:
            deg_inputs = torch.cat([deg_rgb, deg_ir], dim=1)
            bm = [ds.metainfo for ds in data_samples]
            seg_logits = self.encode_decode(deg_inputs, bm)
            B, C, H, W = seg_logits.shape
            preds = seg_logits.argmax(dim=1)

            acc = self._val_deg_accum.setdefault(
                deg_name,
                dict(area_intersect=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_union=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_pred=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_label=torch.zeros(num_classes, device=dev, dtype=torch.long)))

            for b in range(B):
                pred = preds[b]
                label = data_samples[b].gt_sem_seg.data.squeeze().to(dev)
                if pred.shape != label.shape:
                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0).float(),
                        size=label.shape, mode='nearest').squeeze()
                mask = (label != ignore_index)
                pred_m = pred[mask]
                label_m = label[mask]
                for c in range(num_classes):
                    pc = (pred_m == c)
                    lc = (label_m == c)
                    acc['area_intersect'][c] += (pc & lc).sum()
                    acc['area_union'][c] += (pc | lc).sum()
                    acc['area_pred'][c] += pc.sum()
                    acc['area_label'][c] += lc.sum()

        return results

    def reset_val_deg_accum(self):
        self._val_deg_accum = {}

    def compute_val_deg_metrics(self):
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if not hasattr(self, '_val_deg_accum') or not self._val_deg_accum:
            return
        class_names = getattr(self, '_class_names', None)
        if class_names is None:
            class_names = [f'class_{i}' for i in range(self.num_classes)]
        for deg_name, acc in self._val_deg_accum.items():
            iou_per_class = acc['area_intersect'].float() / (
                acc['area_union'].float() + 1e-10) * 100
            valid = acc['area_union'] > 0
            miou = iou_per_class[valid].mean().item() if valid.any() else 0.0
            acc_per_class = acc['area_intersect'].float() / (
                acc['area_label'].float() + 1e-10) * 100
            macc = acc_per_class[valid].mean().item() if valid.any() else 0.0
            all_acc = acc['area_intersect'].sum().float() / (
                acc['area_label'].sum().float() + 1e-10) * 100
            cw = [max(len(str(class_names[i])), 8) for i in range(self.num_classes)]
            header = f'+{"-"*14}+' + '+'.join([f'{"-"*(cw[i]+2)}' for i in range(self.num_classes)]) + f'+{"-"*8}+{"-"*8}+{"-"*8}+'
            sep = '|' + f'{deg_name:^14}|' + '|'.join(
                [f'{class_names[i]:^{cw[i]+2}}' for i in range(self.num_classes)]
            ) + f'|{"mIoU":^8}|{"mAcc":^8}|{"aAcc":^8}|'
            vals = '|' + f'{"IoU(%)":^14}|' + '|'.join(
                [f'{iou_per_class[i].item():^{cw[i]+2}.2f}' if valid[i] else f'{"N/A":^{cw[i]+2}}' for i in range(self.num_classes)]
            ) + f'|{miou:^8.2f}|{macc:^8.2f}|{all_acc:^8.2f}|'
            logger.info('\n' + header + '\n' + sep + '\n' + header + '\n' + vals + '\n' + header)
        self._val_deg_accum = {}

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        B, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(B)]
        for i in range(B):
            img_meta = data_samples[i].metainfo
            ps = img_meta.get('padding_size', [0]*4)
            pl, pr, pt, pb = ps
            i_sl = seg_logits[i:i+1,:,pt:H-pb,pl:W-pr]
            flip = img_meta.get('flip', None)
            if flip:
                fd = img_meta.get('flip_direction',None)
                i_sl = i_sl.flip(dims=(3,) if fd=='horizontal' else (2,))
            from mmseg.models.utils import resize
            i_sl = resize(i_sl, size=img_meta['ori_shape'], mode='bilinear',
                         align_corners=self.align_corners, warning=False).squeeze(0)
            pred = i_sl.argmax(dim=0, keepdim=True) if C > 1 else (i_sl.sigmoid() > 0.5).to(i_sl)
            data_samples[i].set_data({'seg_logits': PixelData(data=i_sl), 'pred_sem_seg': PixelData(data=pred)})
        return data_samples

    def inference(self, inputs, bm):
        assert self.test_cfg.mode in ['slide','whole']
        if self.test_cfg.mode == 'slide': return self.slide_inference(inputs, bm)
        return self.whole_inference(inputs, bm)

    def whole_inference(self, inputs, bm):
        return self.encode_decode(inputs, bm)

    def slide_inference(self, inputs, bm):
        hs, ws = self.test_cfg.stride
        hc, wc = self.test_cfg.crop_size
        B, _, h_img, w_img = inputs.size()
        oc = self.out_channels
        hg = max(h_img-hc+hs-1,0)//hs+1; wg = max(w_img-wc+ws-1,0)//ws+1
        preds = inputs.new_zeros((B,oc,h_img,w_img))
        cnt = inputs.new_zeros((B,1,h_img,w_img))
        for hi in range(hg):
            for wi in range(wg):
                y1=hi*hs; x1=wi*ws; y2=min(y1+hc,h_img); x2=min(x1+wc,w_img)
                y1=max(y2-hc,0); x1=max(x2-wc,0)
                crop = inputs[:,:,y1:y2,x1:x2]
                bm[0]['img_shape']=crop.shape[2:]
                csl = self.encode_decode(crop,bm)
                preds += F.pad(csl,(int(x1),int(preds.shape[3]-x2),int(y1),int(preds.shape[2]-y2)))
                cnt[:,:,y1:y2,x1:x2] += 1
        return preds / cnt
