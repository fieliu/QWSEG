"""Quality-Gated Swin with Mask2Former Decoder.

Same three-branch architecture as QualityGatedMiTMamba, but with Swin
backbones instead of MiT.  Quality masks are converted to per-window
attention bias (cyclic-shift aligned for SW-MSA blocks).
Mask2Former decoder replaces SegformerHead.
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
    gumbel_softmax_hard,
    complementary_fix,
    downsample_mask,
    get_degradation_schedule,
    sample_level,
    _apply_degradation,
    _generate_local_mask,
    compute_cross_modal_contrastive_loss,
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

def _quality_mask_to_swin_bias(D_2d, window_size, shift_size, eps=1e-9):
    B, _, H, W = D_2d.shape
    D = D_2d.squeeze(1)
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        D = F.pad(D, (0, pad_r, 0, pad_b), value=1.0)
    H_pad, W_pad = D.shape[1], D.shape[2]
    if shift_size > 0:
        D = torch.roll(D, shifts=(-shift_size, -shift_size), dims=(1, 2))
    wh = window_size
    nW_h, nW_w = H_pad // wh, W_pad // wh
    D_windows = D.reshape(B, nW_h, wh, nW_w, wh)
    D_windows = D_windows.permute(0, 1, 3, 2, 4).contiguous()
    D_windows = D_windows.view(-1, wh * wh)
    bias = torch.log(D_windows.clamp(min=eps) + eps).clamp(min=-10.0)
    return bias.view(-1, 1, 1, wh * wh).expand(-1, 1, wh * wh, wh * wh).contiguous()


# ---------------------------------------------------------------------------
# Swin quality-aware forward: common dual-modality
# ---------------------------------------------------------------------------

def _forward_swin_common_dual_pruned(swin_branch, input_rgbt, orig_B,
                                      predictors_rgb, predictors_t,
                                      gumbel_tau, training=True,
                                      force_all_keep=False, phase=3):
    """Quality-aware forward through a single Swin backbone for both modalities.

    Processes concatenated RGB+T batch with separate quality predictors
    for each modality.  Analogous to _forward_common_dual_pruned in the
    MiT model, but uses Swin window attention with quality bias.
    """
    window_size = swin_branch.stages[0].blocks[0].attn.window_size
    outs, all_D_rgb, all_D_t, all_q_rgb, all_q_t = [], [], [], [], []
    cum_D_rgb_list, cum_D_t_list = [], []
    cum_D_rgb, cum_D_t = None, None

    stages = swin_branch.stages
    x, (H, W) = swin_branch.patch_embed(input_rgbt)
    if swin_branch.use_abs_pos_embed:
        x = x + swin_branch.absolute_pos_embed
    x = swin_branch.drop_after_pos(x)
    B_tok = x.shape[0]

    for i, stage in enumerate(stages):
        blocks, downsample = stage.blocks, stage.downsample

        prev_mask_rgb = cum_D_rgb if cum_D_rgb is not None else \
            torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        prev_mask_t = cum_D_t if cum_D_t is not None else \
            torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        D_rgb_raw = D_t_raw = q_rgb = q_t = None
        if force_all_keep:
            D_rgb_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            D_t_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            if phase == 1:
                q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
                q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            elif i < len(predictors_rgb) and predictors_rgb[i] is not None:
                x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
                _, q_rgb = predictors_rgb[i](x_2d[:orig_B].detach(), prev_mask_rgb)
                _, q_t = predictors_t[i](x_2d[orig_B:].detach(), prev_mask_t)
            else:
                q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
                q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictors_rgb) and predictors_rgb[i] is not None:
            x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
            gate_rgb, q_rgb = predictors_rgb[i](x_2d[:orig_B].detach(), prev_mask_rgb)
            gate_t, q_t = predictors_t[i](x_2d[orig_B:].detach(), prev_mask_t)
            D_rgb_raw, _ = gumbel_softmax_hard(gate_rgb, tau=gumbel_tau, training=training)
            D_t_raw, _ = gumbel_softmax_hard(gate_t, tau=gumbel_tau, training=training)
        if D_rgb_raw is None:
            D_rgb_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            D_t_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if q_rgb is None:
            q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if q_t is None:
            q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        # cross-stage: D_raw × max_pool(prev_cum_D)
        if cum_D_rgb is not None:
            if cum_D_rgb.shape[2:] != (H, W):
                cum_D_rgb = F.adaptive_max_pool2d(cum_D_rgb.float(), (H, W))
            cum_D_rgb = D_rgb_raw * cum_D_rgb
        else:
            cum_D_rgb = D_rgb_raw
        if cum_D_t is not None:
            if cum_D_t.shape[2:] != (H, W):
                cum_D_t = F.adaptive_max_pool2d(cum_D_t.float(), (H, W))
            cum_D_t = D_t_raw * cum_D_t
        else:
            cum_D_t = D_t_raw

        # complementary correction AFTER cross-stage
        cum_D_rgb, cum_D_t = complementary_fix(cum_D_rgb, cum_D_t, q_rgb, q_t)

        all_D_rgb.append(D_rgb_raw); all_D_t.append(D_t_raw)
        all_q_rgb.append(q_rgb); all_q_t.append(q_t)
        cum_D_rgb_list.append(cum_D_rgb); cum_D_t_list.append(cum_D_t)

        cum_D = torch.min(
            torch.cat([cum_D_rgb, torch.ones_like(cum_D_t)], dim=0),
            torch.cat([torch.ones_like(cum_D_rgb), cum_D_t], dim=0))

        for j, block in enumerate(blocks):
            if cum_D is not None:
                x = x * cum_D.detach().view(B_tok, H * W, 1)  # hard zero before attention (detached)
            shift_size = block.attn.shift_size
            quality_bias = None
            if cum_D is not None:
                quality_bias = _quality_mask_to_swin_bias(
                    cum_D, window_size, shift_size)
            x = block(x, (H, W), quality_bias=quality_bias)
            if cum_D is not None:
                x = x * cum_D.detach().view(B_tok, H * W, 1)

        if i in swin_branch.out_indices:
            norm_layer = getattr(swin_branch, f'norm{i}', None)
            out = norm_layer(x) if norm_layer is not None else x
            if cum_D is not None:
                out = out * cum_D.detach().view(B_tok, H * W, 1)
            stage_out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(stage_out)

        if downsample is not None:
            x, (H, W) = downsample(x, (H, W))

    return outs, all_D_rgb, all_D_t, all_q_rgb, all_q_t, cum_D_rgb_list, cum_D_t_list


# ---------------------------------------------------------------------------
# Swin quality-aware forward: single branch
# ---------------------------------------------------------------------------

def _forward_swin_branch_pruned(swin_branch, img, predictor_list,
                                 gumbel_tau, training=True,
                                 force_all_keep=False, phase=3):
    """Quality-aware forward through a single Swin branch.

    Returns:
        outs:        list of feature maps [B, C_i, H_i, W_i]
        all_D_raw:   list of per-stage hard masks
        all_q_weight: list of per-stage soft weights
        cum_D_list:  list of per-stage cumulative masks
    """
    window_size = swin_branch.stages[0].blocks[0].attn.window_size
    outs, all_D_raw, all_q_weight, cum_D_list = [], [], [], []
    cum_D = None

    stages = swin_branch.stages
    x, (H, W) = swin_branch.patch_embed(img)
    if swin_branch.use_abs_pos_embed:
        x = x + swin_branch.absolute_pos_embed
    x = swin_branch.drop_after_pos(x)
    B_tok = x.shape[0]

    for i, stage in enumerate(stages):
        blocks, downsample = stage.blocks, stage.downsample
        D_raw = q_weight = None
        prev_mask = cum_D if cum_D is not None else \
            torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        if force_all_keep:
            D_raw = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
            if phase == 1:
                q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
            elif i < len(predictor_list) and predictor_list[i] is not None:
                x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
                _, q_weight = predictor_list[i](x_2d.detach(), prev_mask)
            else:
                q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictor_list) and predictor_list[i] is not None:
            x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
            gate_logits, q_weight = predictor_list[i](x_2d.detach(), prev_mask)
            D_raw, _ = gumbel_softmax_hard(gate_logits, tau=gumbel_tau, training=training)

        if D_raw is None:
            D_raw = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
        if q_weight is None:
            q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        if cum_D is not None:
            if cum_D.shape[2:] != (H, W):
                cum_D = F.adaptive_max_pool2d(cum_D.float(), (H, W))
            cum_D = D_raw * cum_D
        else:
            cum_D = D_raw

        all_D_raw.append(D_raw)
        all_q_weight.append(q_weight)
        cum_D_list.append(cum_D)

        for j, block in enumerate(blocks):
            if cum_D is not None:
                x = x * cum_D.detach().view(B_tok, H * W, 1)  # hard zero before attention (detached)
            shift_size = block.attn.shift_size
            quality_bias = None
            if cum_D is not None:
                quality_bias = _quality_mask_to_swin_bias(
                    cum_D, window_size, shift_size)
            x = block(x, (H, W), quality_bias=quality_bias)
            if cum_D is not None:
                x = x * cum_D.detach().view(B_tok, H * W, 1)

        if i in swin_branch.out_indices:
            norm_layer = getattr(swin_branch, f'norm{i}', None)
            out = norm_layer(x) if norm_layer is not None else x
            if cum_D is not None:
                out = out * cum_D.detach().view(B_tok, H * W, 1)
            stage_out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(stage_out)

        if downsample is not None:
            x, (H, W) = downsample(x, (H, W))

    return outs, all_D_raw, all_q_weight, cum_D_list


# ---------------------------------------------------------------------------
# Fusion modules  (same as MiT model)
# ---------------------------------------------------------------------------

class SimpleConcatFusion(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.fuse_conv = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, common_feat, priv_feat):
        x = torch.cat([common_feat, priv_feat], dim=1)
        x = self.fuse_conv(x)
        x = common_feat + self.act(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class BiMambaFusion(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, base_pos_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.pos_table = nn.Parameter(torch.empty(1, d_model, base_pos_size, base_pos_size))
        nn.init.trunc_normal_(self.pos_table, std=0.02)
        self.type_gen = nn.Parameter(torch.zeros(d_model))
        self.type_priv = nn.Parameter(torch.zeros(d_model))

        self._init_mamba_params('fwd')
        self._init_mamba_params('bwd')
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

    def _init_mamba_params(self, suffix):
        d, s, r = self.d_inner, self.d_state, self.dt_rank
        self.add_module(f'in_proj_{suffix}', nn.Linear(self.d_model, d * 2, bias=False))
        self.add_module(f'conv1d_{suffix}', nn.Conv1d(d, d, kernel_size=self.d_conv,
                         groups=d, padding=self.d_conv - 1, bias=False))
        self.add_module(f'x_proj_{suffix}', nn.Linear(d, r + s * 2, bias=False))
        self.add_module(f'dt_proj_{suffix}', nn.Linear(r, d, bias=True))
        A = torch.arange(1, s + 1, dtype=torch.float32).repeat(d, 1)
        self.register_parameter(f'A_log_{suffix}', nn.Parameter(torch.log(A)))
        self.register_parameter(f'D_{suffix}', nn.Parameter(torch.ones(d)))
        self.add_module(f'out_proj_{suffix}', nn.Linear(d, self.d_model, bias=False))

    def _mamba_scan(self, x_bld, suffix, cu_seqlens=None):
        B_b, L, _ = x_bld.shape
        d = self.d_inner
        in_proj = getattr(self, f'in_proj_{suffix}')
        conv1d = getattr(self, f'conv1d_{suffix}')
        x_proj = getattr(self, f'x_proj_{suffix}')
        dt_proj = getattr(self, f'dt_proj_{suffix}')
        A_log = getattr(self, f'A_log_{suffix}')
        D_param = getattr(self, f'D_{suffix}')

        xz = in_proj(x_bld)
        x_proj_in, z = xz.chunk(2, dim=-1)
        x_conv = x_proj_in.transpose(1, 2)
        x_conv = F.silu(conv1d(x_conv)[:, :, :L]).transpose(1, 2)
        x_ssm = x_proj(x_conv)
        dt, B_ssm, C_ssm = x_ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj(dt)
        A = -torch.exp(A_log.float())

        x_scan = x_conv.transpose(1, 2)
        dt_scan, B_scan, C_scan = dt.transpose(1, 2), B_ssm.transpose(1, 2), C_ssm.transpose(1, 2)

        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        y = selective_scan_fn(x_scan, dt_scan, A, B_scan, C_scan, D_param.float(),
                              delta_bias=dt_proj.bias.float() if dt_proj.bias is not None else None,
                              delta_softplus=True)

        y = y.transpose(1, 2) * F.silu(z)
        return getattr(self, f'out_proj_{suffix}')(y)

    def _get_pos_embed(self, H, W, device, dtype):
        pos = F.interpolate(self.pos_table, size=(H, W), mode='bilinear', align_corners=False)
        return pos.reshape(self.d_model, H * W).t().to(dtype=dtype)

    def _build_sequences(self, gen, priv, cu_seqlens, N_g):
        cu = cu_seqlens.cpu().numpy()
        B = len(cu) - 1
        parts, off = [], 0
        for b in range(B):
            start, end = cu[b], cu[b + 1]
            M_b = (end - start) - N_g
            parts.append(gen[b])
            if M_b > 0:
                parts.append(priv[off:off + M_b])
                off += M_b
        return torch.cat(parts, dim=0)

    def _reverse_sequences(self, x_bld, cu_seqlens):
        cu = cu_seqlens.cpu().numpy()
        x_rev = torch.zeros_like(x_bld)
        for s in range(len(cu) - 1):
            start, end = cu[s], cu[s + 1]
            x_rev[0, start:end] = x_bld[0, start:end].flip(0)
        return x_rev

    def _extract_generic(self, out_seq, cu_seqlens, N_g):
        cu = cu_seqlens.cpu().numpy()
        B = len(cu) - 1
        G_list = [out_seq[cu[b]:cu[b] + N_g] for b in range(B)]
        return torch.stack(G_list, dim=0)

    def forward(self, generic_tokens, priv_valid_tokens, stage_H, stage_W, cu_seqlens, priv_indices=None):
        N_g = stage_H * stage_W
        pos_embed = self._get_pos_embed(stage_H, stage_W, generic_tokens.device, generic_tokens.dtype)
        gen = generic_tokens + pos_embed.unsqueeze(0) + self.type_gen.unsqueeze(0).unsqueeze(0)
        priv = priv_valid_tokens
        if priv.shape[0] > 0:
            priv = priv + self.type_priv.unsqueeze(0)
            if priv_indices is not None and priv_indices.shape[0] == priv.shape[0]:
                priv = priv + pos_embed[priv_indices]
        x_seq = self._build_sequences(gen, priv, cu_seqlens, N_g)
        x_bld = x_seq.unsqueeze(0)
        out_fwd = self._mamba_scan(x_bld, 'fwd', cu_seqlens=cu_seqlens)
        x_rev = self._reverse_sequences(x_bld, cu_seqlens)
        out_rev = self._mamba_scan(x_rev, 'bwd', cu_seqlens=cu_seqlens)
        out_rev = self._reverse_sequences(out_rev, cu_seqlens)
        out_seq = (out_fwd + out_rev).squeeze(0)
        G_enhanced = self._extract_generic(out_seq, cu_seqlens, N_g)
        enhanced = self.out_proj(self.norm(G_enhanced))
        return self.norm_final(generic_tokens + enhanced)


class MultiScaleRefine(nn.Module):
    """Multi-scale refinement with dilated depthwise convs + channel attention.

    Uses Pre-Norm: input is normalised first, residual is also normalised input.
    """

    def __init__(self, channels, dilations=(1, 2, 3)):
        super().__init__()
        # 使用 LayerNorm，与 MiT/Swin 骨干保持一致，无需指定 num_groups
        self.norm = nn.LayerNorm(channels)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False)
            for d in dilations])
        self.fuse_conv = nn.Conv2d(channels * len(dilations), channels, 1, bias=False)
        self.act = nn.GELU()
        mid_ca = max(channels // 4, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid_ca, 1, bias=False),
            nn.ReLU(inplace=True), nn.Conv2d(mid_ca, channels, 1, bias=False), nn.Sigmoid())
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. 将特征转为 [B, H, W, C] 进行 LayerNorm，然后转回
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()   # [B, C, H, W]

        # 2. 多尺度深度空洞卷积 + 融合
        multi = [dw(x_norm) for dw in self.dw_convs]
        x_fused = self.fuse_conv(torch.cat(multi, dim=1))

        # 3. 通道注意力 + 激活
        x_fused = self.act(x_fused) * self.channel_attn(x_fused)

        # 4. 残差连接（使用归一化后的 x_norm），保持 Pre-Norm 范式
        return x_norm + self.out_conv(x_fused)

class DualGateEnhancedFusion(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.num_stages = len(in_channels_list)
        self.ch_gates = nn.ModuleList()
        self.sp_gates = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.post_convs = nn.ModuleList()
        for ch in in_channels_list:
            self.ch_gates.append(nn.Sequential(
                nn.Conv2d(ch * 2, ch * 2, 1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(ch * 2, ch * 2, 1, bias=False), nn.Sigmoid()))
            self.sp_gates.append(nn.Sequential(
                nn.Conv2d(ch * 2, 2, 3, padding=1, bias=False), nn.Sigmoid()))
            self.post_norms.append(nn.LayerNorm(ch))
            self.post_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False), nn.GELU()))

    def forward(self, rgb_enhanced_list, t_enhanced_list, common_fused_list):
        fused_list = []
        for i in range(self.num_stages):
            Fr, Ft, Fg = rgb_enhanced_list[i], t_enhanced_list[i], common_fused_list[i]
            # 空间对齐
            ref_h, ref_w = Fg.shape[2], Fg.shape[3]
            if Fr.shape[2:] != (ref_h, ref_w):
                Fr = F.interpolate(Fr, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            if Ft.shape[2:] != (ref_h, ref_w):
                Ft = F.interpolate(Ft, size=(ref_h, ref_w), mode='bilinear', align_corners=False)

            # 双门控融合
            B, C, H, W = Fr.shape
            concat = torch.cat([Fr, Ft], dim=1)
            ch_gate = self.ch_gates[i](concat)
            ch_r, ch_t = ch_gate.split(C, dim=1)
            sp_gate = self.sp_gates[i](concat)
            sp_r, sp_t = sp_gate[:, 0:1], sp_gate[:, 1:2]
            fused = ch_r * sp_r * Fr + ch_t * sp_t * Ft

            # --- 修正点：Pre-Norm 残差连接 ---
            # 对 fused 做 LayerNorm
            fused_norm = fused.permute(0, 2, 3, 1).contiguous()
            fused_norm = self.post_norms[i](fused_norm).permute(0, 3, 1, 2).contiguous()
            # 主分支变换
            out = self.post_convs[i](fused_norm)
            # 残差 = 归一化后的 fused
            fused_list.append(Fg + out)

        return fused_list

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
                 gumbel_tau_init: float = 1.0,
                 gumbel_tau_min: float = 0.1,
                 gumbel_tau_decay: float = 0.995,
                 retention_min: float = 0.3,
                 retention_max: float = 0.7,
                 retention_loss_weight: float = 5.0,
                 phase1_epochs: int = 10,
                 phase2_epochs: int = 20,
                 total_epochs: int = 200,
                 phase_mode: str = 'absolute',  # 'absolute' or 'ratio'
                 loss_align_weight: float = 0.1,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_distill_weight: float = 0.3,
                 distill_temperature: float = 4.0,
                 aux_loss_weight: float = 0.3,
                 loss_invariant_weight: float = 0.03,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
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

        self._build_predictors()
        self._build_fusion(mamba_d_state, mamba_d_conv, mamba_expand)
        self.common_refine = nn.ModuleList([MultiScaleRefine(ch) for ch in self.embed_dims_list])
        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.gumbel_tau = gumbel_tau_init
        self.gumbel_tau_min = gumbel_tau_min
        self.gumbel_tau_decay = gumbel_tau_decay
        self.retention_min, self.retention_max = retention_min, retention_max
        self.retention_loss_weight = retention_loss_weight
        self.phase1_epochs, self.phase2_epochs = phase1_epochs, phase2_epochs
        self.total_epochs = total_epochs
        self.phase_mode = phase_mode
        self.loss_align_weight = loss_align_weight
        self.contrast_tau, self.contrast_num_samples = contrast_tau, contrast_num_samples
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.missing_ratio, self.global_deg_ratio = missing_ratio, global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
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

    def _build_fusion(self, d_state, d_conv, expand):
        self.fuse_rgb = nn.ModuleList()
        self.fuse_t = nn.ModuleList()
        for i, ch in enumerate(self.embed_dims_list):
            if i < 2:
                self.fuse_rgb.append(SimpleConcatFusion(ch))
                self.fuse_t.append(SimpleConcatFusion(ch))
            else:
                self.fuse_rgb.append(BiMambaFusion(d_model=ch, d_state=d_state,
                    d_conv=d_conv, expand=expand))
                self.fuse_t.append(BiMambaFusion(d_model=ch, d_state=d_state,
                    d_conv=d_conv, expand=expand))

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
        if self.phase_mode == 'ratio':
            r = min(epoch / max(self.total_epochs, 1), 1.0)
            if r < 0.1: return 1
            elif r < 0.3: return 2
            return 3
        if epoch < self.phase1_epochs: return 1
        elif epoch < self.phase1_epochs + self.phase2_epochs: return 2
        return 3

    def _update_training_phase(self, epoch):
        phase = self._get_training_phase(epoch)
        all_p = [self.predictors_common_rgb, self.predictors_common_t,
                 self.predictors_priv_rgb, self.predictors_priv_t]
        if phase == 1:
            for pl in all_p:
                for m in pl:
                    for p in m.parameters(): p.requires_grad = False
        elif phase == 2:
            for pl in all_p:
                for m in pl:
                    for p in m.gate_head.parameters(): p.requires_grad = False
                    for n, p in m.named_parameters():
                        if 'gate_head' not in n: p.requires_grad = True
        else:
            for pl in all_p:
                for m in pl:
                    for p in m.parameters(): p.requires_grad = True
        if self.training and phase >= 3:
            self.gumbel_tau = max(self.gumbel_tau * self.gumbel_tau_decay, self.gumbel_tau_min)
        return phase

    def _compute_retention_loss(self, all_D):
        dev = None
        for d in all_D:
            if d is not None:
                dev = d.device
                break
        loss = torch.tensor(0., device=dev or 'cpu')
        cnt = 0
        for D in all_D:
            if D is None: continue
            r = D.mean()
            if r < self.retention_min: loss += (self.retention_min - r)**2
            elif r > self.retention_max: loss += (r - self.retention_max)**2
            cnt += 1
        return loss / max(cnt, 1) if cnt else loss

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        if q_rgb.shape[2:] != zc_rgb.shape[2:]:
            q_rgb = F.interpolate(q_rgb, size=zc_rgb.shape[2:], mode='nearest')
        if q_t.shape[2:] != zc_t.shape[2:]:
            q_t = F.interpolate(q_t, size=zc_t.shape[2:], mode='nearest')
        q_rgb_s, q_t_s = q_rgb.squeeze(1), q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        return (q_rgb_s/q_sum).unsqueeze(1)*zc_rgb + (q_t_s/q_sum).unsqueeze(1)*zc_t

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

    def _mamba_fuse_stage(self, zc, priv, fuse_mod, Dp):
        B, C, H, W = zc.shape
        N_g = H * W
        if Dp.shape[2:] != (H, W): Dp = F.interpolate(Dp.float(), size=(H, W), mode='nearest')
        Ds = (Dp.squeeze(1) > 0.5).float()
        G = zc.permute(0,2,3,1).reshape(B, N_g, C)
        P = priv.permute(0,2,3,1).reshape(B, N_g, C)
        pl, idx_l, cs = [], [], [0]
        for b in range(B):
            vi = Ds[b].reshape(-1).nonzero(as_tuple=True)[0]
            Mb = vi.shape[0]
            if Mb == 0:
                pl.append(torch.zeros(0, C, device=P.device, dtype=P.dtype))
                idx_l.append(torch.zeros(0, dtype=torch.long, device=P.device))
                cs.append(cs[-1] + N_g)
            else:
                pl.append(P[b, vi]); idx_l.append(vi); cs.append(cs[-1]+N_g+Mb)
        pv = torch.cat(pl, dim=0) if pl[0].shape[0] > 0 else torch.zeros(0, C, device=zc.device, dtype=zc.dtype)
        priv_indices = torch.cat(idx_l, dim=0) if idx_l[0].shape[0] > 0 else torch.zeros(0, dtype=torch.long, device=zc.device)
        ct = torch.tensor(cs, dtype=torch.int32, device=zc.device)
        return fuse_mod(G, pv, H, W, ct, priv_indices).reshape(B, H, W, C).permute(0,3,1,2)

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg: super().init_weights()

    def _extract_feat_single(self, rgb, t):
        B = rgb.shape[0]
        epoch = getattr(self, 'current_epoch', 0)
        ph = self._get_training_phase(epoch)
        fa = (ph < 3)
        ug = self.training and not fa

        zc_outs, D_r, D_t, q_r, q_t, cDr, cDt = _forward_swin_common_dual_pruned(
            self.backbone, torch.cat([rgb, t], dim=0), orig_B=B,
            predictors_rgb=self.predictors_common_rgb,
            predictors_t=self.predictors_common_t,
            gumbel_tau=self.gumbel_tau, training=ug, force_all_keep=fa, phase=ph)
        zc_r = [f[:B] for f in zc_outs]; zc_t = [f[B:] for f in zc_outs]

        zp_r, Dpr, qpr, cDpr = _forward_swin_branch_pruned(
            self.private_branch_rgb, rgb, self.predictors_priv_rgb,
            self.gumbel_tau, training=ug, force_all_keep=fa, phase=ph)
        for i in range(len(zp_r)):
            if zp_r[i] is not None and Dpr[i] is not None: zp_r[i] = zp_r[i] * Dpr[i]

        zp_t, Dpt, qpt, cDpt = _forward_swin_branch_pruned(
            self.private_branch_t, t, self.predictors_priv_t,
            self.gumbel_tau, training=ug, force_all_keep=fa, phase=ph)
        for i in range(len(zp_t)):
            if zp_t[i] is not None and Dpt[i] is not None: zp_t[i] = zp_t[i] * Dpt[i]

        zf, re, te = [], [], []
        for i in range(len(self.embed_dims_list)):
            qri = q_r[i] if q_r[i] is not None else torch.ones(B,1,zc_r[i].shape[2],zc_r[i].shape[3],device=zc_r[i].device)
            qti = q_t[i] if q_t[i] is not None else torch.ones(B,1,zc_t[i].shape[2],zc_t[i].shape[3],device=zc_t[i].device)
            zf.append(self.common_refine[i](self._quality_weighted_common_fusion(zc_r[i], zc_t[i], qri, qti)))

            Dpi = Dpr[i] if Dpr[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device)
            if zf[i].shape[2:]==zp_r[i].shape[2:]:
                re.append(zf[i]+self.fuse_rgb[i](zf[i],zp_r[i])-zf[i] if i<2 else self._mamba_fuse_stage(zf[i],zp_r[i],self.fuse_rgb[i],Dpi))
            else: re.append(zp_r[i])

            Dpi = Dpt[i] if Dpt[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device)
            if zf[i].shape[2:]==zp_t[i].shape[2:]:
                te.append(zf[i]+self.fuse_t[i](zf[i],zp_t[i])-zf[i] if i<2 else self._mamba_fuse_stage(zf[i],zp_t[i],self.fuse_t[i],Dpi))
            else: te.append(zp_t[i])

        ff = self.final_fusion(re, te, zf)
        ad = [d for dl in [cDr,cDt,cDpr,cDpt] for d in dl]
        return zc_r, zc_t, zp_r, zp_t, zf, re, te, ff, q_r, q_t, ad, D_r, D_t, Dpr, Dpt, qpr, qpt, cDr, cDt, cDpr, cDpt

    def _train_with_degradation(self, rgb, ir):
        dr, di, _, _ = self._generate_degraded_inputs(rgb, ir)
        return self._extract_feat_single(dr, di)

    def _generate_degraded_inputs(self, rgb, ir):
        B, C, H, W = rgb.shape
        dev = rgb.device
        rm = self.data_preprocessor.mean[:3].to(dev); rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev); iss = self.data_preprocessor.std[3:].to(dev)
        dr, di = rgb.clone(), ir.clone()
        dtr, dtt = ['none']*B, ['none']*B
        ep = getattr(self,'current_epoch',0)
        sched = get_degradation_schedule(min(ep/max(self.total_epochs,1),1.0))
        for b in range(B):
            r = random.random()
            if r < sched['p_missing']:
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,deg_type='missing',level=5)
                    dtr[b]='missing'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,deg_type='missing',level=5)
                    dtt[b]='missing'
            elif r < sched['p_missing']+sched['p_global']:
                lv = sample_level(sched['global_levels'])
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv); dtr[b]='global'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv); dtt[b]='global'
            else:
                lv = sample_level(sched['local_levels'])
                lm = _generate_local_mask(1,H,W,num_regions=3,device=dev,level=lv)
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv,is_local=True,local_mask=lm)
                    dtr[b]='local'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv,is_local=True,local_mask=lm)
                    dtt[b]='local'
        return dr.to(rgb.dtype), di.to(ir.dtype), dtr, dtt

    # ---- loss / predict / inference ----

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:,:3], inputs[:,3:]
        B = rgb.shape[0]
        ep = getattr(self,'current_epoch',0)
        ph = self._get_training_phase(ep)
        self._update_training_phase(ep)
        (zc_r,zc_t,zp_r,zp_t,zf,re,te,ff,q_r,q_t,ad,D_r,D_t,Dpr,Dpt,qpr,qpt,cDr,cDt,cDpr,cDpt) = self._extract_feat_single(rgb,ir)
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
                    Dr = D_r[i].detach() if D_r[i] is not None else torch.ones(B,1,zc_r[i].shape[2],zc_r[i].shape[3],device=zc_r[i].device)
                    Dt = D_t[i].detach() if D_t[i] is not None else torch.ones(B,1,zc_t[i].shape[2],zc_t[i].shape[3],device=zc_t[i].device)
                    qr = q_r[i].detach() if q_r[i] is not None else None
                    qt = q_t[i].detach() if q_t[i] is not None else None
                    lc += compute_cross_modal_contrastive_loss(
                        zc_r[i],zc_t[i],gt,Dr,Dt,qr,qt,
                        tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                        ignore_label=255, pad_mask=pm)
                    cnt += 1
            if cnt: losses['loss_align'] = (lc/cnt)*self.loss_align_weight
        if self.retention_loss_weight > 0: losses['loss_retention'] = self.retention_loss_weight*self._compute_retention_loss(ad)
        if self.training:
            dzcr,dzct,dzpr,dzpt,dzf,drl,dtl,df,dqr,dqt,dad,dDr,dDt,dDpr,dDpt,dqpr,dqpt,dcDr,dcDt,dcDpr,dcDpt = self._train_with_degradation(rgb,ir)
            # NaN guard: if degraded features are NaN, fall back to clean
            if torch.isnan(df[0]).any():
                import logging
                logging.getLogger(__name__).warning(
                    'NaN in degraded features — falling back to clean features for deg losses')
                dzcr,dzct,dzpr,dzpt,dzf,drl,dtl,df,dqr,dqt,dad,dDr,dDt,dDpr,dDpt,dqpr,dqpt,dcDr,dcDt,dcDpr,dcDpt = \
                    zc_r,zc_t,zp_r,zp_t,zf,re,te,ff,q_r,q_t,ad,D_r,D_t,Dpr,Dpt,qpr,qpt,cDr,cDt,cDpr,cDpt
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
                        dDr_i = dDr[i].detach() if dDr is not None and i < len(dDr) and dDr[i] is not None else torch.ones(B,1,dzcr[i].shape[2],dzcr[i].shape[3],device=dzcr[i].device)
                        dDt_i = dDt[i].detach() if dDt is not None and i < len(dDt) and dDt[i] is not None else torch.ones(B,1,dzct[i].shape[2],dzct[i].shape[3],device=dzct[i].device)
                        dqr_i = dqr[i].detach() if dqr is not None and i < len(dqr) and dqr[i] is not None else None
                        dqt_i = dqt[i].detach() if dqt is not None and i < len(dqt) and dqt[i] is not None else None
                        dlc += compute_cross_modal_contrastive_loss(
                            dzcr[i],dzct[i],gt,dDr_i,dDt_i,dqr_i,dqt_i,
                            tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                            ignore_label=255, pad_mask=pm)
                        dcnt += 1
                if dcnt: losses['loss_align_deg'] = (dlc/dcnt)*self.loss_align_weight
            if self.loss_distill_weight > 0 and ph >= 3:
                T = self.distill_temperature
                cl = self._get_seg_logits(ff, data_samples).float()
                dl_ = self._get_seg_logits(df, data_samples).float()
                tp = F.softmax(cl.detach()/T,dim=1); sp = F.log_softmax(dl_/T,dim=1)
                kl = F.kl_div(sp,tp,reduction='none').sum(dim=1)
                losses['loss_distill'] = self.loss_distill_weight*(T*T)*kl.mean()

            if self.loss_invariant_weight > 0 and ph >= 3:
                inv_loss = torch.tensor(0.0, device=ff[0].device)
                cnt = 0
                all_D = dDr is not None and dDt is not None
                for i in range(len(zf)):
                    if zf[i] is not None and dzf is not None and i < len(dzf) and dzf[i] is not None:
                        if zf[i].shape == dzf[i].shape:
                            Dc = torch.max(
                                D_r[i].detach() if D_r[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                D_t[i].detach() if D_t[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            if all_D:
                                Dd = torch.max(
                                    dDr[i].detach() if dDr[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                    dDt[i].detach() if dDt[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            else:
                                Dd = torch.ones_like(Dc)
                            D_gate = ((Dc > 0.5) & (Dd > 0.5)).float()
                            qc = torch.max(
                                q_r[i].detach() if q_r[i] is not None else torch.ones_like(D_gate),
                                q_t[i].detach() if q_t[i] is not None else torch.ones_like(D_gate))
                            qd = torch.max(
                                dqr[i].detach() if dqr is not None and i < len(dqr) and dqr[i] is not None else torch.ones_like(D_gate),
                                dqt[i].detach() if dqt is not None and i < len(dqt) and dqt[i] is not None else torch.ones_like(D_gate))
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
            (zc_r,zc_t,zp_r,zp_t,zf,re,te,ff,q_r,q_t,ad,
             D_r,D_t,Dpr,Dpt,qpr,qpt,cDr,cDt,cDpr,cDpt) = self._extract_feat_single(rgb,t)
            fused = self.neck(ff) if self.with_neck else ff

            deg_rgb, deg_t, deg_type_rgb, deg_type_t = self._generate_degraded_vis_inputs(rgb, t)
            (zc_r_d,zc_t_d,zp_r_d,zp_t_d,zf_d,re_d,te_d,ff_d,
             q_r_d,q_t_d,ad_d,D_r_d,D_t_d,Dpr_d,Dpt_d,
             qpr_d,qpt_d,cDr_d,cDt_d,cDpr_d,cDpt_d) = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(ff_d) if self.with_neck else ff_d

        # align degraded feat spatial sizes to clean counterparts
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
            q_rgb_maps=q_r, q_t_maps=q_t,
            q_rgb_priv=qpr, q_t_priv=qpt,
            D_rgb=D_r, D_t=D_t, D_rgb_priv=Dpr, D_t_priv=Dpt,
            cum_D_rgb=cDr, cum_D_t=cDt, cum_D_rgb_priv=cDpr, cum_D_t_priv=cDpt,
            deg_rgb_img=deg_rgb, deg_t_img=deg_t,
            deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
            deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
            zc_rgb_deg=zc_r_d, zc_t_deg=zc_t_d,
            zc_fused_deg=zf_d,
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d,
            final_fused_deg=fused_d,
            q_rgb_deg=q_r_d, q_t_deg=q_t_d,
            q_rgb_priv_deg=qpr_d, q_t_priv_deg=qpt_d,
            D_rgb_deg=D_r_d, D_t_deg=D_t_d,
            D_rgb_priv_deg=Dpr_d, D_t_priv_deg=Dpt_d,
            cum_D_rgb_deg=cDr_d, cum_D_t_deg=cDt_d,
            cum_D_rgb_priv_deg=cDpr_d, cum_D_t_priv_deg=cDpt_d,
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
        return self._generate_degraded_inputs(rgb, ir)

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
