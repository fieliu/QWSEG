"""Quality-Gated MiT with Mamba Fusion.

Token pruning (DynamicViT-style) on a multi-branch MiT backbone with
Gumbel-Softmax hard gating, quality-modulated fusion, and three-phase
training.  Fully self-contained — only depends on BaseSegmentor + v9_utils.
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

# ---------------------------------------------------------------------------
# Helper: MiT NLC ↔ NCHW
# ---------------------------------------------------------------------------

def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    return x.flatten(2).transpose(1, 2)

# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class SimpleConcatFusion(nn.Module):
    """Shallow-stage fusion: concat, 1x1 conv, LN, GELU, residual."""

    def __init__(self, d_model):
        super().__init__()
        self.fuse_conv = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, common_feat, priv_feat):
        x = torch.cat([common_feat, priv_feat], dim=1)
        x = self.fuse_conv(x)
        x = self.act(self.out_norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return common_feat + x


class BiMambaFusion(nn.Module):
    """Bidirectional mamba fusion (no quality modulation)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, base_pos_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.pos_table = nn.Parameter(torch.randn(1, d_model, base_pos_size, base_pos_size) * 0.02)
        self.type_gen = nn.Parameter(torch.zeros(d_model))
        self.type_priv = nn.Parameter(torch.zeros(d_model))

        self._init_mamba_params('fwd')
        self._init_mamba_params('bwd')
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.LayerNorm(d_model)

    def _init_mamba_params(self, suffix):
        d, s, r = self.d_inner, self.d_state, self.dt_rank
        self.add_module(f'in_proj_{suffix}', nn.Linear(self.d_model, d * 2, bias=False))
        self.add_module(f'conv1d_{suffix}', nn.Conv1d(d, d, kernel_size=self.d_conv,
                         groups=d, padding=self.d_conv - 1, bias=True))
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

        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            y = selective_scan_fn(x_scan, dt_scan, A, B_scan, C_scan, D_param.float(),
                                  delta_bias=dt_proj.bias.float() if dt_proj.bias is not None else None,
                                  delta_softplus=True)
        except ImportError:
            y = _selective_scan_pytorch(x_scan, dt_scan, A, B_scan, C_scan, D_param, cu_seqlens)

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

    def forward(self, generic_tokens, priv_valid_tokens, stage_H, stage_W, cu_seqlens):
        N_g = stage_H * stage_W
        pos_embed = self._get_pos_embed(stage_H, stage_W, generic_tokens.device, generic_tokens.dtype)
        gen = generic_tokens + pos_embed.unsqueeze(0) + self.type_gen.unsqueeze(0).unsqueeze(0)
        priv = priv_valid_tokens
        if priv.shape[0] > 0:
            priv = priv + self.type_priv.unsqueeze(0)
        x_seq = self._build_sequences(gen, priv, cu_seqlens, N_g)
        x_bld = x_seq.unsqueeze(0)
        out_fwd = self._mamba_scan(x_bld, 'fwd', cu_seqlens=cu_seqlens)
        x_rev = self._reverse_sequences(x_bld, cu_seqlens)
        out_rev = self._mamba_scan(x_rev, 'bwd', cu_seqlens=cu_seqlens)
        out_rev = self._reverse_sequences(out_rev, cu_seqlens)
        out_seq = (out_fwd + out_rev).squeeze(0)
        G_enhanced = self._extract_generic(out_seq, cu_seqlens, N_g)
        return generic_tokens + self.out_proj(self.out_norm(G_enhanced))


def _selective_scan_pytorch(x, dt, A, B_ssm, C_ssm, D, cu_seqlens=None):
    batch, d_inner, L = x.shape
    d_state = A.shape[1]
    dt = F.softplus(dt)
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    ys = []
    cu = cu_seqlens.cpu().numpy() if cu_seqlens is not None else None
    seq_starts = set(cu.tolist()) if cu is not None else set()
    for i in range(L):
        if cu is not None and i in seq_starts and i > 0:
            h = torch.zeros_like(h)
        dA = torch.exp(dt[:, :, i].unsqueeze(-1) * A.unsqueeze(0))
        dB = dt[:, :, i].unsqueeze(-1) * B_ssm[:, :, i].unsqueeze(-2)
        h = dA * h + dB * x[:, :, i].unsqueeze(-1)
        ys.append(torch.sum(h * C_ssm[:, :, i].unsqueeze(1), dim=-1))
    y = torch.stack(ys, dim=2)
    return y + D.unsqueeze(0).unsqueeze(-1) * x


class MultiScaleRefine(nn.Module):
    """Multi-scale refinement with dilated depthwise convs + channel attention.
    All normalisation uses LayerNorm (channel-last)."""

    def __init__(self, channels, dilations=(1, 2, 3)):
        super().__init__()
        self.in_norm = nn.LayerNorm(channels)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False)
            for d in dilations])
        self.fuse_conv = nn.Conv2d(channels * len(dilations), channels, 1, bias=False)
        self.act = nn.GELU()
        mid_ca = max(channels // 4, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, mid_ca, 1, bias=False),
            nn.ReLU(inplace=True), nn.Conv2d(mid_ca, channels, 1, bias=False), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(channels)
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.in_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        multi = [dw(x) for dw in self.dw_convs]
        x = self.fuse_conv(torch.cat(multi, dim=1))
        x = self.act(x) * self.channel_attn(x)
        x = self.out_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + self.out_conv(x)


class DualGateEnhancedFusion(nn.Module):
    """Channel + spatial dual-gate fusion for multi-scale features."""

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
            self.post_convs.append(nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.GELU()))

    def forward(self, rgb_enhanced_list, t_enhanced_list, common_fused_list):
        fused_list = []
        for i in range(self.num_stages):
            Fr, Ft, Fg = rgb_enhanced_list[i], t_enhanced_list[i], common_fused_list[i]
            ref_h, ref_w = Fg.shape[2], Fg.shape[3]
            if Fr.shape[2:] != (ref_h, ref_w):
                Fr = F.interpolate(Fr, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            if Ft.shape[2:] != (ref_h, ref_w):
                Ft = F.interpolate(Ft, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            B, C, H, W = Fr.shape
            concat = torch.cat([Fr, Ft], dim=1)
            ch_gate = self.ch_gates[i](concat)
            ch_r, ch_t = ch_gate.split(C, dim=1)
            sp_gate = self.sp_gates[i](concat)
            sp_r, sp_t = sp_gate[:, 0:1], sp_gate[:, 1:2]
            fused = ch_r * sp_r * Fr + ch_t * sp_t * Ft + Fg
            x = fused.permute(0, 2, 3, 1)
            x = self.post_norms[i](x).permute(0, 3, 1, 2)
            fused_list.append(fused + self.post_convs[i](x))
        return fused_list


# ---------------------------------------------------------------------------
# Quality-aware MiT forward functions
# ---------------------------------------------------------------------------

def _forward_common_dual_pruned(backbone, input_rgbt, orig_B,
                                 predictors_rgb, predictors_t,
                                 gumbel_tau, training=True,
                                 force_all_keep=False, phase=3):
    """Quality-aware forward through a single MiT backbone for both modalities."""
    outs, all_D_rgb, all_D_t, all_q_rgb, all_q_t = [], [], [], [], []
    cum_D_rgb_list, cum_D_t_list = [], []
    cum_D_rgb, cum_D_t = None, None

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(input_rgbt)
        H, W = hw_shape
        B_tok = x.shape[0]

        prev_mask_rgb = cum_D_rgb if cum_D_rgb is not None else \
            torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        prev_mask_t = cum_D_t if cum_D_t is not None else \
            torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        # --- quality prediction ---
        D_rgb_raw = D_t_raw = q_rgb = q_t = None
        if force_all_keep:
            D_rgb_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            D_t_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            if phase == 1:
                q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
                q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            elif i < len(predictors_rgb) and predictors_rgb[i] is not None:
                x_2d = nlc_to_nchw(x, hw_shape)
                _, q_rgb = predictors_rgb[i](x_2d[:orig_B].detach(), prev_mask_rgb)
                _, q_t = predictors_t[i](x_2d[orig_B:].detach(), prev_mask_t)
            else:
                q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
                q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictors_rgb) and predictors_rgb[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            gate_rgb, q_rgb = predictors_rgb[i](x_2d[:orig_B].detach(), prev_mask_rgb)
            gate_t, q_t = predictors_t[i](x_2d[orig_B:].detach(), prev_mask_t)
            D_rgb_raw, _ = gumbel_softmax_hard(gate_rgb, tau=gumbel_tau, training=training)
            D_t_raw, _ = gumbel_softmax_hard(gate_t, tau=gumbel_tau, training=training)
            D_rgb_raw, D_t_raw = complementary_fix(D_rgb_raw, D_t_raw, q_rgb, q_t)
        if D_rgb_raw is None:
            D_rgb_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            D_t_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if q_rgb is None:
            q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if q_t is None:
            q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        # cross-stage mask propagation
        if cum_D_rgb is not None:
            if cum_D_rgb.shape[2:] != (H, W):
                cum_D_rgb = F.interpolate(cum_D_rgb.float(), size=(H, W), mode='nearest')
            cum_D_rgb = D_rgb_raw * cum_D_rgb
        else:
            cum_D_rgb = D_rgb_raw
        if cum_D_t is not None:
            if cum_D_t.shape[2:] != (H, W):
                cum_D_t = F.interpolate(cum_D_t.float(), size=(H, W), mode='nearest')
            cum_D_t = D_t_raw * cum_D_t
        else:
            cum_D_t = D_t_raw

        all_D_rgb.append(D_rgb_raw); all_D_t.append(D_t_raw)
        all_q_rgb.append(q_rgb); all_q_t.append(q_t)
        cum_D_rgb_list.append(cum_D_rgb); cum_D_t_list.append(cum_D_t)

        cum_D = torch.min(
            torch.cat([cum_D_rgb, torch.ones_like(cum_D_t)], dim=0),
            torch.cat([torch.ones_like(cum_D_rgb), cum_D_t], dim=0))

        # --- transformer blocks ---
        for j, block in enumerate(blocks):
            residual = x
            x_norm1 = block.norm1(x)
            attn_module = block.attn
            x_q = x_norm1
            if attn_module.sr_ratio > 1:
                x_kv = nlc_to_nchw(x_norm1, hw_shape)
                x_kv = attn_module.sr(x_kv)
                x_kv = nchw_to_nlc(x_kv)
                x_kv = attn_module.norm(x_kv)
            else:
                x_kv = x_norm1

            Bb, N_tok, C_tok = x_q.shape
            num_heads, head_dim = attn_module.num_heads, C_tok // attn_module.num_heads
            hw_shape_k = hw_shape

            w = attn_module.attn.in_proj_weight
            q_w, k_w, v_w = w[:C_tok], w[C_tok:2*C_tok], w[2*C_tok:]
            b = attn_module.attn.in_proj_bias
            q_b = b[:C_tok] if b is not None else None
            k_b = b[C_tok:2*C_tok] if b is not None else None
            v_b = b[2*C_tok:] if b is not None else None

            Q = F.linear(x_q, q_w, q_b)
            K = F.linear(x_norm1, k_w, k_b)
            V = F.linear(x_kv, v_w, v_b)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            if cum_D is not None:
                H_k, W_k = hw_shape_k
                D_k = downsample_mask(cum_D, H_k, W_k).reshape(Bb, 1, 1, -1)
                attn = attn + torch.log(D_k.float() + 1e-6).to(attn.dtype)

            attn = attn.float().softmax(dim=-1).to(V.dtype)
            if hasattr(attn_module, 'dropout_layer') and attn_module.dropout_layer is not None:
                attn = attn_module.dropout_layer(attn)
            attn_out = (attn @ V).transpose(1, 2).reshape(Bb, N_tok, C_tok)
            attn_out = attn_module.proj_drop(attn_module.attn.out_proj(attn_out))
            x = residual + attn_out
            if cum_D is not None:
                D_q = cum_D.reshape(Bb, H * W, 1)
                x = x * D_q

            residual = x
            x_norm2 = block.norm2(x)
            ffn_out = nlc_to_nchw(x_norm2, hw_shape)
            ffn_out = block.ffn.layers(ffn_out)
            ffn_out = nchw_to_nlc(ffn_out)
            ffn_out = block.ffn.dropout_layer(ffn_out)
            x = residual + ffn_out
            if cum_D is not None:
                x = x * D_q

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        input_rgbt = x

    return outs, all_D_rgb, all_D_t, all_q_rgb, all_q_t, cum_D_rgb_list, cum_D_t_list


def _forward_branch_pruned(backbone, img, predictor_list, cum_D_prev_list,
                            gumbel_tau, is_common, other_D_list=None,
                            other_q_list=None, training=True,
                            force_all_keep=False, phase=3):
    """Quality-aware forward through a single-branch MiT backbone."""
    outs, all_D_raw, all_q_weight, cum_D_list = [], [], [], []
    cum_D = None

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape
        B_tok = x.shape[0]

        prev_mask = cum_D if cum_D is not None else \
            torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        D_raw = q_weight = None
        if force_all_keep:
            D_raw = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
            if phase == 1:
                q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
            elif i < len(predictor_list) and predictor_list[i] is not None:
                x_2d = nlc_to_nchw(x, hw_shape)
                _, q_weight = predictor_list[i](x_2d.detach(), prev_mask)
            else:
                q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictor_list) and predictor_list[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            gate_logits, q_weight = predictor_list[i](x_2d.detach(), prev_mask)
            D_raw, _ = gumbel_softmax_hard(gate_logits, tau=gumbel_tau, training=training)
        if D_raw is None:
            D_raw = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
        if q_weight is None:
            q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        if cum_D is not None:
            if cum_D.shape[2:] != (H, W):
                cum_D = F.interpolate(cum_D.float(), size=(H, W), mode='nearest')
            cum_D = D_raw * cum_D
        else:
            cum_D = D_raw
        all_D_raw.append(D_raw); all_q_weight.append(q_weight); cum_D_list.append(cum_D)

        for j, block in enumerate(blocks):
            residual = x
            x_norm1 = block.norm1(x)
            attn_module = block.attn
            x_q = x_norm1
            if attn_module.sr_ratio > 1:
                x_kv = nlc_to_nchw(x_norm1, hw_shape)
                x_kv = attn_module.sr(x_kv)
                x_kv = nchw_to_nlc(x_kv)
                x_kv = attn_module.norm(x_kv)
            else:
                x_kv = x_norm1

            Bb, N_tok, C_tok = x_q.shape
            num_heads, head_dim = attn_module.num_heads, C_tok // attn_module.num_heads
            hw_shape_k = hw_shape

            w = attn_module.attn.in_proj_weight
            q_w, k_w, v_w = w[:C_tok], w[C_tok:2*C_tok], w[2*C_tok:]
            b = attn_module.attn.in_proj_bias
            q_b = b[:C_tok] if b is not None else None
            k_b = b[C_tok:2*C_tok] if b is not None else None
            v_b = b[2*C_tok:] if b is not None else None

            Q = F.linear(x_q, q_w, q_b)
            K = F.linear(x_norm1, k_w, k_b)
            V = F.linear(x_kv, v_w, v_b)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            if cum_D is not None:
                H_k, W_k = hw_shape_k
                D_k = downsample_mask(cum_D, H_k, W_k).reshape(Bb, 1, 1, -1)
                attn = attn + torch.log(D_k.float() + 1e-6).to(attn.dtype)

            attn = attn.float().softmax(dim=-1).to(V.dtype)
            if hasattr(attn_module, 'dropout_layer') and attn_module.dropout_layer is not None:
                attn = attn_module.dropout_layer(attn)
            attn_out = (attn @ V).transpose(1, 2).reshape(Bb, N_tok, C_tok)
            attn_out = attn_module.proj_drop(attn_module.attn.out_proj(attn_out))
            x = residual + attn_out
            if cum_D is not None:
                D_q = cum_D.reshape(Bb, H * W, 1)
                x = x * D_q

            residual = x
            x_norm2 = block.norm2(x)
            ffn_out = nlc_to_nchw(x_norm2, hw_shape)
            ffn_out = block.ffn.layers(ffn_out)
            ffn_out = nchw_to_nlc(ffn_out)
            ffn_out = block.ffn.dropout_layer(ffn_out)
            x = residual + ffn_out
            if cum_D is not None:
                x = x * D_q

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs, all_D_raw, all_q_weight, cum_D_list


@MODELS.register_module()
class QualityGatedMiTMamba(BaseSegmentor):
    """..."""

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
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2,
                 skip_phases: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            backbone['pretrained'] = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]
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
        valid = torch.ones(len(data_samples), h, w, dtype=torch.bool, device=device)
        for i, ds in enumerate(data_samples):
            ps = ds.metainfo.get('padding_size', [0, 0, 0, 0])
            pl, pr, pt, pb = ps
            if pb > 0 or pr > 0:
                valid[i, pt:h - pb, pl:w - pr] = True
                if pt > 0: valid[i, :pt, :] = False
                if pb > 0: valid[i, h - pb:, :] = False
                if pl > 0: valid[i, :, :pl] = False
                if pr > 0: valid[i, :, w - pr:] = False
        return valid

    def _get_training_phase(self, epoch):
        if self.phase_mode == 'ratio':
            r = min(epoch / max(self.total_epochs, 1), 1.0)
            if r < 0.1: return 1
            elif r < 0.3: return 2
            return 3
        # absolute (default)
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

    def _mamba_fuse_stage(self, zc, priv, fuse_mod, Dp):
        B, C, H, W = zc.shape
        N_g = H * W
        if Dp.shape[2:] != (H, W): Dp = F.interpolate(Dp.float(), size=(H, W), mode='nearest')
        Ds = (Dp.squeeze(1) > 0.5).float()
        G = zc.permute(0,2,3,1).reshape(B, N_g, C)
        P = priv.permute(0,2,3,1).reshape(B, N_g, C)
        pl, cs = [], [0]
        for b in range(B):
            vi = Ds[b].reshape(-1).nonzero(as_tuple=True)[0]
            Mb = vi.shape[0]
            if Mb == 0:
                # No valid private tokens: add zero placeholder, skip priv in sequence
                pl.append(torch.zeros(0, C, device=P.device, dtype=P.dtype))
                cs.append(cs[-1] + N_g)
            else:
                pl.append(P[b, vi]); cs.append(cs[-1]+N_g+Mb)
        pv = torch.cat(pl, dim=0) if pl[0].shape[0] > 0 else torch.zeros(0, C, device=zc.device, dtype=zc.dtype)
        ct = torch.tensor(cs, dtype=torch.int32, device=zc.device)
        return fuse_mod(G, pv, H, W, ct).reshape(B, H, W, C).permute(0,3,1,2)

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

        zc_outs, D_r, D_t, q_r, q_t, cDr, cDt = _forward_common_dual_pruned(
            self.backbone, torch.cat([rgb, t], dim=0), orig_B=B,
            predictors_rgb=self.predictors_common_rgb,
            predictors_t=self.predictors_common_t,
            gumbel_tau=self.gumbel_tau, training=ug, force_all_keep=fa, phase=ph)
        zc_r = [f[:B] for f in zc_outs]; zc_t = [f[B:] for f in zc_outs]

        zp_r, Dpr, qpr, cDpr = _forward_branch_pruned(
            self.private_branch_rgb, rgb, self.predictors_priv_rgb,
            None, self.gumbel_tau, False, training=ug, force_all_keep=fa, phase=ph)
        for i in range(len(zp_r)):
            if zp_r[i] is not None and Dpr[i] is not None: zp_r[i] = zp_r[i] * Dpr[i]

        zp_t, Dpt, qpt, cDpt = _forward_branch_pruned(
            self.private_branch_t, t, self.predictors_priv_t,
            None, self.gumbel_tau, False, training=ug, force_all_keep=fa, phase=ph)
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
        drr = self._extract_feat_single(dr, di)
        return drr[7], drr[5], drr[6], drr[4], drr[8], drr[9], drr[2], drr[3], drr[0], drr[1], drr[11], drr[12]

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
        return dr, di, dtr, dtt

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
                    Dr = D_r[i] if D_r[i] is not None else torch.ones(B,1,zc_r[i].shape[2],zc_r[i].shape[3],device=zc_r[i].device)
                    Dt = D_t[i] if D_t[i] is not None else torch.ones(B,1,zc_t[i].shape[2],zc_t[i].shape[3],device=zc_t[i].device)
                    qr = q_r[i]
                    qt = q_t[i]
                    lc += compute_cross_modal_contrastive_loss(
                        zc_r[i],zc_t[i],gt,Dr,Dt,qr,qt,
                        tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                        ignore_label=255, pad_mask=pm)
                    cnt += 1
            if cnt: losses['loss_align'] = (lc/cnt)*self.loss_align_weight
        if self.retention_loss_weight > 0: losses['loss_retention'] = self.retention_loss_weight*self._compute_retention_loss(ad)
        if self.training:
            df,drl,dtl,dzf,dqr,dqt,dzpr,dzpt,dzcr,dzct,dDr,dDt = self._train_with_degradation(rgb,ir)
            # NaN guard: if degraded features are NaN, fall back to clean
            if torch.isnan(df[0]).any():
                import logging
                logging.getLogger(__name__).warning(
                    'NaN in degraded features — falling back to clean features for deg losses')
                df,drl,dtl,dzf,dqr,dqt,dzpr,dzpt,dzcr,dzct,dDr,dDt = \
                    ff,re,te,zf,q_r,q_t,zp_r,zp_t,zc_r,zc_t,D_r,D_t
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
                        dDr_i = dDr[i] if dDr is not None and i < len(dDr) and dDr[i] is not None else torch.ones(B,1,dzcr[i].shape[2],dzcr[i].shape[3],device=dzcr[i].device)
                        dDt_i = dDt[i] if dDt is not None and i < len(dDt) and dDt[i] is not None else torch.ones(B,1,dzct[i].shape[2],dzct[i].shape[3],device=dzct[i].device)
                        dqr_i = dqr[i] if dqr is not None and i < len(dqr) and dqr[i] is not None else None
                        dqt_i = dqt[i] if dqt is not None and i < len(dqt) and dqt[i] is not None else None
                        dlc += compute_cross_modal_contrastive_loss(
                            dzcr[i],dzct[i],gt,dDr_i,dDt_i,dqr_i,dqt_i,
                            tau_c=self.contrast_tau,num_samples=self.contrast_num_samples,
                            ignore_label=255, pad_mask=pm)
                        dcnt += 1
                if dcnt: losses['loss_align_deg'] = (dlc/dcnt)*self.loss_align_weight
            if self.loss_distill_weight > 0 and ph >= 3:
                T = self.distill_temperature
                cl = self.decode_head.forward(ff); dl_ = self.decode_head.forward(df)
                tp = F.softmax(cl.detach()/T,dim=1); sp = F.log_softmax(dl_/T,dim=1)
                kl = F.kl_div(sp,tp,reduction='none').sum(dim=1)
                dm = self._build_pad_mask(data_samples, kl.shape[-2], kl.shape[-1], kl.device)
                dm_f = dm.float()
                losses['loss_distill'] = self.loss_distill_weight*(T*T)*(kl*dm_f).sum()/dm_f.sum().clamp(min=1)

            if self.loss_invariant_weight > 0 and ph >= 3:
                inv_loss = torch.tensor(0.0, device=ff[0].device)
                cnt = 0
                all_D = dDr is not None and dDt is not None
                for i in range(len(zf)):
                    if zf[i] is not None and dzf is not None and i < len(dzf) and dzf[i] is not None:
                        if zf[i].shape == dzf[i].shape:
                            # D-based region gate: union of RGB+T masks, both sides
                            Dc = torch.max(
                                D_r[i] if D_r[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                D_t[i] if D_t[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            if all_D:
                                Dd = torch.max(
                                    dDr[i] if dDr[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device),
                                    dDt[i] if dDt[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device))
                            else:
                                Dd = torch.ones_like(Dc)
                            D_gate = ((Dc > 0.5) & (Dd > 0.5)).float()
                            qc = torch.max(
                                q_r[i] if q_r[i] is not None else torch.ones_like(D_gate),
                                q_t[i] if q_t[i] is not None else torch.ones_like(D_gate))
                            qd = torch.max(
                                dqr[i] if dqr[i] is not None else torch.ones_like(D_gate),
                                dqt[i] if dqt[i] is not None else torch.ones_like(D_gate))
                            D_gate = F.interpolate(D_gate, size=zf[i].shape[2:], mode='nearest') if D_gate.shape[2:] != zf[i].shape[2:] else D_gate
                            qc = F.interpolate(qc, size=zf[i].shape[2:], mode='nearest') if qc.shape[2:] != zf[i].shape[2:] else qc
                            qd = F.interpolate(qd, size=zf[i].shape[2:], mode='nearest') if qd.shape[2:] != zf[i].shape[2:] else qd
                            pm_i = self._build_pad_mask(data_samples, zf[i].shape[2], zf[i].shape[3], zf[i].device).float().unsqueeze(1)
                            q_distill = qc * qd * D_gate * pm_i
                            diff = F.smooth_l1_loss(zf[i], dzf[i], reduction='none')
                            denom = q_distill.sum() + 1e-6
                            inv_loss += (q_distill * diff).sum() / denom
                            cnt += 1
                if cnt:
                    losses['loss_inv'] = self.loss_invariant_weight * inv_loss / cnt
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
        # inputs may be (2B, 3, H, W) from vis hook (batch-concat) or (B, 6, H, W)
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

        # align degraded feat spatial sizes to clean counterparts (padding may differ)
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
            deg_rgb_img=deg_rgb, deg_t_img=deg_t,
            deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
            deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
            zc_rgb_deg=zc_r_d, zc_t_deg=zc_t_d,
            zc_fused_deg=zf_d,
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d,
            final_fused_deg=fused_d,
            q_rgb_deg=q_r_d, q_t_deg=q_t_d,
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
        B, C, H, W = rgb.shape
        dev = rgb.device
        rm = self.data_preprocessor.mean[:3].to(dev); rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev); iss = self.data_preprocessor.std[3:].to(dev)
        dr, di = rgb.clone(), ir.clone()
        dtr, dtt = ['none']*B, ['none']*B
        for b in range(B):
            r = random.random()
            if r < 0.3:
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,deg_type='missing',level=5)
                    dtr[b]='missing'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,deg_type='missing',level=5)
                    dtt[b]='missing'
            elif r < 0.6:
                lv = random.choice([1,2,3])
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv); dtr[b]='global'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv); dtt[b]='global'
            else:
                lv = random.choice([1,2,3])
                lm = _generate_local_mask(1,H,W,num_regions=3,device=dev,level=lv)
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv,is_local=True,local_mask=lm)
                    dtr[b]='local'
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv,is_local=True,local_mask=lm)
                    dtt[b]='local'
        return dr, di, dtr, dtt

    def _forward(self, inputs, data_samples=None):
        rgb, t = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,t)[7]
        return self.decode_head.forward(self.neck(ff) if self.with_neck else ff)

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
            pred = i_sl.argmax(dim=0, keepdim=True) if C > 1 else (i_sl.sigmoid() > self.decode_head.threshold).to(i_sl)
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
