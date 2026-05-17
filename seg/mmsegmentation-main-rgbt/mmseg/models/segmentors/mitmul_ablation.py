import math
import os
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import nlc_to_nchw, nchw_to_nlc
from .base import BaseSegmentor
from .swinmul_v12_quality_disentangle import (
    ChannelAttention, SpatialAttention,
)
from mmseg.datasets.transforms.quality_degradation import (
    apply_quality_degradation_rgb, apply_quality_degradation_t,
    _QUALITY_RGB_DEG_TYPES, _QUALITY_T_DEG_TYPES,
)

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class _AblationBase(BaseSegmentor):

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList(
                    [MODELS.build(h) for h in auxiliary_head])
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_aux_heads(self, common_head_cfg, rgb_private_head_cfg,
                        t_private_head_cfg, auxiliary_head):
        self.common_decode_head = None
        self.rgb_private_decode_head = None
        self.t_private_decode_head = None
        if common_head_cfg is not None:
            self.common_decode_head = MODELS.build(common_head_cfg)
        if rgb_private_head_cfg is not None:
            self.rgb_private_decode_head = MODELS.build(rgb_private_head_cfg)
        if t_private_head_cfg is not None:
            self.t_private_decode_head = MODELS.build(t_private_head_cfg)
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList(
                    [MODELS.build(h) for h in auxiliary_head])
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_auxiliary_head(self):
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def _stack_batch_gt(self, data_samples):
        gt_semantic_segs = [
            ds.gt_sem_seg.data for ds in data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    @staticmethod
    def _build_pad_mask(data_samples, h, w, device):
        valid_mask = torch.ones(len(data_samples), h, w, dtype=torch.bool,
                                device=device)
        for i, ds in enumerate(data_samples):
            ps = ds.metainfo.get('padding_size', [0, 0, 0, 0])
            pl, pr, pt, pb = ps
            if pb > 0 or pr > 0:
                valid_mask[i, pt:h - pb, pl:w - pr] = True
                if pt > 0:
                    valid_mask[i, :pt, :] = False
                if pb > 0:
                    valid_mask[i, h - pb:, :] = False
                if pl > 0:
                    valid_mask[i, :, :pl] = False
                if pr > 0:
                    valid_mask[i, :, w - pr:] = False
        return valid_mask

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
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
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2),
                                int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)


class ChannelAttention(BaseModule):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class MLPFusion(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels) if out_channels % 32 != 0 else 32, out_channels),
            nn.GELU())

    def forward(self, x):
        return self.mlp(x)


class FinalFusionModule(BaseModule):
    def __init__(self, in_channels_list):
        super().__init__()
        self.num_stages = len(in_channels_list)
        self.ca_rgb = nn.ModuleList()
        self.sa_rgb = nn.ModuleList()
        self.ca_t = nn.ModuleList()
        self.sa_t = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i, ch in enumerate(in_channels_list):
            self.ca_rgb.append(ChannelAttention(ch))
            self.sa_rgb.append(SpatialAttention())
            self.ca_t.append(ChannelAttention(ch))
            self.sa_t.append(SpatialAttention())
            self.mlps.append(MLPFusion(ch * 2, ch))

    def forward(self, rgb_enhanced_list, t_enhanced_list):
        fused_list = []
        for i in range(self.num_stages):
            rgb_att = self.sa_rgb[i](self.ca_rgb[i](rgb_enhanced_list[i]))
            t_att = self.sa_t[i](self.ca_t[i](t_enhanced_list[i]))
            concat = torch.cat([rgb_att, t_att], dim=1)
            fused = self.mlps[i](concat)
            fused_list.append(fused)
        return fused_list


class MultiScaleRefine(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 3)):
        super().__init__()
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                      groups=channels, bias=False)
            for d in dilations
        ])
        self.fuse_conv = nn.Conv2d(channels * len(dilations), channels, 1, bias=False)
        self.act = nn.GELU()
        mid_ca = max(channels // 4, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ca, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ca, channels, 1, bias=False),
            nn.Sigmoid())
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        multi = [dw(x) for dw in self.dw_convs]
        x = torch.cat(multi, dim=1)
        x = self.fuse_conv(x)
        x = self.act(x)
        ca = self.channel_attn(x)
        x = x * ca
        x = self.out_conv(x)
        return x + residual


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
                nn.Conv2d(ch * 2, ch * 2, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch * 2, ch * 2, 1, bias=False),
                nn.Sigmoid()))
            self.sp_gates.append(nn.Sequential(
                nn.Conv2d(ch * 2, 2, 3, padding=1, bias=False),
                nn.Sigmoid()))
            self.post_norms.append(nn.LayerNorm(ch))
            self.post_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.GELU()))

    def forward(self, rgb_enhanced_list, t_enhanced_list, common_fused_list):
        fused_list = []
        for i in range(self.num_stages):
            F_enh_rgb = rgb_enhanced_list[i]
            F_enh_t = t_enhanced_list[i]
            F_G = common_fused_list[i]
            ref_h, ref_w = F_G.shape[2], F_G.shape[3]
            if F_enh_rgb.shape[2:] != (ref_h, ref_w):
                F_enh_rgb = F.interpolate(F_enh_rgb, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            if F_enh_t.shape[2:] != (ref_h, ref_w):
                F_enh_t = F.interpolate(F_enh_t, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
            B, C, H, W = F_enh_rgb.shape
            concat = torch.cat([F_enh_rgb, F_enh_t], dim=1)
            ch_gate_out = self.ch_gates[i](concat)
            ch_rgb, ch_t = ch_gate_out.split(C, dim=1)
            sp_gate_out = self.sp_gates[i](concat)
            sp_rgb = sp_gate_out[:, 0:1, :, :]
            sp_t = sp_gate_out[:, 1:2, :, :]
            w_rgb = ch_rgb * sp_rgb
            w_t = ch_t * sp_t
            fused_enh = w_rgb * F_enh_rgb + w_t * F_enh_t + F_G
            residual = fused_enh
            x = fused_enh.permute(0, 2, 3, 1)
            x = self.post_norms[i](x)
            x = x.permute(0, 3, 1, 2)
            x = self.post_convs[i](x)
            fused_list.append(residual + x)
        return fused_list


class SimpleConcatFusion(nn.Module):
    """Shallow-stage fusion: concat gen+priv along channel dim, 1x1 conv, LN, GELU, residual."""

    def __init__(self, d_model):
        super().__init__()
        self.fuse_conv = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, common_feat, priv_feat):
        x = torch.cat([common_feat, priv_feat], dim=1)  # (B, 2C, H, W)
        x = self.fuse_conv(x)                            # (B, C, H, W)
        x = self.act(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return common_feat + x  # residual


class BiMambaFusion(nn.Module):
    """Mamba fusion for deep stages (no quality modulation)."""

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
        self.norm = nn.LayerNorm(d_model)

    def _init_mamba_params(self, suffix):
        d_inner = self.d_inner
        d_state = self.d_state
        dt_rank = self.dt_rank

        self.add_module(f'in_proj_{suffix}', nn.Linear(self.d_model, d_inner * 2, bias=False))
        self.add_module(f'conv1d_{suffix}', nn.Conv1d(
            d_inner, d_inner, kernel_size=self.d_conv,
            groups=d_inner, padding=self.d_conv - 1, bias=True))
        self.add_module(f'x_proj_{suffix}', nn.Linear(d_inner, dt_rank + d_state * 2, bias=False))
        self.add_module(f'dt_proj_{suffix}', nn.Linear(dt_rank, d_inner, bias=True))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.register_parameter(f'A_log_{suffix}', nn.Parameter(torch.log(A)))
        self.register_parameter(f'D_{suffix}', nn.Parameter(torch.ones(d_inner)))
        self.add_module(f'out_proj_{suffix}', nn.Linear(d_inner, self.d_model, bias=False))

    def _mamba_scan(self, x_bld, suffix, cu_seqlens=None):
        B_b, L, _ = x_bld.shape
        d_inner = self.d_inner
        d_state = self.d_state
        dt_rank = self.dt_rank

        in_proj = getattr(self, f'in_proj_{suffix}')
        conv1d = getattr(self, f'conv1d_{suffix}')
        x_proj = getattr(self, f'x_proj_{suffix}')
        dt_proj = getattr(self, f'dt_proj_{suffix}')
        A_log = getattr(self, f'A_log_{suffix}')
        D_param = getattr(self, f'D_{suffix}')

        xz = in_proj(x_bld)
        x_proj_in, z = xz.chunk(2, dim=-1)

        x_conv = x_proj_in.transpose(1, 2)
        x_conv = F.silu(conv1d(x_conv)[:, :, :L])
        x_conv = x_conv.transpose(1, 2)

        x_ssm = x_proj(x_conv)
        dt, B_ssm, C_ssm = x_ssm.split([dt_rank, d_state, d_state], dim=-1)
        dt = dt_proj(dt)

        A = -torch.exp(A_log.float())

        x_scan = x_conv.transpose(1, 2)
        dt_scan = dt.transpose(1, 2)
        B_scan = B_ssm.transpose(1, 2)
        C_scan = C_ssm.transpose(1, 2)

        if _MAMBA_AVAILABLE:
            y = selective_scan_fn(
                x_scan, dt_scan, A, B_scan, C_scan, D_param.float(),
                z=z.transpose(1, 2) if False else None,
                delta_bias=dt_proj.bias.float() if dt_proj.bias is not None else None,
                delta_softplus=True)
        else:
            y = self._selective_scan_pytorch(
                x_scan, dt_scan, A, B_scan, C_scan, D_param, cu_seqlens)

        y = y.transpose(1, 2)
        y = y * F.silu(z)
        y = getattr(self, f'out_proj_{suffix}')(y)
        return y

    def _selective_scan_pytorch(self, x, dt, A, B_ssm, C_ssm, D, cu_seqlens=None):
        batch, d_inner, L = x.shape
        d_state = A.shape[1]
        dt = F.softplus(dt)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        cu_seqlens_cpu = cu_seqlens.cpu().numpy() if cu_seqlens is not None else None
        seq_starts = set(cu_seqlens_cpu.tolist()) if cu_seqlens_cpu is not None else set()
        for i in range(L):
            if cu_seqlens_cpu is not None and i in seq_starts and i > 0:
                h = torch.zeros_like(h)
            dA = torch.exp(dt[:, :, i].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, :, i].unsqueeze(-1) * B_ssm[:, :, i].unsqueeze(-2)
            h = dA * h + dB * x[:, :, i].unsqueeze(-1)
            y = torch.sum(h * C_ssm[:, :, i].unsqueeze(1), dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=2)
        y = y + D.unsqueeze(0).unsqueeze(-1) * x
        return y

    def _get_pos_embed(self, H, W, device, dtype):
        pos = F.interpolate(self.pos_table, size=(H, W), mode='bilinear', align_corners=False)
        pos = pos.reshape(self.d_model, H * W).t()
        return pos.to(dtype=dtype)

    def _build_sequences(self, gen, priv, cu_seqlens, N_g):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        B = len(cu_seqlens_cpu) - 1
        x_seq_parts = []
        priv_offset = 0
        for b in range(B):
            start = cu_seqlens_cpu[b]
            end = cu_seqlens_cpu[b + 1]
            M_b = (end - start) - N_g
            x_seq_parts.append(gen[b])
            if M_b > 0:
                x_seq_parts.append(priv[priv_offset:priv_offset + M_b])
            priv_offset += M_b
        x_seq = torch.cat(x_seq_parts, dim=0)
        return x_seq

    def _reverse_sequences(self, x_bld, cu_seqlens):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        x_rev = torch.zeros_like(x_bld)
        for s in range(len(cu_seqlens_cpu) - 1):
            start = cu_seqlens_cpu[s]
            end = cu_seqlens_cpu[s + 1]
            x_rev[0, start:end] = x_bld[0, start:end].flip(0)
        return x_rev

    def _extract_generic(self, out_seq, cu_seqlens, N_g):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        B = len(cu_seqlens_cpu) - 1
        G_list = []
        for b in range(B):
            start = cu_seqlens_cpu[b]
            g_enh = out_seq[start:start + N_g]
            G_list.append(g_enh)
        return torch.stack(G_list, dim=0)

    def forward(self, generic_tokens, priv_valid_tokens,
                stage_H, stage_W, cu_seqlens):
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
        enhanced = generic_tokens + self.out_proj(self.norm(G_enhanced))
        return enhanced


class QualityAwareBiMambaFusion(nn.Module):
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
        self.quality_delta_proj = nn.Linear(1, self.d_inner, bias=True)

        self._init_mamba_params('fwd')
        self._init_mamba_params('bwd')

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _init_mamba_params(self, suffix):
        d_inner = self.d_inner
        d_state = self.d_state
        dt_rank = self.dt_rank

        self.add_module(f'in_proj_{suffix}', nn.Linear(self.d_model, d_inner * 2, bias=False))
        self.add_module(f'conv1d_{suffix}', nn.Conv1d(
            d_inner, d_inner, kernel_size=self.d_conv,
            groups=d_inner, padding=self.d_conv - 1, bias=True))
        self.add_module(f'x_proj_{suffix}', nn.Linear(d_inner, dt_rank + d_state * 2, bias=False))
        self.add_module(f'dt_proj_{suffix}', nn.Linear(dt_rank, d_inner, bias=True))

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.register_parameter(f'A_log_{suffix}', nn.Parameter(torch.log(A)))
        self.register_parameter(f'D_{suffix}', nn.Parameter(torch.ones(d_inner)))
        self.add_module(f'out_proj_{suffix}', nn.Linear(d_inner, self.d_model, bias=False))

    def _mamba_scan(self, x_bld, suffix, delta_bias=None, cu_seqlens=None):
        B_b, L, _ = x_bld.shape
        d_inner = self.d_inner
        d_state = self.d_state
        dt_rank = self.dt_rank

        in_proj = getattr(self, f'in_proj_{suffix}')
        conv1d = getattr(self, f'conv1d_{suffix}')
        x_proj = getattr(self, f'x_proj_{suffix}')
        dt_proj = getattr(self, f'dt_proj_{suffix}')
        A_log = getattr(self, f'A_log_{suffix}')
        D_param = getattr(self, f'D_{suffix}')

        xz = in_proj(x_bld)
        x_proj_in, z = xz.chunk(2, dim=-1)

        x_conv = x_proj_in.transpose(1, 2)
        x_conv = F.silu(conv1d(x_conv)[:, :, :L])
        x_conv = x_conv.transpose(1, 2)

        x_ssm = x_proj(x_conv)
        dt, B_ssm, C_ssm = x_ssm.split([dt_rank, d_state, d_state], dim=-1)
        dt = dt_proj(dt)

        if delta_bias is not None:
            dt = dt + delta_bias

        A = -torch.exp(A_log.float())

        x_scan = x_conv.transpose(1, 2)
        dt_scan = dt.transpose(1, 2)
        B_scan = B_ssm.transpose(1, 2)
        C_scan = C_ssm.transpose(1, 2)

        if _MAMBA_AVAILABLE:
            y = selective_scan_fn(
                x_scan, dt_scan, A, B_scan, C_scan, D_param.float(),
                z=z.transpose(1, 2) if False else None,
                delta_bias=dt_proj.bias.float() if delta_bias is None else None,
                delta_softplus=True)
        else:
            y = self._selective_scan_pytorch(
                x_scan, dt_scan, A, B_scan, C_scan, D_param, cu_seqlens)

        y = y.transpose(1, 2)
        y = y * F.silu(z)
        y = getattr(self, f'out_proj_{suffix}')(y)
        return y

    def _selective_scan_pytorch(self, x, dt, A, B_ssm, C_ssm, D, cu_seqlens=None):
        batch, d_inner, L = x.shape
        d_state = A.shape[1]
        dt = F.softplus(dt)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        cu_seqlens_cpu = cu_seqlens.cpu().numpy() if cu_seqlens is not None else None
        seq_starts = set(cu_seqlens_cpu.tolist()) if cu_seqlens_cpu is not None else set()
        for i in range(L):
            if cu_seqlens_cpu is not None and i in seq_starts and i > 0:
                h = torch.zeros_like(h)
            dA = torch.exp(dt[:, :, i].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, :, i].unsqueeze(-1) * B_ssm[:, :, i].unsqueeze(-2)
            h = dA * h + dB * x[:, :, i].unsqueeze(-1)
            y = torch.sum(h * C_ssm[:, :, i].unsqueeze(1), dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=2)
        y = y + D.unsqueeze(0).unsqueeze(-1) * x
        return y

    def _get_pos_embed(self, H, W, device, dtype):
        pos = F.interpolate(self.pos_table, size=(H, W), mode='bilinear', align_corners=False)
        pos = pos.reshape(self.d_model, H * W).t()
        return pos.to(dtype=dtype)

    def _build_sequences(self, gen, priv, q_weight_priv, cu_seqlens, N_g):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        B = len(cu_seqlens_cpu) - 1
        x_seq_parts = []
        q_seq_parts = []
        priv_offset = 0
        for b in range(B):
            start = cu_seqlens_cpu[b]
            end = cu_seqlens_cpu[b + 1]
            M_b = (end - start) - N_g
            x_seq_parts.append(gen[b])
            q_g = torch.ones(N_g, 1, device=gen.device, dtype=gen.dtype)
            if M_b > 0:
                x_seq_parts.append(priv[priv_offset:priv_offset + M_b])
                q_p = q_weight_priv[priv_offset:priv_offset + M_b]
                q_seq_parts.append(torch.cat([q_g, q_p], dim=0))
            else:
                q_seq_parts.append(q_g)
            priv_offset += M_b
        x_seq = torch.cat(x_seq_parts, dim=0)
        q_seq = torch.cat(q_seq_parts, dim=0)
        return x_seq, q_seq

    def _reverse_sequences(self, x_bld, cu_seqlens):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        x_rev = torch.zeros_like(x_bld)
        for s in range(len(cu_seqlens_cpu) - 1):
            start = cu_seqlens_cpu[s]
            end = cu_seqlens_cpu[s + 1]
            x_rev[0, start:end] = x_bld[0, start:end].flip(0)
        return x_rev

    def _extract_generic(self, out_seq, cu_seqlens, N_g):
        cu_seqlens_cpu = cu_seqlens.cpu().numpy()
        B = len(cu_seqlens_cpu) - 1
        G_list = []
        for b in range(B):
            start = cu_seqlens_cpu[b]
            g_enh = out_seq[start:start + N_g]
            G_list.append(g_enh)
        return torch.stack(G_list, dim=0)

    def forward(self, generic_tokens, priv_valid_tokens, q_weight_priv,
                stage_H, stage_W, cu_seqlens):
        N_g = stage_H * stage_W

        pos_embed = self._get_pos_embed(stage_H, stage_W, generic_tokens.device, generic_tokens.dtype)
        gen = generic_tokens + pos_embed.unsqueeze(0) + self.type_gen.unsqueeze(0).unsqueeze(0)

        priv = priv_valid_tokens
        if priv.shape[0] > 0:
            priv = priv + self.type_priv.unsqueeze(0)

        x_seq, q_seq = self._build_sequences(gen, priv, q_weight_priv, cu_seqlens, N_g)

        delta_bias = self.quality_delta_proj(q_seq)

        x_bld = x_seq.unsqueeze(0)

        out_fwd = self._mamba_scan(x_bld, 'fwd', delta_bias=delta_bias, cu_seqlens=cu_seqlens)

        x_rev = self._reverse_sequences(x_bld, cu_seqlens)
        delta_rev = self._reverse_sequences(delta_bias.unsqueeze(0), cu_seqlens)
        out_rev = self._mamba_scan(x_rev, 'bwd', delta_bias=delta_rev.squeeze(0), cu_seqlens=cu_seqlens)
        out_rev = self._reverse_sequences(out_rev, cu_seqlens)

        out_seq = (out_fwd + out_rev).squeeze(0)

        G_enhanced = self._extract_generic(out_seq, cu_seqlens, N_g)
        enhanced = generic_tokens + self.out_proj(self.norm(G_enhanced))
        return enhanced


def _denorm_to_01(norm_tensor, mean, std):
    raw = norm_tensor * std + mean
    return (raw / 255.0).clamp(0, 1)


def _renorm_from_01(tensor_01, mean, std):
    raw = tensor_01 * 255.0
    return (raw - mean) / std


def _apply_degradation(img_tensor, modality, mean, std,
                       deg_type=None, level=None,
                       is_local=False, local_mask=None):
    B, C, H, W = img_tensor.shape
    img_01 = _denorm_to_01(img_tensor, mean, std)
    if deg_type is None:
        if modality == 'rgb':
            deg_type = random.choice(_QUALITY_RGB_DEG_TYPES)
        else:
            deg_type = random.choice(_QUALITY_T_DEG_TYPES)
    if level is None:
        level = random.randint(2, 5)
    if modality == 'rgb':
        deg_img_01 = apply_quality_degradation_rgb(img_01, deg_type, level)
    else:
        deg_img_01 = apply_quality_degradation_t(img_01, deg_type, level)
    if is_local and local_mask is not None:
        if local_mask.shape[2:] != (H, W):
            local_mask = F.interpolate(local_mask.float(), size=(H, W), mode='nearest')
        local_mask_01 = local_mask.expand(B, C, H, W)
        deg_img_01 = img_01 * (1 - local_mask_01) + deg_img_01 * local_mask_01
    result = _renorm_from_01(deg_img_01, mean, std)
    return result


def _generate_local_mask(B, H, W, num_regions=3, device='cpu'):
    mask = torch.zeros(B, 1, H, W, device=device)
    region_h = H // 4
    region_w = W // 4
    for b in range(B):
        for _ in range(num_regions):
            rh = random.randint(region_h, H // 2)
            rw = random.randint(region_w, W // 2)
            y1 = random.randint(0, H - rh)
            x1 = random.randint(0, W - rw)
            mask[b, 0, y1:y1 + rh, x1:x1 + rw] = 1.0
    return mask


def _lerp(a, b, t):
    return a + (b - a) * t


def _get_degradation_schedule(r):
    if r < 0.1:
        t = r / 0.1
        p_local = 1.0
        p_global = 0.0
        p_missing = 0.0
        local_levels = {2: _lerp(0.7, 0.7, t), 3: _lerp(0.3, 0.3, t),
                        4: 0.0, 5: 0.0}
        global_levels = {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    elif r < 0.3:
        t = (r - 0.1) / 0.2
        p_local = _lerp(1.0, 0.8, t)
        p_global = _lerp(0.0, 0.2, t)
        p_missing = 0.0
        local_levels = {2: _lerp(0.7, 0.1, t), 3: _lerp(0.3, 0.5, t),
                        4: _lerp(0.0, 0.4, t), 5: 0.0}
        global_levels = {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    elif r < 0.5:
        t = (r - 0.3) / 0.2
        p_local = _lerp(0.8, 0.6, t)
        p_global = 0.2
        p_missing = _lerp(0.0, 0.2, t)
        local_levels = {2: 0.0, 3: _lerp(0.5, 0.3, t),
                        4: 0.4, 5: _lerp(0.0, 0.3, t)}
        global_levels = {2: _lerp(1.0, 0.5, t), 3: _lerp(0.0, 0.5, t),
                         4: 0.0, 5: 0.0}
    elif r < 0.7:
        t = (r - 0.5) / 0.2
        p_local = _lerp(0.6, 0.3, t)
        p_global = _lerp(0.2, 0.25, t)
        p_missing = _lerp(0.2, 0.45, t)
        local_levels = {2: 0.0, 3: 0.2, 4: 0.4, 5: 0.4}
        global_levels = {2: 0.3, 3: 0.4, 4: 0.3, 5: 0.0}
    elif r < 0.9:
        p_local = 0.25
        p_global = 0.25
        p_missing = 0.5
        local_levels = {2: 0.0, 3: 0.1, 4: 0.4, 5: 0.5}
        global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
    else:
        p_local = 0.25
        p_global = 0.25
        p_missing = 0.5
        local_levels = {2: 0.0, 3: 0.1, 4: 0.4, 5: 0.5}
        global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
    return {
        'p_local': p_local, 'p_global': p_global, 'p_missing': p_missing,
        'local_levels': local_levels, 'global_levels': global_levels,
    }


def _sample_level(level_dist):
    levels = []
    probs = []
    for lv in [2, 3, 4, 5]:
        p = level_dist.get(lv, 0.0)
        if p > 0:
            levels.append(lv)
            probs.append(p)
    if not levels:
        return 2
    total = sum(probs)
    probs = [p / total for p in probs]
    return random.choices(levels, weights=probs, k=1)[0]


def _compute_feature_variance_penalty(feat):
    B, C, H, W = feat.shape
    feat_mean = feat.mean(dim=[2, 3], keepdim=True)
    variance = ((feat - feat_mean) ** 2).mean(dim=[1, 2, 3])
    return -variance.mean()


def _compute_cross_modal_contrastive_loss(feat_rgb, feat_t, labels, q_rgb, q_t,
                                          tau_q=0.3, tau_c=0.07,
                                          num_samples=512, pad_mask=None):
    B, C, H, W = feat_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
    if labels.shape[1:] != (H, W):
        labels = F.interpolate(labels.unsqueeze(1).float(), size=(H, W),
                               mode='nearest').squeeze(1).long()

    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)
    quality_mask = (labels > 0) & (q_rgb_s > tau_q) & (q_t_s > tau_q)
    if pad_mask is not None:
        if pad_mask.shape[1:] != (H, W):
            pm = F.interpolate(pad_mask.float().unsqueeze(1), size=(H, W),
                               mode='nearest').squeeze(1).bool()
        else:
            pm = pad_mask
        quality_mask = quality_mask & pm

    f_rgb = F.normalize(feat_rgb, dim=1)
    f_t = F.normalize(feat_t, dim=1)

    all_f_rgb = []
    all_f_t = []
    all_labels = []
    all_w = []

    for b in range(B):
        mask_b = quality_mask[b]
        valid_indices = mask_b.nonzero(as_tuple=False)
        if valid_indices.shape[0] == 0:
            continue

        labels_b = labels[b]
        unique_classes = labels_b[mask_b].unique()

        max_per_class = max(1, math.ceil(num_samples / max(len(unique_classes), 1)))
        sampled_indices = []

        for cls in unique_classes:
            cls_mask = (labels_b == cls) & mask_b
            cls_indices = cls_mask.nonzero(as_tuple=False)
            n_cls = cls_indices.shape[0]
            if n_cls <= max_per_class:
                sampled_indices.append(cls_indices)
            else:
                perm = torch.randperm(n_cls, device=cls_indices.device)[:max_per_class]
                sampled_indices.append(cls_indices[perm])

        if len(sampled_indices) == 0:
            continue

        idx_b = torch.cat(sampled_indices, dim=0)
        h_idx, w_idx = idx_b[:, 0], idx_b[:, 1]

        all_f_rgb.append(f_rgb[b, :, h_idx, w_idx].t())
        all_f_t.append(f_t[b, :, h_idx, w_idx].t())
        all_labels.append(labels_b[h_idx, w_idx])

        q_rgb_b = q_rgb_s[b, h_idx, w_idx]
        q_t_b = q_t_s[b, h_idx, w_idx]
        q_min = torch.min(q_rgb_b, q_t_b)
        q_gap = torch.abs(q_rgb_b - q_t_b)
        w_b = q_min * (1.0 - q_gap)
        all_w.append(w_b)

    if len(all_f_rgb) == 0:
        return torch.tensor(0.0, device=feat_rgb.device, requires_grad=True)

    F_rgb = torch.cat(all_f_rgb, dim=0)
    F_t = torch.cat(all_f_t, dim=0)
    L = torch.cat(all_labels, dim=0)
    W_q = torch.cat(all_w, dim=0)

    N = F_rgb.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=feat_rgb.device, requires_grad=True)

    S = (F_rgb @ F_t.t()) / tau_c

    same_class = (L.unsqueeze(0) == L.unsqueeze(1))

    S_masked = S.clone()
    neg_inf = torch.finfo(S.dtype).min
    off_diag_same = same_class & ~torch.eye(N, dtype=torch.bool, device=S.device)
    S_masked[off_diag_same] = neg_inf

    pos_scores = S.diag()
    S_ext = torch.cat([pos_scores.unsqueeze(1), S_masked], dim=1)
    log_denom = torch.logsumexp(S_ext, dim=1)
    loss_per_point = -pos_scores + log_denom

    total_w = W_q.sum() + 1e-8
    loss = (W_q * loss_per_point).sum() / total_w

    return loss


def _compute_smooth_l1_alignment_loss(feat_rgb, feat_t, q_rgb, q_t, threshold=0.3,
                                      pad_mask=None):
    B, C, H, W = feat_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)
    valid_mask = (q_rgb_s > threshold) & (q_t_s > threshold)
    if pad_mask is not None:
        if pad_mask.shape[1:] != (H, W):
            pm = F.interpolate(pad_mask.float().unsqueeze(1), size=(H, W),
                               mode='nearest').squeeze(1).bool()
        else:
            pm = pad_mask
        valid_mask = valid_mask & pm
    q_min = torch.min(q_rgb_s, q_t_s)
    q_gap = torch.abs(q_rgb_s - q_t_s)
    weight = q_min * (1.0 - q_gap) * valid_mask.float()

    diff = feat_rgb - feat_t
    smooth_l1 = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none')
    smooth_l1_per_pixel = smooth_l1.mean(dim=1)

    total_weight = weight.sum() + 1e-8
    loss = (weight * smooth_l1_per_pixel).sum() / total_weight
    return loss


def _forward_mit_k_quality(backbone, img, q_maps, orig_B=None):
    """Standard MiT forward, only K is quality-weighted in attention."""
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        quality_weight_1d = None
        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            q_s = q_i.squeeze(1)

            if orig_B is not None and q_s.shape[0] == orig_B * 2:
                q_rgb = q_s[:orig_B]
                q_t = q_s[orig_B:]
                q_sum = q_rgb + q_t + 1e-8
                w_rgb = (q_rgb / q_sum).clamp(min=0.1)
                w_t = (q_t / q_sum).clamp(min=0.1)
                q_s = torch.cat([w_rgb, w_t], dim=0)
            else:
                q_s = q_s.clamp(min=0.1)

            quality_weight_1d = q_s.reshape(x.shape[0], -1)

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

            B_tok, N_tok, C_tok = x_q.shape
            num_heads = attn_module.num_heads
            head_dim = C_tok // num_heads

            q_proj = attn_module.attn.in_proj_weight[:C_tok]
            k_proj = attn_module.attn.in_proj_weight[C_tok:2*C_tok]
            v_proj = attn_module.attn.in_proj_weight[2*C_tok:]
            q_bias = attn_module.attn.in_proj_bias[:C_tok] if attn_module.attn.in_proj_bias is not None else None
            k_bias = attn_module.attn.in_proj_bias[C_tok:2*C_tok] if attn_module.attn.in_proj_bias is not None else None
            v_bias = attn_module.attn.in_proj_bias[2*C_tok:] if attn_module.attn.in_proj_bias is not None else None

            Q = F.linear(x_q, q_proj, q_bias)
            K = F.linear(x_norm1, k_proj, k_bias)
            V = F.linear(x_kv, v_proj, v_bias)

            if quality_weight_1d is not None:
                K = K * quality_weight_1d.unsqueeze(-1)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)
            else:
                hw_shape_k = hw_shape

            Q = Q.reshape(B_tok, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = attn_module.dropout_layer(attn) if hasattr(attn_module, 'dropout_layer') else attn
            attn_out = (attn @ V).transpose(1, 2).reshape(B_tok, N_tok, C_tok)

            attn_out = attn_module.proj_drop(attn_module.attn.out_proj(attn_out))
            x = residual + attn_out

            residual = x
            x_norm2 = block.norm2(x)
            ffn_out = nlc_to_nchw(x_norm2, hw_shape)
            ffn_out = block.ffn.layers(ffn_out)
            ffn_out = nchw_to_nlc(ffn_out)
            ffn_out = block.ffn.dropout_layer(ffn_out)
            x = residual + ffn_out

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


def _forward_mit_with_ste(backbone, img, q_maps, threshold, orig_B=None):
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        quality_weight_1d = None
        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            q_s = q_i.squeeze(1)  # (total_B, H, W)

            if orig_B is not None and q_s.shape[0] == orig_B * 2:
                q_rgb = q_s[:orig_B]
                q_t = q_s[orig_B:]
                q_sum = q_rgb + q_t + 1e-8
                w_rgb = (q_rgb / q_sum).clamp(min=0.1)
                w_t = (q_t / q_sum).clamp(min=0.1)
                q_s = torch.cat([w_rgb, w_t], dim=0)
            else:
                q_s = q_s.clamp(min=0.1)

            quality_weight_1d = q_s.reshape(x.shape[0], -1)

        if quality_weight_1d is not None:
            x = x * quality_weight_1d.unsqueeze(-1)

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

            B_tok, N_tok, C_tok = x_q.shape
            num_heads = attn_module.num_heads
            head_dim = C_tok // num_heads

            q_proj = attn_module.attn.in_proj_weight[:C_tok]
            k_proj = attn_module.attn.in_proj_weight[C_tok:2*C_tok]
            v_proj = attn_module.attn.in_proj_weight[2*C_tok:]
            q_bias = attn_module.attn.in_proj_bias[:C_tok] if attn_module.attn.in_proj_bias is not None else None
            k_bias = attn_module.attn.in_proj_bias[C_tok:2*C_tok] if attn_module.attn.in_proj_bias is not None else None
            v_bias = attn_module.attn.in_proj_bias[2*C_tok:] if attn_module.attn.in_proj_bias is not None else None

            Q = F.linear(x_q, q_proj, q_bias)
            K = F.linear(x_norm1, k_proj, k_bias)
            V = F.linear(x_kv, v_proj, v_bias)

            if quality_weight_1d is not None:
                K = K * quality_weight_1d.unsqueeze(-1)
                N_v = x_kv.shape[1]
                if N_v == N_tok:
                    V = V * quality_weight_1d.unsqueeze(-1)
                else:
                    H_v = int(N_v ** 0.5)
                    qw_2d = quality_weight_1d.reshape(B_tok, H, W)
                    qw_2d_resized = F.interpolate(
                        qw_2d.unsqueeze(1), size=(H_v, H_v),
                        mode='nearest').squeeze(1)
                    qw_1d_resized = qw_2d_resized.reshape(B_tok, -1)
                    V = V * qw_1d_resized.unsqueeze(-1)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)
            else:
                hw_shape_k = hw_shape

            Q = Q.reshape(B_tok, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = attn_module.dropout_layer(attn) if hasattr(attn_module, 'dropout_layer') else attn
            attn_out = (attn @ V).transpose(1, 2).reshape(B_tok, N_tok, C_tok)

            attn_out = attn_module.proj_drop(attn_module.attn.out_proj(attn_out))

            x = residual + attn_out

            if quality_weight_1d is not None:
                x = x * quality_weight_1d.unsqueeze(-1)

            residual = x
            x_norm2 = block.norm2(x)
            ffn_out = nlc_to_nchw(x_norm2, hw_shape)
            ffn_out = block.ffn.layers(ffn_out)
            ffn_out = nchw_to_nlc(ffn_out)
            ffn_out = block.ffn.dropout_layer(ffn_out)
            x = residual + ffn_out

            if quality_weight_1d is not None:
                x = x * quality_weight_1d.unsqueeze(-1)

        x = norm(x)
        if quality_weight_1d is not None:
            x = x * quality_weight_1d.unsqueeze(-1)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


class WindowCrossAttention(BaseModule):
    def __init__(self, dim, num_heads=4, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, common_feat, private_feat, keep_mask_1d=None):
        B, C, H, W = common_feat.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            common_feat = F.pad(common_feat, (0, pad_w, 0, pad_h))
            private_feat = F.pad(private_feat, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = common_feat.shape

        common_nlc = nchw_to_nlc(common_feat)
        private_nlc = nchw_to_nlc(private_feat)

        Q = self.q_proj(common_nlc)
        K = self.k_proj(private_nlc)
        V = self.v_proj(private_nlc)

        low_mask_1d = None
        if keep_mask_1d is not None:
            if keep_mask_1d.shape[1] != Hp * Wp:
                km_2d = keep_mask_1d.reshape(B, H, W)
                km_2d = F.pad(km_2d, (0, pad_w, 0, pad_h), value=0.0)
                keep_mask_1d = km_2d.reshape(B, -1)
            low_mask_1d = (keep_mask_1d < 0.5)

        Q = Q.reshape(B, Hp * Wp, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, Hp * Wp, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, Hp * Wp, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        nH_q = Hp // ws
        nW_q = Wp // ws
        Q_win = Q.reshape(B, self.num_heads, nH_q, ws, nW_q, ws, self.head_dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, nH_q * nW_q, ws * ws, self.head_dim)
        K_win = K.reshape(B, self.num_heads, nH_q, ws, nW_q, ws, self.head_dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, nH_q * nW_q, ws * ws, self.head_dim)
        V_win = V.reshape(B, self.num_heads, nH_q, ws, nW_q, ws, self.head_dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, nH_q * nW_q, ws * ws, self.head_dim)

        attn = (Q_win @ K_win.transpose(-2, -1)) * self.scale

        if low_mask_1d is not None:
            low_mask_win = low_mask_1d.reshape(B, nH_q, ws, nW_q, ws).permute(0, 1, 3, 2, 4).reshape(B, nH_q * nW_q, ws * ws)
            attn_bias = torch.zeros_like(attn)
            attn_bias.masked_fill_(low_mask_win.unsqueeze(1).unsqueeze(2), -10000.0)
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        out = (attn @ V_win)
        out = out.reshape(B, self.num_heads, nH_q, nW_q, ws, ws, self.head_dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, Hp * Wp, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        residual = common_nlc + out
        residual = self.norm(residual)
        result = nlc_to_nchw(residual, (Hp, Wp))
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :H, :W]
        return result


def _selective_scan_seq(x, dt, A, B, C, D):
    B_b, N, D_d = x.shape
    d_state = A.shape[1]
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    h = x.new_zeros(B_b, D_d, d_state)
    ys = []
    for t in range(N):
        h = dA[:, t] * h + dBx[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    y = y + D.unsqueeze(0).unsqueeze(0) * x
    return y


class CrossScanSSMFusion(BaseModule):
    def __init__(self, dim, d_state=16, dt_rank='auto', num_directions=4):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = dim // 16 if dt_rank == 'auto' else dt_rank
        self.num_directions = num_directions

        self.dt_proj = nn.Linear(dim, self.dt_rank)
        self.B_proj = nn.Linear(dim, d_state)
        self.C_proj = nn.Linear(dim, d_state)
        self.A_log = nn.Parameter(torch.log(torch.ones(dim, d_state)))
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_up_proj = nn.Linear(self.dt_rank, dim)

        self.private_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim * num_directions, dim)
        self.norm = nn.LayerNorm(dim)

    def _get_ssm_params(self, common_nlc, quality_weight_1d=None):
        dt = self.dt_proj(common_nlc)
        if quality_weight_1d is not None:
            dt = dt * quality_weight_1d.unsqueeze(-1)
        dt_full = F.softplus(self.dt_up_proj(dt))
        B_mat = self.B_proj(common_nlc)
        C_mat = self.C_proj(common_nlc)
        A = -torch.exp(self.A_log)
        return dt_full, A, B_mat, C_mat

    def _scan_4dir(self, x, dt, A, B_mat, C_mat, H, W):
        B_b = x.shape[0]

        y1 = _selective_scan_seq(x, dt, A, B_mat, C_mat, self.D)

        y2 = _selective_scan_seq(x.flip(1), dt.flip(1), A, B_mat.flip(1), C_mat.flip(1), self.D)
        y2 = y2.flip(1)

        x_col = x.reshape(B_b, H, W, -1).permute(0, 2, 1, 3).reshape(B_b, W * H, -1)
        dt_col = dt.reshape(B_b, H, W, -1).permute(0, 2, 1, 3).reshape(B_b, W * H, -1)
        B_col = B_mat.reshape(B_b, H, W, -1).permute(0, 2, 1, 3).reshape(B_b, W * H, -1)
        C_col = C_mat.reshape(B_b, H, W, -1).permute(0, 2, 1, 3).reshape(B_b, W * H, -1)

        y3 = _selective_scan_seq(x_col, dt_col, A, B_col, C_col, self.D)
        y3 = y3.reshape(B_b, W, H, -1).permute(0, 2, 1, 3).reshape(B_b, H * W, -1)

        y4 = _selective_scan_seq(x_col.flip(1), dt_col.flip(1), A, B_col.flip(1), C_col.flip(1), self.D)
        y4 = y4.flip(1).reshape(B_b, W, H, -1).permute(0, 2, 1, 3).reshape(B_b, H * W, -1)

        return y1, y2, y3, y4

    def forward(self, common_feat, private_feat, quality_weight_1d=None):
        B, C_dim, H, W = common_feat.shape

        common_nlc = nchw_to_nlc(common_feat)
        private_nlc = nchw_to_nlc(private_feat)

        dt, A, B_mat, C_mat = self._get_ssm_params(common_nlc, quality_weight_1d)

        x = self.private_proj(private_nlc)

        y1, y2, y3, y4 = self._scan_4dir(x, dt, A, B_mat, C_mat, H, W)

        y = torch.cat([y1, y2, y3, y4], dim=-1)
        y = self.out_proj(y)

        result = self.norm(common_nlc + y)
        return nlc_to_nchw(result, (H, W))


@MODELS.register_module()
class MiTMulABBaseline(_AblationBase):
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone_rgb = MODELS.build(backbone)
        self.backbone_t = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def init_weights(self):
        self.backbone_rgb.init_weights()
        self.backbone_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def extract_feat(self, input_rgb, input_t):
        x_rgb = self.backbone_rgb(input_rgb)
        x_t = self.backbone_t(input_t)
        num_stages = len(x_rgb)
        fused = [(x_rgb[i] + x_t[i]) / 2.0 for i in range(num_stages)]
        if self.with_neck:
            fused = self.neck(fused)
        return fused

    def _extract_feat(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t
        x_rgb_list = self.backbone_rgb(input_rgb)
        x_t_list = self.backbone_t(input_t)
        num_stages = len(x_rgb_list)
        fused_list = [(x_rgb_list[i] + x_t_list[i]) / 2.0 for i in range(num_stages)]
        return (x_rgb_list, x_t_list, fused_list,
                has_rgb, has_t, both_present)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        fused = self.extract_feat(input_rgb, input_t)
        seg_logits = self.decode_head.predict(fused, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        fused = self.extract_feat(input_rgb, input_t)
        losses = self._decode_head_forward_train(fused, data_samples)
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for idx, aux_head in enumerate(self.auxiliary_head):
                    loss_aux = aux_head.loss(fused, data_samples, self.train_cfg)
                    losses.update(add_prefix(loss_aux, f'aux_{idx}'))
            else:
                loss_aux = self.auxiliary_head.loss(fused, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        fused = self.extract_feat(input_rgb, input_t)
        return self.decode_head.forward(fused)


@MODELS.register_module()
class MiTMulABV3(_AblationBase):
    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.1,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.03,
                 loss_distill_weight: float = 0.3,
                 distill_temperature: float = 4.0,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.win_cross_attn_rgb = nn.ModuleList()
        self.win_cross_attn_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            n_heads_i = max(1, ch // 64)
            self.win_cross_attn_rgb.append(
                WindowCrossAttention(ch, num_heads=n_heads_i, window_size=7))
            self.win_cross_attn_t.append(
                WindowCrossAttention(ch, num_heads=n_heads_i, window_size=7))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}', logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys', logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _get_keep_mask_1d(self, q_2d, H, W, B):
        if q_2d.shape[2:] != (H, W):
            q_2d = F.interpolate(q_2d, size=(H, W), mode='nearest')
        q_s = q_2d.squeeze(1).clamp(min=0.1)
        quality_weight_1d = q_s.reshape(B, -1)
        return quality_weight_1d

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_k_quality(
                self.backbone, input_rgbt, q_rgbt_maps, orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_k_quality(
                self.private_branch_rgb, input_rgb, q_rgb_maps)
        if has_t.any():
            zp_t_list = _forward_mit_k_quality(
                self.private_branch_t, input_t, q_t_maps)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(B, self.embed_dims_list[i],
                                       zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                                       zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                                       device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                H_i, W_i = zp_rgb_list[i].shape[2], zp_rgb_list[i].shape[3]
                keep_mask_1d_rgb = self._get_keep_mask_1d(q_rgb_maps[i], H_i, W_i, B)
                rgb_enhanced = self.win_cross_attn_rgb[i](
                    zc_fused, zp_rgb_list[i], keep_mask_1d=keep_mask_1d_rgb)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                H_i, W_i = zp_t_list[i].shape[2], zp_t_list[i].shape[3]
                keep_mask_1d_t = self._get_keep_mask_1d(q_t_maps[i], H_i, W_i, B)
                t_enhanced = self.win_cross_attn_t[i](
                    zc_fused, zp_t_list[i], keep_mask_1d=keep_mask_1d_t)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _compute_quality_masked_seg_loss(self, feats, data_samples, head, q_maps, threshold):
        losses = head.loss(feats, data_samples, self.train_cfg)
        if q_maps is not None and len(q_maps) > 0:
            q = q_maps[0]
            high_ratio = (q >= threshold).float().mean().clamp(min=0.1)
            for k in list(losses.keys()):
                losses[k] = losses[k] * high_ratio
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_feats = self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg
            deg_rgb_logits = self._decode_head_predict_logits(
                deg_rgb_feats, self.rgb_private_decode_head)
            loss_distill_rgb = self._compute_distill_loss(
                clean_rgb_logits, deg_rgb_logits)
            q_rgb_scale = (q_rgb_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_rgb'] = (
                loss_distill_rgb * self.loss_distill_weight * q_rgb_scale)
            loss_rgb_deg = self._compute_quality_masked_seg_loss(
                deg_rgb_feats, data_samples, self.rgb_private_decode_head,
                q_rgb_deg, self.quality_threshold)
            losses.update(add_prefix(loss_rgb_deg, 'rgb_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_feats = self.neck(t_enh_deg) if self.with_neck else t_enh_deg
            deg_t_logits = self._decode_head_predict_logits(
                deg_t_feats, self.t_private_decode_head)
            loss_distill_t = self._compute_distill_loss(
                clean_t_logits, deg_t_logits)
            q_t_scale = (q_t_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_t'] = (
                loss_distill_t * self.loss_distill_weight * q_t_scale)
            loss_t_deg = self._compute_quality_masked_seg_loss(
                deg_t_feats, data_samples, self.t_private_decode_head,
                q_t_deg, self.quality_threshold)
            losses.update(add_prefix(loss_t_deg, 't_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('t_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits



class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto',
                 quality_cond=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.quality_cond = quality_cond

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        if quality_cond:
            self.q_gate_proj = nn.Sequential(
                nn.Linear(1, d_model, bias=True),
                nn.Sigmoid())

    def _selective_scan(self, x, dt, B_ssm, C_ssm, D, cu_seqlens=None):
        if cu_seqlens is not None and _MAMBA_AVAILABLE:
            A = -torch.exp(self.A_log)
            return selective_scan_fn(
                x, dt, A, B_ssm, C_ssm, D,
                cu_seqlens=cu_seqlens,
                delta_softplus=True)
        return self._selective_scan_pytorch(x, dt, B_ssm, C_ssm, D, cu_seqlens)

    def _selective_scan_pytorch(self, x, dt, B_ssm, C_ssm, D, cu_seqlens=None):
        batch, L, d_inner = x.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log)
        dt = F.softplus(dt)

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        if cu_seqlens is not None:
            cu_seqlens_cpu = cu_seqlens.cpu().numpy()
            seq_starts = set(cu_seqlens_cpu.tolist())

        for i in range(L):
            if cu_seqlens is not None and i in seq_starts and i > 0:
                h = torch.zeros_like(h)

            dA = torch.exp(dt[:, i, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = dt[:, i, :].unsqueeze(-1) * B_ssm[:, i, :].unsqueeze(-2)
            h = dA * h + dB * x[:, i, :].unsqueeze(-1)
            y = torch.sum(h * C_ssm[:, i, :].unsqueeze(1), dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + D.unsqueeze(0).unsqueeze(0) * x
        return y

    def forward(self, x, cu_seqlens=None, quality_scores=None, dt_bias=None):
        use_varlen = (cu_seqlens is not None)
        if x.dim() == 2:
            T, C = x.shape
            x = x.unsqueeze(0)
            if quality_scores is not None:
                quality_scores = quality_scores.unsqueeze(0)
            if dt_bias is not None:
                dt_bias = dt_bias.unsqueeze(0)
            was_2d = True
        else:
            was_2d = False

        residual = x
        x = self.norm(x)

        if self.quality_cond and quality_scores is not None:
            q_gate = self.q_gate_proj(quality_scores)
            x = x * q_gate

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.act(self.conv1d(x_conv)[:, :, :x.shape[1]])
        x_conv = x_conv.transpose(1, 2)

        x_ssm = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        if dt_bias is not None:
            dt = dt + dt_bias

        if use_varlen:
            cu_seqlens_cpu = cu_seqlens.cpu().numpy()
            y_fwd = self._selective_scan(x_conv, dt, B_ssm, C_ssm, self.D, cu_seqlens)
            x_rev = torch.zeros_like(x_conv)
            for s in range(len(cu_seqlens_cpu) - 1):
                start = cu_seqlens_cpu[s]
                end = cu_seqlens_cpu[s + 1]
                x_rev[0, start:end] = x_conv[0, start:end].flip(0)
            y_rev = self._selective_scan(x_rev, dt, B_ssm, C_ssm, self.D, cu_seqlens)
            for s in range(len(cu_seqlens_cpu) - 1):
                start = cu_seqlens_cpu[s]
                end = cu_seqlens_cpu[s + 1]
                y_rev[0, start:end] = y_rev[0, start:end].flip(0)
            y = y_fwd + y_rev
        else:
            y_fwd = self._selective_scan(x_conv, dt, B_ssm, C_ssm, self.D)
            x_rev = x_conv.flip(1)
            y_rev = self._selective_scan(x_rev, dt, B_ssm, C_ssm, self.D)
            y_rev = y_rev.flip(1)
            y = y_fwd + y_rev

        y = y * F.silu(z)
        output = self.out_proj(y)

        if was_2d:
            output = output.squeeze(0)
            residual = residual.squeeze(0)

        return output + residual


class PrivateCommonMambaFusion(nn.Module):
    def __init__(self, d_model, num_layers=2, d_state=16, d_conv=4, expand=2,
                 quality_cond=True):
        super().__init__()
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         quality_cond=quality_cond)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cu_seqlens=None, quality_scores=None):
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, quality_scores=quality_scores)
        return self.norm(x)


@MODELS.register_module()
class MiTMulABV8(_AblationBase):
    """V8: V7 backbone (attn bias) + BiMamba fusion for common+private.

    Key difference from V7:
    - Backbone: same as V7 (attention bias for low-quality K)
    - Fusion: instead of concat+MLP, uses BiMamba to fuse common and private
      - Common tokens G + valid private tokens P_valid (quality-gated)
      - Add modality type embeddings (e_gen, e_priv_rgb, e_priv_t)
      - Add spatial position embeddings (per stage)
      - Variable-length sequence with cu_seqlens for BiMamba
      - Extract enhanced common tokens after BiMamba
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 mamba_layers: int = 2,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.mamba_fuse_rgb = nn.ModuleList()
        self.mamba_fuse_t = nn.ModuleList()

        for i, ch in enumerate(self.embed_dims_list):
            self.mamba_fuse_rgb.append(QualityAwareBiMambaFusion(
                d_model=ch, d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand))
            self.mamba_fuse_t.append(QualityAwareBiMambaFusion(
                d_model=ch, d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True

    def _load_quality_pretrained(self):
        if self._quality_pretrained and os.path.exists(self._quality_pretrained):
            state = torch.load(self._quality_pretrained, map_location='cpu')
            if 'state_dict' in state:
                state = state['state_dict']
            q_state = {k.replace('quality_pyramid_net.', ''): v
                       for k, v in state.items()
                       if k.startswith('quality_pyramid_net.')}
            self.quality_pyramid_net.load_state_dict(q_state, strict=False)
            print_log(f'Loaded quality pretrained from {self._quality_pretrained}',
                      logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _mamba_fuse_stage(self, common_feat, priv_feat, q_map,
                          mamba_module, threshold):
        B, C, H, W = common_feat.shape
        N_g = H * W

        if q_map.shape[2:] != (H, W):
            q_map = F.interpolate(q_map, size=(H, W), mode='nearest')
        q_s = q_map.squeeze(1)

        high_mask = (q_s >= threshold)

        G = common_feat.permute(0, 2, 3, 1).reshape(B, N_g, C)
        P = priv_feat.permute(0, 2, 3, 1).reshape(B, N_g, C)

        priv_valid_list = []
        q_weight_priv_list = []
        cu_seqlens = [0]
        valid_counts = []

        for b in range(B):
            mask_b = high_mask[b].reshape(-1)
            valid_idx = mask_b.nonzero(as_tuple=True)[0]
            M_b = valid_idx.shape[0]
            valid_counts.append(M_b)

            p_valid_b = P[b, valid_idx]
            priv_valid_list.append(p_valid_b)

            q_p = q_s[b].reshape(-1)[valid_idx].unsqueeze(-1)
            q_weight_priv_list.append(q_p)

            cu_seqlens.append(cu_seqlens[-1] + N_g + M_b)

        if len(priv_valid_list) > 0 and priv_valid_list[0].shape[0] > 0:
            priv_valid_tokens = torch.cat(priv_valid_list, dim=0)
            q_weight_priv = torch.cat(q_weight_priv_list, dim=0)
        else:
            priv_valid_tokens = torch.zeros(0, C, device=common_feat.device, dtype=common_feat.dtype)
            q_weight_priv = torch.zeros(0, 1, device=common_feat.device, dtype=common_feat.dtype)

        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32,
                                         device=common_feat.device)

        G_enhanced = mamba_module(G, priv_valid_tokens, q_weight_priv,
                                  H, W, cu_seqlens_tensor)
        G_enhanced = G_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return G_enhanced

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_attn_bias(
                self.backbone, input_rgbt, q_rgbt_maps,
                self.quality_threshold, orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_attn_bias(
                self.private_branch_rgb, input_rgb, q_rgb_maps,
                self.quality_threshold)
        if has_t.any():
            zp_t_list = _forward_mit_attn_bias(
                self.private_branch_t, input_t, q_t_maps,
                self.quality_threshold)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                ref_feat = zp_rgb_list[i] if zp_rgb_list[i] is not None else zp_t_list[i]
                if ref_feat is not None:
                    zc_fused = torch.zeros(
                        B, self.embed_dims_list[i],
                        ref_feat.shape[2], ref_feat.shape[3],
                        device=input_rgb.device)
                else:
                    zc_fused = torch.zeros(
                        B, self.embed_dims_list[i], 1, 1,
                        device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_enhanced = self._mamba_fuse_stage(
                    zc_fused, zp_rgb_list[i], q_rgb_maps[i],
                    self.mamba_fuse_rgb[i],
                    self.quality_threshold)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_enhanced = self._mamba_fuse_stage(
                    zc_fused, zp_t_list[i], q_t_maps[i],
                    self.mamba_fuse_t[i],
                    self.quality_threshold)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        B = input_rgb.shape[0]
        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        self._update_quality_freeze_status(epoch)

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = clean_results

        losses = {}

        seg_label = self._stack_batch_gt(data_samples)
        for ds in data_samples:
            if hasattr(ds, 'gt_sem_seg'):
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)

        main_loss = self.decode_head.loss(final_fused, data_samples, self.train_cfg)
        losses.update(add_prefix(main_loss, 'decode'))

        if self.common_decode_head is not None and zc_fused_list is not None:
            common_loss = self.common_decode_head.loss(
                zc_fused_list, data_samples, self.train_cfg)
            losses.update(add_prefix(common_loss, 'common_decode'))

        if self.rgb_private_decode_head is not None and zp_rgb_list is not None:
            rgb_priv_loss = self.rgb_private_decode_head.loss(
                zp_rgb_list, data_samples, self.train_cfg)
            losses.update(add_prefix(rgb_priv_loss, 'rgb_private_decode'))

        if self.t_private_decode_head is not None and zp_t_list is not None:
            t_priv_loss = self.t_private_decode_head.loss(
                zp_t_list, data_samples, self.train_cfg)
            losses.update(add_prefix(t_priv_loss, 't_private_decode'))

        if self.loss_align_weight > 0 and both_present.any():
            gt_labels = seg_label.squeeze(1).long()
            pad_mask = self._build_pad_mask(
                data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            loss_contrast = 0.0
            loss_smooth = 0.0
            count = 0
            for i in range(len(zc_rgb_list)):
                if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                    loss_contrast += _compute_cross_modal_contrastive_loss(
                        zc_rgb_list[i], zc_t_list[i], gt_labels,
                        q_rgb_maps[i], q_t_maps[i],
                        tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                        num_samples=self.contrast_num_samples,
                        pad_mask=pad_mask)
                    loss_smooth += _compute_smooth_l1_alignment_loss(
                        zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i],
                        threshold=self.quality_threshold,
                        pad_mask=pad_mask)
                    count += 1
            if count > 0:
                losses['loss_align'] = (loss_contrast / count) * self.loss_align_weight
                losses['loss_smooth_align'] = (loss_smooth / count) * self.loss_smooth_align_weight

        if self.loss_invariant_weight > 0:
            inv_loss = self._compute_invariant_loss(
                rgb_enhanced_list, t_enhanced_list, both_present)
            losses['loss_invariant'] = self.loss_invariant_weight * inv_loss

        if self.loss_distill_weight > 0:
            distill_loss = self._compute_distill_loss(
                final_fused, zc_fused_list, zp_rgb_list, zp_t_list,
                both_present)
            losses['loss_distill'] = self.loss_distill_weight * distill_loss

        if self.missing_ratio > 0 and self.training:
            deg_results = self._train_with_degradation(input_rgb, input_ir)
            if deg_results is not None:
                deg_final, deg_rgb_list, deg_t_list, deg_both = deg_results
                deg_loss = self.decode_head.loss(
                    deg_final, data_samples, self.train_cfg)
                losses['loss_deg'] = self.aux_loss_weight * sum(deg_loss.values())

        return losses

    def _compute_invariant_loss(self, rgb_enhanced, t_enhanced, both_present):
        loss = torch.tensor(0.0, device=rgb_enhanced[0].device)
        count = 0
        for r, t in zip(rgb_enhanced, t_enhanced):
            if r is not None and t is not None and r.shape == t.shape:
                diff = (r - t) ** 2
                if both_present.any():
                    mask = both_present.float().view(-1, 1, 1, 1)
                    loss = loss + (diff * mask).sum() / (mask.sum() * r.shape[1] + 1e-8)
                else:
                    loss = loss + diff.mean()
                count += 1
        return loss / max(count, 1)

    def _compute_distill_loss(self, final_fused, zc_fused, zp_rgb, zp_t,
                              both_present):
        loss = torch.tensor(0.0, device=final_fused[0].device)
        count = 0
        for i in range(len(final_fused)):
            if final_fused[i].shape != zc_fused[i].shape:
                continue
            diff = (final_fused[i].detach() - zc_fused[i]) ** 2
            loss = loss + diff.mean()
            count += 1
        return loss / max(count, 1)

    def _get_deg_schedule(self, r):
        if r < 0.1:
            t = r / 0.1
            p_local = 1.0
            p_global = 0.0
            p_missing = 0.0
            local_levels = {2: 0.7, 3: 0.3}
            global_levels = {2: 1.0}
        elif r < 0.3:
            t = (r - 0.1) / 0.2
            p_local = 1.0 - 0.2 * t
            p_global = 0.2 * t
            p_missing = 0.0
            local_levels = {
                2: 0.7 - 0.6 * t,
                3: 0.3 + 0.2 * t,
                4: 0.4 * t}
            global_levels = {2: 1.0}
        elif r < 0.5:
            t = (r - 0.3) / 0.2
            p_local = 0.8 - 0.2 * t
            p_global = 0.2
            p_missing = 0.2 * t
            local_levels = {
                3: 0.5 - 0.2 * t,
                4: 0.4,
                5: 0.3 * t}
            global_levels = {
                2: 1.0 - 0.5 * t,
                3: 0.5 * t}
        elif r < 0.7:
            t = (r - 0.5) / 0.2
            p_local = 0.6 - 0.3 * t
            p_global = 0.2 + 0.05 * t
            p_missing = 0.2 + 0.25 * t
            local_levels = {3: 0.2, 4: 0.4, 5: 0.4}
            global_levels = {2: 0.3, 3: 0.4, 4: 0.3}
        elif r < 0.9:
            p_local = 0.25
            p_global = 0.25
            p_missing = 0.5
            local_levels = {3: 0.1, 4: 0.4, 5: 0.5}
            global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
        else:
            p_local = 0.25
            p_global = 0.25
            p_missing = 0.5
            local_levels = {3: 0.1, 4: 0.4, 5: 0.5}
            global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
        return p_local, p_global, p_missing, local_levels, global_levels

    @staticmethod
    def _sample_level(level_dist):
        levels = list(level_dist.keys())
        probs = [level_dist[l] for l in levels]
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(levels, weights=probs, k=1)[0]

    def _train_with_degradation(self, input_rgb, input_ir):
        deg_result = self._generate_degraded_inputs(input_rgb, input_ir)
        if deg_result is None:
            return None
        deg_input_rgb, deg_input_ir = deg_result[0], deg_result[1]

        deg_results = self._extract_feat_single(deg_input_rgb, deg_input_ir)
        return (deg_results[7], deg_results[5], deg_results[6], deg_results[12],
                deg_results[4])

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        max_epochs = 200
        r = min(epoch / max_epochs, 1.0)
        p_local, p_global, p_missing, local_levels, global_levels = \
            self._get_deg_schedule(r)

        deg_rgb = input_rgb.clone()
        deg_ir = input_ir.clone()
        deg_type_rgb = ['none'] * B
        deg_type_t = ['none'] * B
        any_deg = False

        for b in range(B):
            rand_val = random.random()
            if rand_val < p_missing:
                if random.random() < 0.5:
                    deg_rgb[b] = 0.0
                    deg_type_rgb[b] = 'missing'
                else:
                    deg_ir[b] = 0.0
                    deg_type_t[b] = 'missing'
                any_deg = True
            elif rand_val < p_missing + p_global:
                level = self._sample_level(global_levels)
                if random.random() < 0.5:
                    deg_rgb[b:b+1] = _apply_degradation(
                        input_rgb[b:b+1], 'rgb', rgb_mean, rgb_std,
                        level=level)
                    deg_type_rgb[b] = f'degraded_l{level}'
                else:
                    deg_ir[b:b+1] = _apply_degradation(
                        input_ir[b:b+1], 'thermal', ir_mean, ir_std,
                        level=level)
                    deg_type_t[b] = f'degraded_l{level}'
                any_deg = True
            else:
                level = self._sample_level(local_levels)
                local_mask = _generate_local_mask(1, H, W, num_regions=3,
                                                  device=device)
                if random.random() < 0.5:
                    deg_rgb[b:b+1] = _apply_degradation(
                        input_rgb[b:b+1], 'rgb', rgb_mean, rgb_std,
                        deg_type=None, level=level,
                        is_local=True, local_mask=local_mask)
                    deg_type_rgb[b] = f'local_l{level}'
                else:
                    deg_ir[b:b+1] = _apply_degradation(
                        input_ir[b:b+1], 'thermal', ir_mean, ir_std,
                        deg_type=None, level=level,
                        is_local=True, local_mask=local_mask)
                    deg_type_t[b] = f'local_l{level}'
                any_deg = True

        if not any_deg:
            return None
        return deg_rgb, deg_ir, deg_type_rgb, deg_type_t

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_ir.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _extract_feat(self, input_rgb, input_t):
        return self._extract_feat_single(input_rgb, input_t)

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)



class TokenPrunePredictor(nn.Module):
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True))
        self.gate_head = nn.Conv2d(mid_channels, 2, 1, bias=True)
        self.weight_head = nn.Conv2d(mid_channels, 1, 1, bias=True)

    def forward(self, x):
        feat = self.shared(x)
        gate_logits = self.gate_head(feat)
        q_weight = torch.sigmoid(self.weight_head(feat))
        return gate_logits, q_weight


def _gumbel_softmax_hard(gate_logits, tau=1.0, training=True):
    B, C2, H, W = gate_logits.shape
    gate_logits_2d = gate_logits.permute(0, 2, 3, 1).reshape(-1, 2)
    if training:
        y_soft = F.gumbel_softmax(gate_logits_2d, tau=tau, hard=False)
    else:
        y_soft = F.softmax(gate_logits_2d / tau, dim=-1)
    hard_idx = y_soft.argmax(dim=-1, keepdim=True)
    hard_onehot = torch.zeros_like(y_soft).scatter_(1, hard_idx, 1.0)
    if training:
        y_hard = (hard_onehot - y_soft).detach() + y_soft
    else:
        y_hard = hard_onehot
    y_hard = y_hard.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    y_soft = y_soft.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    D_raw = 1.0 - y_hard[:, 1:2, :, :]
    y_soft_keep = y_soft[:, 0:1, :, :]
    D_raw = D_raw + y_soft_keep - y_soft_keep.detach()
    return D_raw, y_soft_keep


def _complementary_fix(D_rgb_raw, D_t_raw, q_rgb_weight, q_t_weight):
    both_zero = (D_rgb_raw < 0.5) & (D_t_raw < 0.5)
    if not both_zero.any():
        return D_rgb_raw.clone(), D_t_raw.clone()
    rgb_better = q_rgb_weight >= q_t_weight
    D_rgb_fixed = D_rgb_raw.clone()
    D_t_fixed = D_t_raw.clone()
    fix_rgb = both_zero & rgb_better
    fix_t = both_zero & ~rgb_better
    D_rgb_fixed = torch.where(fix_rgb, torch.ones_like(D_rgb_fixed), D_rgb_fixed)
    D_t_fixed = torch.where(fix_rgb, torch.zeros_like(D_t_fixed), D_t_fixed)
    D_rgb_fixed = torch.where(fix_t, torch.zeros_like(D_rgb_fixed), D_rgb_fixed)
    D_t_fixed = torch.where(fix_t, torch.ones_like(D_t_fixed), D_t_fixed)
    D_rgb = D_rgb_fixed + D_rgb_raw - D_rgb_raw.detach()
    D_t = D_t_fixed + D_t_raw - D_t_raw.detach()
    return D_rgb, D_t


def _downsample_mask(D, H_k, W_k):
    if D.shape[2] == H_k and D.shape[3] == W_k:
        return D
    return F.interpolate(D.float(), size=(H_k, W_k), mode='nearest')


def _forward_branch_pruned(backbone, img, predictor_list, cum_D_prev_list,
                           gumbel_tau, is_common,
                           other_D_list=None, other_q_list=None,
                           training=True, force_all_keep=False):
    outs = []
    all_D_raw = []
    all_q_weight = []
    cum_D_list = []

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape
        B_tok = x.shape[0]

        D_raw = None
        q_weight = None
        cum_D = None

        if force_all_keep:
            D_raw = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
            q_weight = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictor_list) and predictor_list[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            gate_logits, q_weight = predictor_list[i](x_2d.detach())
            D_raw, _ = _gumbel_softmax_hard(
                gate_logits, tau=gumbel_tau, training=training)

            if is_common and other_D_list is not None and i < len(other_D_list) and other_D_list[i] is not None:
                other_D_i = other_D_list[i]
                if other_D_i.shape[2:] != (H, W):
                    other_D_i = F.interpolate(other_D_i.float(), size=(H, W), mode='nearest')
                other_q_i = other_q_list[i]
                if other_q_i.shape[2:] != (H, W):
                    other_q_i = F.interpolate(other_q_i.float(), size=(H, W), mode='nearest')
                D_raw, _ = _complementary_fix(D_raw, other_D_i, q_weight, other_q_i)

            if cum_D_prev_list is not None and i < len(cum_D_prev_list) and cum_D_prev_list[i] is not None:
                prev_D = cum_D_prev_list[i]
                if prev_D.shape[2:] != (H, W):
                    prev_D = F.interpolate(prev_D.float(), size=(H, W), mode='nearest')
                cum_D = D_raw * prev_D
            else:
                cum_D = D_raw

        all_D_raw.append(D_raw)
        all_q_weight.append(q_weight)
        cum_D_list.append(cum_D)

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
            num_heads = attn_module.num_heads
            head_dim = C_tok // num_heads

            q_proj_w = attn_module.attn.in_proj_weight[:C_tok]
            k_proj_w = attn_module.attn.in_proj_weight[C_tok:2*C_tok]
            v_proj_w = attn_module.attn.in_proj_weight[2*C_tok:]
            q_bias = attn_module.attn.in_proj_bias[:C_tok] if attn_module.attn.in_proj_bias is not None else None
            k_bias = attn_module.attn.in_proj_bias[C_tok:2*C_tok] if attn_module.attn.in_proj_bias is not None else None
            v_bias = attn_module.attn.in_proj_bias[2*C_tok:] if attn_module.attn.in_proj_bias is not None else None

            Q = F.linear(x_q, q_proj_w, q_bias)
            K = F.linear(x_norm1, k_proj_w, k_bias)
            V = F.linear(x_kv, v_proj_w, v_bias)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)
            else:
                hw_shape_k = hw_shape

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            if cum_D is not None:
                H_k, W_k = hw_shape_k
                D_k = _downsample_mask(cum_D, H_k, W_k)
                D_k_1d = D_k.reshape(Bb, 1, 1, -1)
                attn = attn + torch.log(D_k_1d + 1e-9)

            attn = attn.softmax(dim=-1)
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


def _forward_common_dual_pruned(backbone, input_rgbt, orig_B,
                                predictors_rgb, predictors_t,
                                gumbel_tau, training=True,
                                force_all_keep=False):
    outs = []
    all_D_rgb = []
    all_D_t = []
    all_q_rgb = []
    all_q_t = []
    cum_D_rgb_list = []
    cum_D_t_list = []

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(input_rgbt)
        H, W = hw_shape
        B_tok = x.shape[0]

        D_rgb_raw = None
        D_t_raw = None
        q_rgb = None
        q_t = None
        cum_D_rgb = None
        cum_D_t = None

        if force_all_keep:
            D_rgb_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            D_t_raw = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            q_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
            q_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        elif i < len(predictors_rgb) and predictors_rgb[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            x_rgb_2d = x_2d[:orig_B]
            x_t_2d = x_2d[orig_B:]

            gate_logits_rgb, q_rgb = predictors_rgb[i](x_rgb_2d.detach())
            D_rgb_raw, _ = _gumbel_softmax_hard(
                gate_logits_rgb, tau=gumbel_tau, training=training)

            gate_logits_t, q_t = predictors_t[i](x_t_2d.detach())
            D_t_raw, _ = _gumbel_softmax_hard(
                gate_logits_t, tau=gumbel_tau, training=training)

            D_rgb_fixed, D_t_fixed = _complementary_fix(
                D_rgb_raw, D_t_raw, q_rgb, q_t)
            D_rgb_raw = D_rgb_fixed
            D_t_raw = D_t_fixed

        if cum_D_rgb_list and i < len(cum_D_rgb_list) and cum_D_rgb_list[i] is not None:
            prev_D_rgb = cum_D_rgb_list[i]
            if prev_D_rgb.shape[2:] != (H, W):
                prev_D_rgb = F.interpolate(prev_D_rgb.float(), size=(H, W), mode='nearest')
            cum_D_rgb = D_rgb_raw * prev_D_rgb if D_rgb_raw is not None else None
        else:
            cum_D_rgb = D_rgb_raw

        if cum_D_t_list and i < len(cum_D_t_list) and cum_D_t_list[i] is not None:
            prev_D_t = cum_D_t_list[i]
            if prev_D_t.shape[2:] != (H, W):
                prev_D_t = F.interpolate(prev_D_t.float(), size=(H, W), mode='nearest')
            cum_D_t = D_t_raw * prev_D_t if D_t_raw is not None else None
        else:
            cum_D_t = D_t_raw

        all_D_rgb.append(D_rgb_raw)
        all_D_t.append(D_t_raw)
        all_q_rgb.append(q_rgb)
        all_q_t.append(q_t)
        cum_D_rgb_list.append(cum_D_rgb)
        cum_D_t_list.append(cum_D_t)

        cum_D = None
        if cum_D_rgb is not None and cum_D_t is not None:
            D_rgb_full = torch.cat([cum_D_rgb, torch.ones_like(cum_D_t)], dim=0)
            D_t_full = torch.cat([torch.ones_like(cum_D_rgb), cum_D_t], dim=0)
            cum_D = torch.min(D_rgb_full, D_t_full)

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
            num_heads = attn_module.num_heads
            head_dim = C_tok // num_heads

            q_proj_w = attn_module.attn.in_proj_weight[:C_tok]
            k_proj_w = attn_module.attn.in_proj_weight[C_tok:2*C_tok]
            v_proj_w = attn_module.attn.in_proj_weight[2*C_tok:]
            q_bias = attn_module.attn.in_proj_bias[:C_tok] if attn_module.attn.in_proj_bias is not None else None
            k_bias = attn_module.attn.in_proj_bias[C_tok:2*C_tok] if attn_module.attn.in_proj_bias is not None else None
            v_bias = attn_module.attn.in_proj_bias[2*C_tok:] if attn_module.attn.in_proj_bias is not None else None

            Q = F.linear(x_q, q_proj_w, q_bias)
            K = F.linear(x_norm1, k_proj_w, k_bias)
            V = F.linear(x_kv, v_proj_w, v_bias)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)
            else:
                hw_shape_k = hw_shape

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            if cum_D is not None:
                H_k, W_k = hw_shape_k
                D_k = _downsample_mask(cum_D, H_k, W_k)
                D_k_1d = D_k.reshape(Bb, 1, 1, -1)
                attn = attn + torch.log(D_k_1d + 1e-9)

            attn = attn.softmax(dim=-1)
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

    return outs, all_D_rgb, all_D_t, all_q_rgb, all_q_t


class QualityModulatedMambaFusion(nn.Module):
    def __init__(self, d_model, num_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         quality_cond=False)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.q_to_dt_bias = nn.Linear(1, self.d_inner, bias=True)

    def forward(self, x, cu_seqlens=None, quality_scores=None):
        if quality_scores is not None:
            dt_bias = self.q_to_dt_bias(quality_scores)
        else:
            dt_bias = None
        for layer in self.layers:
            x = layer(x, cu_seqlens=cu_seqlens, dt_bias=dt_bias)
        return self.norm(x)


@MODELS.register_module()
class MiTMulABV9(_AblationBase):
    """V9: Token pruning (DynamicViT-style) + quality-modulated Mamba fusion.

    Key innovations over V8:
    - No external quality network; lightweight TokenPrunePredictor per stage
    - Gumbel-Softmax hard gating with STE for differentiable pruning
    - Complementary mask correction for common branch (at least one modality kept)
    - Attention masking via log(D_k + eps) in every Transformer block
    - Hard zeroing after attention and FFN in every block
    - Cross-stage mask propagation (downsample + logical AND)
    - Quality-modulated Mamba fusion (q_weight -> delta bias)
    - Retention rate regularization loss
    - Three-phase training strategy
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
                 prune_mid_channels: int = 32,
                 gumbel_tau_init: float = 1.0,
                 gumbel_tau_min: float = 0.1,
                 gumbel_tau_decay: float = 0.995,
                 retention_min: float = 0.3,
                 retention_max: float = 0.7,
                 retention_loss_weight: float = 0.01,
                 phase1_epochs: int = 10,
                 phase2_epochs: int = 20,
                 loss_align_weight: float = 0.1,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.03,
                 loss_distill_weight: float = 0.3,
                 distill_temperature: float = 4.0,
                 aux_loss_weight: float = 0.3,
                 total_epochs: int = 200,
                 mamba_layers: int = 2,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.predictors_common_rgb = nn.ModuleList([
            TokenPrunePredictor(ch, prune_mid_channels)
            for ch in self.embed_dims_list])
        self.predictors_common_t = nn.ModuleList([
            TokenPrunePredictor(ch, prune_mid_channels)
            for ch in self.embed_dims_list])
        self.predictors_priv_rgb = nn.ModuleList([
            TokenPrunePredictor(ch, prune_mid_channels)
            for ch in self.embed_dims_list])
        self.predictors_priv_t = nn.ModuleList([
            TokenPrunePredictor(ch, prune_mid_channels)
            for ch in self.embed_dims_list])

        self.fuse_rgb = nn.ModuleList()
        self.fuse_t = nn.ModuleList()

        for i, ch in enumerate(self.embed_dims_list):
            if i < 2:
                # Shallow stages (0-1): simple concat + 1x1 conv fusion
                self.fuse_rgb.append(SimpleConcatFusion(ch))
                self.fuse_t.append(SimpleConcatFusion(ch))
            else:
                # Deep stages (2-3): bidirectional mamba fusion (no quality)
                self.fuse_rgb.append(BiMambaFusion(
                    d_model=ch, d_state=mamba_d_state, d_conv=mamba_d_conv,
                    expand=mamba_expand))
                self.fuse_t.append(BiMambaFusion(
                    d_model=ch, d_state=mamba_d_state, d_conv=mamba_d_conv,
                    expand=mamba_expand))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.gumbel_tau = gumbel_tau_init
        self.gumbel_tau_min = gumbel_tau_min
        self.gumbel_tau_decay = gumbel_tau_decay
        self.retention_min = retention_min
        self.retention_max = retention_max
        self.retention_loss_weight = retention_loss_weight
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.total_epochs = total_epochs

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def _get_training_phase(self, epoch):
        if epoch < self.phase1_epochs:
            return 1
        elif epoch < self.phase1_epochs + self.phase2_epochs:
            return 2
        else:
            return 3

    def _update_training_phase(self, epoch):
        phase = self._get_training_phase(epoch)
        if phase == 1:
            for pred_list in [self.predictors_common_rgb, self.predictors_common_t,
                              self.predictors_priv_rgb, self.predictors_priv_t]:
                for pred in pred_list:
                    for p in pred.parameters():
                        p.requires_grad = False
        elif phase == 2:
            for pred_list in [self.predictors_common_rgb, self.predictors_common_t,
                              self.predictors_priv_rgb, self.predictors_priv_t]:
                for pred in pred_list:
                    for p in pred.shared.parameters():
                        p.requires_grad = False
                    for p in pred.gate_head.parameters():
                        p.requires_grad = False
                    for p in pred.weight_head.parameters():
                        p.requires_grad = True
        else:
            for pred_list in [self.predictors_common_rgb, self.predictors_common_t,
                              self.predictors_priv_rgb, self.predictors_priv_t]:
                for pred in pred_list:
                    for p in pred.parameters():
                        p.requires_grad = True
        if self.training and phase >= 3:
            self.gumbel_tau = max(self.gumbel_tau * self.gumbel_tau_decay,
                                  self.gumbel_tau_min)
        return phase

    def _compute_retention_loss(self, all_D_raw):
        loss = torch.tensor(0.0, device=all_D_raw[0].device if all_D_raw[0] is not None else 'cpu')
        count = 0
        for D in all_D_raw:
            if D is None:
                continue
            r = D.mean()
            if r < self.retention_min:
                loss = loss + (self.retention_min - r) ** 2
            elif r > self.retention_max:
                loss = loss + (r - self.retention_max) ** 2
            count += 1
        if count > 0:
            loss = loss / count
        return loss

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _simple_fuse_stage(self, common_feat, priv_feat, fuse_module, D_gen):
        """Shallow-stage fusion: priv zero-masked, then concat+1x1 conv."""
        B, C, H, W = common_feat.shape
        if D_gen.shape[2:] != (H, W):
            D_gen = F.interpolate(D_gen.float(), size=(H, W), mode='nearest')

        priv_masked = priv_feat * D_gen  # zero out invalid positions
        enhanced = fuse_module(common_feat, priv_masked)
        enhanced = enhanced * D_gen       # hard zeroing safety net
        return enhanced

    def _mamba_fuse_stage(self, common_feat, priv_feat, fuse_module, D_gen):
        """Deep-stage fusion: variable-length mamba (no quality modulation)."""
        B, C, H, W = common_feat.shape
        N_g = H * W

        if D_gen.shape[2:] != (H, W):
            D_gen = F.interpolate(D_gen.float(), size=(H, W), mode='nearest')
        D_gen_s = (D_gen.squeeze(1) > 0.5).float()

        G = common_feat.permute(0, 2, 3, 1).reshape(B, N_g, C)
        P = priv_feat.permute(0, 2, 3, 1).reshape(B, N_g, C)

        priv_valid_list = []
        cu_seqlens = [0]

        for b in range(B):
            mask_b = D_gen_s[b].reshape(-1)
            valid_idx = mask_b.nonzero(as_tuple=True)[0]
            M_b = valid_idx.shape[0]

            if M_b == 0:
                valid_idx = torch.arange(N_g, device=mask_b.device)
                M_b = N_g

            p_valid_b = P[b, valid_idx]
            priv_valid_list.append(p_valid_b)

            cu_seqlens.append(cu_seqlens[-1] + N_g + M_b)

        if len(priv_valid_list) > 0 and priv_valid_list[0].shape[0] > 0:
            priv_valid_tokens = torch.cat(priv_valid_list, dim=0)
        else:
            priv_valid_tokens = torch.zeros(0, C, device=common_feat.device, dtype=common_feat.dtype)

        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32,
                                         device=common_feat.device)

        G_enhanced = fuse_module(G, priv_valid_tokens, H, W, cu_seqlens_tensor)
        G_enhanced = G_enhanced.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if D_gen.shape[2:] != (H, W):
            D_gen = F.interpolate(D_gen.float(), size=(H, W), mode='nearest')
        G_enhanced = G_enhanced * D_gen

        return G_enhanced

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        phase = self._get_training_phase(epoch)
        force_all_keep = (phase < 3)
        use_gumbel = self.training and not force_all_keep

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4
        D_rgb_list = [None] * 4
        D_t_list = [None] * 4
        q_rgb_list = [None] * 4
        q_t_list = [None] * 4
        D_rgb_priv_list = [None] * 4
        D_t_priv_list = [None] * 4
        q_rgb_priv_list = [None] * 4
        q_t_priv_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)

            zc_rgbt_outs, D_rgb_raw, D_t_raw, q_rgb_raw, q_t_raw = \
                _forward_common_dual_pruned(
                    self.backbone, input_rgbt, orig_B=B,
                    predictors_rgb=self.predictors_common_rgb,
                    predictors_t=self.predictors_common_t,
                    gumbel_tau=self.gumbel_tau,
                    training=use_gumbel,
                    force_all_keep=force_all_keep)

            zc_rgb_list = [feat[:B] for feat in zc_rgbt_outs]
            zc_t_list = [feat[B:] for feat in zc_rgbt_outs]
            D_rgb_list = D_rgb_raw
            D_t_list = D_t_raw
            q_rgb_list = q_rgb_raw
            q_t_list = q_t_raw

        if has_rgb.any():
            zp_rgb_outs, D_rgb_priv_raw, q_rgb_priv_raw, _ = _forward_branch_pruned(
                self.private_branch_rgb, input_rgb,
                self.predictors_priv_rgb,
                cum_D_prev_list=None,
                gumbel_tau=self.gumbel_tau,
                is_common=False,
                training=use_gumbel,
                force_all_keep=force_all_keep)
            zp_rgb_list = zp_rgb_outs
            D_rgb_priv_list = D_rgb_priv_raw
            q_rgb_priv_list = q_rgb_priv_raw

            for i in range(len(zp_rgb_list)):
                if zp_rgb_list[i] is not None and D_rgb_priv_list[i] is not None:
                    zp_rgb_list[i] = zp_rgb_list[i] * D_rgb_priv_list[i]

        if has_t.any():
            zp_t_outs, D_t_priv_raw, q_t_priv_raw, _ = _forward_branch_pruned(
                self.private_branch_t, input_t,
                self.predictors_priv_t,
                cum_D_prev_list=None,
                gumbel_tau=self.gumbel_tau,
                is_common=False,
                training=use_gumbel,
                force_all_keep=force_all_keep)
            zp_t_list = zp_t_outs
            D_t_priv_list = D_t_priv_raw
            q_t_priv_list = q_t_priv_raw

            for i in range(len(zp_t_list)):
                if zp_t_list[i] is not None and D_t_priv_list[i] is not None:
                    zp_t_list[i] = zp_t_list[i] * D_t_priv_list[i]

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                q_rgb_i = q_rgb_list[i] if q_rgb_list[i] is not None else torch.ones(
                    B, 1, zc_rgb_list[i].shape[2], zc_rgb_list[i].shape[3],
                    device=zc_rgb_list[i].device)
                q_t_i = q_t_list[i] if q_t_list[i] is not None else torch.ones(
                    B, 1, zc_t_list[i].shape[2], zc_t_list[i].shape[3],
                    device=zc_t_list[i].device)
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_i, q_t_i)
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(
                    B, self.embed_dims_list[i],
                    zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                    zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                    device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            D_gen_rgb = D_rgb_list[i] if D_rgb_list[i] is not None else torch.ones(
                B, 1, zc_fused.shape[2], zc_fused.shape[3], device=zc_fused.device)
            D_gen_t = D_t_list[i] if D_t_list[i] is not None else torch.ones(
                B, 1, zc_fused.shape[2], zc_fused.shape[3], device=zc_fused.device)
            D_gen = torch.min(D_gen_rgb, D_gen_t)

            q_rgb_priv_i = q_rgb_priv_list[i] if q_rgb_priv_list[i] is not None else torch.ones(
                B, 1, zc_fused.shape[2], zc_fused.shape[3], device=zc_fused.device)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                if i < 2:
                    rgb_enhanced = self._simple_fuse_stage(
                        zc_fused, zp_rgb_list[i], self.fuse_rgb[i], D_gen)
                else:
                    rgb_enhanced = self._mamba_fuse_stage(
                        zc_fused, zp_rgb_list[i], self.fuse_rgb[i], D_gen)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            q_t_priv_i = q_t_priv_list[i] if q_t_priv_list[i] is not None else torch.ones(
                B, 1, zc_fused.shape[2], zc_fused.shape[3], device=zc_fused.device)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                if i < 2:
                    t_enhanced = self._simple_fuse_stage(
                        zc_fused, zp_t_list[i], self.fuse_t[i], D_gen)
                else:
                    t_enhanced = self._mamba_fuse_stage(
                        zc_fused, zp_t_list[i], self.fuse_t[i], D_gen)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        all_D_for_loss = []
        for d_list in [D_rgb_list, D_t_list, D_rgb_priv_list, D_t_priv_list]:
            all_D_for_loss.extend(d_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_list, q_t_list,
                has_rgb, has_t, both_present,
                all_D_for_loss,
                D_rgb_list, D_t_list,
                D_rgb_priv_list, D_t_priv_list,
                q_rgb_priv_list, q_t_priv_list)

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        B = input_rgb.shape[0]
        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        phase = self._get_training_phase(epoch)
        self._update_training_phase(epoch)

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_list, q_t_list,
         has_rgb, has_t, both_present,
         all_D_for_loss,
         D_rgb_list, D_t_list,
         D_rgb_priv_list, D_t_priv_list,
         q_rgb_priv_list, q_t_priv_list) = clean_results

        losses = {}

        seg_label = self._stack_batch_gt(data_samples)
        for ds in data_samples:
            if hasattr(ds, 'gt_sem_seg'):
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)

        main_loss = self.decode_head.loss(final_fused, data_samples, self.train_cfg)
        losses.update(add_prefix(main_loss, 'decode'))

        if self.common_decode_head is not None and zc_fused_list is not None:
            common_loss = self.common_decode_head.loss(
                zc_fused_list, data_samples, self.train_cfg)
            losses.update(add_prefix(common_loss, 'common_decode'))

        if self.rgb_private_decode_head is not None and zp_rgb_list is not None:
            rgb_priv_loss = self.rgb_private_decode_head.loss(
                zp_rgb_list, data_samples, self.train_cfg)
            losses.update(add_prefix(rgb_priv_loss, 'rgb_private_decode'))

        if self.t_private_decode_head is not None and zp_t_list is not None:
            t_priv_loss = self.t_private_decode_head.loss(
                zp_t_list, data_samples, self.train_cfg)
            losses.update(add_prefix(t_priv_loss, 't_private_decode'))

        if self.loss_align_weight > 0 and both_present.any():
            gt_labels = seg_label.squeeze(1).long()
            pad_mask = self._build_pad_mask(
                data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            loss_contrast = 0.0
            count = 0
            for i in range(len(zc_rgb_list)):
                if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                    q_rgb_i = q_rgb_list[i] if q_rgb_list[i] is not None else torch.ones(
                        B, 1, zc_rgb_list[i].shape[2], zc_rgb_list[i].shape[3],
                        device=zc_rgb_list[i].device)
                    q_t_i = q_t_list[i] if q_t_list[i] is not None else torch.ones(
                        B, 1, zc_t_list[i].shape[2], zc_t_list[i].shape[3],
                        device=zc_t_list[i].device)
                    loss_contrast += _compute_cross_modal_contrastive_loss(
                        zc_rgb_list[i], zc_t_list[i], gt_labels,
                        q_rgb_i, q_t_i,
                        tau_q=0.3, tau_c=self.contrast_tau,
                        num_samples=self.contrast_num_samples,
                        pad_mask=pad_mask)
                    count += 1
            if count > 0:
                losses['loss_align'] = (loss_contrast / count) * self.loss_align_weight

        if self.retention_loss_weight > 0:
            ret_loss = self._compute_retention_loss(all_D_for_loss)
            losses['loss_retention'] = self.retention_loss_weight * ret_loss

        if self.training:
            deg_results = self._train_with_degradation(input_rgb, input_ir)
            deg_final, deg_rgb_list, deg_t_list, deg_both, deg_zc_fused, \
                deg_q_rgb_list, deg_q_t_list = deg_results

            deg_seg_loss = self.decode_head.loss(
                deg_final, data_samples, self.train_cfg)
            losses['loss_deg_seg'] = sum(deg_seg_loss.values())

            if self.loss_distill_weight > 0 and phase >= 3:
                clean_logits = self.decode_head.forward(final_fused)
                deg_logits = self.decode_head.forward(deg_final)
                T = self.distill_temperature
                teacher_prob = F.softmax(clean_logits.detach() / T, dim=1)
                student_log_prob = F.log_softmax(deg_logits / T, dim=1)
                kl_per_pixel = F.kl_div(student_log_prob, teacher_prob,
                                        reduction='none').sum(dim=1)
                deg_conf = torch.max(
                    deg_q_rgb_list[-1] if deg_q_rgb_list[-1] is not None else torch.ones(
                        B, 1, kl_per_pixel.shape[1], kl_per_pixel.shape[2],
                        device=kl_per_pixel.device),
                    deg_q_t_list[-1] if deg_q_t_list[-1] is not None else torch.ones(
                        B, 1, kl_per_pixel.shape[1], kl_per_pixel.shape[2],
                        device=kl_per_pixel.device))
                deg_conf_s = deg_conf.squeeze(1)
                if deg_conf_s.shape != kl_per_pixel.shape:
                    deg_conf_s = F.interpolate(
                        deg_conf_s.unsqueeze(1).float(), size=kl_per_pixel.shape[1:],
                        mode='nearest').squeeze(1)
                mask = (deg_conf_s > self.quality_threshold).float()
                if mask.sum() > 0:
                    kl_masked = (kl_per_pixel * mask).sum() / mask.sum()
                else:
                    kl_masked = kl_per_pixel.mean()
                losses['loss_distill'] = self.loss_distill_weight * (T * T) * kl_masked

            if self.loss_invariant_weight > 0 and phase >= 3:
                q_fused_clean = []
                q_fused_degraded = []
                for i in range(len(zc_fused_list)):
                    q_c = torch.max(
                        q_rgb_list[i] if q_rgb_list[i] is not None else torch.ones(
                            B, 1, zc_fused_list[i].shape[2], zc_fused_list[i].shape[3],
                            device=zc_fused_list[i].device),
                        q_t_list[i] if q_t_list[i] is not None else torch.ones(
                            B, 1, zc_fused_list[i].shape[2], zc_fused_list[i].shape[3],
                            device=zc_fused_list[i].device))
                    q_d = torch.max(
                        deg_q_rgb_list[i] if deg_q_rgb_list[i] is not None else torch.ones(
                            B, 1, zc_fused_list[i].shape[2], zc_fused_list[i].shape[3],
                            device=zc_fused_list[i].device),
                        deg_q_t_list[i] if deg_q_t_list[i] is not None else torch.ones(
                            B, 1, zc_fused_list[i].shape[2], zc_fused_list[i].shape[3],
                            device=zc_fused_list[i].device))
                    q_fused_clean.append(q_c)
                    q_fused_degraded.append(q_d)
                inv_loss = torch.tensor(0.0, device=zc_fused_list[0].device)
                count = 0
                for i in range(len(zc_fused_list)):
                    if zc_fused_list[i] is not None and deg_zc_fused[i] is not None:
                        if zc_fused_list[i].shape == deg_zc_fused[i].shape:
                            q_c = q_fused_clean[i]
                            q_d = q_fused_degraded[i]
                            if q_c.shape[2:] != zc_fused_list[i].shape[2:]:
                                q_c = F.interpolate(q_c, size=zc_fused_list[i].shape[2:],
                                                    mode='nearest')
                            if q_d.shape[2:] != zc_fused_list[i].shape[2:]:
                                q_d = F.interpolate(q_d, size=zc_fused_list[i].shape[2:],
                                                    mode='nearest')
                            q_distill = q_c * q_d
                            diff = F.smooth_l1_loss(
                                zc_fused_list[i].detach(), deg_zc_fused[i],
                                reduction='none')
                            inv_loss = inv_loss + (q_distill * diff).sum() / (q_distill.sum() + 1e-6)
                            count += 1
                if count > 0:
                    losses['loss_inv'] = self.loss_invariant_weight * inv_loss / count

        return losses

    def _compute_invariant_loss(self, rgb_enhanced, t_enhanced, both_present):
        loss = torch.tensor(0.0, device=rgb_enhanced[0].device)
        count = 0
        for r, t in zip(rgb_enhanced, t_enhanced):
            if r is not None and t is not None and r.shape == t.shape:
                diff = (r - t) ** 2
                if both_present.any():
                    mask = both_present.float().view(-1, 1, 1, 1)
                    loss = loss + (diff * mask).sum() / (mask.sum() * r.shape[1] + 1e-8)
                else:
                    loss = loss + diff.mean()
                count += 1
        return loss / max(count, 1)

    def _compute_distill_loss(self, final_fused, zc_fused, zp_rgb, zp_t,
                              both_present):
        loss = torch.tensor(0.0, device=final_fused[0].device)
        count = 0
        for i in range(len(final_fused)):
            if final_fused[i].shape != zc_fused[i].shape:
                continue
            diff = (final_fused[i].detach() - zc_fused[i]) ** 2
            loss = loss + diff.mean()
            count += 1
        return loss / max(count, 1)

    def _train_with_degradation(self, input_rgb, input_ir):
        deg_input_rgb, deg_input_ir = self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_input_rgb, deg_input_ir)
        return (deg_results[7], deg_results[5], deg_results[6], deg_results[12], deg_results[4],
                deg_results[9], deg_results[10])

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_ir = input_ir.clone()

        epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        r = min(epoch / max(self.total_epochs, 1), 1.0)
        schedule = _get_degradation_schedule(r)
        p_local = schedule['p_local']
        p_global = schedule['p_global']
        p_missing = schedule['p_missing']
        local_levels = schedule['local_levels']
        global_levels = schedule['global_levels']

        for b in range(B):
            rand = random.random()
            if rand < p_missing:
                if random.random() < 0.5:
                    deg_rgb[b] = 0.0
                else:
                    deg_ir[b] = 0.0
            elif rand < p_missing + p_global:
                level = _sample_level(global_levels)
                if random.random() < 0.5:
                    deg_rgb[b:b+1] = _apply_degradation(
                        input_rgb[b:b+1], 'rgb', rgb_mean, rgb_std, level=level)
                else:
                    deg_ir[b:b+1] = _apply_degradation(
                        input_ir[b:b+1], 'thermal', ir_mean, ir_std, level=level)
            else:
                level = _sample_level(local_levels)
                local_mask = _generate_local_mask(1, H, W, num_regions=3, device=device)
                if random.random() < 0.5:
                    deg_rgb[b:b+1] = _apply_degradation(
                        input_rgb[b:b+1], 'rgb', rgb_mean, rgb_std,
                        level=level, is_local=True, local_mask=local_mask)
                else:
                    deg_ir[b:b+1] = _apply_degradation(
                        input_ir[b:b+1], 'thermal', ir_mean, ir_std,
                        level=level, is_local=True, local_mask=local_mask)

        return deg_rgb, deg_ir

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_ir.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_list, q_t_list,
         has_rgb, has_t, both_present,
         all_D_for_loss,
         D_rgb_list, D_t_list,
         D_rgb_priv_list, D_t_priv_list,
         q_rgb_priv_list, q_t_priv_list) = results

        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _extract_feat(self, input_rgb, input_t):
        return self._extract_feat_single(input_rgb, input_t)

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)




@MODELS.register_module()
class MiTMulABV7(_AblationBase):
    """V7: Attention bias for low-quality K tokens in every block.

    - No token replacement
    - In every attention block, after computing attention scores,
      add -10000 bias to columns corresponding to low-quality K tokens
    - Applied to all 3 branches (common, private_rgb, private_t)
    - Fusion: V1-style concat+MLP for common+private
    - Quality net frozen, loss same as V3
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.common_private_fuse_rgb = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])
        self.common_private_fuse_t = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}',
                      logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys',
                  logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_attn_bias(
                self.backbone, input_rgbt, q_rgbt_maps,
                self.quality_threshold, orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_attn_bias(
                self.private_branch_rgb, input_rgb, q_rgb_maps,
                self.quality_threshold)
        if has_t.any():
            zp_t_list = _forward_mit_attn_bias(
                self.private_branch_t, input_t, q_t_maps,
                self.quality_threshold)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(
                    B, self.embed_dims_list[i],
                    zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                    zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                    device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_concat = torch.cat([zp_rgb_list[i], zc_fused], dim=1)
                rgb_enhanced = self.common_private_fuse_rgb[i](rgb_concat)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_concat = torch.cat([zp_t_list[i], zc_fused], dim=1)
                t_enhanced = self.common_private_fuse_t[i](t_concat)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _compute_quality_masked_seg_loss(self, feats, data_samples, head, q_maps, threshold):
        losses = head.loss(feats, data_samples, self.train_cfg)
        if q_maps is not None and len(q_maps) > 0:
            q = q_maps[0]
            high_ratio = (q >= threshold).float().mean().clamp(min=0.1)
            for k in list(losses.keys()):
                losses[k] = losses[k] * high_ratio
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_feats = self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg
            deg_rgb_logits = self._decode_head_predict_logits(
                deg_rgb_feats, self.rgb_private_decode_head)
            loss_distill_rgb = self._compute_distill_loss(
                clean_rgb_logits, deg_rgb_logits)
            q_rgb_scale = (q_rgb_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_rgb'] = (
                loss_distill_rgb * self.loss_distill_weight * q_rgb_scale)
            loss_rgb_deg = self._compute_quality_masked_seg_loss(
                deg_rgb_feats, data_samples, self.rgb_private_decode_head,
                q_rgb_deg, self.quality_threshold)
            losses.update(add_prefix(loss_rgb_deg, 'rgb_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_feats = self.neck(t_enh_deg) if self.with_neck else t_enh_deg
            deg_t_logits = self._decode_head_predict_logits(
                deg_t_feats, self.t_private_decode_head)
            loss_distill_t = self._compute_distill_loss(
                clean_t_logits, deg_t_logits)
            q_t_scale = (q_t_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_t'] = (
                loss_distill_t * self.loss_distill_weight * q_t_scale)
            loss_t_deg = self._compute_quality_masked_seg_loss(
                deg_t_feats, data_samples, self.t_private_decode_head,
                q_t_deg, self.quality_threshold)
            losses.update(add_prefix(loss_t_deg, 't_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('t_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits


@MODELS.register_module()
class MiTMulABV6(_AblationBase):
    """V6: Token replacement after patch embedding with cross-modality strategy.

    - Private branch: low-quality tokens replaced by learnable token
    - Common branch: cross-modality replacement
      - Both high: keep original
      - One low: replace with the other modality's token
      - Both low: replace with learnable shared token
    - Fusion: V1-style concat+MLP for common+private
    - Quality net frozen, loss same as V3
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.common_replace_token_shared = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.priv_replace_token_rgb = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.priv_replace_token_t = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])

        self.common_private_fuse_rgb = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])
        self.common_private_fuse_t = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_replace_tokens(self):
        for p in self.common_replace_token_shared:
            nn.init.xavier_uniform_(p)
        for p in self.priv_replace_token_rgb:
            nn.init.xavier_uniform_(p)
        for p in self.priv_replace_token_t:
            nn.init.xavier_uniform_(p)

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}',
                      logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys',
                  logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self._init_replace_tokens()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_token_replace_common_v6(
                self.backbone, input_rgbt, q_rgbt_maps,
                self.quality_threshold,
                self.common_replace_token_shared,
                orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_token_replace_private(
                self.private_branch_rgb, input_rgb, q_rgb_maps,
                self.quality_threshold,
                self.priv_replace_token_rgb)
        if has_t.any():
            zp_t_list = _forward_mit_token_replace_private(
                self.private_branch_t, input_t, q_t_maps,
                self.quality_threshold,
                self.priv_replace_token_t)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(
                    B, self.embed_dims_list[i],
                    zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                    zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                    device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_concat = torch.cat([zp_rgb_list[i], zc_fused], dim=1)
                rgb_enhanced = self.common_private_fuse_rgb[i](rgb_concat)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_concat = torch.cat([zp_t_list[i], zc_fused], dim=1)
                t_enhanced = self.common_private_fuse_t[i](t_concat)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _compute_quality_masked_seg_loss(self, feats, data_samples, head, q_maps, threshold):
        losses = head.loss(feats, data_samples, self.train_cfg)
        if q_maps is not None and len(q_maps) > 0:
            q = q_maps[0]
            high_ratio = (q >= threshold).float().mean().clamp(min=0.1)
            for k in list(losses.keys()):
                losses[k] = losses[k] * high_ratio
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_feats = self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg
            deg_rgb_logits = self._decode_head_predict_logits(
                deg_rgb_feats, self.rgb_private_decode_head)
            loss_distill_rgb = self._compute_distill_loss(
                clean_rgb_logits, deg_rgb_logits)
            q_rgb_scale = (q_rgb_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_rgb'] = (
                loss_distill_rgb * self.loss_distill_weight * q_rgb_scale)
            loss_rgb_deg = self._compute_quality_masked_seg_loss(
                deg_rgb_feats, data_samples, self.rgb_private_decode_head,
                q_rgb_deg, self.quality_threshold)
            losses.update(add_prefix(loss_rgb_deg, 'rgb_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_feats = self.neck(t_enh_deg) if self.with_neck else t_enh_deg
            deg_t_logits = self._decode_head_predict_logits(
                deg_t_feats, self.t_private_decode_head)
            loss_distill_t = self._compute_distill_loss(
                clean_t_logits, deg_t_logits)
            q_t_scale = (q_t_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_t'] = (
                loss_distill_t * self.loss_distill_weight * q_t_scale)
            loss_t_deg = self._compute_quality_masked_seg_loss(
                deg_t_feats, data_samples, self.t_private_decode_head,
                q_t_deg, self.quality_threshold)
            losses.update(add_prefix(loss_t_deg, 't_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('t_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits


@MODELS.register_module()
class MiTMulABV5(_AblationBase):
    """V5: Token replacement after patch embedding.

    - Private branch: low-quality tokens replaced by learnable token
    - Common branch: each modality's low-quality tokens replaced by
      modality-specific learnable token
    - Fusion: V1-style concat+MLP for common+private
    - Quality net frozen, loss same as V3
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.common_replace_token_rgb = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.common_replace_token_t = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.priv_replace_token_rgb = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.priv_replace_token_t = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])

        self.common_private_fuse_rgb = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])
        self.common_private_fuse_t = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_replace_tokens(self):
        for p in self.common_replace_token_rgb:
            nn.init.xavier_uniform_(p)
        for p in self.common_replace_token_t:
            nn.init.xavier_uniform_(p)
        for p in self.priv_replace_token_rgb:
            nn.init.xavier_uniform_(p)
        for p in self.priv_replace_token_t:
            nn.init.xavier_uniform_(p)

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}',
                      logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys',
                  logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self._init_replace_tokens()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_token_replace_common_v5(
                self.backbone, input_rgbt, q_rgbt_maps,
                self.quality_threshold,
                self.common_replace_token_rgb, self.common_replace_token_t,
                orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_token_replace_private(
                self.private_branch_rgb, input_rgb, q_rgb_maps,
                self.quality_threshold,
                self.priv_replace_token_rgb)
        if has_t.any():
            zp_t_list = _forward_mit_token_replace_private(
                self.private_branch_t, input_t, q_t_maps,
                self.quality_threshold,
                self.priv_replace_token_t)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(
                    B, self.embed_dims_list[i],
                    zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                    zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                    device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_concat = torch.cat([zp_rgb_list[i], zc_fused], dim=1)
                rgb_enhanced = self.common_private_fuse_rgb[i](rgb_concat)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_concat = torch.cat([zp_t_list[i], zc_fused], dim=1)
                t_enhanced = self.common_private_fuse_t[i](t_concat)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _compute_quality_masked_seg_loss(self, feats, data_samples, head, q_maps, threshold):
        losses = head.loss(feats, data_samples, self.train_cfg)
        if q_maps is not None and len(q_maps) > 0:
            q = q_maps[0]
            high_ratio = (q >= threshold).float().mean().clamp(min=0.1)
            for k in list(losses.keys()):
                losses[k] = losses[k] * high_ratio
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_feats = self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg
            deg_rgb_logits = self._decode_head_predict_logits(
                deg_rgb_feats, self.rgb_private_decode_head)
            loss_distill_rgb = self._compute_distill_loss(
                clean_rgb_logits, deg_rgb_logits)
            q_rgb_scale = (q_rgb_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_rgb'] = (
                loss_distill_rgb * self.loss_distill_weight * q_rgb_scale)
            loss_rgb_deg = self._compute_quality_masked_seg_loss(
                deg_rgb_feats, data_samples, self.rgb_private_decode_head,
                q_rgb_deg, self.quality_threshold)
            losses.update(add_prefix(loss_rgb_deg, 'rgb_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_feats = self.neck(t_enh_deg) if self.with_neck else t_enh_deg
            deg_t_logits = self._decode_head_predict_logits(
                deg_t_feats, self.t_private_decode_head)
            loss_distill_t = self._compute_distill_loss(
                clean_t_logits, deg_t_logits)
            q_t_scale = (q_t_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_t'] = (
                loss_distill_t * self.loss_distill_weight * q_t_scale)
            loss_t_deg = self._compute_quality_masked_seg_loss(
                deg_t_feats, data_samples, self.t_private_decode_head,
                q_t_deg, self.quality_threshold)
            losses.update(add_prefix(loss_t_deg, 't_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('t_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits


def _forward_mit_token_replace_private(backbone, img, q_maps, threshold,
                                       replace_token, orig_B=None):
    """V5/V6 private branch: replace low-quality tokens after patch embedding.

    Args:
        replace_token: learnable token (1, C, 1, 1) for this stage
    """
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            low_mask = (q_i < threshold).squeeze(1)
            if low_mask.any():
                B_x = x.shape[0]
                C_x = x.shape[2]
                learn_tok = replace_token[i].expand(B_x, -1, H, W)
                learn_tok_nlc = nchw_to_nlc(learn_tok)
                low_mask_1d = low_mask.reshape(B_x, -1)
                x = torch.where(
                    low_mask_1d.unsqueeze(-1),
                    learn_tok_nlc, x)

        for j, block in enumerate(blocks):
            x = block(x, hw_shape)

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


def _forward_mit_token_replace_common_v5(backbone, img, q_maps, threshold,
                                         replace_token_rgb, replace_token_t,
                                         orig_B):
    """V5 common branch: each modality independently replaced by learnable token."""
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            q_s = q_i.squeeze(1)
            q_rgb = q_s[:orig_B]
            q_t = q_s[orig_B:]

            rgb_low = (q_rgb < threshold)
            t_low = (q_t < threshold)

            C_x = x.shape[2]
            learn_rgb = replace_token_rgb[i].expand(orig_B, -1, H, W)
            learn_t = replace_token_t[i].expand(orig_B, -1, H, W)
            learn_rgb_nlc = nchw_to_nlc(learn_rgb)
            learn_t_nlc = nchw_to_nlc(learn_t)

            rgb_low_1d = rgb_low.reshape(orig_B, -1)
            t_low_1d = t_low.reshape(orig_B, -1)
            low_mask_1d = torch.cat([rgb_low_1d, t_low_1d], dim=0)
            learn_nlc = torch.cat([learn_rgb_nlc, learn_t_nlc], dim=0)

            x = torch.where(
                low_mask_1d.unsqueeze(-1),
                learn_nlc, x)

        for j, block in enumerate(blocks):
            x = block(x, hw_shape)

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


def _forward_mit_token_replace_common_v6(backbone, img, q_maps, threshold,
                                         replace_token_shared, orig_B):
    """V6 common branch: cross-modality replacement strategy.

    - Both high: keep original
    - One low: replace with the other modality's token
    - Both low: replace with learnable shared token
    """
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            q_s = q_i.squeeze(1)
            q_rgb = q_s[:orig_B]
            q_t = q_s[orig_B:]

            rgb_low = (q_rgb < threshold)
            t_low = (q_t < threshold)

            x_rgb = x[:orig_B]
            x_t = x[orig_B:]
            C_x = x.shape[2]

            learn_shared = replace_token_shared[i].expand(orig_B, -1, H, W)
            learn_shared_nlc = nchw_to_nlc(learn_shared)

            rgb_low_1d = rgb_low.reshape(orig_B, -1)
            t_low_1d = t_low.reshape(orig_B, -1)
            both_low_1d = rgb_low_1d & t_low_1d
            only_rgb_low_1d = rgb_low_1d & ~t_low_1d
            only_t_low_1d = t_low_1d & ~rgb_low_1d

            x_rgb_nlc = x_rgb
            x_t_nlc = x_t

            x_rgb_nlc = torch.where(only_rgb_low_1d.unsqueeze(-1), x_t_nlc, x_rgb_nlc)
            x_rgb_nlc = torch.where(both_low_1d.unsqueeze(-1), learn_shared_nlc, x_rgb_nlc)

            x_t_nlc = torch.where(only_t_low_1d.unsqueeze(-1), x_rgb_nlc, x_t_nlc)
            x_t_nlc = torch.where(both_low_1d.unsqueeze(-1), learn_shared_nlc, x_t_nlc)

            x = torch.cat([x_rgb_nlc, x_t_nlc], dim=0)

        for j, block in enumerate(blocks):
            x = block(x, hw_shape)

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


def _forward_mit_attn_bias(backbone, img, q_maps, threshold, orig_B=None):
    """V7: add attention bias for low-quality K tokens in every block.

    After computing attention scores, add -10000 bias to columns
    corresponding to low-quality K tokens, so they are ignored.
    """
    outs = []
    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape

        quality_mask_1d = None
        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (H, W):
                q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
            q_s = q_i.squeeze(1)

            if orig_B is not None and q_s.shape[0] == orig_B * 2:
                q_rgb = q_s[:orig_B]
                q_t = q_s[orig_B:]
                q_s = torch.cat([q_rgb, q_t], dim=0)

            quality_mask_1d = (q_s < threshold).reshape(x.shape[0], -1)

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

            B_tok, N_tok, C_tok = x_q.shape
            num_heads = attn_module.num_heads
            head_dim = C_tok // num_heads

            q_proj = attn_module.attn.in_proj_weight[:C_tok]
            k_proj = attn_module.attn.in_proj_weight[C_tok:2*C_tok]
            v_proj = attn_module.attn.in_proj_weight[2*C_tok:]
            q_bias = attn_module.attn.in_proj_bias[:C_tok] if attn_module.attn.in_proj_bias is not None else None
            k_bias = attn_module.attn.in_proj_bias[C_tok:2*C_tok] if attn_module.attn.in_proj_bias is not None else None
            v_bias = attn_module.attn.in_proj_bias[2*C_tok:] if attn_module.attn.in_proj_bias is not None else None

            Q = F.linear(x_q, q_proj, q_bias)
            K = F.linear(x_norm1, k_proj, k_bias)
            V = F.linear(x_kv, v_proj, v_bias)

            if attn_module.sr_ratio > 1:
                K = nlc_to_nchw(K, hw_shape)
                K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]
                K = nchw_to_nlc(K)
                K = attn_module.norm(K)
            else:
                hw_shape_k = hw_shape

            Q = Q.reshape(B_tok, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(B_tok, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            if quality_mask_1d is not None:
                N_k = K.shape[2]
                if N_k == N_tok:
                    k_low = quality_mask_1d
                else:
                    H_k, W_k = hw_shape_k
                    qm_2d = quality_mask_1d.reshape(B_tok, H, W).float().unsqueeze(1)
                    qm_pooled = F.max_pool2d(qm_2d.float(), kernel_size=(H // H_k, W // W_k))
                    k_low = (qm_pooled.squeeze(1) < threshold)
                    k_low = k_low.reshape(B_tok, -1)

                attn_bias = torch.zeros_like(attn)
                k_low_mask = k_low.unsqueeze(1).unsqueeze(2)
                attn_bias.masked_fill_(k_low_mask, -10000.0)
                attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = attn_module.dropout_layer(attn) if hasattr(attn_module, 'dropout_layer') else attn
            attn_out = (attn @ V).transpose(1, 2).reshape(B_tok, N_tok, C_tok)

            attn_out = attn_module.proj_drop(attn_module.attn.out_proj(attn_out))
            x = residual + attn_out

            residual = x
            x_norm2 = block.norm2(x)
            ffn_out = nlc_to_nchw(x_norm2, hw_shape)
            ffn_out = block.ffn.layers(ffn_out)
            ffn_out = nchw_to_nlc(ffn_out)
            ffn_out = block.ffn.dropout_layer(ffn_out)
            x = residual + ffn_out

        x = norm(x)
        x = nlc_to_nchw(x, hw_shape)
        if i in backbone.out_indices:
            outs.append(x)
        img = x

    return outs


@MODELS.register_module()
class MiTMulABV3Replace(_AblationBase):
    """V3 variant: token replacement instead of attention K weighting.

    Post-backbone token replacement based on quality scores:
    - Shared branch: low-quality tokens replaced by other modality's token,
      or learnable token if both below threshold.
    - Private branch: low-quality tokens replaced by modality-specific learnable token.
    - Fusion: V1-style concat+MLP for common+private, quality-weighted sum for common.
    """

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        # Learnable replacement tokens (one per stage)
        self.shared_replace_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.rgb_priv_replace_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])
        self.t_priv_replace_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, 1, 1))
            for ch in self.embed_dims_list
        ])

        # V1-style fusion: concat+MLP for common+private (replaces WindowCrossAttention)
        self.common_private_fuse_rgb = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])
        self.common_private_fuse_t = nn.ModuleList([
            MLPFusion(ch * 2, ch) for ch in self.embed_dims_list
        ])

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_replace_tokens(self):
        for p in self.shared_replace_token:
            nn.init.xavier_uniform_(p)
        for p in self.rgb_priv_replace_token:
            nn.init.xavier_uniform_(p)
        for p in self.t_priv_replace_token:
            nn.init.xavier_uniform_(p)

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}',
                      logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys',
                  logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self._init_replace_tokens()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _replace_low_quality_tokens(self, zc_rgb_list, zc_t_list,
                                    zp_rgb_list, zp_t_list,
                                    q_rgb_maps, q_t_maps):
        """Replace low-quality tokens post-backbone.

        Shared branch:
          - only rgb low → use t's token
          - only t low → use rgb's token
          - both low → learnable shared token
          - both high → unchanged

        Private branch:
          - low → modality-specific learnable token
          - high → unchanged
        """
        num_stages = len(self.embed_dims_list)
        zc_rgb_new = [None] * num_stages
        zc_t_new = [None] * num_stages
        zp_rgb_new = [None] * num_stages
        zp_t_new = [None] * num_stages

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                B, C, H, W = zc_rgb_list[i].shape

                q_rgb = q_rgb_maps[i]
                q_t = q_t_maps[i]
                if q_rgb.shape[2:] != (H, W):
                    q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
                if q_t.shape[2:] != (H, W):
                    q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

                rgb_low = q_rgb < self.quality_threshold   # (B, 1, H, W)
                t_low = q_t < self.quality_threshold
                both_low = rgb_low & t_low
                only_rgb_low = rgb_low & ~t_low
                only_t_low = t_low & ~rgb_low

                learnable_shared = self.shared_replace_token[i].expand(B, C, H, W)

                # Shared RGB: replace low tokens
                zc_rgb_i = torch.where(
                    only_rgb_low.expand(B, C, H, W), zc_t_list[i], zc_rgb_list[i])
                zc_rgb_i = torch.where(
                    both_low.expand(B, C, H, W), learnable_shared, zc_rgb_i)
                zc_rgb_new[i] = zc_rgb_i

                # Shared T: replace low tokens
                zc_t_i = torch.where(
                    only_t_low.expand(B, C, H, W), zc_rgb_list[i], zc_t_list[i])
                zc_t_i = torch.where(
                    both_low.expand(B, C, H, W), learnable_shared, zc_t_i)
                zc_t_new[i] = zc_t_i

            elif zc_rgb_list[i] is not None:
                zc_rgb_new[i] = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_t_new[i] = zc_t_list[i]

            # Private RGB: replace low tokens
            if zp_rgb_list[i] is not None:
                Bp, Cp, Hp, Wp = zp_rgb_list[i].shape
                q_rgb_p = q_rgb_maps[i]
                if q_rgb_p.shape[2:] != (Hp, Wp):
                    q_rgb_p = F.interpolate(
                        q_rgb_p, size=(Hp, Wp), mode='nearest')
                rgb_low_p = q_rgb_p < self.quality_threshold
                learnable_rgb = self.rgb_priv_replace_token[i].expand(Bp, Cp, Hp, Wp)
                zp_rgb_new[i] = torch.where(
                    rgb_low_p.expand(Bp, Cp, Hp, Wp), learnable_rgb, zp_rgb_list[i])

            # Private T: replace low tokens
            if zp_t_list[i] is not None:
                Bp, Cp, Hp, Wp = zp_t_list[i].shape
                q_t_p = q_t_maps[i]
                if q_t_p.shape[2:] != (Hp, Wp):
                    q_t_p = F.interpolate(
                        q_t_p, size=(Hp, Wp), mode='nearest')
                t_low_p = q_t_p < self.quality_threshold
                learnable_t = self.t_priv_replace_token[i].expand(Bp, Cp, Hp, Wp)
                zp_t_new[i] = torch.where(
                    t_low_p.expand(Bp, Cp, Hp, Wp), learnable_t, zp_t_list[i])

        return zc_rgb_new, zc_t_new, zp_rgb_new, zp_t_new

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        # Standard backbone forward (no quality intervention)
        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            zc_rgbt_list = self.backbone(input_rgbt)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = self.private_branch_rgb(input_rgb)
        if has_t.any():
            zp_t_list = self.private_branch_t(input_t)

        # Token replacement based on quality scores
        zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list = \
            self._replace_low_quality_tokens(
                zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                q_rgb_maps, q_t_maps)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(
                    B, self.embed_dims_list[i],
                    zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                    zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                    device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            # V1-style fusion: concat + MLP
            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_concat = torch.cat([zp_rgb_list[i], zc_fused], dim=1)
                rgb_enhanced = self.common_private_fuse_rgb[i](rgb_concat)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_concat = torch.cat([zp_t_list[i], zc_fused], dim=1)
                t_enhanced = self.common_private_fuse_t[i](t_concat)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _compute_quality_masked_seg_loss(self, feats, data_samples, head, q_maps, threshold):
        losses = head.loss(feats, data_samples, self.train_cfg)
        if q_maps is not None and len(q_maps) > 0:
            q = q_maps[0]
            high_ratio = (q >= threshold).float().mean().clamp(min=0.1)
            for k in list(losses.keys()):
                losses[k] = losses[k] * high_ratio
        return losses

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(
                common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(
                rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(
                t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_feats = self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg
            deg_rgb_logits = self._decode_head_predict_logits(
                deg_rgb_feats, self.rgb_private_decode_head)
            loss_distill_rgb = self._compute_distill_loss(
                clean_rgb_logits, deg_rgb_logits)
            q_rgb_scale = (q_rgb_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_rgb'] = (
                loss_distill_rgb * self.loss_distill_weight * q_rgb_scale)
            loss_rgb_deg = self._compute_quality_masked_seg_loss(
                deg_rgb_feats, data_samples, self.rgb_private_decode_head,
                q_rgb_deg, self.quality_threshold)
            losses.update(add_prefix(loss_rgb_deg, 'rgb_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_feats = self.neck(t_enh_deg) if self.with_neck else t_enh_deg
            deg_t_logits = self._decode_head_predict_logits(
                deg_t_feats, self.t_private_decode_head)
            loss_distill_t = self._compute_distill_loss(
                clean_t_logits, deg_t_logits)
            q_t_scale = (q_t_deg[0] >= self.quality_threshold).float().mean().clamp(min=0.1)
            losses['loss_distill_t'] = (
                loss_distill_t * self.loss_distill_weight * q_t_scale)
            loss_t_deg = self._compute_quality_masked_seg_loss(
                deg_t_feats, data_samples, self.t_private_decode_head,
                q_t_deg, self.quality_threshold)
            losses.update(add_prefix(loss_t_deg, 't_private_deg'))
            for k in list(losses.keys()):
                if k.startswith('t_private_deg.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits


@MODELS.register_module()
class MiTMulABV4(_AblationBase):
    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 loss_invariant_weight: float = 0.3,
                 loss_distill_weight: float = 0.2,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 d_state: int = 16,
                 dt_rank: str = 'auto',
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.cross_scan_ssm_rgb = nn.ModuleList()
        self.cross_scan_ssm_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            self.cross_scan_ssm_rgb.append(
                CrossScanSSMFusion(ch, d_state=d_state, dt_rank=dt_rank))
            self.cross_scan_ssm_t.append(
                CrossScanSSMFusion(ch, d_state=d_state, dt_rank=dt_rank))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)
        self.common_refine = nn.ModuleList([
            MultiScaleRefine(ch) for ch in self.embed_dims_list])

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_distill_weight = loss_distill_weight
        self.aux_loss_weight = aux_loss_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True
        else:
            should_freeze = True
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}', logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys', logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _get_quality_weight_1d(self, q_2d, H, W, B):
        if q_2d.shape[2:] != (H, W):
            q_2d = F.interpolate(q_2d, size=(H, W), mode='nearest')
        q_s = q_2d.squeeze(1).clamp(min=0.1)
        quality_weight_1d = q_s.reshape(B, -1)
        return quality_weight_1d

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_with_ste(
                self.backbone, input_rgbt, q_rgbt_maps, self.quality_threshold,
                orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_with_ste(
                self.private_branch_rgb, input_rgb, q_rgb_maps, self.quality_threshold)
        if has_t.any():
            zp_t_list = _forward_mit_with_ste(
                self.private_branch_t, input_t, q_t_maps, self.quality_threshold)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(B, self.embed_dims_list[i],
                                       zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                                       zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                                       device=input_rgb.device)
            zc_fused = self.common_refine[i](zc_fused)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                H_i, W_i = zp_rgb_list[i].shape[2], zp_rgb_list[i].shape[3]
                quality_weight_1d_rgb = self._get_quality_weight_1d(q_rgb_maps[i], H_i, W_i, B)
                rgb_enhanced = self.cross_scan_ssm_rgb[i](
                    zc_fused, zp_rgb_list[i], quality_weight_1d=quality_weight_1d_rgb)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                H_i, W_i = zp_t_list[i].shape[2], zp_t_list[i].shape[3]
                quality_weight_1d_t = self._get_quality_weight_1d(q_t_maps[i], H_i, W_i, B)
                t_enhanced = self.cross_scan_ssm_t[i](
                    zc_fused, zp_t_list[i], quality_weight_1d=quality_weight_1d_t)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def _generate_degraded_inputs(self, input_rgb, input_ir):
        B, C, H, W = input_rgb.shape
        device = input_rgb.device
        rgb_mean = self.data_preprocessor.mean[:3].to(device)
        rgb_std = self.data_preprocessor.std[:3].to(device)
        ir_mean = self.data_preprocessor.mean[3:].to(device)
        ir_std = self.data_preprocessor.std[3:].to(device)

        deg_rgb = input_rgb.clone()
        deg_t = input_ir.clone()
        deg_type_rgb = 'none'
        deg_type_t = 'none'

        r = random.random()
        if r < self.missing_ratio:
            if random.random() < 0.5:
                deg_rgb = (-rgb_mean / rgb_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_rgb = 'missing'
            else:
                deg_t = (-ir_mean / ir_std).view(1, 3, 1, 1).expand(B, C, H, W).clone()
                deg_type_t = 'missing'
        elif r < self.missing_ratio + self.global_deg_ratio:
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb', rgb_mean, rgb_std,
                                             is_local=True, local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't', ir_mean, ir_std,
                                           is_local=True, local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            if clean_zc_list[i] is not None and deg_zc_list[i] is not None:
                loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        batch_img_metas = [
            dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
        ] * feats[0].shape[0]
        seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean,
         has_rgb_clean, has_t_clean, both_present_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_clean)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_clean[i] is not None and zc_t_clean[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_clean[i], zc_t_clean[i], gt_labels,
                    q_rgb_clean[i], q_t_clean[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_clean[i], zc_t_clean[i], q_rgb_clean[i], q_t_clean[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg,
         has_rgb_deg, has_t_deg, both_present_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        clean_final_logits = self._decode_head_predict_logits(final_neck).detach()

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            clean_common_feats = self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean
            clean_common_logits = self._decode_head_predict_logits(
                clean_common_feats, self.common_decode_head).detach()
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits, deg_common_logits) *
                self.loss_distill_weight)

        if self.rgb_private_decode_head is not None:
            clean_rgb_feats = self.neck(rgb_enh_clean) if self.with_neck else rgb_enh_clean
            clean_rgb_logits = self._decode_head_predict_logits(
                clean_rgb_feats, self.rgb_private_decode_head).detach()
            deg_rgb_logits = self._decode_head_predict_logits(
                self.neck(rgb_enh_deg) if self.with_neck else rgb_enh_deg,
                self.rgb_private_decode_head)
            losses['loss_distill_rgb'] = (
                self._compute_distill_loss(clean_rgb_logits, deg_rgb_logits) *
                self.loss_distill_weight)

        if self.t_private_decode_head is not None:
            clean_t_feats = self.neck(t_enh_clean) if self.with_neck else t_enh_clean
            clean_t_logits = self._decode_head_predict_logits(
                clean_t_feats, self.t_private_decode_head).detach()
            deg_t_logits = self._decode_head_predict_logits(
                self.neck(t_enh_deg) if self.with_neck else t_enh_deg,
                self.t_private_decode_head)
            losses['loss_distill_t'] = (
                self._compute_distill_loss(clean_t_logits, deg_t_logits) *
                self.loss_distill_weight)

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

@MODELS.register_module()
class MiTMulABV2(_AblationBase):
    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
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
                 quality_threshold: float = 0.3,
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 aux_loss_weight: float = 0.3,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 10,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)
        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.win_cross_attn_rgb = nn.ModuleList()
        self.win_cross_attn_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            n_heads_i = max(1, ch // 64)
            self.win_cross_attn_rgb.append(
                WindowCrossAttention(ch, num_heads=n_heads_i, window_size=7))
            self.win_cross_attn_t.append(
                WindowCrossAttention(ch, num_heads=n_heads_i, window_size=7))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.aux_loss_weight = aux_loss_weight
        self.quality_freeze_epochs = quality_freeze_epochs
        self._quality_frozen = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = True  # permanently frozen
        else:
            should_freeze = True  # permanently frozen
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch})', logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})', logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: {self._quality_pretrained}', logger='current')
            return
        ckpt = torch.load(self._quality_pretrained, map_location='cpu')
        q_state = ckpt.get('model_state_dict', ckpt)
        model_state = self.quality_pyramid_net.state_dict()
        loaded = {k: v for k, v in q_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
        self.quality_pyramid_net.load_state_dict(loaded, strict=False)
        print_log(f'Quality net loaded {len(loaded)}/{len(model_state)} keys', logger='current')

    def init_weights(self):
        self._load_quality_pretrained()
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)
        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)
        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    def _get_keep_mask_1d(self, q_2d, H, W, B):
        if q_2d.shape[2:] != (H, W):
            q_2d = F.interpolate(q_2d, size=(H, W), mode='nearest')
        q_s = q_2d.squeeze(1).clamp(min=0.1)
        quality_weight_1d = q_s.reshape(B, -1)
        return quality_weight_1d

    def _extract_feat(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        q_rgb_maps = [q.detach() for q in self.quality_pyramid_net.forward_rgb(input_rgb)]
        q_t_maps = [q.detach() for q in self.quality_pyramid_net.forward_thermal(input_t)]

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                           for q_r, q_t in zip(q_rgb_maps, q_t_maps)]
            zc_rgbt_list = _forward_mit_with_ste(
                self.backbone, input_rgbt, q_rgbt_maps, self.quality_threshold,
                orig_B=B)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = _forward_mit_with_ste(
                self.private_branch_rgb, input_rgb, q_rgb_maps, self.quality_threshold)
        if has_t.any():
            zp_t_list = _forward_mit_with_ste(
                self.private_branch_t, input_t, q_t_maps, self.quality_threshold)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = self._quality_weighted_common_fusion(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i])
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(B, self.embed_dims_list[i],
                                       zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                                       zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                                       device=input_rgb.device)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                H_i, W_i = zp_rgb_list[i].shape[2], zp_rgb_list[i].shape[3]
                keep_mask_1d_rgb = self._get_keep_mask_1d(q_rgb_maps[i], H_i, W_i, B)
                rgb_enhanced = self.win_cross_attn_rgb[i](
                    zc_fused, zp_rgb_list[i], keep_mask_1d=keep_mask_1d_rgb)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                H_i, W_i = zp_t_list[i].shape[2], zp_t_list[i].shape[3]
                keep_mask_1d_t = self._get_keep_mask_1d(q_t_maps[i], H_i, W_i, B)
                t_enhanced = self.win_cross_attn_t[i](
                    zc_fused, zp_t_list[i], keep_mask_1d=keep_mask_1d_t)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, q_rgb_maps, q_t_maps,
                has_rgb, has_t, both_present)

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused)
        else:
            final_neck = final_fused
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_list
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enhanced_list
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enhanced_list
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_list)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_list[i], zc_t_list[i], gt_labels,
                    q_rgb_maps[i], q_t_maps[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, q_rgb_maps, q_t_maps,
         has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits


@MODELS.register_module()
class MiTMulABV1(_AblationBase):
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
                 loss_align_weight: float = 0.5,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_smooth_align_weight: float = 0.3,
                 aux_loss_weight: float = 0.3,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head, rgb_private_decode_head,
                             t_private_decode_head, auxiliary_head)

        num_heads = backbone.get('num_heads', [1, 2, 5, 8])
        embed_dims_base = backbone.get('embed_dims', 64)
        self.embed_dims_list = [embed_dims_base * h for h in num_heads]

        self.common_private_fuse_rgb = nn.ModuleList()
        self.common_private_fuse_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            self.common_private_fuse_rgb.append(MLPFusion(ch * 2, ch))
            self.common_private_fuse_t.append(MLPFusion(ch * 2, ch))

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_smooth_align_weight = loss_smooth_align_weight
        self.aux_loss_weight = aux_loss_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _extract_feat(self, input_rgb, input_t):
        B = input_rgb.shape[0]
        has_rgb = (input_rgb.abs().sum(dim=[1, 2, 3]) > 1e-6)
        has_t = (input_t.abs().sum(dim=[1, 2, 3]) > 1e-6)
        both_present = has_rgb & has_t

        zc_rgb_list = [None] * 4
        zc_t_list = [None] * 4
        zp_rgb_list = [None] * 4
        zp_t_list = [None] * 4

        if both_present.any():
            input_rgbt = torch.cat([input_rgb, input_t], dim=0)
            zc_rgbt_list = self.backbone(input_rgbt)
            zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
            zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        if has_rgb.any():
            zp_rgb_list = self.private_branch_rgb(input_rgb)
        if has_t.any():
            zp_t_list = self.private_branch_t(input_t)

        num_stages = len(self.embed_dims_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                zc_fused = zc_rgb_list[i] + zc_t_list[i]
            elif zc_rgb_list[i] is not None:
                zc_fused = zc_rgb_list[i]
            elif zc_t_list[i] is not None:
                zc_fused = zc_t_list[i]
            else:
                zc_fused = torch.zeros(B, self.embed_dims_list[i],
                                       zp_rgb_list[i].shape[2] if zp_rgb_list[i] is not None else 1,
                                       zp_rgb_list[i].shape[3] if zp_rgb_list[i] is not None else 1,
                                       device=input_rgb.device)
            zc_fused_list.append(zc_fused)

            if zp_rgb_list[i] is not None and zc_fused.shape[2:] == zp_rgb_list[i].shape[2:]:
                rgb_concat = torch.cat([zp_rgb_list[i], zc_fused], dim=1)
                rgb_enhanced = self.common_private_fuse_rgb[i](rgb_concat)
            elif zp_rgb_list[i] is not None:
                rgb_enhanced = zp_rgb_list[i]
            else:
                rgb_enhanced = zc_fused
            rgb_enhanced_list.append(rgb_enhanced)

            if zp_t_list[i] is not None and zc_fused.shape[2:] == zp_t_list[i].shape[2:]:
                t_concat = torch.cat([zp_t_list[i], zc_fused], dim=1)
                t_enhanced = self.common_private_fuse_t[i](t_concat)
            elif zp_t_list[i] is not None:
                t_enhanced = zp_t_list[i]
            else:
                t_enhanced = zc_fused
            t_enhanced_list.append(t_enhanced)

        final_fused = self.final_fusion(rgb_enhanced_list, t_enhanced_list, zc_fused_list)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list, t_enhanced_list,
                final_fused, has_rgb, has_t, both_present)

    def loss(self, inputs, data_samples):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, has_rgb, has_t, both_present) = results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused)
        else:
            final_neck = final_fused
        loss_final = self.decode_head.loss(final_neck, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_list
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self.common_decode_head.loss(common_feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enhanced_list
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self.rgb_private_decode_head.loss(rgb_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enhanced_list
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self.t_private_decode_head.loss(t_pf, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        gt_labels = self._stack_batch_gt(data_samples).squeeze(1)

        pad_mask = self._build_pad_mask(
            data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)

        num_stages = len(zc_rgb_list)
        loss_contrast_total = 0.0
        loss_smooth_total = 0.0
        count = 0
        for i in range(num_stages):
            if zc_rgb_list[i] is not None and zc_t_list[i] is not None:
                loss_contrast_total += _compute_cross_modal_contrastive_loss(
                    zc_rgb_list[i], zc_t_list[i], gt_labels,
                    q_rgb_maps[i], q_t_maps[i],
                    tau_q=self.quality_threshold, tau_c=self.contrast_tau,
                    num_samples=self.contrast_num_samples,
                    pad_mask=pad_mask)
                loss_smooth_total += _compute_smooth_l1_alignment_loss(
                    zc_rgb_list[i], zc_t_list[i], q_rgb_maps[i], q_t_maps[i],
                    threshold=self.quality_threshold,
                    pad_mask=pad_mask)
                count += 1
        if count > 0:
            losses['loss_align'] = (loss_contrast_total / count) * self.loss_align_weight
            losses['loss_smooth_align'] = (loss_smooth_total / count) * self.loss_smooth_align_weight

        return losses

    def extract_feat(self, inputs):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return final_fused

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_t = inputs[:, 3:, :, :]
        results = self._extract_feat(input_rgb, input_t)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list, t_enhanced_list,
         final_fused, has_rgb, has_t, both_present) = results

        B = input_rgb.shape[0]
        if both_present.all():
            feats = final_fused
        elif has_rgb.all() and not has_t.any():
            feats = rgb_enhanced_list
        elif has_t.all() and not has_rgb.any():
            feats = t_enhanced_list
        else:
            feats = final_fused

        if self.with_neck:
            feats = self.neck(feats)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits



