"""Quality-Gated MiT with DualGate Fusion.

Three-branch architecture + DualGateFusion (channel+spatial gating).
Smooth startup via QP bias=4.0 and clamp_min=0.1 for first 5 epochs.
"""

import copy
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
    cascade_quality_suppress,
    downsample_mask,
    get_missing_schedule,
    _apply_missing_degradation,
    _apply_local_missing,
    _generate_single_rect_mask,
    compute_cross_modal_contrastive_loss,
    apply_mask_to_gt,
    compute_missing_loss,
)
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         SampleList, add_prefix)


def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    B, _, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Quality-aware MiT forward functions (soft-only)
# ---------------------------------------------------------------------------

def _forward_common_dual_pruned(backbone, input_rgbt, orig_B,
                                 predictors_rgb, predictors_t,
                                 clamp_min=0.0,
                                 tau=0.3, alpha=10.0):
    outs, all_s_rgb, all_s_t = [], [], []
    cum_rgb, cum_t = None, None

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(input_rgbt)
        H, W = hw_shape
        B_tok = x.shape[0]

        s_rgb = s_t = None
        if i < len(predictors_rgb) and predictors_rgb[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            s_rgb = predictors_rgb[i](x_2d[:orig_B],
                                       torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype))
            s_t = predictors_t[i](x_2d[orig_B:],
                                   torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype))
        if s_rgb is None:
            s_rgb = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)
        if s_t is None:
            s_t = torch.ones(orig_B, 1, H, W, device=x.device, dtype=x.dtype)

        s_rgb_adj, cum_rgb = cascade_quality_suppress(s_rgb, cum_rgb, H, W, clamp_min=clamp_min)
        s_t_adj, cum_t = cascade_quality_suppress(s_t, cum_t, H, W, clamp_min=clamp_min)
        cum_rgb = s_rgb_adj
        cum_t = s_t_adj

        all_s_rgb.append(s_rgb_adj)
        all_s_t.append(s_t_adj)

        bias_rgb = f_attn(s_rgb_adj, tau=tau, alpha=alpha)
        bias_t = f_attn(s_t_adj, tau=tau, alpha=alpha)
        bias_combined = torch.cat([bias_rgb, bias_t], dim=0)

        for block in blocks:
            residual = x
            x_norm1 = block.norm1(x)
            attn_module = block.attn
            x_q = x_norm1
            if attn_module.sr_ratio > 1:
                x_kv = nlc_to_nchw(x_norm1, hw_shape)
                x_kv = attn_module.sr(x_kv); x_kv = nchw_to_nlc(x_kv)
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
                K = nlc_to_nchw(K, hw_shape); K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]; K = nchw_to_nlc(K)
                K = attn_module.norm(K)

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            H_k, W_k = hw_shape_k
            s_k = downsample_mask(bias_combined, H_k, W_k)
            attn_bias = s_k.reshape(Bb, 1, 1, -1)
            attn = attn + attn_bias.to(attn.dtype)

            attn = attn.float().softmax(dim=-1).to(V.dtype)
            if hasattr(attn_module, 'dropout_layer') and attn_module.dropout_layer is not None:
                attn = attn_module.dropout_layer(attn)
            attn_out = (attn @ V).transpose(1, 2).reshape(Bb, N_tok, C_tok)
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
        input_rgbt = x

    return outs, all_s_rgb, all_s_t


def _forward_branch_pruned(backbone, img, predictor_list,
                            clamp_min=0.0,
                            tau=0.3, alpha=10.0):
    outs, all_s = [], []
    cum_s = None

    for i, layer in enumerate(backbone.layers):
        patch_embed, blocks, norm = layer[0], layer[1], layer[2]
        x, hw_shape = patch_embed(img)
        H, W = hw_shape
        B_tok = x.shape[0]

        s = None
        if i < len(predictor_list) and predictor_list[i] is not None:
            x_2d = nlc_to_nchw(x, hw_shape)
            s = predictor_list[i](x_2d,
                                   torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype))
        if s is None:
            s = torch.ones(B_tok, 1, H, W, device=x.device, dtype=x.dtype)

        s_adj, cum_s = cascade_quality_suppress(s, cum_s, H, W, clamp_min=clamp_min)
        cum_s = s_adj
        all_s.append(s_adj)

        for block in blocks:
            residual = x
            x_norm1 = block.norm1(x)
            attn_module = block.attn
            x_q = x_norm1
            if attn_module.sr_ratio > 1:
                x_kv = nlc_to_nchw(x_norm1, hw_shape)
                x_kv = attn_module.sr(x_kv); x_kv = nchw_to_nlc(x_kv)
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
                K = nlc_to_nchw(K, hw_shape); K = attn_module.sr(K)
                hw_shape_k = K.shape[2:]; K = nchw_to_nlc(K)
                K = attn_module.norm(K)

            Q = Q.reshape(Bb, N_tok, num_heads, head_dim).permute(0, 2, 1, 3)
            K = K.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)
            V = V.reshape(Bb, -1, num_heads, head_dim).permute(0, 2, 1, 3)

            scale = head_dim ** -0.5
            attn = (Q @ K.transpose(-2, -1)) * scale

            H_k, W_k = hw_shape_k
            s_k = downsample_mask(s_adj, H_k, W_k)
            attn_bias = f_attn(s_k, tau=tau, alpha=alpha).reshape(Bb, 1, 1, -1)
            attn = attn + attn_bias.to(attn.dtype)

            attn = attn.float().softmax(dim=-1).to(V.dtype)
            if hasattr(attn_module, 'dropout_layer') and attn_module.dropout_layer is not None:
                attn = attn_module.dropout_layer(attn)
            attn_out = (attn @ V).transpose(1, 2).reshape(Bb, N_tok, C_tok)
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

    return outs, all_s


# ===================================================================
# DualGateFusion
# ===================================================================

class DualGateFusion(nn.Module):
    """Channel- and spatial-gated residual fusion module."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels // 4, 64)
        self.ch_gate_conv1 = nn.Conv2d(channels * 3, mid_ch * 2, 1, bias=False)
        self.ch_gate_conv2 = nn.Conv2d(mid_ch * 2, channels * 2, 1, bias=False)
        self.sp_gate = nn.Conv2d(channels * 3, 2, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, zc_r, zc_t, zp_r, zp_t, w_r, w_t, w_pr, w_pt):
        Fc_r = zc_r * w_r; Fc_t = zc_t * w_t
        Fp_r = zp_r * w_pr; Fp_t = zp_t * w_pt

        F_common = (Fc_r + Fc_t) / (w_r + w_t + 1e-8)
        F_common = F_common.permute(0, 2, 3, 1).contiguous()
        F_common = F.layer_norm(F_common, [F_common.size(-1)])
        F_common = F_common.permute(0, 3, 1, 2).contiguous()

        gate_input = torch.cat([F_common, Fp_r, Fp_t], dim=1)

        ch_feat = F.adaptive_avg_pool2d(gate_input, 1)
        ch_feat = self.ch_gate_conv1(ch_feat); ch_feat = F.relu(ch_feat)
        ch_feat = self.ch_gate_conv2(ch_feat)
        ch_gate_r, ch_gate_t = ch_feat.sigmoid().chunk(2, dim=1)

        sp_gate_r, sp_gate_t = self.sp_gate(gate_input).sigmoid().chunk(2, dim=1)

        Pr_gated = Fp_r * ch_gate_r * sp_gate_r
        Pt_gated = Fp_t * ch_gate_t * sp_gate_t

        F_final = F_common + Pr_gated + Pt_gated
        F_final = F_final.permute(0, 2, 3, 1).contiguous()
        F_final = F.layer_norm(F_final, [F_final.size(-1)])
        F_final = F_final.permute(0, 3, 1, 2).contiguous()

        return F_final


# ===================================================================
# QualityGatedMiTMamba
# ===================================================================

@MODELS.register_module()
class QualityGatedMiTMamba(BaseSegmentor):
    """Quality-gated MiT with DualGateFusion and smooth startup."""

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
                 total_epochs: int = 200,
                 loss_align_weight: float = 0.1,
                 contrast_tau: float = 0.07,
                 contrast_num_samples: int = 512,
                 loss_distill_weight: float = 0.3,
                 distill_temperature: float = 4.0,
                 aux_loss_weight: float = 0.5,
                 loss_invariant_weight: float = 0.03,
                 loss_missing_weight: float = 0.5,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 tau: float = 0.3,
                 alpha: float = 10.0,
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
        self.tau = tau; self.alpha = alpha
        self._build_predictors()
        self.total_epochs = total_epochs
        self.loss_align_weight = loss_align_weight
        self.contrast_tau, self.contrast_num_samples = contrast_tau, contrast_num_samples
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_missing_weight = loss_missing_weight

        self.dual_gate_fusions = nn.ModuleList(
            [DualGateFusion(ch) for ch in self.embed_dims_list])
        final_dim = self.embed_dims_list[-1]
        self.final_conv = nn.Conv2d(final_dim, final_dim, 1, bias=False)

    def _build_predictors(self):
        self.predictors_common_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_common_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_priv_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])
        self.predictors_priv_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.embed_dims_list])

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

    @staticmethod
    def _clamp_min_for_epoch(epoch):
        if epoch is None: return 0.0
        if epoch < 5: return 0.1
        return 0.0

    @staticmethod
    def _make_quality_gate(score, tau, target_h, target_w):
        gate = (score > tau).float().detach()
        gate_sq = gate.squeeze(1)
        if gate_sq.shape[-2:] != (target_h, target_w):
            gate_sq = F.interpolate(
                gate_sq.unsqueeze(1).float(),
                size=(target_h, target_w), mode='nearest').squeeze(1)
        return gate_sq

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg: super().init_weights()

    # ---- Core feature extraction ----

    def _extract_feat_single(self, rgb, t):
        B = rgb.shape[0]
        ep = getattr(self, 'current_epoch', 0)
        clamp_min = self._clamp_min_for_epoch(ep)

        zc_outs, s_r, s_t = _forward_common_dual_pruned(
            self.backbone, torch.cat([rgb, t], dim=0), orig_B=B,
            predictors_rgb=self.predictors_common_rgb,
            predictors_t=self.predictors_common_t,
            clamp_min=clamp_min, tau=self.tau, alpha=self.alpha)
        zc_r = [f[:B] for f in zc_outs]; zc_t = [f[B:] for f in zc_outs]

        zp_r, spr = _forward_branch_pruned(
            self.private_branch_rgb, rgb, self.predictors_priv_rgb,
            clamp_min=clamp_min, tau=self.tau, alpha=self.alpha)

        zp_t, spt = _forward_branch_pruned(
            self.private_branch_t, t, self.predictors_priv_t,
            clamp_min=clamp_min, tau=self.tau, alpha=self.alpha)

        re, te, ff, zf_weighted = [], [], [], []
        for i in range(len(self.embed_dims_list)):
            zc_ri = zc_r[i]; zc_ti = zc_t[i]; zp_ri = zp_r[i]; zp_ti = zp_t[i]
            dev = zc_ri.device; B1 = B

            w_r = s_r[i] if s_r[i] is not None else torch.ones(B1, 1, zc_ri.shape[2], zc_ri.shape[3], device=dev)
            w_t = s_t[i] if s_t[i] is not None else torch.ones(B1, 1, zc_ti.shape[2], zc_ti.shape[3], device=dev)
            w_pr = spr[i] if spr[i] is not None else torch.ones(B1, 1, zc_ri.shape[2], zc_ri.shape[3], device=dev)
            w_pt = spt[i] if spt[i] is not None else torch.ones(B1, 1, zc_ti.shape[2], zc_ti.shape[3], device=dev)

            # Aux heads: raw average + LN
            re_i = (zc_ri + zp_ri) / 2.0
            re_i = re_i.permute(0, 2, 3, 1).contiguous()
            re_i = F.layer_norm(re_i, [re_i.size(-1)])
            re_i = re_i.permute(0, 3, 1, 2).contiguous()
            re.append(re_i)

            te_i = (zc_ti + zp_ti) / 2.0
            te_i = te_i.permute(0, 2, 3, 1).contiguous()
            te_i = F.layer_norm(te_i, [te_i.size(-1)])
            te_i = te_i.permute(0, 3, 1, 2).contiguous()
            te.append(te_i)

            # zf_weighted for invariance loss
            zf_w = (w_r * zc_ri + w_t * zc_ti) / (w_r + w_t + 1e-8)
            zf_w = zf_w.permute(0, 2, 3, 1).contiguous()
            zf_w = F.layer_norm(zf_w, [zf_w.size(-1)])
            zf_w = zf_w.permute(0, 3, 1, 2).contiguous()
            zf_weighted.append(zf_w)

            # DualGateFusion
            ff_i = self.dual_gate_fusions[i](zc_ri, zc_ti, zp_ri, zp_ti, w_r, w_t, w_pr, w_pt)
            if i == len(self.embed_dims_list) - 1:
                ff_i = self.final_conv(ff_i)
                ff_i = ff_i.permute(0, 2, 3, 1).contiguous()
                ff_i = F.layer_norm(ff_i, [ff_i.size(-1)])
                ff_i = ff_i.permute(0, 3, 1, 2).contiguous()
            ff.append(ff_i)

        all_s = [s for sl in [s_r, s_t, spr, spt] for s in sl]
        return zc_r, zc_t, zp_r, zp_t, re, te, ff, zf_weighted, s_r, s_t, all_s, spr, spt

    # ---- Degradation ----

    def _train_with_degradation(self, rgb, ir):
        dr, di, _, _, miss_rgb, miss_t = self._generate_degraded_inputs(rgb, ir)
        (zc_r, zc_t, zp_r, zp_t, re, te, ff, zf_weighted,
         s_r, s_t, all_s, spr, spt) = self._extract_feat_single(dr, di)
        return (zc_r, zc_t, zp_r, zp_t, re, te, ff, zf_weighted,
                s_r, s_t, all_s, spr, spt, miss_rgb, miss_t)

    def _generate_degraded_inputs(self, rgb, ir):
        B, C, H, W = rgb.shape
        dev = rgb.device
        rm = self.data_preprocessor.mean[:3].to(dev); rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev); iss = self.data_preprocessor.std[3:].to(dev)
        dr, di = rgb.clone(), ir.clone()
        dtr, dtt = ['none'] * B, ['none'] * B
        miss_rgb = torch.zeros(B, 1, H, W, device=dev)
        miss_t = torch.zeros(B, 1, H, W, device=dev)
        ep = getattr(self, 'current_epoch', 0)
        sched = get_missing_schedule(min(ep / max(self.total_epochs, 1), 1.0))
        for b in range(B):
            modality = 'rgb' if torch.rand(1, device=dev).item() < 0.5 else 'thermal'
            r = torch.rand(1, device=dev).item()
            if r < sched['p_global']:
                if modality == 'rgb':
                    dr[b:b+1] = _apply_missing_degradation(rgb[b:b+1], rm, rs)
                    dtr[b] = 'global_missing'; miss_rgb[b:b+1] = 1.0
                else:
                    di[b:b+1] = _apply_missing_degradation(ir[b:b+1], im, iss)
                    dtt[b] = 'global_missing'; miss_t[b:b+1] = 1.0
            elif r < sched['p_global'] + sched['p_local']:
                area = sched['local_area']
                if area <= 0: continue
                lm = _generate_single_rect_mask(1, H, W, area, device=dev)
                if modality == 'rgb':
                    dr[b:b+1] = _apply_local_missing(rgb[b:b+1], rm, rs, lm)
                    dtr[b] = f'local_missing_{area:.0%}'; miss_rgb[b:b+1] = lm
                else:
                    di[b:b+1] = _apply_local_missing(ir[b:b+1], im, iss, lm)
                    dtt[b] = f'local_missing_{area:.0%}'; miss_t[b:b+1] = lm
        return dr.to(rgb.dtype), di.to(ir.dtype), dtr, dtt, miss_rgb, miss_t

    # ---- Loss ----

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        B = rgb.shape[0]

        (zc_r, zc_t, zp_r, zp_t, re, te, ff, zf_weighted,
         s_r, s_t, all_s, spr, spt) = self._extract_feat_single(rgb, ir)

        losses = {}
        sl = self._stack_batch_gt(data_samples)
        for ds in data_samples:
            if hasattr(ds, 'gt_sem_seg'): ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)

        # Main decoder (clean)
        losses.update(add_prefix(self.decode_head.loss(ff, data_samples, self.train_cfg), 'decode'))

        # Common auxiliary
        if self.common_decode_head and zc_r and zc_t:
            zc_common = [torch.cat([zc_r[i], zc_t[i]], dim=0) for i in range(len(zc_r))]
            data_samples_x2 = data_samples + [copy.deepcopy(ds) for ds in data_samples]
            losses.update(add_prefix(
                self.common_decode_head.loss(zc_common, data_samples_x2, self.train_cfg), 'common_decode'))

        # Private aux heads
        if self.rgb_private_decode_head and re:
            losses.update(add_prefix(
                self.rgb_private_decode_head.loss(re, data_samples, self.train_cfg), 'rgb_private_decode'))
        if self.t_private_decode_head and te:
            losses.update(add_prefix(
                self.t_private_decode_head.loss(te, data_samples, self.train_cfg), 't_private_decode'))

        # Contrastive
        if self.loss_align_weight > 0:
            gt = sl.squeeze(1).long(); lc, cnt = 0., 0
            pm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            for i in range(len(zc_r)):
                if zc_r[i] is not None and zc_t[i] is not None:
                    Dr = (s_r[i] > self.tau).float().detach() if s_r[i] is not None else torch.ones(
                        B, 1, zc_r[i].shape[2], zc_r[i].shape[3], device=zc_r[i].device)
                    Dt = (s_t[i] > self.tau).float().detach() if s_t[i] is not None else torch.ones(
                        B, 1, zc_t[i].shape[2], zc_t[i].shape[3], device=zc_t[i].device)
                    lc += compute_cross_modal_contrastive_loss(
                        zc_r[i], zc_t[i], gt, Dr, Dt,
                        s_r[i].detach() if s_r[i] is not None else None,
                        s_t[i].detach() if s_t[i] is not None else None,
                        tau_c=self.contrast_tau, num_samples=self.contrast_num_samples,
                        ignore_label=255, pad_mask=pm)
                    cnt += 1
            if cnt: losses['loss_align'] = (lc / cnt) * self.loss_align_weight

        # Degradation branch (always active during training)
        if self.training:
            (dzcr, dzct, dzpr, dzpt, dre, dte, dff, dzf_weighted,
             ds_r, ds_t, dall_s, dspr, dspt,
             miss_rgb, miss_t) = self._train_with_degradation(rgb, ir)

            gt_h, gt_w = sl.shape[2], sl.shape[3]

            # Deg main decoder
            losses.update(add_prefix(
                self.decode_head.loss(dff, data_samples, self.train_cfg), 'deg_decode'))

            # Deg common aux: each head uses its own deepest quality score
            if self.common_decode_head and dzcr and dzct:
                for feats_d, scores_d, prefix in [
                    (dzcr, ds_r, 'deg_common_rgb_decode'),
                    (dzct, ds_t, 'deg_common_t_decode'),
                ]:
                    gate_sq = self._make_quality_gate(scores_d[-1], self.tau, gt_h, gt_w)
                    masked_gt = apply_mask_to_gt(data_samples, gate_sq)
                    ld = self.common_decode_head.loss(feats_d, masked_gt, self.train_cfg)
                    losses.update(add_prefix(
                        {k: v * self.aux_loss_weight for k, v in ld.items()}, prefix))

            # Deg private aux
            for head, feats_d, scores_d, prefix in [
                (self.rgb_private_decode_head, dre, dspr, 'deg_rgb_private_decode'),
                (self.t_private_decode_head,   dte, dspt, 'deg_t_private_decode'),
            ]:
                if head and feats_d:
                    gate_sq = self._make_quality_gate(scores_d[-1], self.tau, gt_h, gt_w)
                    masked_gt = apply_mask_to_gt(data_samples, gate_sq)
                    ld = head.loss(feats_d, masked_gt, self.train_cfg)
                    losses.update(add_prefix(
                        {k: v * self.aux_loss_weight for k, v in ld.items()}, prefix))

            # Contrastive (deg)
            if self.loss_align_weight > 0 and dzcr is not None and dzct is not None:
                dlc, dcnt = 0., 0
                for i in range(len(dzcr)):
                    if dzcr[i] is not None and dzct[i] is not None:
                        dDr = (ds_r[i] > self.tau).float().detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones(
                            B, 1, dzcr[i].shape[2], dzcr[i].shape[3], device=dzcr[i].device)
                        dDt = (ds_t[i] > self.tau).float().detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones(
                            B, 1, dzct[i].shape[2], dzct[i].shape[3], device=dzct[i].device)
                        dlc += compute_cross_modal_contrastive_loss(
                            dzcr[i], dzct[i], gt, dDr, dDt,
                            ds_r[i].detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else None,
                            ds_t[i].detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else None,
                            tau_c=self.contrast_tau, num_samples=self.contrast_num_samples,
                            ignore_label=255, pad_mask=pm)
                        dcnt += 1
                if dcnt: losses['loss_align_deg'] = (dlc / dcnt) * self.loss_align_weight

            # Distillation
            if self.loss_distill_weight > 0:
                T = self.distill_temperature
                cl = self.decode_head.forward(ff).float()
                dl_ = self.decode_head.forward(dff).float()
                tp = F.softmax(cl.detach() / T, dim=1)
                sp = F.log_softmax(dl_ / T, dim=1)
                kl = F.kl_div(sp, tp, reduction='none').sum(dim=1)
                dm = self._build_pad_mask(data_samples, kl.shape[-2], kl.shape[-1], kl.device).float()
                losses['loss_distill'] = self.loss_distill_weight * (T * T) * (kl * dm).sum() / dm.sum().clamp(min=1)

            # Invariance loss (on zf_weighted)
            if self.loss_invariant_weight > 0:
                inv_loss = torch.tensor(0.0, device=ff[0].device); cnt = 0
                for i in range(len(zf_weighted)):
                    if (zf_weighted[i] is not None and dzf_weighted is not None
                            and i < len(dzf_weighted) and dzf_weighted[i] is not None):
                        if zf_weighted[i].shape == dzf_weighted[i].shape:
                            Dc = torch.max(
                                (s_r[i] > self.tau).float().detach() if s_r[i] is not None else torch.ones(
                                    B, 1, zf_weighted[i].shape[2], zf_weighted[i].shape[3], device=zf_weighted[i].device),
                                (s_t[i] > self.tau).float().detach() if s_t[i] is not None else torch.ones(
                                    B, 1, zf_weighted[i].shape[2], zf_weighted[i].shape[3], device=zf_weighted[i].device))
                            Dd = torch.max(
                                (ds_r[i] > self.tau).float().detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones(
                                    B, 1, zf_weighted[i].shape[2], zf_weighted[i].shape[3], device=zf_weighted[i].device),
                                (ds_t[i] > self.tau).float().detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones(
                                    B, 1, zf_weighted[i].shape[2], zf_weighted[i].shape[3], device=zf_weighted[i].device))
                            D_gate = Dc * Dd
                            qc = torch.max(
                                s_r[i].detach() if s_r[i] is not None else torch.ones_like(D_gate),
                                s_t[i].detach() if s_t[i] is not None else torch.ones_like(D_gate))
                            qd = torch.max(
                                ds_r[i].detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones_like(D_gate),
                                ds_t[i].detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones_like(D_gate))
                            D_gate = F.interpolate(D_gate, size=zf_weighted[i].shape[2:], mode='nearest') if D_gate.shape[2:] != zf_weighted[i].shape[2:] else D_gate
                            qc = F.interpolate(qc, size=zf_weighted[i].shape[2:], mode='nearest') if qc.shape[2:] != zf_weighted[i].shape[2:] else qc
                            qd = F.interpolate(qd, size=zf_weighted[i].shape[2:], mode='nearest') if qd.shape[2:] != zf_weighted[i].shape[2:] else qd
                            pm_i = self._build_pad_mask(data_samples, zf_weighted[i].shape[2], zf_weighted[i].shape[3], zf_weighted[i].device).float().unsqueeze(1)
                            q_distill = qc * qd * D_gate * pm_i
                            diff = F.smooth_l1_loss(zf_weighted[i], dzf_weighted[i], reduction='none')
                            denom = q_distill.sum() + 1e-6
                            inv_loss += (q_distill * diff).sum() / denom; cnt += 1
                if cnt: losses['loss_inv'] = self.loss_invariant_weight * inv_loss / cnt

            # Missing guidance
            if self.loss_missing_weight > 0:
                l_miss = compute_missing_loss(
                    ds_r, ds_t, dspr, dspt, miss_rgb, miss_t,
                    num_stages=len(self.embed_dims_list))
                losses['loss_missing'] = self.loss_missing_weight * l_miss

        for key in list(losses.keys()):
            if not torch.isfinite(losses[key]):
                losses[key] = torch.tensor(0.0, device=losses[key].device)
            else:
                losses[key] = torch.clamp(losses[key], max=100.0)
        return losses

    # ---- Inference ----

    def encode_decode(self, inputs, bm):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, ir)[6]
        if self.with_neck: ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def extract_feat(self, inputs):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, t)[6]
        return self.neck(ff) if self.with_neck else ff

    def extract_feat_vis(self, inputs):
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2; rgb, t = inputs[:B], inputs[B:]
        with torch.no_grad():
            (zc_r, zc_t, zp_r, zp_t, re, te, ff, zf_weighted,
             s_r, s_t, all_s, spr, spt) = self._extract_feat_single(rgb, t)
            fused = self.neck(ff) if self.with_neck else ff

            deg_rgb, deg_t, deg_type_rgb, deg_type_t, miss_rgb, miss_t = \
                self._generate_degraded_inputs(rgb, t)
            (zc_r_d, zc_t_d, zp_r_d, zp_t_d, re_d, te_d, ff_d, zf_weighted_d,
             ds_r_d, ds_t_d, dall_s_d, dspr_d, dspt_d) = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(ff_d) if self.with_neck else ff_d

        for i in range(len(ff)):
            if zc_r_d[i].shape[-2:] != zc_r[i].shape[-2:]:
                zc_r_d[i] = F.interpolate(zc_r_d[i], size=zc_r[i].shape[-2:], mode='bilinear')
                zc_t_d[i] = F.interpolate(zc_t_d[i], size=zc_t[i].shape[-2:], mode='bilinear')
                zp_r_d[i] = F.interpolate(zp_r_d[i], size=zp_r[i].shape[-2:], mode='bilinear')
                zp_t_d[i] = F.interpolate(zp_t_d[i], size=zp_t[i].shape[-2:], mode='bilinear')
                re_d[i] = F.interpolate(re_d[i], size=re[i].shape[-2:], mode='bilinear')
                te_d[i] = F.interpolate(te_d[i], size=te[i].shape[-2:], mode='bilinear')
                ff_d[i] = F.interpolate(ff_d[i], size=ff[i].shape[-2:], mode='bilinear')
                zf_weighted_d[i] = F.interpolate(zf_weighted_d[i], size=zf_weighted[i].shape[-2:], mode='bilinear')

        return dict(
            zc_rgb=zc_r, zc_t=zc_t,
            zc_fused=[(zc_r[i] + zc_t[i]) / 2.0 for i in range(len(zc_r))],
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
            zc_fused_deg=[(zc_r_d[i] + zc_t_d[i]) / 2.0 for i in range(len(zc_r_d))],
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d,
            final_fused_deg=fused_d,
            s_rgb_deg=ds_r_d, s_t_deg=ds_t_d,
            s_rgb_priv_deg=dspr_d, s_t_priv_deg=dspt_d,
            q_rgb_deg=ds_r_d, q_t_deg=ds_t_d,
            q_rgb_priv_deg=dspr_d, q_t_priv_deg=dspt_d,
        )

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None: head = self.decode_head
        if hasattr(head, 'pixel_decoder'):
            batch_img_metas = [
                dict(ori_shape=feats[0].shape[2:], img_shape=feats[0].shape[2:])
            ] * feats[0].shape[0]
            seg_logits = head.predict(feats, batch_img_metas, self.test_cfg)
        else:
            seg_logits = head(feats)
        return seg_logits

    def _forward(self, inputs, data_samples=None):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, t)[6]
        feats = self.neck(ff) if self.with_neck else ff
        return self.decode_head.forward(feats)

    def predict(self, inputs, data_samples=None):
        if data_samples:
            bm = [ds.metainfo for ds in data_samples]
        else:
            bm = [dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                       pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])] * inputs.shape[0]
        sl = self.encode_decode(inputs, bm)
        return self.postprocess_result(sl, data_samples)

    @torch.no_grad()
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        results = self._run_forward(data, mode='predict')

        if not hasattr(self, '_val_deg_accum'): self._val_deg_accum = {}
        if data_samples is None: return results

        if not hasattr(self, '_class_names') and data_samples:
            metainfo = data_samples[0].metainfo
            if 'classes' in metainfo: self._class_names = list(metainfo['classes'])

        dev = inputs.device
        rm = self.data_preprocessor.mean[:3].to(dev); rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev); iss = self.data_preprocessor.std[3:].to(dev)
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        num_classes = self.num_classes; ignore_index = 255

        for deg_name, deg_rgb, deg_ir in [
            ('rgb_missing', _apply_missing_degradation(rgb, rm, rs), ir),
            ('thermal_missing', rgb, _apply_missing_degradation(ir, im, iss)),
        ]:
            deg_inputs = torch.cat([deg_rgb, deg_ir], dim=1)
            bm = [ds.metainfo for ds in data_samples]
            seg_logits = self.encode_decode(deg_inputs, bm)
            preds = seg_logits.argmax(dim=1)

            acc = self._val_deg_accum.setdefault(
                deg_name,
                dict(area_intersect=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_union=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_pred=torch.zeros(num_classes, device=dev, dtype=torch.long),
                     area_label=torch.zeros(num_classes, device=dev, dtype=torch.long)))

            for b in range(seg_logits.shape[0]):
                pred = preds[b]
                label = data_samples[b].gt_sem_seg.data.squeeze().to(dev)
                if pred.shape != label.shape:
                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0).float(),
                        size=label.shape, mode='nearest').squeeze()
                mask = (label != ignore_index)
                pred_m = pred[mask]; label_m = label[mask]
                for c in range(num_classes):
                    pc = (pred_m == c); lc = (label_m == c)
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
        if not hasattr(self, '_val_deg_accum') or not self._val_deg_accum: return
        class_names = getattr(self, '_class_names', None)
        if class_names is None: class_names = [f'class_{i}' for i in range(self.num_classes)]
        for deg_name, acc in self._val_deg_accum.items():
            iou_per_class = acc['area_intersect'].float() / (acc['area_union'].float() + 1e-10) * 100
            valid = acc['area_union'] > 0
            miou = iou_per_class[valid].mean().item() if valid.any() else 0.0
            acc_per_class = acc['area_intersect'].float() / (acc['area_label'].float() + 1e-10) * 100
            macc = acc_per_class[valid].mean().item() if valid.any() else 0.0
            all_acc = acc['area_intersect'].sum().float() / (acc['area_label'].sum().float() + 1e-10) * 100
            cw = [max(len(str(class_names[i])), 8) for i in range(self.num_classes)]
            header = f'+{"-"*14}+' + '+'.join([f'{"-"*(cw[i]+2)}' for i in range(self.num_classes)]) + f'+{"-"*8}+{"-"*8}+{"-"*8}+'
            sep = '|' + f'{deg_name:^14}|' + '|'.join(
                [f'{class_names[i]:^{cw[i]+2}}' for i in range(self.num_classes)]) + f'|{"mIoU":^8}|{"mAcc":^8}|{"aAcc":^8}|'
            vals = '|' + f'{"IoU(%)":^14}|' + '|'.join(
                [f'{iou_per_class[i].item():^{cw[i]+2}.2f}' if valid[i] else f'{"N/A":^{cw[i]+2}}' for i in range(self.num_classes)]
            ) + f'|{miou:^8.2f}|{macc:^8.2f}|{all_acc:^8.2f}|'
            logger.info('\n' + header + '\n' + sep + '\n' + header + '\n' + vals + '\n' + header)
        self._val_deg_accum = {}

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        B, C, H, W = seg_logits.shape
        if data_samples is None: data_samples = [SegDataSample() for _ in range(B)]
        for i in range(B):
            img_meta = data_samples[i].metainfo
            ps = img_meta.get('padding_size', [0] * 4)
            pl, pr, pt, pb = ps
            i_sl = seg_logits[i:i+1, :, pt:H-pb, pl:W-pr]
            flip = img_meta.get('flip', None)
            if flip:
                fd = img_meta.get('flip_direction', None)
                i_sl = i_sl.flip(dims=(3,) if fd == 'horizontal' else (2,))
            from mmseg.models.utils import resize
            i_sl = resize(i_sl, size=img_meta['ori_shape'], mode='bilinear',
                         align_corners=self.align_corners, warning=False).squeeze(0)
            pred = i_sl.argmax(dim=0, keepdim=True) if C > 1 else (i_sl.sigmoid() > 0.5).to(i_sl)
            data_samples[i].set_data({'seg_logits': PixelData(data=i_sl), 'pred_sem_seg': PixelData(data=pred)})
        return data_samples

    def inference(self, inputs, bm):
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide': return self.slide_inference(inputs, bm)
        return self.whole_inference(inputs, bm)

    def whole_inference(self, inputs, bm):
        return self.encode_decode(inputs, bm)

    def slide_inference(self, inputs, bm):
        hs, ws = self.test_cfg.stride
        hc, wc = self.test_cfg.crop_size
        B, _, h_img, w_img = inputs.size()
        oc = self.out_channels
        hg = max(h_img - hc + hs - 1, 0) // hs + 1; wg = max(w_img - wc + ws - 1, 0) // ws + 1
        preds = inputs.new_zeros((B, oc, h_img, w_img))
        cnt = inputs.new_zeros((B, 1, h_img, w_img))
        for hi in range(hg):
            for wi in range(wg):
                y1 = hi * hs; x1 = wi * ws
                y2 = min(y1 + hc, h_img); x2 = min(x1 + wc, w_img)
                y1 = max(y2 - hc, 0); x1 = max(x2 - wc, 0)
                crop = inputs[:, :, y1:y2, x1:x2]
                bm[0]['img_shape'] = crop.shape[2:]
                csl = self.encode_decode(crop, bm)
                preds += F.pad(csl, (int(x1), int(preds.shape[3] - x2),
                                     int(y1), int(preds.shape[2] - y2)))
                cnt[:, :, y1:y2, x1:x2] += 1
        return preds / cnt
