"""Swin Multi-Branch Baseline (No Quality Head).

Three-branch architecture same as QualityGatedSwinMask2Former but
with all quality-related components removed:
  - No QualityPredictor, no gumbel mask, no complementary fix
  - No quality bias injection into attention
  - No retention loss, no training phase control
  - Simple average fusion for common branch
  - No mask gating for private branch fusion

Kept:
  - Three-branch architecture (common + private RGB + private T)
  - Degradation training pipeline
  - Cross-modal contrastive loss
  - Knowledge distillation loss
  - Feature invariant loss
  - Mask2Former decoder
"""

import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.v9_utils import (
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


def _forward_swin_plain(backbone, img):
    """Plain Swin forward without any quality masking."""
    x, (H, W) = backbone.patch_embed(img)
    if backbone.use_abs_pos_embed:
        x = x + backbone.absolute_pos_embed
    x = backbone.drop_after_pos(x)
    outs = []
    for i, stage in enumerate(backbone.stages):
        for block in stage.blocks:
            x = block(x, (H, W))
        if i in backbone.out_indices:
            norm_layer = getattr(backbone, f'norm{i}', None)
            out = norm_layer(x) if norm_layer is not None else x
            out = out.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        if stage.downsample is not None:
            x, (H, W) = stage.downsample(x, (H, W))
    return outs


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
        self.in_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.proj_down = nn.Linear(d_model, self.d_inner, bias=False)
        self.proj_up = nn.Linear(self.d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_inner + 1).float().repeat(2)))
        self.D_param = nn.Parameter(torch.ones(self.d_inner * 2))
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank * 2, bias=False)
        nn.init.zeros_(self.x_proj.weight)

    def _get_pos_bias(self, H, W, device):
        base_h, base_w = self.pos_table.shape[2], self.pos_table.shape[3]
        if H <= base_h and W <= base_w:
            pos = self.pos_table[:, :, :H, :W]
        else:
            pos = F.interpolate(self.pos_table, size=(H, W), mode='bilinear', align_corners=False)
        return pos.reshape(1, self.d_model, H * W).permute(0, 2, 1)

    def _mamba_scan(self, x, H, W, direction='forward'):
        B, N, D = x.shape
        if direction == 'forward':
            return x
        elif direction == 'backward':
            return x.flip(dims=[1])
        elif direction == 'flatten':
            return x.reshape(B, H, W, D).reshape(B, H * W, D)
        elif direction == 'column':
            return x.reshape(B, H, W, D).permute(0, 2, 1, 3).reshape(B, H * W, D)
        return x

    def forward(self, G, P, H, W, cum_starts=None, priv_indices=None):
        B, N_g, C = G.shape
        pos = self._get_pos_bias(H, W, G.device)
        G = G + pos
        G = G + self.type_gen.unsqueeze(0).unsqueeze(0)

        if P is not None and P.shape[0] > 0:
            if priv_indices is not None and priv_indices.shape[0] > 0:
                P = P + self.type_priv.unsqueeze(0).unsqueeze(0)
                x_full = G.clone()
                for b in range(B):
                    if cum_starts is not None and b < len(cum_starts) - 1:
                        s_priv = cum_starts[b] + N_g
                        e_priv = cum_starts[b + 1] + N_g
                        if s_priv < e_priv and e_priv <= P.shape[0] + N_g * B:
                            n_priv = e_priv - s_priv
                            if n_priv <= P.shape[0]:
                                idx_b = priv_indices[cum_starts[b]:cum_starts[b + 1]] if cum_starts[b] < priv_indices.shape[0] else priv_indices[:0]
                                if idx_b.shape[0] > 0 and idx_b.max() < N_g:
                                    x_full[b, idx_b] = x_full[b, idx_b] + P[cum_starts[b]:cum_starts[b + 1]][:idx_b.shape[0]]
            else:
                x_full = G
        else:
            x_full = G

        x_norm = self.in_norm(x_full)
        x_down = self.proj_down(x_norm)
        x_down = x_down.permute(0, 2, 1)
        x_down = self.conv1d(x_down)[:, :, :N_g]
        x_down = x_down.permute(0, 2, 1)
        x_down = F.silu(x_down)
        x_up = self.proj_up(x_down)
        x_out = self.out_norm(x_up)
        return x_out


import math

class MultiScaleRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.act(self.conv(x_norm))
        return residual + out


class DualGateEnhancedFusion(nn.Module):
    def __init__(self, embed_dims_list):
        super().__init__()
        self.ch_gates = nn.ModuleList()
        self.sp_gates = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.post_convs = nn.ModuleList()
        for ch in embed_dims_list:
            self.ch_gates.append(nn.Sequential(
                nn.Conv2d(ch * 2, ch * 2, 1, bias=False),
                nn.Sigmoid()))
            self.sp_gates.append(nn.Sequential(
                nn.Conv2d(ch * 2, 2, 1, bias=True),
                nn.Sigmoid()))
            self.post_norms.append(nn.LayerNorm(ch))
            self.post_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False)))

    def forward(self, rgb_feats, t_feats, common_feats):
        fused_list = []
        ref_h, ref_w = common_feats[0].shape[2], common_feats[0].shape[3]
        for i, (Fr, Ft, Fg) in enumerate(zip(rgb_feats, t_feats, common_feats)):
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
            fused = ch_r * sp_r * Fr + ch_t * sp_t * Ft
            fused_norm = fused.permute(0, 2, 3, 1).contiguous()
            fused_norm = self.post_norms[i](fused_norm).permute(0, 3, 1, 2).contiguous()
            out = self.post_convs[i](fused_norm)
            fused_list.append(Fg + out)
        return fused_list


@MODELS.register_module()
class SwinMultiBranchNoQuality(BaseSegmentor):
    """Swin three-branch baseline without quality heads.

    Architecture:
      - Common: one Swin backbone processing RGB+T concatenated -> zc_rgb, zc_t
      - Private: two Swin branches (RGB, T) -> zp_rgb, zp_t
      - Simple average fusion for common branch
      - SimpleConcatFusion / BiMambaFusion for private branch fusion
      - DualGateEnhancedFusion for final fusion
      - Mask2FormerHead main decoder
      - SegformerHead auxiliary decoders
      - Degradation training + contrastive + distillation + invariant losses
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

        embed_dims = backbone.get('embed_dims', 96)
        depths = backbone.get('depths', [2, 2, 6, 2])
        self.embed_dims_list = [embed_dims * (2 ** i) for i in range(len(depths))]

        self._build_fusion(mamba_d_state, mamba_d_conv, mamba_expand)
        self.common_refine = nn.ModuleList([MultiScaleRefine(ch) for ch in self.embed_dims_list])
        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.loss_align_weight = loss_align_weight
        self.contrast_tau, self.contrast_num_samples = contrast_tau, contrast_num_samples
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.missing_ratio, self.global_deg_ratio = missing_ratio, global_deg_ratio
        self.local_deg_ratio = local_deg_ratio

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
                else nn.ModuleList([MODELS.build(c) for c in aux_cfg])

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg:
            super().init_weights()

    def _simple_common_fusion(self, zc_rgb, zc_t):
        return (zc_rgb + zc_t) / 2.0

    def _mamba_fuse_stage(self, zc, priv, fuse_mod):
        B, C, H, W = zc.shape
        N_g = H * W
        G = zc.permute(0, 2, 3, 1).reshape(B, N_g, C)
        P = priv.permute(0, 2, 3, 1).reshape(B, N_g, C)
        cs = torch.zeros(B + 1, dtype=torch.int32, device=zc.device)
        for b in range(B):
            cs[b + 1] = cs[b] + N_g + N_g
        priv_indices = torch.arange(N_g, dtype=torch.long, device=zc.device).unsqueeze(0).expand(B, -1).reshape(-1)
        return fuse_mod(G, P.reshape(B * N_g, C), H, W, cs, priv_indices).reshape(B, H, W, C).permute(0, 3, 1, 2)

    def _extract_feat_single(self, rgb, t):
        B = rgb.shape[0]

        input_rgbt = torch.cat([rgb, t], dim=0)
        zc_outs = _forward_swin_plain(self.backbone, input_rgbt)
        zc_r = [f[:B] for f in zc_outs]
        zc_t = [f[B:] for f in zc_outs]

        zp_r = _forward_swin_plain(self.private_branch_rgb, rgb)
        zp_t = _forward_swin_plain(self.private_branch_t, t)

        zf, re, te = [], [], []
        for i in range(len(self.embed_dims_list)):
            zf.append(self.common_refine[i](self._simple_common_fusion(zc_r[i], zc_t[i])))

            if zf[i].shape[2:] == zp_r[i].shape[2:]:
                re.append(self.fuse_rgb[i](zf[i], zp_r[i]) if i < 2
                          else self._mamba_fuse_stage(zf[i], zp_r[i], self.fuse_rgb[i]))
            else:
                re.append(zp_r[i])

            if zf[i].shape[2:] == zp_t[i].shape[2:]:
                te.append(self.fuse_t[i](zf[i], zp_t[i]) if i < 2
                          else self._mamba_fuse_stage(zf[i], zp_t[i], self.fuse_t[i]))
            else:
                te.append(zp_t[i])

        ff = self.final_fusion(re, te, zf)
        return zc_r, zc_t, zp_r, zp_t, zf, re, te, ff

    def _generate_degraded_inputs(self, rgb, ir):
        B, C, H, W = rgb.shape
        dev = rgb.device
        rm = self.data_preprocessor.mean[:3].to(dev)
        rs = self.data_preprocessor.std[:3].to(dev)
        im = self.data_preprocessor.mean[3:].to(dev)
        iss = self.data_preprocessor.std[3:].to(dev)
        dr, di = rgb.clone(), ir.clone()
        dtr, dtt = ['none'] * B, ['none'] * B
        ep = getattr(self, 'current_epoch', 0)
        total_ep = getattr(self, 'total_epochs', 200)
        sched = get_degradation_schedule(min(ep / max(total_ep, 1), 1.0))
        for b in range(B):
            r = random.random()
            if r < sched['p_missing']:
                if random.random() < 0.5:
                    dr[b:b + 1] = _apply_degradation(rgb[b:b + 1], 'rgb', rm, rs, deg_type='missing', level=5)
                    dtr[b] = 'missing'
                else:
                    di[b:b + 1] = _apply_degradation(ir[b:b + 1], 'thermal', im, iss, deg_type='missing', level=5)
                    dtt[b] = 'missing'
            elif r < sched['p_missing'] + sched['p_global']:
                lv = sample_level(sched['global_levels'])
                if random.random() < 0.5:
                    dr[b:b + 1] = _apply_degradation(rgb[b:b + 1], 'rgb', rm, rs, level=lv)
                    dtr[b] = 'global'
                else:
                    di[b:b + 1] = _apply_degradation(ir[b:b + 1], 'thermal', im, iss, level=lv)
                    dtt[b] = 'global'
            else:
                lv = sample_level(sched['local_levels'])
                lm = _generate_local_mask(1, H, W, num_regions=3, device=dev, level=lv)
                if random.random() < 0.5:
                    dr[b:b + 1] = _apply_degradation(rgb[b:b + 1], 'rgb', rm, rs, level=lv, is_local=True, local_mask=lm)
                    dtr[b] = 'local'
                else:
                    di[b:b + 1] = _apply_degradation(ir[b:b + 1], 'thermal', im, iss, level=lv, is_local=True, local_mask=lm)
                    dtt[b] = 'local'
        return dr.to(rgb.dtype), di.to(ir.dtype), dtr, dtt

    def _train_with_degradation(self, rgb, ir):
        dr, di, _, _ = self._generate_degraded_inputs(rgb, ir)
        return self._extract_feat_single(dr, di)

    def _build_pad_mask(self, data_samples, H, W, device):
        B = len(data_samples)
        valid = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        for i, ds in enumerate(data_samples):
            if hasattr(ds, 'img_shape') and ds.img_shape is not None:
                h, w = ds.img_shape[:2]
            elif hasattr(ds, 'metainfo') and ds.metainfo is not None:
                h, w = ds.metainfo.get('img_shape', (H, W))[:2]
            else:
                h, w = H, W
            pt, pb = 0, max(0, H - h)
            pl, pr = 0, max(0, W - w)
            valid[i, pt:h - pb, pl:w - pr] = True
        return valid

    def _stack_batch_gt(self, data_samples):
        from mmseg.models.segmentors.v9_utils import _stack_batch_gt as _sbg
        return _sbg(data_samples, self.num_classes, 255)

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
        all_cls_score, all_mask_pred = self.decode_head(features, batch_data_samples)
        mask_cls_results = all_cls_score[-1].float()
        mask_pred_results = all_mask_pred[-1].float()
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        B = rgb.shape[0]
        (zc_r, zc_t, zp_r, zp_t, zf, re, te, ff) = self._extract_feat_single(rgb, ir)

        losses = {}
        sl = self._stack_batch_gt(data_samples)
        for ds in data_samples:
            if hasattr(ds, 'gt_sem_seg'):
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)

        losses.update(add_prefix(self.decode_head.loss(ff, data_samples, self.train_cfg), 'decode'))
        if self.common_decode_head and zf:
            losses.update(add_prefix(self.common_decode_head.loss(zf, data_samples, self.train_cfg), 'common_decode'))
        if self.rgb_private_decode_head and re:
            losses.update(add_prefix(self.rgb_private_decode_head.loss(re, data_samples, self.train_cfg), 'rgb_private_decode'))
        if self.t_private_decode_head and te:
            losses.update(add_prefix(self.t_private_decode_head.loss(te, data_samples, self.train_cfg), 't_private_decode'))

        if self.loss_align_weight > 0:
            gt = sl.squeeze(1).long()
            lc, cnt = 0., 0
            pm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            for i in range(len(zc_r)):
                if zc_r[i] is not None and zc_t[i] is not None:
                    Dr = torch.ones(B, 1, zc_r[i].shape[2], zc_r[i].shape[3], device=zc_r[i].device)
                    Dt = torch.ones(B, 1, zc_t[i].shape[2], zc_t[i].shape[3], device=zc_t[i].device)
                    lc += compute_cross_modal_contrastive_loss(
                        zc_r[i], zc_t[i], gt, Dr, Dt, None, None,
                        tau_c=self.contrast_tau, num_samples=self.contrast_num_samples,
                        ignore_label=255, pad_mask=pm)
                    cnt += 1
            if cnt:
                losses['loss_align'] = (lc / cnt) * self.loss_align_weight

        if self.training:
            dzcr, dzct, dzpr, dzpt, dzf, drl, dtl, df = self._train_with_degradation(rgb, ir)
            if torch.isnan(df[0]).any():
                import logging
                logging.getLogger(__name__).warning('NaN in degraded features — falling back to clean')
                dzcr, dzct, dzpr, dzpt, dzf, drl, dtl, df = zc_r, zc_t, zp_r, zp_t, zf, re, te, ff

            losses.update(add_prefix(self.decode_head.loss(df, data_samples, self.train_cfg), 'deg_decode'))
            for head, feats, pfx in [
                (self.common_decode_head, dzf, 'deg_common_decode'),
                (self.rgb_private_decode_head, drl, 'deg_rgb_private_decode'),
                (self.t_private_decode_head, dtl, 'deg_t_private_decode'),
            ]:
                if head and feats:
                    ld = {k: v * self.aux_loss_weight for k, v in head.loss(feats, data_samples, self.train_cfg).items()}
                    losses.update(add_prefix(ld, pfx))

            if self.loss_align_weight > 0 and dzcr is not None and dzct is not None:
                dlc, dcnt = 0., 0
                for i in range(len(dzcr)):
                    if dzcr[i] is not None and dzct[i] is not None:
                        dDr = torch.ones(B, 1, dzcr[i].shape[2], dzcr[i].shape[3], device=dzcr[i].device)
                        dDt = torch.ones(B, 1, dzct[i].shape[2], dzct[i].shape[3], device=dzct[i].device)
                        dlc += compute_cross_modal_contrastive_loss(
                            dzcr[i], dzct[i], gt, dDr, dDt, None, None,
                            tau_c=self.contrast_tau, num_samples=self.contrast_num_samples,
                            ignore_label=255, pad_mask=pm)
                        dcnt += 1
                if dcnt:
                    losses['loss_align_deg'] = (dlc / dcnt) * self.loss_align_weight

            if self.loss_distill_weight > 0:
                T = self.distill_temperature
                cl = self._get_seg_logits(ff, data_samples).float()
                dl_ = self._get_seg_logits(df, data_samples).float()
                tp = F.softmax(cl.detach() / T, dim=1)
                sp = F.log_softmax(dl_ / T, dim=1)
                kl = F.kl_div(sp, tp, reduction='none').sum(dim=1)
                losses['loss_distill'] = self.loss_distill_weight * (T * T) * kl.mean()

            if self.loss_invariant_weight > 0:
                inv_loss = torch.tensor(0.0, device=ff[0].device)
                cnt = 0
                for i in range(len(zf)):
                    if zf[i] is not None and dzf is not None and i < len(dzf) and dzf[i] is not None:
                        if zf[i].shape == dzf[i].shape:
                            diff = F.smooth_l1_loss(zf[i], dzf[i], reduction='none')
                            inv_loss += diff.mean()
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
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, ir)[7]
        if self.with_neck:
            ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def _forward(self, inputs, data_samples=None):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, ir)[7]
        if self.with_neck:
            ff = self.neck(ff)
        return self.decode_head.forward(ff)

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

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

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:],
                     img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:],
                     padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        from mmseg.models.utils import resize
        C, H, W = seg_logits.shape[1:]
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(seg_logits.shape[0])]
        for i in range(seg_logits.shape[0]):
            if i < len(data_samples) and data_samples[i].metainfo is not None:
                img_meta = data_samples[i].metainfo
                padding_size = img_meta.get('padding_size', img_meta.get('img_padding_size', [0] * 4))
                pl, pr, pt, pb = padding_size
                i_seg = seg_logits[i:i + 1, :, pt:H - pb, pl:W - pr]
                flip = img_meta.get('flip', None)
                if flip:
                    fd = img_meta.get('flip_direction', 'horizontal')
                    i_seg = i_seg.flip(dims=(3,) if fd == 'horizontal' else (2,))
                i_seg = resize(i_seg, size=img_meta['ori_shape'], mode='bilinear',
                               align_corners=self.align_corners, warning=False).squeeze(0)
            else:
                i_seg = seg_logits[i]
            if C > 1:
                i_pred = i_seg.argmax(dim=0, keepdim=True)
            else:
                i_pred = (i_seg.sigmoid() > self.decode_head.threshold).to(i_seg)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_seg),
                'pred_sem_seg': PixelData(data=i_pred),
            })
        return data_samples

    def extract_feat_vis(self, inputs):
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]
        with torch.no_grad():
            (zc_r, zc_t, zp_r, zp_t, zf, re, te, ff) = self._extract_feat_single(rgb, t)
            fused = self.neck(ff) if self.with_neck else ff
            deg_rgb, deg_t, deg_type_rgb, deg_type_t = self._generate_degraded_inputs(rgb, t)
            (zc_r_d, zc_t_d, zp_r_d, zp_t_d, zf_d, re_d, te_d, ff_d) = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(ff_d) if self.with_neck else ff_d
        return dict(
            zc_rgb=zc_r, zc_t=zc_t, zc_fused=zf,
            zp_rgb=zp_r, zp_t=zp_t, rgb_pf=re, t_pf=te,
            final_fused=fused,
            deg_rgb_img=deg_rgb, deg_t_img=deg_t,
            deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
            deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
            zc_rgb_deg=zc_r_d, zc_t_deg=zc_t_d, zc_fused_deg=zf_d,
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d, final_fused_deg=fused_d,
        )
