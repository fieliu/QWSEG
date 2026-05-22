"""MiT Baseline Mamba — No Quality Awareness.

Three-branch architecture identical to QualityGatedMiTMamba but with
all quality-related components removed.  Serves as the "no quality awareness"
ablation baseline.

Kept:
  - Three-branch architecture (common + private RGB + private T)
  - MixVisionTransformer backbones (plain forward, no quality bias injection)
  - DualGateEnhancedFusion (same as quality model)
  - SegformerHead main decoder + auxiliary decoders
  - Degradation training pipeline
  - Cross-modal contrastive loss (all-1 quality gates)
  - Knowledge distillation loss
  - Feature invariant loss (all-1 quality gates)

Removed:
  - QualityPredictor, quality bias injection, retention loss
  - Training phase control (force_all_keep, phase1/2/3)
  - Quality-weighted fusion (replaced by simple average + LayerNorm)
  - Private quality distillation loss
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            fused = ch_r * sp_r * Fr + ch_t * sp_t * Ft

            fused_norm = fused.permute(0, 2, 3, 1).contiguous()
            fused_norm = self.post_norms[i](fused_norm).permute(0, 3, 1, 2).contiguous()
            out = self.post_convs[i](fused_norm)
            fused_out = Fg + out
            fused_out = fused_out.permute(0, 2, 3, 1).contiguous()
            fused_out = F.layer_norm(fused_out, [fused_out.size(-1)])
            fused_out = fused_out.permute(0, 3, 1, 2).contiguous()
            fused_list.append(fused_out)
        return fused_list


@MODELS.register_module()
class MiTBaselineMamba(BaseSegmentor):
    """MiT three-branch baseline without quality awareness.

    Architecture:
      - Common:  one MiT backbone, RGB+T batch-concatenated -> zc_rgb, zc_t
      - Private: two MiT branches (RGB, T) -> zp_rgb, zp_t
      - Common fusion:  (zc_rgb + zc_t) / 2 + LayerNorm
      - Private enhance: zf + zp + LayerNorm
      - Final fusion: DualGateEnhancedFusion (same as quality model)
      - SegformerHead main decoder + auxiliary decoders
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
                 total_epochs: int = 200,
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

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.loss_align_weight = loss_align_weight
        self.contrast_tau = contrast_tau
        self.contrast_num_samples = contrast_num_samples
        self.loss_distill_weight = loss_distill_weight
        self.distill_temperature = distill_temperature
        self.aux_loss_weight = aux_loss_weight
        self.loss_invariant_weight = loss_invariant_weight
        self.missing_ratio = missing_ratio
        self.global_deg_ratio = global_deg_ratio
        self.local_deg_ratio = local_deg_ratio
        self.total_epochs = total_epochs

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

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

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

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg:
            super().init_weights()

    def _extract_feat_single(self, rgb, t):
        B = rgb.shape[0]

        input_rgbt = torch.cat([rgb, t], dim=0)
        zc_outs = self.backbone(input_rgbt)
        zc_r = [f[:B] for f in zc_outs]
        zc_t = [f[B:] for f in zc_outs]

        zp_r = self.private_branch_rgb(rgb)
        zp_t = self.private_branch_t(t)

        zf, re, te = [], [], []
        for i in range(len(self.embed_dims_list)):
            fused = (zc_r[i] + zc_t[i]) / 2.0
            fused = fused.permute(0, 2, 3, 1).contiguous()
            fused = F.layer_norm(fused, [fused.size(-1)])
            fused = fused.permute(0, 3, 1, 2).contiguous()
            zf.append(fused)

            if zf[i].shape[2:] == zp_r[i].shape[2:]:
                out_r = zf[i] + zp_r[i]
            else:
                zp_r_resized = F.interpolate(zp_r[i], size=zf[i].shape[2:], mode='bilinear', align_corners=False)
                out_r = zf[i] + zp_r_resized
            out_r = out_r.permute(0, 2, 3, 1).contiguous()
            out_r = F.layer_norm(out_r, [out_r.size(-1)])
            out_r = out_r.permute(0, 3, 1, 2).contiguous()
            re.append(out_r)

            if zf[i].shape[2:] == zp_t[i].shape[2:]:
                out_t = zf[i] + zp_t[i]
            else:
                zp_t_resized = F.interpolate(zp_t[i], size=zf[i].shape[2:], mode='bilinear', align_corners=False)
                out_t = zf[i] + zp_t_resized
            out_t = out_t.permute(0, 2, 3, 1).contiguous()
            out_t = F.layer_norm(out_t, [out_t.size(-1)])
            out_t = out_t.permute(0, 3, 1, 2).contiguous()
            te.append(out_t)

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
        sched = get_degradation_schedule(min(ep / max(self.total_epochs, 1), 1.0))
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
        seg_logits = self.decode_head(features)
        return seg_logits

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        B = rgb.shape[0]

        zc_r, zc_t, zp_r, zp_t, zf, re, te, ff = self._extract_feat_single(rgb, ir)

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
                logging.getLogger(__name__).warning(
                    'NaN in degraded features — falling back to clean features for deg losses')
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
                cl = self._get_seg_logits(ff).float()
                dl_ = self._get_seg_logits(df).float()
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
                            D_gate = torch.ones(B, 1, zf[i].shape[2], zf[i].shape[3], device=zf[i].device)
                            diff = F.smooth_l1_loss(zf[i], dzf[i], reduction='none')
                            denom = D_gate.sum() + 1e-6
                            inv_loss += (D_gate * diff).sum() / denom
                            cnt += 1
                if cnt:
                    losses['loss_inv'] = self.loss_invariant_weight * inv_loss / cnt

        for key in list(losses.keys()):
            if not torch.isfinite(losses[key]):
                losses[key] = torch.tensor(0.0, device=losses[key].device)
            else:
                losses[key] = torch.clamp(losses[key], max=100.0)
        return losses

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        seg_logits = head(feats)
        return seg_logits

    def _generate_degraded_vis_inputs(self, rgb, ir):
        return self._generate_degraded_inputs(rgb, ir)

    def extract_feat_vis(self, inputs):
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]
        with torch.no_grad():
            zc_r, zc_t, zp_r, zp_t, zf, re, te, ff = self._extract_feat_single(rgb, t)
            fused = self.neck(ff) if self.with_neck else ff

            deg_rgb, deg_t, deg_type_rgb, deg_type_t = self._generate_degraded_vis_inputs(rgb, t)
            zc_r_d, zc_t_d, zp_r_d, zp_t_d, zf_d, re_d, te_d, ff_d = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(ff_d) if self.with_neck else ff_d

        for i in range(len(zf)):
            if zc_r_d[i].shape[-2:] != zc_r[i].shape[-2:]:
                zc_r_d[i] = F.interpolate(zc_r_d[i], size=zc_r[i].shape[-2:], mode='bilinear', align_corners=False)
                zc_t_d[i] = F.interpolate(zc_t_d[i], size=zc_t[i].shape[-2:], mode='bilinear', align_corners=False)
                zf_d[i] = F.interpolate(zf_d[i], size=zf[i].shape[-2:], mode='bilinear', align_corners=False)
                zp_r_d[i] = F.interpolate(zp_r_d[i], size=zp_r[i].shape[-2:], mode='bilinear', align_corners=False)
                zp_t_d[i] = F.interpolate(zp_t_d[i], size=zp_t[i].shape[-2:], mode='bilinear', align_corners=False)
                re_d[i] = F.interpolate(re_d[i], size=re[i].shape[-2:], mode='bilinear', align_corners=False)
                te_d[i] = F.interpolate(te_d[i], size=te[i].shape[-2:], mode='bilinear', align_corners=False)
                ff_d[i] = F.interpolate(ff_d[i], size=ff[i].shape[-2:], mode='bilinear', align_corners=False)

        return dict(
            zc_rgb=zc_r, zc_t=zc_t,
            zc_fused=zf,
            zp_rgb=zp_r, zp_t=zp_t,
            rgb_pf=re, t_pf=te,
            final_fused=fused,
            clean_rgb_img=rgb, clean_t_img=t,
            deg_rgb_img=deg_rgb, deg_t_img=deg_t,
            deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
            deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
            zc_rgb_deg=zc_r_d, zc_t_deg=zc_t_d,
            zc_fused_deg=zf_d,
            zp_rgb_deg=zp_r_d, zp_t_deg=zp_t_d,
            rgb_pf_deg=re_d, t_pf_deg=te_d,
            final_fused_deg=fused_d,
        )

    def encode_decode(self, inputs, bm):
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, ir)[7]
        if self.with_neck:
            ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def extract_feat(self, inputs):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, t)[7]
        return self.neck(ff) if self.with_neck else ff

    def _forward(self, inputs, data_samples=None):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        ff = self._extract_feat_single(rgb, t)[7]
        feats = self.neck(ff) if self.with_neck else ff
        return self._get_seg_logits(feats)

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
        if data_samples:
            bm = [ds.metainfo for ds in data_samples]
        else:
            bm = [dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                       pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])] * inputs.shape[0]
        seg_logits = self.encode_decode(inputs, bm)
        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        from mmseg.models.utils import resize
        B, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(B)]
        for i in range(B):
            img_meta = data_samples[i].metainfo
            ps = img_meta.get('padding_size', [0] * 4)
            pl, pr, pt, pb = ps
            i_sl = seg_logits[i:i + 1, :, pt:H - pb, pl:W - pr]
            flip = img_meta.get('flip', None)
            if flip:
                fd = img_meta.get('flip_direction', None)
                i_sl = i_sl.flip(dims=(3,) if fd == 'horizontal' else (2,))
            i_sl = resize(i_sl, size=img_meta['ori_shape'], mode='bilinear',
                          align_corners=self.align_corners, warning=False).squeeze(0)
            pred = i_sl.argmax(dim=0, keepdim=True) if C > 1 else (i_sl.sigmoid() > 0.5).to(i_sl)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_sl),
                'pred_sem_seg': PixelData(data=pred)
            })
        return data_samples
