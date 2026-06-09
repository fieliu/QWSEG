"""Quality-Gated Swin-L/T Asymmetric with Mask2Former Decoder.

Same three-branch architecture as QualityGatedSwinMask2Former, but with
asymmetric backbones:
  - Common branch:  Swin-L (embed_dims=192, channels=[192, 384, 768, 1536])
  - Private branches: Swin-T (embed_dims=96, channels=[96, 192, 384, 768])

Projection layers are added to align private features to the common branch's
channel dimensions before fusion.
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
    f_hard_mask,
    downsample_mask,
    get_degradation_schedule,
    sample_level,
    _apply_degradation,
    _generate_local_mask,
    compute_cross_modal_contrastive_loss,
    compute_quality_anchor_loss,
    compute_quality_rank_loss,
)
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         SampleList, add_prefix)
from mmseg.models.backbones.swin import (
    ShiftWindowMSA, SwinBlock, SwinBlockSequence, WindowMSA,
)
from mmengine.utils import to_2tuple

# Reuse quality-aware Swin components from the symmetric model
from mmseg.models.segmentors.swin_quality_mask2former import (
    _QualityWindowMSA,
    _QualityShiftWindowMSA,
    QualitySwinBlock,
    _replace_swin_blocks_with_quality,
    _quality_score_to_swin_bias,
    _forward_swin_common_dual_pruned,
    _forward_swin_branch_pruned,
    DualGateEnhancedFusion,
)


# ---------------------------------------------------------------------------
# Projection layers: align private (Swin-T) features to common (Swin-L) dims
# ---------------------------------------------------------------------------

class PrivateProjectionLayer(nn.Module):
    """Project private branch features from Swin-T channels to Swin-L channels.

    Uses a lightweight Conv1x1 + LayerNorm + GELU projection per stage.
    """

    def __init__(self, priv_channels_list, common_channels_list):
        super().__init__()
        self.num_stages = len(priv_channels_list)
        self.projs = nn.ModuleList()
        for priv_ch, common_ch in zip(priv_channels_list, common_channels_list):
            self.projs.append(nn.Sequential(
                nn.Conv2d(priv_ch, common_ch, 1, bias=False),
                nn.GroupNorm(min(32, common_ch) if common_ch % 32 != 0 else 32, common_ch),
                nn.GELU(),
            ))

    def forward(self, priv_feat_list):
        """Project each stage's private features to common channel dims.

        Args:
            priv_feat_list: list of [B, C_priv_i, H_i, W_i]
        Returns:
            projected: list of [B, C_common_i, H_i, W_i]
        """
        return [self.projs[i](feat) for i, feat in enumerate(priv_feat_list)]


# ===================================================================
# Asymmetric Swin-L/T Model
# ===================================================================

@MODELS.register_module()
class QualityGatedSwinLTMask2Former(BaseSegmentor):
    """Quality-gated Swin-L/T asymmetric with Mask2Former decoder.

    Three-branch architecture with asymmetric backbones:
      - Common:  Swin-L backbone processing RGB+T concatenated -> zc_rgb, zc_t
      - Private: two Swin-T branches (RGB, T) -> zp_rgb, zp_t
      - Projection layers align private features to common channel dims
      - QualityPredictor x 16 (4 stages x 4 predictor sets)
      - Mask2FormerHead main decoder
      - SegformerHead auxiliary decoders (at common channel dims)
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
                 loss_quality_weight: float = 0.5,
                 loss_rank_weight: float = 0.3,
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
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 tau: float = 0.3,
                 alpha: float = 10.0,
                 tau_hard: float = 0.2,
                 fuse_epsilon: float = 1e-3,
                 fuse_beta: float = 6.0,
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

        # Common branch channel dims (Swin-L)
        common_embed_dims = backbone.get('embed_dims', 192)
        common_depths = backbone.get('depths', [2, 2, 18, 2])
        self.common_embed_dims_list = [common_embed_dims * (2 ** i) for i in range(len(common_depths))]

        # Private branch channel dims (Swin-T)
        priv_embed_dims = private_branch_rgb.get('embed_dims', 96)
        priv_depths = private_branch_rgb.get('depths', [2, 2, 6, 2])
        self.priv_embed_dims_list = [priv_embed_dims * (2 ** i) for i in range(len(priv_depths))]

        # Use common dims as the canonical embed_dims_list for downstream modules
        self.embed_dims_list = self.common_embed_dims_list

        self.tau = tau
        self.alpha = alpha
        self.tau_hard = tau_hard
        self.fuse_epsilon = fuse_epsilon
        self.fuse_beta = fuse_beta

        # QualityPredictors: common branch uses common dims, private uses private dims
        self._build_predictors()

        # Projection layers: private (Swin-T) -> common (Swin-L) channels
        self.priv_proj_rgb = PrivateProjectionLayer(
            self.priv_embed_dims_list, self.common_embed_dims_list)
        self.priv_proj_t = PrivateProjectionLayer(
            self.priv_embed_dims_list, self.common_embed_dims_list)

        self.final_fusion = DualGateEnhancedFusion(self.embed_dims_list)

        self.loss_quality_weight = loss_quality_weight
        self.loss_rank_weight = loss_rank_weight
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
        # Common predictors use common (Swin-L) channel dims
        self.predictors_common_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.common_embed_dims_list])
        self.predictors_common_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.common_embed_dims_list])
        # Private predictors use private (Swin-T) channel dims
        self.predictors_priv_rgb = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.priv_embed_dims_list])
        self.predictors_priv_t = nn.ModuleList(
            [QualityPredictor(ch) for ch in self.priv_embed_dims_list])

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
        if phase == 1 or phase == 2:
            for pl in all_p:
                for m in pl:
                    for p in m.parameters(): p.requires_grad = False
        else:
            for pl in all_p:
                for m in pl:
                    for p in m.parameters(): p.requires_grad = True
        return phase

    def _quality_anchor_all(self, s_r, s_t, spr, spt, level_rgb, level_t, pad_mask):
        """Anchor loss on all 16 quality maps (4 stages x 4 predictor sets)."""
        loss = torch.tensor(0.0, device=level_rgb.device)
        cnt = 0
        for i in range(len(s_r)):
            for s_list, lvl in [([s_r[i], spr[i]], level_rgb), ([s_t[i], spt[i]], level_t)]:
                for s in s_list:
                    if s is not None:
                        pm = pad_mask
                        if pm.shape[2:] != s.shape[2:]:
                            pm = F.interpolate(pm.float(), size=s.shape[2:], mode='nearest')
                        loss += compute_quality_anchor_loss(s, lvl, pad_mask=pm)
                        cnt += 1
        return loss / max(cnt, 1) if cnt else loss

    def _quality_rank_all(self, s_r, s_t, spr, spt,
                          ds_r, ds_t, dspr, dspt,
                          level_rgb, level_t,
                          level_rgb_deg, level_t_deg, pad_mask):
        """Rank loss: clean vs degraded, all 16 maps with dynamic margin."""
        loss = torch.tensor(0.0, device=level_rgb.device)
        cnt = 0
        for i in range(len(s_r)):
            for s_clean_list, s_deg_list, lvl_c, lvl_d in [
                ([s_r[i], spr[i]], [ds_r[i], dspt[i]], level_rgb, level_rgb_deg),
                ([s_t[i], spt[i]], [ds_t[i], dspt[i]], level_t, level_t_deg),
            ]:
                for sc, sd in zip(s_clean_list, s_deg_list):
                    if sc is not None and sd is not None:
                        pm = pad_mask
                        if pm.shape[2:] != sc.shape[2:]:
                            pm = F.interpolate(pm.float(), size=sc.shape[2:], mode='nearest')
                        loss += compute_quality_rank_loss(sc, sd, lvl_c, lvl_d, pad_mask=pm)
                        cnt += 1
        return loss / max(cnt, 1) if cnt else loss

    def _quality_weighted_common_fusion(self, zc_rgb, zc_t, s_rgb, s_t):
        if s_rgb.shape[2:] != zc_rgb.shape[2:]:
            s_rgb = F.interpolate(s_rgb, size=zc_rgb.shape[2:], mode='nearest')
        if s_t.shape[2:] != zc_t.shape[2:]:
            s_t = F.interpolate(s_t, size=zc_t.shape[2:], mode='nearest')
        w_rgb = f_fuse(s_rgb, tau=self.tau, epsilon=self.fuse_epsilon, beta=self.fuse_beta)
        w_t = f_fuse(s_t, tau=self.tau, epsilon=self.fuse_epsilon, beta=self.fuse_beta)
        w_sum = w_rgb + w_t + 1e-8
        return (w_rgb / w_sum) * zc_rgb + (w_t / w_sum) * zc_t

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
        epoch = getattr(self, 'current_epoch', 0)
        ph = self._get_training_phase(epoch)
        fa = (ph < 3)

        # Common branch: Swin-L, dual-modality forward
        zc_outs, s_r, s_t = _forward_swin_common_dual_pruned(
            self.backbone, torch.cat([rgb, t], dim=0), orig_B=B,
            predictors_rgb=self.predictors_common_rgb,
            predictors_t=self.predictors_common_t,
            training=self.training, force_all_keep=fa, phase=ph,
            tau=self.tau, alpha=self.alpha, tau_hard=self.tau_hard)
        zc_r = [f[:B] for f in zc_outs]; zc_t = [f[B:] for f in zc_outs]

        # Private branches: Swin-T, single-modality forward
        zp_r_raw, spr = _forward_swin_branch_pruned(
            self.private_branch_rgb, rgb, self.predictors_priv_rgb,
            training=self.training, force_all_keep=fa, phase=ph,
            tau=self.tau, alpha=self.alpha, tau_hard=self.tau_hard)

        zp_t_raw, spt = _forward_swin_branch_pruned(
            self.private_branch_t, t, self.predictors_priv_t,
            training=self.training, force_all_keep=fa, phase=ph,
            tau=self.tau, alpha=self.alpha, tau_hard=self.tau_hard)

        # Project private features from Swin-T dims to Swin-L dims
        zp_r = self.priv_proj_rgb(zp_r_raw)
        zp_t = self.priv_proj_t(zp_t_raw)

        # Fusion: A-scheme — common fused as sole base, private as quality-weighted residual
        zf, re, te = [], [], []
        for i in range(len(self.embed_dims_list)):
            sri = s_r[i] if s_r[i] is not None else torch.ones(B,1,zc_r[i].shape[2],zc_r[i].shape[3],device=zc_r[i].device)
            sti = s_t[i] if s_t[i] is not None else torch.ones(B,1,zc_t[i].shape[2],zc_t[i].shape[3],device=zc_t[i].device)
            fused = self._quality_weighted_common_fusion(zc_r[i], zc_t[i], sri, sti)
            zf_i = fused.permute(0,2,3,1).contiguous()
            zf_i = F.layer_norm(zf_i, [zf_i.size(-1)])
            zf_i = zf_i.permute(0,3,1,2).contiguous()
            zf.append(zf_i)

            # RGB private: projected zp_r already at common dims
            if zf[i].shape[2:] == zp_r[i].shape[2:]:
                spr_i = spr[i] if spr[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device)
                w_priv = f_fuse(spr_i, tau=self.tau, epsilon=self.fuse_epsilon, beta=self.fuse_beta)
                priv_mod = zp_r[i] * w_priv
                out = zf[i] + priv_mod
                out = out.permute(0,2,3,1).contiguous()
                out = F.layer_norm(out, [out.size(-1)])
                out = out.permute(0,3,1,2).contiguous()
                re.append(out)
            else:
                re.append(zp_r[i])

            # T private: projected zp_t already at common dims
            if zf[i].shape[2:] == zp_t[i].shape[2:]:
                spt_i = spt[i] if spt[i] is not None else torch.ones(B,1,zf[i].shape[2],zf[i].shape[3],device=zf[i].device)
                w_priv = f_fuse(spt_i, tau=self.tau, epsilon=self.fuse_epsilon, beta=self.fuse_beta)
                priv_mod = zp_t[i] * w_priv
                out = zf[i] + priv_mod
                out = out.permute(0,2,3,1).contiguous()
                out = F.layer_norm(out, [out.size(-1)])
                out = out.permute(0,3,1,2).contiguous()
                te.append(out)
            else:
                te.append(zp_t[i])

        # A-scheme fusion: common fused as sole base, private as pure residual
        ff = self.final_fusion(zp_r, zp_t, zf)
        all_s = [s for sl in [s_r, s_t, spr, spt] for s in sl]
        return zc_r, zc_t, zp_r_raw, zp_t_raw, zp_r, zp_t, zf, re, te, ff, s_r, s_t, all_s, spr, spt

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
        # level map: [B,1,H,W], clean=1, missing/global=L, local=1+(L-1)*lm
        level_rgb = torch.ones(B, 1, H, W, device=dev, dtype=rgb.dtype)
        level_t = torch.ones(B, 1, H, W, device=dev, dtype=ir.dtype)
        ep = getattr(self,'current_epoch',0)
        sched = get_degradation_schedule(min(ep/max(self.total_epochs,1),1.0))
        for b in range(B):
            r = random.random()
            if r < sched['p_missing']:
                lv = 5
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,deg_type='missing',level=lv)
                    dtr[b]='missing'
                    level_rgb[b:b+1] = lv
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,deg_type='missing',level=lv)
                    dtt[b]='missing'
                    level_t[b:b+1] = lv
            elif r < sched['p_missing']+sched['p_global']:
                lv = sample_level(sched['global_levels'])
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv); dtr[b]='global'
                    level_rgb[b:b+1] = lv
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv); dtt[b]='global'
                    level_t[b:b+1] = lv
            else:
                lv = sample_level(sched['local_levels'])
                lm = _generate_local_mask(1,H,W,num_regions=3,device=dev,level=lv)
                if random.random() < 0.5:
                    dr[b:b+1] = _apply_degradation(rgb[b:b+1],'rgb',rm,rs,level=lv,is_local=True,local_mask=lm)
                    dtr[b]='local'
                    level_rgb[b:b+1] = 1 + (lv - 1) * lm
                else:
                    di[b:b+1] = _apply_degradation(ir[b:b+1],'thermal',im,iss,level=lv,is_local=True,local_mask=lm)
                    dtt[b]='local'
                    level_t[b:b+1] = 1 + (lv - 1) * lm
        self._last_level_rgb = level_rgb
        self._last_level_t = level_t
        return dr.to(rgb.dtype), di.to(ir.dtype), dtr, dtt

    # ---- loss / predict / inference ----

    def loss(self, inputs, data_samples):
        rgb, ir = inputs[:,:3], inputs[:,3:]
        B = rgb.shape[0]
        ep = getattr(self,'current_epoch',0)
        ph = self._get_training_phase(ep)
        self._update_training_phase(ep)

        # Helper: split a list/tensor into clean[:B] and degraded[B:]
        def _split(lst):
            if lst is None: return None, None
            if isinstance(lst, list):
                return [x[:B] for x in lst], [x[B:] for x in lst]
            return lst[:B], lst[B:]

        # Phase 3: batch clean + degraded into a single forward pass
        # Phase 1/2: clean only
        if self.training and ph >= 3:
            dr, di, _, _ = self._generate_degraded_inputs(rgb, ir)
            cat_rgb = torch.cat([rgb, dr], dim=0)
            cat_t = torch.cat([ir, di], dim=0)
            (cat_zcr,cat_zct,cat_zpr_raw,cat_zpt_raw,cat_zpr,cat_zpt,
             cat_zf,cat_re,cat_te,cat_ff,cat_sr,cat_st,cat_all_s,cat_spr,cat_spt) = \
                self._extract_feat_single(cat_rgb, cat_t)
            (zc_r, dzc_r) = _split(cat_zcr)
            (zc_t, dzc_t) = _split(cat_zct)
            (zp_r_raw, dzp_r_raw) = _split(cat_zpr_raw)
            (zp_t_raw, dzp_t_raw) = _split(cat_zpt_raw)
            (zp_r, dzp_r) = _split(cat_zpr)
            (zp_t, dzp_t) = _split(cat_zpt)
            (zf, dzf) = _split(cat_zf)
            (re, drl) = _split(cat_re)
            (te, dtl) = _split(cat_te)
            (ff, df) = _split(cat_ff)
            (s_r, ds_r) = _split(cat_sr)
            (s_t, ds_t) = _split(cat_st)
            (spr, dspr) = _split(cat_spr)
            (spt, dspt) = _split(cat_spt)
        else:
            (zc_r,zc_t,zp_r_raw,zp_t_raw,zp_r,zp_t,
             zf,re,te,ff,s_r,s_t,all_s,spr,spt) = self._extract_feat_single(rgb,ir)
            dzc_r=dzc_t=dzp_r_raw=dzp_t_raw=dzp_r=dzp_t=dzf=drl=dtl=df=ds_r=ds_t=dspr=dspt=None

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

        # Quality anchor loss on clean forward (only in Phase 3 when predictors are unfrozen)
        if self.loss_quality_weight > 0 and ph >= 3:
            pm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
            level_clean_rgb = torch.ones(B, 1, inputs.shape[-2], inputs.shape[-1],
                                         device=inputs.device, dtype=inputs.dtype)
            level_clean_t = torch.ones(B, 1, inputs.shape[-2], inputs.shape[-1],
                                       device=inputs.device, dtype=inputs.dtype)
            losses['loss_quality'] = self.loss_quality_weight * self._quality_anchor_all(
                s_r, s_t, spr, spt, level_clean_rgb, level_clean_t,
                pad_mask=pm.float().unsqueeze(1) if pm.ndim == 3 else pm.float())

        if self.training and ph >= 3:
            if torch.isnan(df[0]).any():
                import logging
                logging.getLogger(__name__).warning(
                    'NaN in degraded features — falling back to clean features for deg losses')
                dzc_r,dzc_t,dzp_r_raw,dzp_t_raw,dzp_r,dzp_t = zc_r,zc_t,zp_r_raw,zp_t_raw,zp_r,zp_t
                dzf,drl,dtl,df = zf,re,te,ff
                ds_r,ds_t,dspr,dspt = s_r,s_t,spr,spt
            losses.update(add_prefix(self.decode_head.loss(df,data_samples,self.train_cfg),'deg_decode'))
            for head, feats, pfx in [
                (self.common_decode_head, dzf, 'deg_common_decode'),
                (self.rgb_private_decode_head, drl, 'deg_rgb_private_decode'),
                (self.t_private_decode_head, dtl, 'deg_t_private_decode'),
            ]:
                if head and feats:
                    ld = {k: v*self.aux_loss_weight for k,v in head.loss(feats,data_samples,self.train_cfg).items()}
                    losses.update(add_prefix(ld, pfx))
            if self.loss_align_weight > 0 and dzc_r is not None and dzc_t is not None:
                dlc, dcnt = 0., 0
                for i in range(len(dzc_r)):
                    if dzc_r[i] is not None and dzc_t[i] is not None:
                        dDr_i = (ds_r[i] > self.tau).float().detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else torch.ones(B,1,dzc_r[i].shape[2],dzc_r[i].shape[3],device=dzc_r[i].device)
                        dDt_i = (ds_t[i] > self.tau).float().detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else torch.ones(B,1,dzc_t[i].shape[2],dzc_t[i].shape[3],device=dzc_t[i].device)
                        dsr_i = ds_r[i].detach() if ds_r is not None and i < len(ds_r) and ds_r[i] is not None else None
                        dst_i = ds_t[i].detach() if ds_t is not None and i < len(ds_t) and ds_t[i] is not None else None
                        dlc += compute_cross_modal_contrastive_loss(
                            dzc_r[i],dzc_t[i],gt,dDr_i,dDt_i,dsr_i,dst_i,
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

            # Quality anchor loss on degraded forward (level-5 → push < 0.2)
            if self.loss_quality_weight > 0 and hasattr(self, '_last_level_rgb'):
                dpm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
                losses['loss_quality_deg'] = self.loss_quality_weight * self._quality_anchor_all(
                    ds_r, ds_t, dspr, dspt,
                    self._last_level_rgb, self._last_level_t,
                    pad_mask=dpm.float().unsqueeze(1) if dpm.ndim == 3 else dpm.float())

            # Quality rank loss (clean vs degraded, all 16 maps)
            if self.loss_rank_weight > 0 and hasattr(self, '_last_level_rgb'):
                rpm = self._build_pad_mask(data_samples, inputs.shape[-2], inputs.shape[-1], inputs.device)
                level_clean_rgb = torch.ones(B, 1, inputs.shape[-2], inputs.shape[-1],
                                             device=inputs.device, dtype=inputs.dtype)
                level_clean_t = torch.ones(B, 1, inputs.shape[-2], inputs.shape[-1],
                                           device=inputs.device, dtype=inputs.dtype)
                losses['loss_rank'] = self.loss_rank_weight * self._quality_rank_all(
                    s_r, s_t, spr, spt,
                    ds_r, ds_t, dspr, dspt,
                    level_clean_rgb, level_clean_t,
                    self._last_level_rgb, self._last_level_t,
                    pad_mask=rpm.float().unsqueeze(1) if rpm.ndim == 3 else rpm.float())

        for key in list(losses.keys()):
            if not torch.isfinite(losses[key]):
                losses[key] = torch.tensor(0.0, device=losses[key].device)
            else:
                losses[key] = torch.clamp(losses[key], max=100.0)
        return losses

    def encode_decode(self, inputs, bm):
        rgb, ir = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,ir)[9]
        if self.with_neck: ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def encode_decode_with_missing(self, inputs, bm, mask_rgb=False, mask_t=False):
        """Encode-decode with one modality forced to missing."""
        rgb, t = inputs[:, :3], inputs[:, 3:]
        dev = inputs.device
        if mask_rgb:
            mean = self.data_preprocessor.mean[:3].view(1, 3, 1, 1).to(dev)
            std = self.data_preprocessor.std[:3].view(1, 3, 1, 1).to(dev)
            rgb = torch.full_like(rgb, 0) - mean / std
        if mask_t:
            mean = self.data_preprocessor.mean[3:].view(1, 3, 1, 1).to(dev)
            std = self.data_preprocessor.std[3:].view(1, 3, 1, 1).to(dev)
            t = torch.full_like(t, 0) - mean / std
        ff = self._extract_feat_single(rgb, t)[9]
        if self.with_neck:
            ff = self.neck(ff)
        return self.decode_head.predict(ff, bm, self.test_cfg)

    def extract_feat(self, inputs):
        rgb, t = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,t)[9]
        return self.neck(ff) if self.with_neck else ff

    def extract_feat_vis(self, inputs):
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]
        with torch.no_grad():
            (zc_r,zc_t,zp_r_raw,zp_t_raw,zp_r,zp_t,
             zf,re,te,ff,s_r,s_t,all_s,spr,spt) = self._extract_feat_single(rgb,t)
            fused = self.neck(ff) if self.with_neck else ff

            deg_rgb, deg_t, deg_type_rgb, deg_type_t = self._generate_degraded_vis_inputs(rgb, t)
            (dzc_r,dzc_t,dzp_r_raw,dzp_t_raw,dzp_r,dzp_t,
             dzf,drl,dtl,df,ds_r,ds_t,dall_s,dspr,dspt) = self._extract_feat_single(deg_rgb, deg_t)
            fused_d = self.neck(df) if self.with_neck else df

        for i in range(len(zf)):
            if dzc_r[i].shape[-2:] != zc_r[i].shape[-2:]:
                dzc_r[i] = F.interpolate(dzc_r[i], size=zc_r[i].shape[-2:], mode='bilinear')
                dzc_t[i] = F.interpolate(dzc_t[i], size=zc_t[i].shape[-2:], mode='bilinear')
                dzf[i] = F.interpolate(dzf[i], size=zf[i].shape[-2:], mode='bilinear')
                dzp_r[i] = F.interpolate(dzp_r[i], size=zp_r[i].shape[-2:], mode='bilinear')
                dzp_t[i] = F.interpolate(dzp_t[i], size=zp_t[i].shape[-2:], mode='bilinear')
                drl[i] = F.interpolate(drl[i], size=re[i].shape[-2:], mode='bilinear')
                dtl[i] = F.interpolate(dtl[i], size=te[i].shape[-2:], mode='bilinear')
                df[i] = F.interpolate(df[i], size=ff[i].shape[-2:], mode='bilinear')

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
            zc_rgb_deg=dzc_r, zc_t_deg=dzc_t,
            zc_fused_deg=dzf,
            zp_rgb_deg=dzp_r, zp_t_deg=dzp_t,
            rgb_pf_deg=drl, t_pf_deg=dtl,
            final_fused_deg=fused_d,
            s_rgb_deg=ds_r, s_t_deg=ds_t,
            s_rgb_priv_deg=dspr, s_t_priv_deg=dspt,
            q_rgb_deg=ds_r, q_t_deg=ds_t,
            q_rgb_priv_deg=dspr, q_t_priv_deg=dspt,
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
        return self._generate_degraded_inputs(rgb, ir)

    def _forward(self, inputs, data_samples=None):
        rgb, t = inputs[:,:3], inputs[:,3:]
        ff = self._extract_feat_single(rgb,t)[9]
        feats = self.neck(ff) if self.with_neck else ff
        return self._get_seg_logits(feats, data_samples)

    def predict(self, inputs, data_samples=None):
        if data_samples:
            bm = [ds.metainfo for ds in data_samples]
        else:
            bm = [dict(ori_shape=inputs.shape[2:],img_shape=inputs.shape[2:],pad_shape=inputs.shape[2:],padding_size=[0,0,0,0])]*inputs.shape[0]
        sl = self.encode_decode(inputs, bm)
        return self.postprocess_result(sl, data_samples)

    def predict_with_missing(self, inputs, data_samples, mask_rgb=False, mask_t=False):
        bm = [ds.metainfo for ds in data_samples]
        sl = self.encode_decode_with_missing(
            inputs, bm, mask_rgb=mask_rgb, mask_t=mask_t)
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
