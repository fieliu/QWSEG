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
from .base import BaseSegmentor
from .swinmul_v12_quality_disentangle import (
    ChannelAttention, SpatialAttention,
    compute_quality_weighted_alignment_loss,
    _forward_swin_with_quality,
    _get_keep_mask_1d,
)
from mmseg.datasets.transforms.quality_degradation import (
    apply_quality_degradation_rgb, apply_quality_degradation_t,
    _QUALITY_RGB_DEG_TYPES, _QUALITY_T_DEG_TYPES,
)


class CrossAttentionEnhance(BaseModule):

    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, private_feat, common_feat, keep_mask_1d=None):
        B, C, H, W = private_feat.shape
        N = H * W

        priv_nlc = private_feat.flatten(2).transpose(1, 2)
        comm_nlc = common_feat.flatten(2).transpose(1, 2)

        q = self.q_proj(priv_nlc).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(comm_nlc).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(comm_nlc).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        residual = priv_nlc + out
        if keep_mask_1d is not None:
            residual = residual * keep_mask_1d.unsqueeze(-1)

        residual = self.norm(residual)
        if keep_mask_1d is not None:
            residual = residual * keep_mask_1d.unsqueeze(-1)

        return residual.transpose(1, 2).reshape(B, C, H, W)


class QualityWeightedFinalFusion(BaseModule):

    def __init__(self, in_channels_list, num_classes=9):
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
            self.mlps.append(nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.GroupNorm(min(32, ch) if ch % 32 != 0 else 32, ch),
                nn.GELU(),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.GroupNorm(min(32, ch) if ch % 32 != 0 else 32, ch),
            ))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, rgb_enhanced_list, t_enhanced_list, q_rgb_maps, q_t_maps, threshold):
        fused_list = []
        for i in range(self.num_stages):
            rgb_feat = rgb_enhanced_list[i]
            t_feat = t_enhanced_list[i]
            B, C, H, W = rgb_feat.shape

            q_rgb = q_rgb_maps[i]
            q_t = q_t_maps[i]
            if q_rgb.shape[2:] != (H, W):
                q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
            if q_t.shape[2:] != (H, W):
                q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

            q_rgb_s = q_rgb.squeeze(1)
            q_t_s = q_t.squeeze(1)

            both_low = (q_rgb_s < threshold) & (q_t_s < threshold)
            rgb_better = q_rgb_s >= q_t_s

            q_sum = q_rgb_s + q_t_s + 1e-8
            w_rgb = (q_rgb_s / q_sum).unsqueeze(1)
            w_t = (q_t_s / q_sum).unsqueeze(1)

            fused = w_rgb * rgb_feat + w_t * t_feat

            if both_low.any():
                both_low_3d = both_low.unsqueeze(1).expand_as(fused)
                rgb_better_3d = rgb_better.unsqueeze(1).expand_as(fused)
                fused = torch.where(
                    both_low_3d,
                    torch.where(rgb_better_3d, rgb_feat, t_feat),
                    fused)

            rgb_att = self.sa_rgb[i](self.ca_rgb[i](rgb_feat))
            t_att = self.sa_t[i](self.ca_t[i](t_feat))
            fused = fused + rgb_att + t_att

            fused = fused + self.mlps[i](fused)

            fused_list.append(fused)

        return fused_list


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
            local_mask = F.interpolate(local_mask.float(), size=(H, W),
                                       mode='nearest')
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


@MODELS.register_module()
class SwinMulV12DQualityDisentangle(BaseSegmentor):

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
                 loss_invariant_weight: float = 1.0,
                 loss_distill_weight: float = 1.0,
                 aux_loss_weight: float = 0.3,
                 missing_ratio: float = 0.3,
                 global_deg_ratio: float = 0.3,
                 local_deg_ratio: float = 0.4,
                 quality_pretrained: Optional[str] = None,
                 quality_freeze_epochs: int = 0,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_aux_heads(common_decode_head,
                             rgb_private_decode_head,
                             t_private_decode_head,
                             auxiliary_head)

        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        embed_dims = backbone.get('embed_dims', 96)
        depths = backbone.get('depths', [2, 2, 6, 2])
        self.embed_dims_list = [embed_dims * (2 ** i) for i in range(len(depths))]

        self.cross_attn_rgb = nn.ModuleList()
        self.cross_attn_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            num_heads_i = max(1, ch // 64)
            self.cross_attn_rgb.append(
                CrossAttentionEnhance(ch, num_heads=num_heads_i))
            self.cross_attn_t.append(
                CrossAttentionEnhance(ch, num_heads=num_heads_i))

        self.final_fusion = QualityWeightedFinalFusion(self.embed_dims_list)

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
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

        self._last_cache = {}

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

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
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def train(self, mode=True):
        super().train(mode)
        if self._quality_frozen and mode:
            self.quality_pyramid_net.eval()
        return self

    def _update_quality_freeze_status(self, epoch=None):
        if epoch is not None:
            should_freeze = epoch < self.quality_freeze_epochs
        else:
            should_freeze = self.quality_freeze_epochs > 0
        if should_freeze and not self._quality_frozen:
            self._quality_frozen = True
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self.quality_pyramid_net.eval()
            print_log(f'Quality net FROZEN (epoch {epoch}, '
                      f'freeze_epochs={self.quality_freeze_epochs})',
                      logger='current')
        elif not should_freeze and self._quality_frozen:
            self._quality_frozen = False
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self.quality_pyramid_net.train()
            print_log(f'Quality net UNFROZEN (epoch {epoch})',
                      logger='current')

    def _load_quality_pretrained(self):
        if not self._quality_pretrained:
            return
        if not os.path.exists(self._quality_pretrained):
            print_log(f'Quality pretrained NOT FOUND: '
                      f'{self._quality_pretrained}', logger='current')
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

        both_low = (q_rgb_s < self.quality_threshold) & (q_t_s < self.quality_threshold)
        rgb_better = q_rgb_s >= q_t_s

        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)

        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t

        if both_low.any():
            both_low_expand = both_low.unsqueeze(1).expand_as(fused)
            rgb_better_expand = rgb_better.unsqueeze(1).expand_as(fused)
            fused = torch.where(
                both_low_expand,
                torch.where(rgb_better_expand, zc_rgb, zc_t),
                fused)

        return fused

    @staticmethod
    def _apply_cross_modal_masking(zc_rgb, zc_t, q_rgb, q_t, threshold):
        B, C, H, W = zc_rgb.shape
        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)

        rgb_low = q_rgb_s < threshold
        t_low = q_t_s < threshold
        both_low = rgb_low & t_low

        rgb_better = q_rgb_s >= q_t_s

        masked_rgb = zc_rgb.clone()
        masked_t = zc_t.clone()

        rgb_zero_mask = rgb_low & (~both_low | ~rgb_better)
        t_zero_mask = t_low & (~both_low | rgb_better)

        if rgb_zero_mask.any():
            zero_expand = rgb_zero_mask.unsqueeze(1).expand_as(masked_rgb)
            masked_rgb = masked_rgb.masked_fill(zero_expand, 0.0)

        if t_zero_mask.any():
            zero_expand = t_zero_mask.unsqueeze(1).expand_as(masked_t)
            masked_t = masked_t.masked_fill(zero_expand, 0.0)

        return masked_rgb, masked_t

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]

        q_rgb_maps = self.quality_pyramid_net.forward_rgb(input_rgb)
        q_t_maps = self.quality_pyramid_net.forward_thermal(input_t)

        input_rgbt = torch.cat([input_rgb, input_t], dim=0)
        q_rgbt_maps = [torch.cat([q_r, q_t], dim=0)
                       for q_r, q_t in zip(q_rgb_maps, q_t_maps)]

        zc_rgbt_list = _forward_swin_with_quality(
            self.backbone, input_rgbt, q_rgbt_maps,
            self.quality_threshold, orig_B=B)

        zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
        zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        zp_rgb_list = _forward_swin_with_quality(
            self.private_branch_rgb, input_rgb, q_rgb_maps,
            self.quality_threshold, q_other_maps=q_t_maps)
        zp_t_list = _forward_swin_with_quality(
            self.private_branch_t, input_t, q_t_maps,
            self.quality_threshold, q_other_maps=q_rgb_maps)

        num_stages = len(zc_rgb_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            zc_fused = self._quality_weighted_common_fusion(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i])
            zc_fused_list.append(zc_fused)

            q_rgb_i = q_rgb_maps[i]
            q_t_i = q_t_maps[i]
            H, W = zp_rgb_list[i].shape[2], zp_rgb_list[i].shape[3]
            if q_rgb_i.shape[2:] != (H, W):
                q_rgb_i = F.interpolate(q_rgb_i, size=(H, W), mode='nearest')
            if q_t_i.shape[2:] != (H, W):
                q_t_i = F.interpolate(q_t_i, size=(H, W), mode='nearest')

            rgb_keep = _get_keep_mask_1d(
                q_rgb_i, self.quality_threshold, H, W, B,
                q_other_2d=q_t_i)
            t_keep = _get_keep_mask_1d(
                q_t_i, self.quality_threshold, H, W, B,
                q_other_2d=q_rgb_i)

            rgb_enh = self.cross_attn_rgb[i](
                zp_rgb_list[i], zc_fused, keep_mask_1d=rgb_keep)
            t_enh = self.cross_attn_t[i](
                zp_t_list[i], zc_fused, keep_mask_1d=t_keep)
            rgb_enhanced_list.append(rgb_enh)
            t_enhanced_list.append(t_enh)

        final_fused = self.final_fusion(
            rgb_enhanced_list, t_enhanced_list,
            q_rgb_maps, q_t_maps, self.quality_threshold)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list,
                t_enhanced_list, final_fused,
                q_rgb_maps, q_t_maps)

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
                deg_rgb = _apply_degradation(input_rgb, 'rgb',
                                             rgb_mean, rgb_std)
                deg_type_rgb = 'global_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't',
                                           ir_mean, ir_std)
                deg_type_t = 'global_deg'
        else:
            local_mask = _generate_local_mask(B, H, W, device=device)
            if random.random() < 0.5:
                deg_rgb = _apply_degradation(input_rgb, 'rgb',
                                             rgb_mean, rgb_std,
                                             is_local=True,
                                             local_mask=local_mask)
                deg_type_rgb = 'local_deg'
            else:
                deg_t = _apply_degradation(input_ir, 't',
                                           ir_mean, ir_std,
                                           is_local=True,
                                           local_mask=local_mask)
                deg_type_t = 'local_deg'

        return deg_rgb, deg_t, deg_type_rgb, deg_type_t

    def extract_feat(self, input_rgb: torch.Tensor,
                     input_t: torch.Tensor):
        results = self._extract_feat_single(input_rgb, input_t)

        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list,
         t_enhanced_list, final_fused,
         q_rgb_maps, q_t_maps) = results

        self._last_cache = {
            'zc_rgb': [f.detach() for f in zc_rgb_list],
            'zc_t': [f.detach() for f in zc_t_list],
            'zp_rgb': [f.detach() for f in zp_rgb_list],
            'zp_t': [f.detach() for f in zp_t_list],
            'zc_fused': [f.detach() for f in zc_fused_list],
            'rgb_private_fused': [f.detach() for f in rgb_enhanced_list],
            't_private_fused': [f.detach() for f in t_enhanced_list],
            'final_fused': [f.detach() for f in final_fused],
            'q_rgb': [f.detach() for f in q_rgb_maps],
            'q_t': [f.detach() for f in q_t_maps],
        }

        return (zc_fused_list, rgb_enhanced_list,
                t_enhanced_list, final_fused)

    def _compute_invariant_loss(self, clean_zc_list, deg_zc_list):
        num_stages = len(clean_zc_list)
        loss = 0.0
        for i in range(num_stages):
            loss = loss + F.mse_loss(clean_zc_list[i].detach(), deg_zc_list[i])
        return loss / num_stages

    def _compute_distill_loss(self, clean_logits, deg_logits):
        clean_prob = F.softmax(clean_logits.detach(), dim=1)
        deg_log_prob = F.log_softmax(deg_logits, dim=1)
        loss = -(clean_prob * deg_log_prob).sum(dim=1).mean()
        return loss

    @staticmethod
    def _check_feats_valid(feats, name='feats'):
        for i, feat in enumerate(feats):
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                raise ValueError(
                    f'{name}[{i}] contains NaN/Inf! '
                    f'shape={feat.shape} '
                    f'range=[{feat.min().item():.4f}, {feat.max().item():.4f}] '
                    f'nan={torch.isnan(feat).sum().item()} '
                    f'inf={torch.isinf(feat).sum().item()}')

    def _decode_head_forward(self, feats, data_samples, head=None):
        if head is None:
            head = self.decode_head
        losses = head.loss(feats, data_samples, self.train_cfg)
        return losses

    def _decode_head_predict_logits(self, feats, head=None):
        if head is None:
            head = self.decode_head
        if hasattr(head, 'pixel_decoder'):
            batch_img_metas = [
                dict(ori_shape=feats[0].shape[2:],
                     img_shape=feats[0].shape[2:])
            ] * feats[0].shape[0]
            seg_logits = head.predict(feats, batch_img_metas,
                                      self.test_cfg)
        else:
            seg_logits = head(feats)
        return seg_logits

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        clean_results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_clean, zc_t_clean, zp_rgb_clean, zp_t_clean,
         zc_fused_clean, rgb_enh_clean, t_enh_clean,
         final_fused_clean, q_rgb_clean, q_t_clean) = clean_results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused_clean)
        else:
            final_neck = final_fused_clean

        self._check_feats_valid(final_neck, 'final_neck')

        loss_final = self._decode_head_forward(final_neck, data_samples)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_clean
            if self.with_neck:
                common_feats = self.neck(common_feats)
            self._check_feats_valid(common_feats, 'common_feats')
            loss_common = self._decode_head_forward(
                common_feats, data_samples, self.common_decode_head)
            losses.update(add_prefix(loss_common, 'common'))
            for k in list(losses.keys()):
                if k.startswith('common.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_enh_clean
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            self._check_feats_valid(rgb_pf, 'rgb_pf')
            rgb_pf_detached = [f.detach() for f in rgb_pf]
            loss_rgb_pf = self._decode_head_forward(
                rgb_pf_detached, data_samples, self.rgb_private_decode_head)
            q_rgb_scale = (q_rgb_clean[0] >= 0.5).float().mean().clamp(min=0.1)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight * q_rgb_scale

        if self.t_private_decode_head is not None:
            t_pf = t_enh_clean
            if self.with_neck:
                t_pf = self.neck(t_pf)
            self._check_feats_valid(t_pf, 't_pf')
            t_pf_detached = [f.detach() for f in t_pf]
            loss_t_pf = self._decode_head_forward(
                t_pf_detached, data_samples, self.t_private_decode_head)
            q_t_scale = (q_t_clean[0] >= 0.5).float().mean().clamp(min=0.1)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight * q_t_scale

        num_stages = len(zc_rgb_clean)
        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += compute_quality_weighted_alignment_loss(
                zc_rgb_clean[i], zc_t_clean[i],
                q_rgb_clean[i], q_t_clean[i],
                threshold=self.quality_threshold)
        losses['loss_align'] = (loss_align_total / num_stages *
                                self.loss_align_weight)

        deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
            self._generate_degraded_inputs(input_rgb, input_ir)

        deg_results = self._extract_feat_single(deg_rgb, deg_t)
        (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
         zc_fused_deg, rgb_enh_deg, t_enh_deg,
         final_fused_deg, q_rgb_deg, q_t_deg) = deg_results

        loss_invariant = 0.0
        if deg_type_rgb != 'none' or deg_type_t != 'none':
            loss_invariant += self._compute_invariant_loss(
                zc_fused_clean, zc_fused_deg)
        losses['loss_invariant'] = loss_invariant * self.loss_invariant_weight

        with torch.no_grad():
            clean_final_logits = self._decode_head_predict_logits(
                final_neck)

        if self.with_neck:
            deg_final_neck = self.neck(final_fused_deg)
        else:
            deg_final_neck = final_fused_deg

        self._check_feats_valid(deg_final_neck, 'deg_final_neck')
        deg_final_logits = self._decode_head_predict_logits(deg_final_neck)
        losses['loss_distill_final'] = (
            self._compute_distill_loss(clean_final_logits, deg_final_logits) *
            self.loss_distill_weight)

        if self.common_decode_head is not None:
            with torch.no_grad():
                clean_common_logits = self._decode_head_predict_logits(
                    self.neck(zc_fused_clean) if self.with_neck else zc_fused_clean,
                    self.common_decode_head)
            deg_common_logits = self._decode_head_predict_logits(
                self.neck(zc_fused_deg) if self.with_neck else zc_fused_deg,
                self.common_decode_head)
            losses['loss_distill_common'] = (
                self._compute_distill_loss(clean_common_logits,
                                           deg_common_logits) *
                self.loss_distill_weight)

        self._last_cache = {
            'zc_rgb': [f.detach() for f in zc_rgb_clean],
            'zc_t': [f.detach() for f in zc_t_clean],
            'zp_rgb': [f.detach() for f in zp_rgb_clean],
            'zp_t': [f.detach() for f in zp_t_clean],
            'zc_fused': [f.detach() for f in zc_fused_clean],
            'rgb_private_fused': [f.detach() for f in rgb_enh_clean],
            't_private_fused': [f.detach() for f in t_enh_clean],
            'final_fused': [f.detach() for f in final_fused_clean],
            'q_rgb': [f.detach() for f in q_rgb_clean],
            'q_t': [f.detach() for f in q_t_clean],
            'deg_type_rgb': deg_type_rgb,
            'deg_type_t': deg_type_t,
        }

        return losses

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_ir)
        final_fused = results[7]
        if self.with_neck:
            final_fused = self.neck(final_fused)
        return self.decode_head.forward(final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self._extract_feat_single(input_rgb, input_ir)
        final_fused = results[7]

        if self.with_neck:
            final_fused = self.neck(final_fused)

        seg_logits = self.decode_head.predict(
            final_fused, batch_img_metas, self.test_cfg)
        return seg_logits

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
                crop_seg_logit = self.encode_decode(
                    crop_img, batch_img_metas)
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

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
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
        from mmseg.structures.seg_data_sample import SegDataSample
        from mmseg.models.utils import resize
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom = \
                    padding_size
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3,))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2,))
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(data=i_seg_logits),
                'pred_sem_seg':
                PixelData(data=i_seg_pred)
            })
        return data_samples
