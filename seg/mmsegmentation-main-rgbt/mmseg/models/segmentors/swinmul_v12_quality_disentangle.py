import os
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class STARSFusionBlock(BaseModule):

    def __init__(self, common_dim, private_dim):
        super().__init__()
        self.private_proj = nn.Sequential(
            nn.Conv2d(private_dim, common_dim, 1, bias=False),
            nn.GroupNorm(32, common_dim),
            nn.GELU())

        self.gate = nn.Sequential(
            nn.Conv2d(common_dim * 2, common_dim, 1, bias=False),
            nn.GroupNorm(32, common_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(common_dim, common_dim, 1, bias=False),
            nn.Sigmoid())

        self.refine = nn.Sequential(
            nn.Conv2d(common_dim * 2, common_dim, 1, bias=False),
            nn.GroupNorm(32, common_dim),
            nn.GELU())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, common_feat, private_feat):
        private_proj = self.private_proj(private_feat)
        gate = self.gate(torch.cat([common_feat, private_proj], dim=1))
        gated_private = gate * private_proj
        fused = common_feat + gated_private
        fused = self.refine(torch.cat([fused, private_proj], dim=1)) + fused
        return fused


class SoftModNet(BaseModule):

    def __init__(self, channels, mid_ratio=4):
        super().__init__()
        mid_ch = max(channels // mid_ratio, 8)
        self.modulate = nn.Sequential(
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.GroupNorm(max(mid_ch // 8, 1), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, channels, 1, bias=False))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x, alpha, q, threshold):
        B, C, H, W = x.shape
        if q.shape[2:] != (H, W):
            q = F.interpolate(q, size=(H, W), mode='nearest')
        gate = torch.relu(torch.tanh(alpha * (q - threshold)))
        delta = self.modulate(x)
        return x + gate * delta


class FinalFusionBlock(BaseModule):

    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        if isinstance(out_channels_list, int):
            out_channels_list = [out_channels_list] * len(in_channels_list)
        self.num_stages = len(in_channels_list)
        self.out_channels_list = out_channels_list
        self.ca_rgb = nn.ModuleList()
        self.sa_rgb = nn.ModuleList()
        self.ca_t = nn.ModuleList()
        self.sa_t = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i, ch in enumerate(in_channels_list):
            self.ca_rgb.append(ChannelAttention(ch))
            self.sa_rgb.append(SpatialAttention())
            self.ca_t.append(ChannelAttention(ch))
            self.sa_t.append(SpatialAttention())
            self.projs.append(nn.Sequential(
                nn.Conv2d(ch * 2, out_channels_list[i], 1, bias=False),
                nn.GroupNorm(
                    min(32, out_channels_list[i])
                    if out_channels_list[i] % 32 != 0
                    else 32,
                    out_channels_list[i]),
                nn.GELU()))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, rgb_private_fused, t_private_fused):
        fused_list = []
        for i in range(self.num_stages):
            rgb_enhanced = self.sa_rgb[i](self.ca_rgb[i](rgb_private_fused[i]))
            t_enhanced = self.sa_t[i](self.ca_t[i](t_private_fused[i]))
            concat = torch.cat([rgb_enhanced, t_enhanced], dim=1)
            fused = self.projs[i](concat)
            fused_list.append(fused)
        return fused_list


def _repair_tokens_before_downsample(x_2d, x_other_2d, q_self, q_other,
                                     channel_proj, threshold):
    B, C, H, W = x_2d.shape
    if q_self.shape[2:] != (H, W):
        q_self = F.interpolate(q_self, size=(H, W), mode='nearest')
    if q_other.shape[2:] != (H, W):
        q_other = F.interpolate(q_other, size=(H, W), mode='nearest')
    if x_other_2d.shape[2:] != (H, W):
        x_other_2d = F.interpolate(x_other_2d, size=(H, W), mode='bilinear',
                                   align_corners=False)

    low_self = (q_self.squeeze(1) < threshold)
    high_other = (q_other.squeeze(1) >= threshold)

    repaired = x_2d.clone()

    cross_repair_mask = low_self & high_other
    if cross_repair_mask.any():
        projected_other = channel_proj(x_other_2d)
        mask_2d = cross_repair_mask.float().unsqueeze(1).expand_as(repaired)
        repaired = torch.where(mask_2d.bool(), projected_other, repaired)

    still_low = low_self & (~high_other)
    if still_low.any():
        high_self = ~low_self
        high_count = high_self.float().sum(dim=[1, 2], keepdim=True).unsqueeze(1)
        high_sum = (x_2d * high_self.unsqueeze(1).float()).sum(
            dim=[2, 3], keepdim=True)
        global_mean = high_sum / (high_count + 1e-8)

        kernel_size = min(H, W, 5)
        pad = kernel_size // 2
        weight = torch.ones(C, 1, kernel_size, kernel_size,
                            device=x_2d.device) / (kernel_size ** 2)
        local_avg = F.conv2d(
            x_2d * high_self.unsqueeze(1).float(), weight,
            groups=C, padding=pad)
        norm_conv = F.conv2d(
            high_self.unsqueeze(1).float(),
            torch.ones(1, 1, kernel_size, kernel_size, device=x_2d.device) /
            (kernel_size ** 2), padding=pad)
        local_avg = local_avg / (norm_conv + 1e-8)
        local_avg = torch.where(norm_conv > 0.5, local_avg, global_mean)

        interp_mask = still_low.float().unsqueeze(1).expand_as(repaired)
        repaired = torch.where(interp_mask.bool(), local_avg, repaired)

    return repaired


def compute_quality_weighted_alignment_loss(zc_rgb, zc_t, q_rgb, q_t,
                                            threshold=0.3):
    B, C, H, W = zc_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)

    valid_mask = (q_rgb_s > threshold) & (q_t_s > threshold)

    q_min = torch.min(q_rgb_s, q_t_s)
    q_gap = torch.abs(q_rgb_s - q_t_s)
    q_gap_normalized = 1.0 - q_gap / (q_gap.max() + 1e-8)
    weight = q_min * q_gap_normalized * valid_mask.float()

    token_dist = ((zc_rgb - zc_t) ** 2).mean(dim=1)

    total_weight = weight.sum() + 1e-8
    loss = (weight * token_dist).sum() / total_weight

    return loss


def _build_quality_attn_mask(q_2d, threshold, H, W, B,
                             window_size=7, shift_size=0,
                             num_heads=1, orig_B=None,
                             q_other_2d=None):
    if q_2d.shape[2:] != (H, W):
        q_2d = F.interpolate(q_2d, size=(H, W), mode='nearest')
    q_s = q_2d.squeeze(1)

    if orig_B is not None and B == orig_B * 2:
        q_rgb_s = q_s[:orig_B]
        q_t_s = q_s[orig_B:]

        rgb_low = q_rgb_s < threshold
        t_low = q_t_s < threshold
        both_low = rgb_low & t_low
        rgb_better = q_rgb_s >= q_t_s

        rgb_low_final = rgb_low & ~(both_low & rgb_better)
        t_low_final = t_low & ~(both_low & ~rgb_better)

        low_quality_mask = torch.cat(
            [rgb_low_final.float(), t_low_final.float()], dim=0)
    elif q_other_2d is not None:
        if q_other_2d.shape[2:] != (H, W):
            q_other_2d = F.interpolate(q_other_2d, size=(H, W),
                                       mode='nearest')
        q_other_s = q_other_2d.squeeze(1)

        cur_low = q_s < threshold
        other_low = q_other_s < threshold
        both_low = cur_low & other_low
        cur_better = q_s >= q_other_s

        low_quality_mask = (cur_low & ~(both_low & cur_better)).float()
    else:
        low_quality_mask = (q_s < threshold).float()

    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        low_quality_mask = F.pad(low_quality_mask.unsqueeze(1),
                                 (0, pad_r, 0, pad_b), mode='constant',
                                 value=1.0).squeeze(1)

    Hp, Wp = low_quality_mask.shape[1], low_quality_mask.shape[2]

    if shift_size > 0:
        low_quality_mask = torch.roll(
            low_quality_mask,
            shifts=(-shift_size, -shift_size), dims=(1, 2))

    nW_h = Hp // window_size
    nW_w = Wp // window_size
    nW = nW_h * nW_w
    N = window_size * window_size

    mask_windows = low_quality_mask.unfold(1, window_size,
                                           window_size).unfold(
        2, window_size, window_size)
    mask_windows = mask_windows.contiguous().view(B, nW, N)

    row_mask = mask_windows.unsqueeze(3).expand(-1, -1, -1, N)
    col_mask = mask_windows.unsqueeze(2).expand(-1, -1, N, -1)
    combined = torch.clamp(row_mask + col_mask, max=1.0)

    attn_bias = torch.where(combined.bool(),
                            torch.tensor(-50.0,
                                         device=q_2d.device),
                            torch.tensor(0.0,
                                         device=q_2d.device))

    attn_bias = attn_bias.reshape(B * nW, 1, N, N).expand(
        B * nW, num_heads, N, N).contiguous()
    return attn_bias


def _get_keep_mask_1d(q_2d, threshold, H, W, B, orig_B=None,
                      q_other_2d=None):
    if q_2d.shape[2:] != (H, W):
        q_2d = F.interpolate(q_2d, size=(H, W), mode='nearest')
    q_s = q_2d.squeeze(1)

    if orig_B is not None and B == orig_B * 2:
        q_rgb_s = q_s[:orig_B]
        q_t_s = q_s[orig_B:]

        rgb_low = q_rgb_s < threshold
        t_low = q_t_s < threshold
        both_low = rgb_low & t_low
        rgb_better = q_rgb_s >= q_t_s

        rgb_keep = (q_rgb_s >= threshold) | (both_low & rgb_better)
        t_keep = (q_t_s >= threshold) | (both_low & ~rgb_better)

        rgb_mask = torch.where(rgb_keep,
                               torch.ones_like(q_rgb_s),
                               torch.zeros_like(q_rgb_s))
        t_mask = torch.where(t_keep,
                             torch.ones_like(q_t_s),
                             torch.zeros_like(q_t_s))

        keep = torch.cat([rgb_mask, t_mask], dim=0)
    elif q_other_2d is not None:
        if q_other_2d.shape[2:] != (H, W):
            q_other_2d = F.interpolate(q_other_2d, size=(H, W),
                                       mode='nearest')
        q_other_s = q_other_2d.squeeze(1)

        cur_low = q_s < threshold
        other_low = q_other_s < threshold
        both_low = cur_low & other_low
        cur_better = q_s >= q_other_s

        cur_keep = (~cur_low) | (both_low & cur_better)
        keep = torch.where(cur_keep,
                           torch.ones_like(q_s),
                           torch.zeros_like(q_s))
    else:
        keep = torch.where(q_s >= threshold,
                           torch.ones_like(q_s),
                           torch.zeros_like(q_s))

    keep_1d = keep.reshape(B, -1)
    return keep_1d


def _forward_swin_with_quality(backbone, img, q_maps, threshold,
                            orig_B=None, q_other_maps=None):
    x, hw_shape = backbone.patch_embed(img)

    if backbone.use_abs_pos_embed:
        x = x + backbone.absolute_pos_embed
    x = backbone.drop_after_pos(x)

    outs = []
    cur_h, cur_w = hw_shape

    for i, stage in enumerate(backbone.stages):
        keep_mask_1d = None
        q_i_stage = None
        q_i_other = None
        if i < len(q_maps):
            q_i = q_maps[i]
            if q_i.shape[2:] != (cur_h, cur_w):
                q_i = F.interpolate(q_i, size=(cur_h, cur_w),
                                    mode='nearest')
            if q_other_maps is not None and i < len(q_other_maps):
                q_i_other = q_other_maps[i]
                if q_i_other.shape[2:] != (cur_h, cur_w):
                    q_i_other = F.interpolate(
                        q_i_other, size=(cur_h, cur_w), mode='nearest')
            keep_mask_1d = _get_keep_mask_1d(
                q_i, threshold, cur_h, cur_w, x.shape[0],
                orig_B=orig_B, q_other_2d=q_i_other)
            q_i_stage = q_i

        for j, block in enumerate(stage.blocks):
            attn_mask = None
            if q_i_stage is not None:
                blk_attn = getattr(block, 'attn', None)
                blk_shift = getattr(blk_attn, 'shift_size', 0)
                blk_ws = getattr(blk_attn, 'window_size', 7)
                blk_heads = getattr(blk_attn, 'w_msa', None)
                blk_num_heads = getattr(blk_heads, 'num_heads', 1) if blk_heads is not None else 1
                attn_mask = _build_quality_attn_mask(
                    q_i_stage, threshold, cur_h, cur_w, x.shape[0],
                    window_size=blk_ws, shift_size=blk_shift,
                    num_heads=blk_num_heads, orig_B=orig_B,
                    q_other_2d=q_i_other)
            x = block(x, hw_shape, attn_mask=attn_mask)

            if keep_mask_1d is not None:
                x = x * keep_mask_1d.unsqueeze(-1)

        if i in backbone.out_indices:
            norm_layer = getattr(backbone, f'norm{i}')
            out = norm_layer(x)
            if keep_mask_1d is not None:
                out = out * keep_mask_1d.unsqueeze(-1)
            out = out.view(-1, *hw_shape, x.shape[-1]).permute(
                0, 3, 1, 2).contiguous()
            outs.append(out)

        if stage.downsample is not None:
            x, hw_shape = stage.downsample(x, hw_shape)
            cur_h, cur_w = hw_shape

    return outs


@MODELS.register_module()
class SwinMulV12QualityDisentangle(BaseSegmentor):

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
                 soft_alpha: float = 10.0,
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

        self.soft_mod_common = nn.ModuleList()
        self.soft_mod_private = nn.ModuleList()
        self.cross_proj_rgb_to_t = nn.ModuleList()
        self.cross_proj_t_to_rgb = nn.ModuleList()
        for ch in self.embed_dims_list:
            self.soft_mod_common.append(SoftModNet(ch, mid_ratio=4))
            self.soft_mod_private.append(SoftModNet(ch, mid_ratio=4))
            self.cross_proj_rgb_to_t.append(nn.Conv2d(ch, ch, 1, bias=False))
            self.cross_proj_t_to_rgb.append(nn.Conv2d(ch, ch, 1, bias=False))

        self.stars_fusion_rgb = nn.ModuleList()
        self.stars_fusion_t = nn.ModuleList()
        for ch in self.embed_dims_list:
            self.stars_fusion_rgb.append(STARSFusionBlock(ch, ch))
            self.stars_fusion_t.append(STARSFusionBlock(ch, ch))

        self.final_fusion = FinalFusionBlock(self.embed_dims_list,
                                             self.embed_dims_list)

        self.quality_threshold = quality_threshold
        self.loss_align_weight = loss_align_weight
        self.soft_alpha = soft_alpha
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

        q_sum = q_rgb_s + q_t_s + 1e-8
        q_rgb_norm = (q_rgb_s / q_sum).unsqueeze(1)
        q_t_norm = (q_t_s / q_sum).unsqueeze(1)

        fused = q_rgb_norm * zc_rgb + q_t_norm * zc_t
        return fused

    @staticmethod
    def _zero_low_quality(feat_list, q_maps, threshold):
        result = []
        for i, feat in enumerate(feat_list):
            q_i = q_maps[i]
            if q_i.shape[2:] != feat.shape[2:]:
                q_i = F.interpolate(q_i, size=feat.shape[2:],
                                    mode='nearest')
            soft_mask = torch.sigmoid(10.0 * (q_i - threshold))
            soft_mask = torch.clamp(soft_mask, min=0.05)
            result.append(feat * soft_mask)
        return result

    def extract_feat(self, input_rgb: torch.Tensor,
                     input_t: torch.Tensor):
        B = input_rgb.shape[0]

        q_rgb_maps = self.quality_pyramid_net.forward_rgb(input_rgb)
        q_t_maps = self.quality_pyramid_net.forward_thermal(input_t)

        zc_rgb_list = _forward_swin_with_quality(
            self.backbone, input_rgb, q_rgb_maps, self.quality_threshold)
        zc_t_list = _forward_swin_with_quality(
            self.backbone, input_t, q_t_maps, self.quality_threshold)

        zc_rgb_repaired = []
        zc_t_repaired = []
        for i in range(len(zc_rgb_list)):
            zc_rgb_r = _repair_tokens_before_downsample(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i],
                self.cross_proj_t_to_rgb[i], self.quality_threshold)
            zc_t_r = _repair_tokens_before_downsample(
                zc_t_list[i], zc_rgb_list[i],
                q_t_maps[i], q_rgb_maps[i],
                self.cross_proj_rgb_to_t[i], self.quality_threshold)
            zc_rgb_repaired.append(zc_rgb_r)
            zc_t_repaired.append(zc_t_r)

        zc_rgb_repaired = self._zero_low_quality(
            zc_rgb_repaired, q_rgb_maps, self.quality_threshold)
        zc_t_repaired = self._zero_low_quality(
            zc_t_repaired, q_t_maps, self.quality_threshold)

        zp_rgb_list = _forward_swin_with_quality(
            self.private_branch_rgb, input_rgb, q_rgb_maps,
            self.quality_threshold)
        zp_t_list = _forward_swin_with_quality(
            self.private_branch_t, input_t, q_t_maps,
            self.quality_threshold)

        zp_rgb_list = self._zero_low_quality(
            zp_rgb_list, q_rgb_maps, self.quality_threshold)
        zp_t_list = self._zero_low_quality(
            zp_t_list, q_t_maps, self.quality_threshold)

        num_stages = len(zc_rgb_repaired)
        zc_fused_list = []
        rgb_private_fused_list = []
        t_private_fused_list = []

        for i in range(num_stages):
            zc_fused = self._quality_weighted_common_fusion(
                zc_rgb_repaired[i], zc_t_repaired[i],
                q_rgb_maps[i], q_t_maps[i])
            zc_fused_list.append(zc_fused)

            rgb_pf = self.stars_fusion_rgb[i](zc_fused, zp_rgb_list[i])
            t_pf = self.stars_fusion_t[i](zc_fused, zp_t_list[i])
            rgb_private_fused_list.append(rgb_pf)
            t_private_fused_list.append(t_pf)

        final_fused = self.final_fusion(rgb_private_fused_list,
                                        t_private_fused_list)

        self._last_cache = {
            'zc_rgb': [f.detach() for f in zc_rgb_repaired],
            'zc_t': [f.detach() for f in zc_t_repaired],
            'zp_rgb': [f.detach() for f in zp_rgb_list],
            'zp_t': [f.detach() for f in zp_t_list],
            'zc_fused': [f.detach() for f in zc_fused_list],
            'rgb_private_fused': [f.detach() for f in rgb_private_fused_list],
            't_private_fused': [f.detach() for f in t_private_fused_list],
            'final_fused': [f.detach() for f in final_fused],
            'q_rgb': [f.detach() for f in q_rgb_maps],
            'q_t': [f.detach() for f in q_t_maps],
        }

        return (zc_fused_list, rgb_private_fused_list,
                t_private_fused_list, final_fused)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        zc_fused = results[0]

        if self.with_neck:
            zc_fused = self.neck(zc_fused)

        seg_logits = self.decode_head.predict(
            zc_fused, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples, head=None):
        if head is None:
            head = self.decode_head
        inputs = [torch.clamp(torch.nan_to_num(f, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
                  for f in inputs]
        loss_decode = head.loss(inputs, data_samples, self.train_cfg)
        return loss_decode

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        (zc_fused_list, rgb_private_fused_list,
         t_private_fused_list, final_fused) = self.extract_feat(
            input_rgb, input_ir)

        cache = self._last_cache
        q_rgb_maps = cache['q_rgb']
        q_t_maps = cache['q_t']

        losses = dict()

        if self.with_neck:
            final_fused_neck = self.neck(zc_fused_list)
        else:
            final_fused_neck = zc_fused_list

        loss_final = self._decode_head_forward_train(
            final_fused_neck, data_samples)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_list
            if self.with_neck:
                common_feats = self.neck(common_feats)
            loss_common = self._decode_head_forward_train(
                common_feats, data_samples, self.common_decode_head)
            losses.update(add_prefix(loss_common, 'common'))

        if self.rgb_private_decode_head is not None:
            rgb_pf = rgb_private_fused_list
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            loss_rgb_pf = self._decode_head_forward_train(
                rgb_pf, data_samples, self.rgb_private_decode_head)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))

        if self.t_private_decode_head is not None:
            t_pf = t_private_fused_list
            if self.with_neck:
                t_pf = self.neck(t_pf)
            loss_t_pf = self._decode_head_forward_train(
                t_pf, data_samples, self.t_private_decode_head)
            losses.update(add_prefix(loss_t_pf, 't_private'))

        zc_rgb_list = cache['zc_rgb']
        zc_t_list = cache['zc_t']
        num_stages = len(zc_rgb_list)

        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += compute_quality_weighted_alignment_loss(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i],
                threshold=self.quality_threshold)
        losses['loss_align'] = (loss_align_total / num_stages *
                                self.loss_align_weight)

        return losses

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        zc_fused = results[0]
        if self.with_neck:
            zc_fused = self.neck(zc_fused)
        return self.decode_head.forward(zc_fused)

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
                x1 = w_idx * h_stride
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
