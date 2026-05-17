import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import nlc_to_nchw
from .base import BaseSegmentor


class CrossModalRectification(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid())
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid())

    def forward(self, feat_to_rectify, guide_feat):
        c_gate = self.channel_gate(guide_feat)
        feat_c = feat_to_rectify * c_gate
        avg_out = torch.mean(feat_c, dim=1, keepdim=True)
        max_out, _ = torch.max(feat_c, dim=1, keepdim=True)
        s_gate = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        return feat_c * s_gate + feat_to_rectify


class ComplementaryChannelFilter(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, private_feat, universal_guide):
        c_gate = self.channel_gate(universal_guide)
        return private_feat * c_gate + private_feat


class EfficientCrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8, sr_ratio=1, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, hw_shape):
        B, N_q, D = query.shape
        residual = query

        q = self.q_proj(query).reshape(
            B, N_q, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1 and hw_shape is not None:
            kv = key_value.transpose(1, 2).reshape(
                B, D, hw_shape[0], hw_shape[1])
            kv = self.sr(kv)
            kv = kv.flatten(2).transpose(1, 2)
            kv = self.sr_norm(kv)
        else:
            kv = key_value

        N_kv = kv.shape[1]
        k = self.k_proj(kv).reshape(
            B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).reshape(
            B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        out = self.norm(out + residual)
        return out


class QualityGatedFusionBlock(nn.Module):

    def __init__(self, shared_dim, private_dim, num_heads=8,
                 sr_ratio=1, use_cross_attn=True):
        super().__init__()
        self.shared_dim = shared_dim

        self.private_proj_rgb = nn.Sequential(
            nn.Conv2d(private_dim, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True))
        self.private_proj_t = nn.Sequential(
            nn.Conv2d(private_dim, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True))

        self.universal_refine = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1,
                      groups=shared_dim, bias=False),
            nn.Conv2d(shared_dim, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim))

        self.private_gate_rgb = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, shared_dim, 1, bias=False),
            nn.Sigmoid())

        self.private_gate_t = nn.Sequential(
            nn.Conv2d(shared_dim * 2, shared_dim, 1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_dim, shared_dim, 1, bias=False),
            nn.Sigmoid())

    def forward(self, zc_rgb, zc_t, zp_rgb, zp_t,
                q_rgb=None, q_t=None, threshold=0.1):
        B, C, H, W = zc_rgb.shape

        if q_rgb is not None and q_t is not None:
            if q_rgb.shape[2:] != (H, W):
                q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
            if q_t.shape[2:] != (H, W):
                q_t = F.interpolate(q_t, size=(H, W), mode='nearest')
            q_rgb_s = q_rgb.squeeze(1)
            q_t_s = q_t.squeeze(1)

            q_sum = q_rgb_s + q_t_s + 1e-8
            alpha = (q_rgb_s / q_sum).unsqueeze(1)

            rgb_valid = (q_rgb_s >= threshold).float().unsqueeze(1)
            t_valid = (q_t_s >= threshold).float().unsqueeze(1)
            both_valid = rgb_valid * t_valid
        else:
            alpha = torch.full((B, 1, H, W), 0.5, device=zc_rgb.device)
            both_valid = torch.ones(B, 1, H, W, device=zc_rgb.device)
            rgb_valid = torch.ones(B, 1, H, W, device=zc_rgb.device)
            t_valid = torch.ones(B, 1, H, W, device=zc_rgb.device)

        zc_avg = alpha * zc_rgb + (1 - alpha) * zc_t

        refine = self.universal_refine(
            torch.cat([zc_rgb, zc_t], dim=1))
        zc_fused = zc_avg + both_valid * refine

        zp_rgb_proj = self.private_proj_rgb(zp_rgb)
        zp_t_proj = self.private_proj_t(zp_t)

        gate_rgb = self.private_gate_rgb(
            torch.cat([zc_fused, zp_rgb_proj], dim=1))
        gate_t = self.private_gate_t(
            torch.cat([zc_fused, zp_t_proj], dim=1))

        fused = zc_fused + rgb_valid * gate_rgb * zp_rgb_proj + \
            t_valid * gate_t * zp_t_proj

        return fused, zc_fused


def compute_orthogonal_loss(zc, zp):
    B, C, N = zc.shape
    zc_norm = F.normalize(zc, dim=2)
    zp_norm = F.normalize(zp, dim=2)
    corr = torch.bmm(zc_norm, zp_norm.transpose(1, 2))
    loss = (corr ** 2).mean()
    return loss


def compute_complementarity_loss(zc, zp, q_map=None, threshold=0.1):
    B, C, N = zc.shape
    zc_norm = F.normalize(zc, dim=1)
    zp_norm = F.normalize(zp, dim=1)
    sim = (zc_norm * zp_norm).sum(dim=1)
    redundancy_loss = F.relu(sim - 0.3).mean()
    zp_var = zp.var(dim=2).mean()
    diversity_loss = F.relu(1e-4 - zp_var).mean()
    return redundancy_loss + diversity_loss


def compute_infonce_loss(zc_rgb, zc_t, temperature=0.07):
    B = zc_rgb.shape[0]
    zc_rgb_gap = F.adaptive_avg_pool2d(zc_rgb, 1).flatten(1)
    zc_t_gap = F.adaptive_avg_pool2d(zc_t, 1).flatten(1)
    zc_rgb_norm = F.normalize(zc_rgb_gap, dim=1)
    zc_t_norm = F.normalize(zc_t_gap, dim=1)
    sim = torch.mm(zc_rgb_norm, zc_t_norm.t()) / temperature
    labels = torch.arange(B, device=zc_rgb.device)
    loss = (F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.t(), labels)) / 2
    return loss


def compute_quality_weighted_alignment_loss(zc_rgb, zc_t, q_rgb, q_t,
                                             threshold=0.1):
    B, C, H, W = zc_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)

    valid_mask = (q_rgb_s > threshold) & (q_t_s > threshold)

    q_min = torch.min(q_rgb_s, q_t_s)
    weight = q_min * valid_mask.float()

    token_dist = ((zc_rgb - zc_t) ** 2).mean(dim=1)

    total_weight = weight.sum() + 1e-8
    loss = (weight * token_dist).sum() / total_weight

    return loss


@MODELS.register_module()
class MiTMulV10QualityEmbed(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_pyramid_net: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 universal_embed_dims=None,
                 private_embed_dims=None,
                 loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
                 disentangle_strategy='complementarity',
                 loss_invariant_weight=0.5,
                 loss_align_weight=0.5,
                 quality_threshold=0.1,
                 quality_pretrained=None,
                 quality_freeze_epochs=0,
                 use_cross_attn=True,
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
        self._init_auxiliary_head(auxiliary_head)

        if universal_embed_dims is None:
            universal_embed_dims = [64, 128, 320, 512]
        if private_embed_dims is None:
            private_embed_dims = [32, 64, 128, 256]
        self.universal_embed_dims = universal_embed_dims
        self.private_embed_dims = private_embed_dims

        num_stages = len(universal_embed_dims)

        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)
        self._quality_pretrained = quality_pretrained

        attn_sr_ratios = [8, 4, 2, 1]
        attn_num_heads = [1, 2, 4, 8]
        self.fusion_blocks = nn.ModuleList()
        for i in range(num_stages):
            sr = attn_sr_ratios[i] if i < len(attn_sr_ratios) else 1
            nh = attn_num_heads[i] if i < len(attn_num_heads) else 8
            self.fusion_blocks.append(
                QualityGatedFusionBlock(
                    shared_dim=universal_embed_dims[i],
                    private_dim=private_embed_dims[i],
                    num_heads=nh,
                    sr_ratio=sr,
                    use_cross_attn=use_cross_attn))

        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.disentangle_strategy = disentangle_strategy
        self.loss_invariant_weight = loss_invariant_weight
        self.loss_align_weight = loss_align_weight
        self.quality_threshold = quality_threshold
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

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
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

    def _load_quality_pretrained(self):
        quality_pretrained = self._quality_pretrained
        if not quality_pretrained:
            print_log(
                'No quality pretrained weight specified',
                logger='current')
            return

        if not os.path.exists(quality_pretrained):
            print_log(
                f'Quality pretrained weight NOT FOUND: '
                f'{quality_pretrained}',
                logger='current')
            return

        ckpt = torch.load(quality_pretrained, map_location='cpu')
        if 'model_state_dict' in ckpt:
            q_state_dict = ckpt['model_state_dict']
        else:
            q_state_dict = ckpt

        model_state = self.quality_pyramid_net.state_dict()
        matched_keys = []
        mismatched_keys = []
        missing_keys = []
        for k, v in q_state_dict.items():
            if k not in model_state:
                missing_keys.append(k)
            elif model_state[k].shape != v.shape:
                mismatched_keys.append(
                    (k, v.shape, model_state[k].shape))
            else:
                matched_keys.append(k)

        new_state_dict = {
            k: v for k, v in q_state_dict.items() if k in matched_keys}
        self.quality_pyramid_net.load_state_dict(new_state_dict, strict=False)

        print_log(
            f'Quality pyramid net: {quality_pretrained}', logger='current')
        print_log(
            f'  Matched: {len(matched_keys)}/{len(model_state)} keys',
            logger='current')
        if mismatched_keys:
            print_log(
                f'  Shape mismatched: {len(mismatched_keys)} keys',
                logger='current')
            for key, src_shape, dst_shape in mismatched_keys[:5]:
                print_log(
                    f'    {key}: {src_shape} -> {dst_shape}',
                    logger='current')
        if missing_keys:
            print_log(
                f'  Extra in checkpoint: {len(missing_keys)} keys',
                logger='current')
        if not mismatched_keys and not missing_keys:
            print_log(
                '  Quality pyramid net loaded successfully '
                '(all keys matched)', logger='current')

    def init_weights(self):
        self._load_quality_pretrained()

        if self.init_cfg is not None and 'checkpoint' in self.init_cfg:
            checkpoint_path = self.init_cfg['checkpoint']
            print_log(
                f'Loading checkpoint from: {checkpoint_path}',
                logger='current')

            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt

                new_state_dict = {}
                loaded_keys = []
                skipped_keys = []

                skip_prefixes = (
                    'quality_pyramid_net.',
                    'fusion_blocks.',
                    'decode_head.', 'auxiliary_head.')

                for k, v in state_dict.items():
                    if any(k.startswith(p) for p in skip_prefixes):
                        continue

                    if k in self.state_dict().keys():
                        if self.state_dict()[k].shape == v.shape:
                            new_state_dict[k] = v
                            loaded_keys.append(k)
                        else:
                            skipped_keys.append(
                                (k, v.shape, self.state_dict()[k].shape))

                print_log(
                    f'Loaded {len(loaded_keys)} keys from checkpoint',
                    logger='current')
                if skipped_keys:
                    print_log(
                        f'Skipped {len(skipped_keys)} keys due to shape '
                        f'mismatch:', logger='current')
                    for key, src_shape, dst_shape in skipped_keys[:10]:
                        print_log(
                            f'  {key}: {src_shape} -> {dst_shape}',
                            logger='current')

                self.load_state_dict(new_state_dict, strict=False)
                print_log(
                    'Successfully initialized from checkpoint',
                    logger='current')

            except Exception as e:
                print_log(
                    f'Failed to load checkpoint: {e}', logger='current',
                    level=logging.WARNING)
                super().init_weights()
        else:
            super().init_weights()

    @staticmethod
    def _build_attn_mask(valid_mask, num_heads, N_q, hw_shape, sr_ratio,
                         device):
        """Build attention mask from valid token mask.

        Args:
            valid_mask: [B, N] bool tensor, True = valid token
            num_heads: number of attention heads
            N_q: number of query tokens
            hw_shape: (H, W) spatial shape of query tokens
            sr_ratio: spatial reduction ratio for K/V
            device: torch device

        Returns:
            attn_mask: [B*num_heads, N_q, N_kv] float tensor,
                       -inf for masked positions, 0 for valid
        """
        B = valid_mask.shape[0]
        if sr_ratio > 1 and hw_shape is not None:
            H, W = hw_shape
            H_kv = H // sr_ratio
            W_kv = W // sr_ratio
            N_kv = H_kv * W_kv
            kv_valid = valid_mask.reshape(B, H, W).float()
            kv_valid = F.max_pool2d(
                kv_valid.unsqueeze(1),
                kernel_size=sr_ratio, stride=sr_ratio)
            kv_valid = (kv_valid.squeeze(1) > 0.5).reshape(B, N_kv)
        else:
            kv_valid = valid_mask
            N_kv = N_q

        kv_mask = ~kv_valid
        kv_mask = kv_mask.unsqueeze(1).expand(B, N_q, N_kv)
        attn_mask = torch.zeros_like(kv_mask, dtype=torch.float32)
        attn_mask.masked_fill_(kv_mask, float('-inf'))
        attn_mask = attn_mask.repeat(num_heads, 1, 1)
        return attn_mask

    def _forward_backbone_with_quality(self, backbone, x, q_maps,
                                        q_other_maps=None,
                                        threshold=0.1):
        outs = []
        for i, layer in enumerate(backbone.layers):
            x, hw_shape = layer[0](x)
            H, W = hw_shape
            N = H * W

            attn_mask = None
            rescued_valid = None

            if i < len(q_maps):
                q_i = q_maps[i]
                if q_i.shape[2:] != (H, W):
                    q_i = F.interpolate(q_i, size=(H, W), mode='nearest')
                q_flat = q_i.squeeze(1).reshape(x.shape[0], -1)

                if q_flat.shape[1] == N:
                    self_valid = (q_flat >= threshold)

                    if q_other_maps is not None and i < len(q_other_maps):
                        q_other_i = q_other_maps[i]
                        if q_other_i.shape[2:] != (H, W):
                            q_other_i = F.interpolate(
                                q_other_i, size=(H, W), mode='nearest')
                        q_other_flat = q_other_i.squeeze(1).reshape(
                            x.shape[0], -1)
                        other_valid = (q_other_flat >= threshold)
                        both_invalid = ~self_valid & ~other_valid
                        rescue = both_invalid & (q_flat >= q_other_flat)
                        rescued_valid = self_valid | rescue
                    else:
                        rescued_valid = self_valid

                    if i == 0:
                        q_token = q_flat.unsqueeze(2).clamp(min=0.0)
                        q_token[~rescued_valid] = 0.0
                        x = x * q_token

                    num_heads = layer[1][0].attn.num_heads
                    sr_ratio = getattr(layer[1][0].attn, 'sr_ratio', 1)
                    attn_mask = self._build_attn_mask(
                        rescued_valid, num_heads, N, hw_shape,
                        sr_ratio, x.device)

            for block in layer[1]:
                x = block(x, hw_shape, attn_mask=attn_mask)

            x = layer[2](x)

            if rescued_valid is not None:
                x[~rescued_valid] = 0.0

            x = nlc_to_nchw(x, hw_shape)
            if i in backbone.out_indices:
                outs.append(x)
        return outs

    def extract_feat(self, input_rgb: torch.Tensor,
                     input_t: torch.Tensor):
        B = input_rgb.shape[0]

        q_rgb_maps = self.quality_pyramid_net.forward_rgb(input_rgb)
        q_t_maps = self.quality_pyramid_net.forward_thermal(input_t)

        zc_rgb_list = self._forward_backbone_with_quality(
            self.backbone, input_rgb, q_rgb_maps, q_t_maps,
            self.quality_threshold)
        zc_t_list = self._forward_backbone_with_quality(
            self.backbone, input_t, q_t_maps, q_rgb_maps,
            self.quality_threshold)

        zp_rgb_list = self._forward_backbone_with_quality(
            self.private_branch_rgb, input_rgb, q_rgb_maps, q_t_maps,
            self.quality_threshold)
        zp_t_list = self._forward_backbone_with_quality(
            self.private_branch_t, input_t, q_t_maps, q_rgb_maps,
            self.quality_threshold)

        num_stages = len(zc_rgb_list)

        fused_feats = []
        zc_fused_feats = []
        for i in range(num_stages):
            fused, zc_fused = self.fusion_blocks[i](
                zc_rgb_list[i], zc_t_list[i],
                zp_rgb_list[i], zp_t_list[i],
                q_rgb=q_rgb_maps[i], q_t=q_t_maps[i],
                threshold=self.quality_threshold)
            fused_feats.append(fused)
            zc_fused_feats.append(zc_fused)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
            zc_fused_feats = self.neck(zc_fused_feats)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_feats, fused_feats,
                q_rgb_maps, q_t_maps)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        fused_feats = results[5]
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(
                    inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(
                inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def _update_quality_freeze_status(self):
        if self.quality_freeze_epochs <= 0:
            return
        epoch = getattr(self, 'epoch', 0)
        if epoch < self.quality_freeze_epochs and not self._quality_frozen:
            self.quality_pyramid_net.eval()
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = False
            self._quality_frozen = True
            print_log(
                f'Quality pyramid net FROZEN (epoch {epoch}/'
                f'{self.quality_freeze_epochs})', logger='current')
        elif epoch >= self.quality_freeze_epochs and self._quality_frozen:
            self.quality_pyramid_net.train()
            for p in self.quality_pyramid_net.parameters():
                p.requires_grad = True
            self._quality_frozen = False
            print_log(
                f'Quality pyramid net UNFROZEN (epoch {epoch}/'
                f'{self.quality_freeze_epochs})', logger='current')

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_feats, fused_feats,
         q_rgb_maps, q_t_maps) = self.extract_feat(input_rgb, input_ir)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        num_stages = len(zc_rgb_list)

        loss_invariant_total = 0.0
        for i in range(num_stages):
            loss_invariant_total += compute_infonce_loss(
                zc_rgb_list[i], zc_t_list[i])
        losses['loss_invariant'] = loss_invariant_total / num_stages * \
            self.loss_invariant_weight

        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += compute_quality_weighted_alignment_loss(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i],
                threshold=self.quality_threshold)
        losses['loss_align'] = loss_align_total / num_stages * \
            self.loss_align_weight

        for i in range(num_stages):
            zp_rgb_aligned = self.fusion_blocks[i].private_proj_rgb(
                zp_rgb_list[i])
            zp_t_aligned = self.fusion_blocks[i].private_proj_t(
                zp_t_list[i])

            zc_feat = zc_fused_feats[i].flatten(2)
            zp_rgb_feat = zp_rgb_aligned.flatten(2)
            zp_t_feat = zp_t_aligned.flatten(2)

            w = self.loss_disentangle_weights[i] \
                if i < len(self.loss_disentangle_weights) \
                else self.loss_disentangle_weights[-1]

            if self.disentangle_strategy == 'orthogonal':
                loss_d_rgb = compute_orthogonal_loss(zc_feat, zp_rgb_feat)
                loss_d_t = compute_orthogonal_loss(zc_feat, zp_t_feat)
            elif self.disentangle_strategy == 'complementarity':
                loss_d_rgb = compute_complementarity_loss(
                    zc_feat, zp_rgb_feat)
                loss_d_t = compute_complementarity_loss(
                    zc_feat, zp_t_feat)
            else:
                loss_d_rgb = torch.tensor(0.0, device=zc_feat.device)
                loss_d_t = torch.tensor(0.0, device=zc_feat.device)

            losses[f'loss_disentangle_s{i}'] = \
                (loss_d_rgb + loss_d_t) * w

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: torch.Tensor,
                data_samples: OptSampleList = None) -> SampleList:
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

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        results = self.extract_feat(input_rgb, input_ir)
        fused_feats = results[5]
        return self.decode_head.forward(fused_feats)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros(
            (batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros(
            (batch_size, 1, h_img, w_img))
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
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        return seg_logit

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        batch_size, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
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
