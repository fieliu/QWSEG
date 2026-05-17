import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class SelfAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, sr_ratio=1):
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

    def forward(self, x, hw_shape=None, attn_bias=None):
        B, N, D = x.shape
        q = self.q_proj(x).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr_ratio > 1 and hw_shape is not None:
            x_kv = x.transpose(1, 2).reshape(B, D, hw_shape[0], hw_shape[1])
            x_kv = self.sr(x_kv)
            x_kv = x_kv.flatten(2).transpose(1, 2)
            x_kv = self.sr_norm(x_kv)
        else:
            x_kv = x

        k = self.k_proj(x_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_kv).reshape(
            B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(
            q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + x)
        return out


class QualityModulatedInject(nn.Module):

    def __init__(self, fused_dim, private_dim):
        super().__init__()
        self.proj_private = nn.Sequential(
            nn.Conv2d(private_dim, fused_dim, 1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.ReLU(inplace=True))
        self.inject_gate = nn.Sequential(
            nn.Conv2d(fused_dim * 2, fused_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim, fused_dim, 1),
            nn.Sigmoid())

    def forward(self, fused_feat, private_feat, quality):
        private_proj = self.proj_private(private_feat)
        if private_proj.shape[2:] != fused_feat.shape[2:]:
            private_proj = F.interpolate(
                private_proj, size=fused_feat.shape[2:],
                mode='bilinear', align_corners=False)

        if quality.shape[2:] != fused_feat.shape[2:]:
            quality = F.interpolate(
                quality, size=fused_feat.shape[2:], mode='nearest')

        gate_input = torch.cat([fused_feat, private_proj], dim=1)
        gate = self.inject_gate(gate_input)

        inject_weight = gate * quality

        return fused_feat + inject_weight * private_proj


class QualityGatedFusion(nn.Module):

    def __init__(self, dim, learnable_threshold=True, init_threshold=0.1):
        super().__init__()
        self.learnable_threshold = learnable_threshold
        if learnable_threshold:
            self.threshold_raw = nn.Parameter(
                torch.tensor(float(init_threshold)))
        else:
            self.register_buffer(
                'threshold_val', torch.tensor(float(init_threshold)))

        self.proj_rgb = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))
        self.proj_t = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True))

    def forward(self, zc_rgb, zc_t, q_rgb, q_t):
        B, C, H, W = zc_rgb.shape

        if q_rgb.shape[2:] != (H, W):
            q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
        if q_t.shape[2:] != (H, W):
            q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

        q_rgb_s = q_rgb.squeeze(1)
        q_t_s = q_t.squeeze(1)

        threshold = self.threshold_raw if self.learnable_threshold else self.threshold_val

        rgb_gate = torch.sigmoid((q_rgb_s - threshold) * 20.0)
        t_gate = torch.sigmoid((q_t_s - threshold) * 20.0)

        rgb_proj = self.proj_rgb(zc_rgb)
        t_proj = self.proj_t(zc_t)

        rgb_gated = rgb_gate.unsqueeze(1) * rgb_proj
        t_gated = t_gate.unsqueeze(1) * t_proj

        both_low = (1 - rgb_gate) * (1 - t_gate)
        rgb_higher = torch.sigmoid((q_rgb_s - q_t_s) * 20.0)
        t_higher = 1 - rgb_higher

        fallback_rgb = both_low.unsqueeze(1) * rgb_higher.unsqueeze(1) * rgb_proj
        fallback_t = both_low.unsqueeze(1) * t_higher.unsqueeze(1) * t_proj

        q_sum = q_rgb_s + q_t_s + 1e-8
        w_rgb = q_rgb_s / q_sum
        w_t = q_t_s / q_sum

        fused = (w_rgb.unsqueeze(1) * rgb_gated + w_t.unsqueeze(1) * t_gated
                 + fallback_rgb + fallback_t)

        return fused


class FusionEnhanceBlock(nn.Module):

    def __init__(self, dim, sr_ratio=1):
        super().__init__()
        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention()
        self.self_attn = SelfAttentionBlock(
            embed_dim=dim, num_heads=max(1, dim // 64),
            sr_ratio=sr_ratio)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.self_attn(tokens, hw_shape=(H, W))
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        return x


def compute_quality_aware_alignment_loss(zc_rgb, zc_t, q_rgb, q_t,
                                          threshold=0.1):
    B, C, H, W = zc_rgb.shape
    if q_rgb.shape[2:] != (H, W):
        q_rgb = F.interpolate(q_rgb, size=(H, W), mode='nearest')
    if q_t.shape[2:] != (H, W):
        q_t = F.interpolate(q_t, size=(H, W), mode='nearest')

    q_rgb_s = q_rgb.squeeze(1)
    q_t_s = q_t.squeeze(1)

    valid_mask = (q_rgb_s > threshold) & (q_t_s > threshold)

    q_diff = torch.abs(q_rgb_s - q_t_s)
    q_min = torch.min(q_rgb_s, q_t_s)

    similarity_weight = torch.exp(-q_diff * 5.0)
    quality_weight = q_min

    weight = similarity_weight * quality_weight * valid_mask.float()

    token_dist = ((zc_rgb - zc_t) ** 2).mean(dim=1)

    total_weight = weight.sum() + 1e-8
    loss = (weight * token_dist).sum() / total_weight

    return loss


@MODELS.register_module()
class MiTMulV9QualityGated(BaseSegmentor):

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
                 learnable_threshold=True,
                 init_threshold=0.1,
                 loss_align_weight=0.5,
                 quality_pretrained=None,
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
        self.init_threshold = init_threshold

        num_stages = len(universal_embed_dims)

        self.quality_pyramid_net = MODELS.build(quality_pyramid_net)

        if quality_pretrained and os.path.exists(quality_pretrained):
            ckpt = torch.load(quality_pretrained, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.quality_pyramid_net.load_state_dict(
                    ckpt['model_state_dict'], strict=False)
            else:
                self.quality_pyramid_net.load_state_dict(
                    ckpt, strict=False)
            print_log(
                f'Loaded quality pyramid net from {quality_pretrained}',
                logger='current')

        self.quality_gated_fusion = nn.ModuleList()
        self.private_inject_rgb = nn.ModuleList()
        self.private_inject_t = nn.ModuleList()
        self.fusion_enhance = nn.ModuleList()
        attn_sr_ratios = [8, 4, 2, 1]
        for i in range(num_stages):
            self.quality_gated_fusion.append(
                QualityGatedFusion(
                    dim=universal_embed_dims[i],
                    learnable_threshold=learnable_threshold,
                    init_threshold=init_threshold))
            self.private_inject_rgb.append(
                QualityModulatedInject(
                    fused_dim=universal_embed_dims[i],
                    private_dim=private_embed_dims[i]))
            self.private_inject_t.append(
                QualityModulatedInject(
                    fused_dim=universal_embed_dims[i],
                    private_dim=private_embed_dims[i]))
            sr = attn_sr_ratios[i] if i < len(attn_sr_ratios) else 1
            self.fusion_enhance.append(
                FusionEnhanceBlock(
                    dim=universal_embed_dims[i], sr_ratio=sr))

        self.loss_align_weight = loss_align_weight

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

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

    def init_weights(self):
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

                for k, v in state_dict.items():
                    skip_prefixes = (
                        'quality_pyramid_net.', 'quality_gated_fusion.',
                        'private_inject_rgb.', 'private_inject_t.',
                        'fusion_enhance.',
                        'decode_head.', 'auxiliary_head.')
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

    def extract_feat(self, inputs: torch.Tensor):
        B = inputs.shape[0] // 2
        input_rgb = inputs[:B]
        input_t = inputs[B:]

        q_rgb_maps = self.quality_pyramid_net.forward_rgb(input_rgb)
        q_t_maps = self.quality_pyramid_net.forward_thermal(input_t)

        x_rgbt = self.backbone(inputs)
        zc_rgb_list = [feat[:B] for feat in x_rgbt]
        zc_t_list = [feat[B:] for feat in x_rgbt]

        zp_rgb_list = self.private_branch_rgb(input_rgb)
        zp_t_list = self.private_branch_t(input_t)

        num_stages = len(zc_rgb_list)

        fused_feats = []
        for i in range(num_stages):
            fused_shared = self.quality_gated_fusion[i](
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i])

            fused = self.private_inject_rgb[i](
                fused_shared, zp_rgb_list[i], q_rgb_maps[i])
            fused = self.private_inject_t[i](
                fused, zp_t_list[i], q_t_maps[i])

            fused = self.fusion_enhance[i](fused)
            fused_feats.append(fused)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return (fused_feats, zc_rgb_list, zc_t_list,
                zp_rgb_list, zp_t_list, q_rgb_maps, q_t_maps)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
        results = self.extract_feat(input_rgbt)
        fused_feats = results[0]
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

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)

        (fused_feats, zc_rgb_list, zc_t_list,
         zp_rgb_list, zp_t_list, q_rgb_maps, q_t_maps) = \
            self.extract_feat(input_rgbt)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        num_stages = len(zc_rgb_list)

        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += compute_quality_aware_alignment_loss(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i],
                threshold=self.init_threshold)
        losses['loss_align'] = loss_align_total / num_stages * \
            self.loss_align_weight

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
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
        results = self.extract_feat(input_rgbt)
        fused_feats = results[0]
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
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)
