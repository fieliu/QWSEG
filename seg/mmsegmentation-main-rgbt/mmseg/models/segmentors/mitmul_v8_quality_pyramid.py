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
from .mitmul_v6_disentangle import (MiTMulV6Disentangle,
                                     ChannelAttention,
                                     SpatialAttention,
                                     compute_quality_weighted_infonce_loss)


@MODELS.register_module()
class MiTMulV8QualityPyramid(MiTMulV6Disentangle):

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
                 loss_seg_zc_weight=0.3,
                 loss_invariant_weight=0.5,
                 quality_pretrained=None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            private_branch_rgb=private_branch_rgb,
            private_branch_t=private_branch_t,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            universal_embed_dims=universal_embed_dims,
            private_embed_dims=private_embed_dims,
            loss_seg_zc_weight=loss_seg_zc_weight,
            loss_seg_zp_residual_weight=0.0,
            loss_disentangle_weights=(0,) * 4,
            loss_invariant_weight=loss_invariant_weight,
            loss_modality_weight=0.0,
            loss_variance_weight=0.0,
            init_cfg=init_cfg)

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

        num_stages = len(universal_embed_dims)

        self.channel_attn_rgb = nn.ModuleList()
        self.channel_attn_t = nn.ModuleList()
        self.spatial_attn_rgb = nn.ModuleList()
        self.spatial_attn_t = nn.ModuleList()
        self.fusion_rgb_proj = nn.ModuleList()
        self.fusion_t_proj = nn.ModuleList()
        self.fusion_enhance = nn.ModuleList()
        for i in range(num_stages):
            self.channel_attn_rgb.append(
                ChannelAttention(universal_embed_dims[i]))
            self.channel_attn_t.append(
                ChannelAttention(universal_embed_dims[i]))
            self.spatial_attn_rgb.append(SpatialAttention())
            self.spatial_attn_t.append(SpatialAttention())
            self.fusion_rgb_proj.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))
            self.fusion_t_proj.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))
            self.fusion_enhance.append(nn.Sequential(
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          3, padding=1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(universal_embed_dims[i], universal_embed_dims[i],
                          3, padding=1, bias=False),
                nn.BatchNorm2d(universal_embed_dims[i]),
                nn.ReLU(inplace=True)))

    def _interpolate_quality(self, q_map, target_size):
        if q_map.shape[2:] != target_size:
            return F.interpolate(q_map, size=target_size, mode='nearest')
        return q_map

    def extract_feat_with_quality(self, inputs: torch.Tensor):
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

        zc_enhanced_feats = []
        rgb_weighted_list = []
        t_weighted_list = []
        for i in range(num_stages):
            H_i, W_i = zc_rgb_list[i].shape[-2:]

            q_rgb_i = self._interpolate_quality(q_rgb_maps[i], (H_i, W_i))
            q_t_i = self._interpolate_quality(q_t_maps[i], (H_i, W_i))

            zc_rgb_attn = self.channel_attn_rgb[i](zc_rgb_list[i])
            zc_rgb_attn = self.spatial_attn_rgb[i](zc_rgb_attn)
            zc_t_attn = self.channel_attn_t[i](zc_t_list[i])
            zc_t_attn = self.spatial_attn_t[i](zc_t_attn)

            rgb_proj = self.fusion_rgb_proj[i](zc_rgb_attn)
            t_proj = self.fusion_t_proj[i](zc_t_attn)

            q_sum = q_rgb_i + q_t_i + 1e-6
            w_rgb = q_rgb_i / q_sum
            w_t = q_t_i / q_sum

            rgb_weighted = w_rgb * rgb_proj
            t_weighted = w_t * t_proj
            rgb_weighted_list.append(rgb_weighted)
            t_weighted_list.append(t_weighted)

            fused_feat = rgb_weighted + t_weighted
            fused_feat = self.fusion_enhance[i](fused_feat)
            zc_enhanced_feats.append(fused_feat)

        universal_tokens_list = []
        universal_enhanced_list = []
        zp_rgb_aligned_list = []
        zp_t_aligned_list = []
        for i in range(num_stages):
            H_i, W_i = zc_enhanced_feats[i].shape[-2:]
            hw_shape_i = (H_i, W_i)

            q_rgb_i = self._interpolate_quality(q_rgb_maps[i], (H_i, W_i))
            q_t_i = self._interpolate_quality(q_t_maps[i], (H_i, W_i))

            q_sum = q_rgb_i + q_t_i + 1e-6
            w_rgb_i = q_rgb_i / q_sum
            w_t_i = q_t_i / q_sum

            q_combined = (q_rgb_i + q_t_i) / 2.0

            sr_ratio = self.self_attn_universal[i].sr_ratio
            if sr_ratio > 1:
                q_combined_ds = F.avg_pool2d(
                    q_combined,
                    kernel_size=sr_ratio,
                    stride=sr_ratio)
            else:
                q_combined_ds = q_combined

            q_bias = torch.log(q_combined_ds + 1e-6).squeeze(1)
            q_bias = q_bias.reshape(B, 1, 1, -1)
            q_bias = q_bias.expand(-1, self.self_attn_universal[i].num_heads,
                                    -1, -1)

            universal_tokens = zc_enhanced_feats[i].flatten(
                2).transpose(1, 2)
            universal_tokens = self.self_attn_universal[i](
                universal_tokens, hw_shape_i, attn_bias=q_bias)
            universal_tokens_list.append(universal_tokens)

            universal_enhanced = universal_tokens.transpose(
                1, 2).reshape(B, -1, H_i, W_i)
            universal_enhanced_list.append(universal_enhanced)

            zp_rgb_aligned = self.private_proj_rgb[i](zp_rgb_list[i])
            zp_t_aligned = self.private_proj_t[i](zp_t_list[i])

            zp_rgb_weighted = w_rgb_i * zp_rgb_aligned
            zp_t_weighted = w_t_i * zp_t_aligned

            zp_rgb_tokens = zp_rgb_weighted.flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_weighted.flatten(2).transpose(1, 2)
            zp_rgb_tokens = self.self_attn_rgb[i](
                zp_rgb_tokens, hw_shape_i)
            zp_t_tokens = self.self_attn_t[i](
                zp_t_tokens, hw_shape_i)
            zp_rgb_aligned_list.append(zp_rgb_tokens)
            zp_t_aligned_list.append(zp_t_tokens)

        fused_feats = []
        for i in range(num_stages):
            H_i, W_i = zc_enhanced_feats[i].shape[-2:]
            hw_shape_i = (H_i, W_i)
            universal_tokens = universal_tokens_list[i]
            zp_rgb_tokens = zp_rgb_aligned_list[i]
            zp_t_tokens = zp_t_aligned_list[i]

            cross_rgb = self.cross_attn_rgb[i](
                universal_tokens, zp_rgb_tokens, zp_rgb_tokens, hw_shape_i)
            rgb_enhanced = cross_rgb.permute(0, 2, 1).reshape(
                B, -1, H_i, W_i)
            rgb_enhanced = self.cross_proj_rgb[i](rgb_enhanced)
            rgb_enhanced_tokens = rgb_enhanced.flatten(2).transpose(1, 2)

            cross_t = self.cross_attn_t[i](
                rgb_enhanced_tokens, zp_t_tokens, zp_t_tokens, hw_shape_i)
            t_enhanced = cross_t.permute(0, 2, 1).reshape(
                B, -1, H_i, W_i)
            t_enhanced = self.cross_proj_t[i](t_enhanced)
            fused_feats.append(t_enhanced)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
            zc_enhanced_feats_neck = self.neck(zc_enhanced_feats)
        else:
            zc_enhanced_feats_neck = zc_enhanced_feats

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_enhanced_feats_neck, fused_feats, q_rgb_maps, q_t_maps,
                rgb_weighted_list, t_weighted_list, universal_enhanced_list)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)

        results = self.extract_feat_with_quality(input_rgbt)
        fused_feats = results[5]

        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)

        losses = dict()

        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_enhanced_feats, fused_feats,
         q_rgb_maps, q_t_maps,
         rgb_weighted_list, t_weighted_list,
         universal_enhanced_list) = self.extract_feat_with_quality(
            input_rgbt)

        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        loss_zc_seg, zc_logits = self._zc_mlp_seg_forward_train(
            zc_enhanced_feats, data_samples)
        losses['loss_seg_zc'] = loss_zc_seg * self.loss_seg_zc_weight

        num_stages = len(zc_rgb_list)

        loss_invariant_total = 0.0
        for i in range(num_stages):
            loss_invariant_total += compute_quality_weighted_infonce_loss(
                zc_rgb_list[i], zc_t_list[i],
                q_rgb_maps[i], q_t_maps[i])
        losses['loss_invariant'] = loss_invariant_total / num_stages * \
            self.loss_invariant_weight

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples)
            losses.update(loss_aux)

        return losses
