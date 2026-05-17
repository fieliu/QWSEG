import logging
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from .mitmul_v2_disentangle import compute_hsic
from .mitmul_v1_baseline import CrossAttentionFusion
from ..utils import nchw_to_nlc, nlc_to_nchw
from ..utils.quality_network_v4 import (
    QualityNetworkV4, RankingMarginLoss, LevelConsistencyLoss,
    AnomalyPenaltyLoss, CrossModalQualityLoss,
    ImagePseudoLabelGenerator, ThermalPseudoLabelGenerator)
from ..utils.spatial_degradation_generator import SpatialDegradationGenerator


@MODELS.register_module()
class MiTMulV4Degradation(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 quality_network: ConfigType,
                 decode_head: ConfigType,
                 zc_seg_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 fusion_embed_dims=None,
                 private_embed_dims=None,
                 loss_seg_zc_weight=0.3,
                 loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
                 loss_invariance_weight=0.1,
                 loss_ranking_weight=0.5,
                 loss_level_consistency_weight=0.3,
                 loss_anomaly_weight=0.2,
                 loss_cross_modal_weight=0.3,
                 level_margin=0.1,
                 anomaly_threshold=0.3,
                 anomaly_ceiling=0.6,
                 cross_modal_margin=0.1,
                 min_low_quality=0.1,
                 min_clean_quality=0.5,
                 num_degradation_levels=5,
                 quality_prune_threshold=0.3,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)
        self.quality_network = MODELS.build(quality_network)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_zc_seg_head(zc_seg_head)
        self._init_auxiliary_head(auxiliary_head)

        if fusion_embed_dims is None:
            fusion_embed_dims = [64, 128, 320, 512]
        if private_embed_dims is None:
            private_embed_dims = [32, 64, 128, 256]

        num_stages = len(fusion_embed_dims)
        self.cross_attn_rgb = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=d)
             for d in fusion_embed_dims])
        self.cross_attn_t = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=d)
             for d in fusion_embed_dims])

        self.channel_proj = nn.ModuleList()
        for i in range(num_stages):
            self.channel_proj.append(
                nn.Conv2d(fusion_embed_dims[i], private_embed_dims[i], 1))

        self.loss_seg_zc_weight = loss_seg_zc_weight
        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.loss_invariance_weight = loss_invariance_weight
        self.loss_ranking_weight = loss_ranking_weight
        self.loss_level_consistency_weight = loss_level_consistency_weight
        self.loss_anomaly_weight = loss_anomaly_weight
        self.loss_cross_modal_weight = loss_cross_modal_weight
        self.quality_prune_threshold = quality_prune_threshold
        self.num_degradation_levels = num_degradation_levels

        self.ranking_loss_fn = RankingMarginLoss(margin=level_margin)
        self.level_consistency_fn = LevelConsistencyLoss(
            num_levels=num_degradation_levels,
            level_margin=level_margin,
            min_low_quality=min_low_quality,
            min_clean_quality=min_clean_quality)

        self.anomaly_fn = AnomalyPenaltyLoss(
            anomaly_threshold=anomaly_threshold,
            quality_ceiling=anomaly_ceiling)

        self.cross_modal_fn = CrossModalQualityLoss(
            margin=cross_modal_margin)

        self.pseudo_label_gen = ImagePseudoLabelGenerator(patch_size=16)
        self.thermal_pseudo_label_gen = ThermalPseudoLabelGenerator(patch_size=16)

        self.degradation_generator = SpatialDegradationGenerator(
            num_regions_range=(2, 6),
            region_size_range=(32, 80),
            num_levels=num_degradation_levels,
            num_stages=4)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fusion_embed_dims = fusion_embed_dims
        self.private_embed_dims = private_embed_dims

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_zc_seg_head(self, zc_seg_head: ConfigType) -> None:
        self.zc_seg_head = MODELS.build(zc_seg_head)

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _get_mit_intermediate(self, x):
        outs = []
        hw_shapes = []
        tokens_list = []
        backbone = self.backbone
        for i, layer in enumerate(backbone.layers):
            patch_embed, transformer_layers, norm = layer[0], layer[1], layer[2]
            x, hw_shape = patch_embed(x)
            for block in transformer_layers:
                x = block(x, hw_shape)
            x = norm(x)
            tokens_list.append(x)
            hw_shapes.append(hw_shape)
            B, _, C = x.shape
            out = nlc_to_nchw(x, hw_shape)
            outs.append(out)
        return outs, tokens_list, hw_shapes

    def extract_feat(self, inputs: torch.Tensor):
        B = inputs.shape[0] // 2
        x_rgbt = self.backbone(inputs)

        x_rgb_list = [feat[:B] for feat in x_rgbt]
        x_t_list = [feat[B:] for feat in x_rgbt]

        _, tokens_rgbt, hw_shapes = self._get_mit_intermediate(inputs)
        tokens_rgb_list = [t[:B] for t in tokens_rgbt]
        tokens_t_list = [t[B:] for t in tokens_rgbt]

        zp_rgb_list = self.private_branch_rgb(tokens_rgb_list, hw_shapes)
        zp_t_list = self.private_branch_t(tokens_t_list, hw_shapes)

        zc_rgb_tokens = x_rgb_list[-1].flatten(2).transpose(1, 2)
        zc_t_tokens = x_t_list[-1].flatten(2).transpose(1, 2)
        q_rgb = self.quality_network.forward_rgb(zc_rgb_tokens)
        q_t = self.quality_network.forward_thermal(zc_t_tokens)

        num_stages = len(x_rgb_list)
        zc_fused_feats = []
        for i in range(num_stages):
            zc_fused_feats.append(x_rgb_list[i] + x_t_list[i])

        rgb_keep_mask = (q_rgb >= self.quality_prune_threshold).float()
        t_keep_mask = (q_t >= self.quality_prune_threshold).float()

        fused_feats = []
        for i in range(num_stages):
            H_i, W_i = x_rgb_list[i].shape[-2:]
            zc_sum_tokens = zc_fused_feats[i].flatten(2).transpose(1, 2)
            zp_rgb_tokens = zp_rgb_list[i].flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_list[i].flatten(2).transpose(1, 2)

            pruned_zp_rgb = zp_rgb_tokens * rgb_keep_mask.unsqueeze(-1)
            pruned_zp_t = zp_t_tokens * t_keep_mask.unsqueeze(-1)

            fused_rgb = self.cross_attn_rgb[i](
                zc_sum_tokens, pruned_zp_rgb, pruned_zp_rgb)
            fused_t = self.cross_attn_t[i](
                fused_rgb, pruned_zp_t, pruned_zp_t)

            fused_feat = (fused_rgb + fused_t) / 2.0
            fused_feat = fused_feat.permute(0, 2, 1).reshape(
                B, -1, H_i, W_i)
            fused_feats.append(fused_feat)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
            zc_fused_feats_neck = self.neck(zc_fused_feats)
        else:
            zc_fused_feats_neck = zc_fused_feats

        return (x_rgb_list, x_t_list, zp_rgb_list, zp_t_list,
                zc_fused_feats_neck, fused_feats, q_rgb, q_t)

    def _compute_quality_training_losses(self, input_rgb, input_t):
        B, C, H, W = input_rgb.shape

        all_rgb_q = []
        all_t_q = []
        rgb_token_mask = None
        t_token_mask = None

        for level in range(1, self.num_degradation_levels + 1):
            if level == 1:
                rgb_deg = input_rgb
                t_deg = input_t
            else:
                deg_type_rgb = random.choice(
                    ['motion_blur', 'gaussian_noise', 'salt_pepper',
                     'overexposure', 'lowlight', 'stain', 'bad_block'])
                deg_type_t = random.choice(
                    ['gaussian_noise', 'salt_pepper', 'stripe_noise',
                     'thermal_contrast', 'thermal_halo', 'stain',
                     'bad_block'])

                rgb_deg, rgb_mask = self.degradation_generator.generate_degraded_image(
                    input_rgb, modality='rgb',
                    deg_type=deg_type_rgb, level=level)
                t_deg, t_mask = self.degradation_generator.generate_degraded_image(
                    input_t, modality='thermal',
                    deg_type=deg_type_t, level=level)
                if rgb_token_mask is None:
                    tok_H = zc_rgb.shape[1] ** 0.5
                    tok_H = int(tok_H) if tok_H == int(tok_H) else int(tok_H ** 0.5)
                    rgb_token_mask = F.adaptive_avg_pool2d(
                        rgb_mask, (tok_H, tok_H)).reshape(rgb_mask.shape[0], -1)
                if t_token_mask is None:
                    tok_H = zc_t.shape[1] ** 0.5
                    tok_H = int(tok_H) if tok_H == int(tok_H) else int(tok_H ** 0.5)
                    t_token_mask = F.adaptive_avg_pool2d(
                        t_mask, (tok_H, tok_H)).reshape(t_mask.shape[0], -1)

            rgbt_deg = torch.cat([rgb_deg, t_deg], dim=0)
            x_deg = self.backbone(rgbt_deg)
            zc_rgb_deg = x_deg[-1][:B].flatten(2).transpose(1, 2)
            zc_t_deg = x_deg[-1][B:].flatten(2).transpose(1, 2)

            q_rgb_deg = self.quality_network.forward_rgb(zc_rgb_deg)
            q_t_deg = self.quality_network.forward_thermal(zc_t_deg)

            all_rgb_q.append(q_rgb_deg)
            all_t_q.append(q_t_deg)

        losses = dict()

        loss_rgb_level = self.level_consistency_fn(all_rgb_q, mask=rgb_token_mask)
        loss_t_level = self.level_consistency_fn(all_t_q, mask=t_token_mask)
        losses['loss_level_rgb'] = loss_rgb_level * \
            self.loss_level_consistency_weight
        losses['loss_level_t'] = loss_t_level * \
            self.loss_level_consistency_weight

        for i in range(len(all_rgb_q) - 1):
            loss_rank_rgb = self.ranking_loss_fn(
                all_rgb_q[i], all_rgb_q[i + 1], mask=rgb_token_mask)
            loss_rank_t = self.ranking_loss_fn(
                all_t_q[i], all_t_q[i + 1], mask=t_token_mask)
            losses[f'loss_rank_rgb_l{i}'] = loss_rank_rgb * \
                self.loss_ranking_weight / (len(all_rgb_q) - 1)
            losses[f'loss_rank_t_l{i}'] = loss_rank_t * \
                self.loss_ranking_weight / (len(all_rgb_q) - 1)

        rgb_pseudo = self.pseudo_label_gen(input_rgb * 255.0)
        t_pseudo = self.thermal_pseudo_label_gen(input_t * 255.0)
        loss_anomaly_rgb = self.anomaly_fn(all_rgb_q[0], rgb_pseudo)
        loss_anomaly_t = self.anomaly_fn(all_t_q[0], t_pseudo)
        losses['loss_anomaly_rgb'] = loss_anomaly_rgb * self.loss_anomaly_weight
        losses['loss_anomaly_t'] = loss_anomaly_t * self.loss_anomaly_weight

        if rgb_token_mask is not None and t_token_mask is not None:
            loss_cross = self.cross_modal_fn(
                all_rgb_q[0], all_t_q[0],
                rgb_token_mask, t_token_mask)
            losses['loss_cross_modal'] = loss_cross * self.loss_cross_modal_weight

        return losses

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
        results = self.extract_feat(input_rgbt)
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

    def _zc_seg_head_forward_train(self, zc_fused_feats, data_samples):
        losses = dict()
        loss_zc = self.zc_seg_head.loss(
            zc_fused_feats, data_samples, self.train_cfg)
        loss_val = loss_zc.get('loss_ce', list(loss_zc.values())[0])
        losses['loss_seg_zc'] = loss_val
        return losses

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
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

        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_feats, fused_feats,
         q_rgb, q_t) = self.extract_feat(input_rgbt)

        losses = dict()
        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        with torch.no_grad():
            zc_fused_feats_detached = [f.detach() for f in zc_fused_feats]
        loss_zc = self._zc_seg_head_forward_train(
            zc_fused_feats_detached, data_samples)
        losses['loss_seg_zc'] = loss_zc['loss_seg_zc'] * \
            self.loss_seg_zc_weight

        num_stages = len(zc_rgb_list)
        for i in range(num_stages):
            zc_rgb_proj = self.channel_proj[i](zc_rgb_list[i])
            zc_t_proj = self.channel_proj[i](zc_t_list[i])
            zp_rgb_tokens_i = zp_rgb_list[i].flatten(2).transpose(1, 2).float()
            zp_t_tokens_i = zp_t_list[i].flatten(2).transpose(1, 2).float()
            zc_rgb_tokens_i = zc_rgb_proj.flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_proj.flatten(2).transpose(1, 2).float()

            w = self.loss_disentangle_weights[i] \
                if i < len(self.loss_disentangle_weights) \
                else self.loss_disentangle_weights[-1]

            loss_hsic_rgb = compute_hsic(zc_rgb_tokens_i, zp_rgb_tokens_i)
            loss_hsic_t = compute_hsic(zc_t_tokens_i, zp_t_tokens_i)
            losses[f'loss_disentangle_s{i}'] = \
                (loss_hsic_rgb + loss_hsic_t) * w

        loss_invariance = torch.tensor(0.0, device=inputs.device)
        for i in range(num_stages):
            zc_rgb_proj = self.channel_proj[i](zc_rgb_list[i])
            zc_t_proj = self.channel_proj[i](zc_t_list[i])
            zc_rgb_tokens_i = zc_rgb_proj.flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_proj.flatten(2).transpose(1, 2).float()
            loss_invariance = loss_invariance + \
                F.mse_loss(zc_rgb_tokens_i, zc_t_tokens_i.detach()) + \
                F.mse_loss(zc_t_tokens_i, zc_rgb_tokens_i.detach())
        losses['loss_invariance'] = loss_invariance * \
            self.loss_invariance_weight

        quality_losses = self._compute_quality_training_losses(
            input_rgb, input_ir)
        losses.update(quality_losses)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs, data_samples=None):
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

    def _forward(self, inputs, data_samples=None):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
        fused_feats = self.extract_feat(input_rgbt)
        return self.decode_head.forward(fused_feats)

    def inference(self, inputs, batch_img_metas):
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        return hasattr(self, 'decode_head') and self.decode_head is not None
