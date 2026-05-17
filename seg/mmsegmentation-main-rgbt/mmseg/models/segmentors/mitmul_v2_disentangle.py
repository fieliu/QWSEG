import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from .mitmul_v1_baseline import CrossAttentionFusion
from ..utils import nchw_to_nlc, nlc_to_nchw


def compute_hsic(zc, zp, sigma=1.0):
    B, N, D = zc.shape
    zc_flat = zc.reshape(B, N * D).float()
    zp_flat = zp.reshape(B, N * D).float()

    zc_norm = F.normalize(zc_flat, dim=1)
    zp_norm = F.normalize(zp_flat, dim=1)

    zc_centered = zc_norm - zc_norm.mean(dim=0, keepdim=True)
    zp_centered = zp_norm - zp_norm.mean(dim=0, keepdim=True)

    K = zc_centered @ zc_centered.T
    L = zp_centered @ zp_centered.T
    H = torch.eye(B, device=zc.device, dtype=torch.float32) - 1.0 / B

    hsic = torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)
    return hsic


def compute_infonce(zc_rgb, zc_t, temperature=0.1):
    B, N, D = zc_rgb.shape
    zc_rgb_flat = F.normalize(zc_rgb.reshape(B, N * D).float(), dim=-1)
    zc_t_flat = F.normalize(zc_t.reshape(B, N * D).float(), dim=-1)

    sim_matrix = torch.matmul(
        zc_rgb_flat, zc_t_flat.transpose(0, 1)) / temperature
    sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)
    labels = torch.arange(B, device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)


@MODELS.register_module()
class MiTMulV2Disentangle(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
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
                 loss_modal_weight=0.2,
                 loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
                 loss_invariance_weight=0.01,
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
        self.loss_modal_weight = loss_modal_weight
        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.loss_invariance_weight = loss_invariance_weight

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
            tokens_list.append(x.clone())
            hw_shapes.append(hw_shape)
            x = nlc_to_nchw(x, hw_shape)
            if i in backbone.out_indices:
                outs.append(x)
            x = nchw_to_nlc(x)
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

        num_stages = len(x_rgb_list)
        zc_fused_feats = []
        for i in range(num_stages):
            zc_fused_feats.append(x_rgb_list[i] + x_t_list[i])

        fused_feats = []
        for i in range(num_stages):
            H_i, W_i = x_rgb_list[i].shape[-2:]
            zc_sum_tokens = zc_fused_feats[i].flatten(2).transpose(1, 2)
            zp_rgb_tokens = zp_rgb_list[i].flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_list[i].flatten(2).transpose(1, 2)

            fused_rgb = self.cross_attn_rgb[i](
                zc_sum_tokens, zp_rgb_tokens, zp_rgb_tokens)
            fused_t = self.cross_attn_t[i](
                fused_rgb, zp_t_tokens, zp_t_tokens)

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
                zc_fused_feats_neck, fused_feats)

    def encode_decode(self, inputs, batch_img_metas):
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]
        input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
        (_, _, _, _, _, fused_feats) = self.extract_feat(input_rgbt)
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
         zc_fused_feats, fused_feats) = self.extract_feat(input_rgbt)

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

            loss_modal_i = compute_infonce(zc_rgb_tokens_i, zc_t_tokens_i)
            losses[f'loss_modal_s{i}'] = loss_modal_i * \
                self.loss_modal_weight

        loss_invariance = torch.tensor(0.0, device=inputs.device)
        for i in range(num_stages):
            zc_rgb_proj = self.channel_proj[i](zc_rgb_list[i])
            zc_t_proj = self.channel_proj[i](zc_t_list[i])
            zc_rgb_tokens_i = zc_rgb_proj.flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_proj.flatten(2).transpose(1, 2).float()
            zc_rgb_norm = F.normalize(zc_rgb_tokens_i, dim=-1)
            zc_t_norm = F.normalize(zc_t_tokens_i, dim=-1)
            loss_invariance = loss_invariance + \
                F.mse_loss(zc_rgb_norm, zc_t_norm.detach()) + \
                F.mse_loss(zc_t_norm, zc_rgb_norm.detach())
        losses['loss_invariance'] = loss_invariance * \
            self.loss_invariance_weight

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
        (_, _, _, _, _, fused_feats) = self.extract_feat(input_rgbt)
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
