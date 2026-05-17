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
from .swinmul_v12d_quality_disentangle import (
    CrossAttentionEnhance,
    QualityWeightedFinalFusion,
)
from .swinmul_v12_quality_disentangle import (
    ChannelAttention, SpatialAttention,
)


def _forward_swin_no_quality(backbone, img):
    x, hw_shape = backbone.patch_embed(img)

    if backbone.use_abs_pos_embed:
        x = x + backbone.absolute_pos_embed
    x = backbone.drop_after_pos(x)

    outs = []

    for i, stage in enumerate(backbone.stages):
        for j, block in enumerate(stage.blocks):
            x = block(x, hw_shape)

        if i in backbone.out_indices:
            norm_layer = getattr(backbone, f'norm{i}')
            out = norm_layer(x)
            out = out.view(-1, *hw_shape, x.shape[-1]).permute(
                0, 3, 1, 2).contiguous()
            outs.append(out)

        if stage.downsample is not None:
            x, hw_shape = stage.downsample(x, hw_shape)

    return outs


@MODELS.register_module()
class SwinMulV12DisentangleOnly(BaseSegmentor):

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
                 loss_align_weight: float = 0.5,
                 aux_loss_weight: float = 0.3,
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

        self.loss_align_weight = loss_align_weight
        self.aux_loss_weight = aux_loss_weight

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

    def init_weights(self):
        self.backbone.init_weights()
        self.private_branch_rgb.init_weights()
        self.private_branch_t.init_weights()
        if self.init_cfg is not None:
            super().init_weights()

    def _simple_common_fusion(self, zc_rgb, zc_t):
        return (zc_rgb + zc_t) / 2.0

    def _extract_feat_single(self, input_rgb, input_t):
        B = input_rgb.shape[0]

        input_rgbt = torch.cat([input_rgb, input_t], dim=0)
        zc_rgbt_list = _forward_swin_no_quality(self.backbone, input_rgbt)

        zc_rgb_list = [feat[:B] for feat in zc_rgbt_list]
        zc_t_list = [feat[B:] for feat in zc_rgbt_list]

        zp_rgb_list = _forward_swin_no_quality(
            self.private_branch_rgb, input_rgb)
        zp_t_list = _forward_swin_no_quality(
            self.private_branch_t, input_t)

        num_stages = len(zc_rgb_list)
        zc_fused_list = []
        rgb_enhanced_list = []
        t_enhanced_list = []

        for i in range(num_stages):
            zc_fused = self._simple_common_fusion(
                zc_rgb_list[i], zc_t_list[i])
            zc_fused_list.append(zc_fused)

            rgb_enh = self.cross_attn_rgb[i](
                zp_rgb_list[i], zc_fused)
            t_enh = self.cross_attn_t[i](
                zp_t_list[i], zc_fused)
            rgb_enhanced_list.append(rgb_enh)
            t_enhanced_list.append(t_enh)

        ones_maps = [torch.ones(B, 1, f.shape[2], f.shape[3],
                                device=f.device, dtype=f.dtype)
                     for f in rgb_enhanced_list]
        final_fused = self.final_fusion(
            rgb_enhanced_list, t_enhanced_list,
            ones_maps, ones_maps, 0.5)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_list, rgb_enhanced_list,
                t_enhanced_list, final_fused)

    def extract_feat(self, input_rgb: torch.Tensor,
                     input_t: torch.Tensor):
        results = self._extract_feat_single(input_rgb, input_t)

        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enhanced_list,
         t_enhanced_list, final_fused) = results

        self._last_cache = {
            'zc_rgb': [f.detach() for f in zc_rgb_list],
            'zc_t': [f.detach() for f in zc_t_list],
            'zp_rgb': [f.detach() for f in zp_rgb_list],
            'zp_t': [f.detach() for f in zp_t_list],
            'zc_fused': [f.detach() for f in zc_fused_list],
            'rgb_private_fused': [f.detach() for f in rgb_enhanced_list],
            't_private_fused': [f.detach() for f in t_enhanced_list],
            'final_fused': [f.detach() for f in final_fused],
        }

        return (zc_fused_list, rgb_enhanced_list,
                t_enhanced_list, final_fused)

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

    def _compute_alignment_loss(self, zc_rgb, zc_t):
        return F.mse_loss(zc_rgb, zc_t)

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        results = self._extract_feat_single(input_rgb, input_ir)
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_list, rgb_enh_list, t_enh_list,
         final_fused) = results

        losses = dict()

        if self.with_neck:
            final_neck = self.neck(final_fused)
        else:
            final_neck = final_fused

        self._check_feats_valid(final_neck, 'final_neck')

        loss_final = self._decode_head_forward(final_neck, data_samples)
        losses.update(add_prefix(loss_final, 'final'))

        if self.common_decode_head is not None:
            common_feats = zc_fused_list
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
            rgb_pf = rgb_enh_list
            if self.with_neck:
                rgb_pf = self.neck(rgb_pf)
            self._check_feats_valid(rgb_pf, 'rgb_pf')
            rgb_pf_detached = [f.detach() for f in rgb_pf]
            loss_rgb_pf = self._decode_head_forward(
                rgb_pf_detached, data_samples, self.rgb_private_decode_head)
            losses.update(add_prefix(loss_rgb_pf, 'rgb_private'))
            for k in list(losses.keys()):
                if k.startswith('rgb_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        if self.t_private_decode_head is not None:
            t_pf = t_enh_list
            if self.with_neck:
                t_pf = self.neck(t_pf)
            self._check_feats_valid(t_pf, 't_pf')
            t_pf_detached = [f.detach() for f in t_pf]
            loss_t_pf = self._decode_head_forward(
                t_pf_detached, data_samples, self.t_private_decode_head)
            losses.update(add_prefix(loss_t_pf, 't_private'))
            for k in list(losses.keys()):
                if k.startswith('t_private.'):
                    losses[k] = losses[k] * self.aux_loss_weight

        num_stages = len(zc_rgb_list)
        loss_align_total = 0.0
        for i in range(num_stages):
            loss_align_total += self._compute_alignment_loss(
                zc_rgb_list[i], zc_t_list[i])
        losses['loss_align'] = (loss_align_total / num_stages *
                                self.loss_align_weight)

        self._last_cache = {
            'zc_rgb': [f.detach() for f in zc_rgb_list],
            'zc_t': [f.detach() for f in zc_t_list],
            'zp_rgb': [f.detach() for f in zp_rgb_list],
            'zp_t': [f.detach() for f in zp_t_list],
            'zc_fused': [f.detach() for f in zc_fused_list],
            'rgb_private_fused': [f.detach() for f in rgb_enh_list],
            't_private_fused': [f.detach() for f in t_enh_list],
            'final_fused': [f.detach() for f in final_fused],
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
