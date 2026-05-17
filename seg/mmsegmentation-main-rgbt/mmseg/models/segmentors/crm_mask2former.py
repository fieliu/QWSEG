from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..losses.crm_loss import MaskGenerator


@MODELS.register_module()
class CRMMask2Former(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 mws_weight: float = 1.0,
                 sdc_weight: float = 1.0,
                 sdn_weight: float = 1.0,
                 mask_enabled: bool = True,
                 mask_size: tuple = (256, 320),
                 mask_patch_size: int = 32,
                 model_patch_size: int = 4,
                 mask_ratio: float = 0.5,
                 mask_type: str = 'patch',
                 mask_strategy: str = 'rand_comp'):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.mws_weight = mws_weight
        self.sdc_weight = sdc_weight
        self.sdn_weight = sdn_weight
        self.mask_enabled = mask_enabled

        assert self.with_decode_head

        if mask_enabled:
            self.mask_generator = MaskGenerator(
                input_size=mask_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                mask_ratio=mask_ratio,
                mask_type=mask_type,
                strategy=mask_strategy)

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

    def extract_feat(self, inputs: torch.Tensor):
        fused_feats = self.backbone(inputs)
        rgb_feats = self.backbone._rgb_feats
        thr_feats = self.backbone._thr_feats

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return fused_feats, rgb_feats, thr_feats

    def _fuse_rgb_thr(self, rgb_feats, thr_feats, fusion_type='add'):
        fused = []
        for i in range(len(rgb_feats)):
            if fusion_type == 'add':
                fused.append(rgb_feats[i] + thr_feats[i])
            elif fusion_type == 'max':
                fused.append(torch.max(rgb_feats[i], thr_feats[i]))
            else:
                fused.append(rgb_feats[i] + thr_feats[i])
        return fused

    def _decode_head_loss(self, feats, data_samples, prefix=''):
        losses = dict()
        loss_decode = self.decode_head.loss(feats, data_samples, self.train_cfg)
        if prefix:
            losses.update(add_prefix(loss_decode, prefix))
        else:
            losses.update(loss_decode)
        return losses

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        B = inputs.shape[0]
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        fused_feats, rgb_feats, thr_feats = self.extract_feat(inputs)

        losses = dict()
        losses_rgbt = self.decode_head.loss(fused_feats, data_samples, self.train_cfg)
        losses.update(add_prefix(losses_rgbt, 'rgbt'))

        if self.with_neck:
            rgb_fused = self.neck(rgb_feats)
            thr_fused = self.neck(thr_feats)
        else:
            rgb_fused = rgb_feats
            thr_fused = thr_feats

        losses_rgb = self.decode_head.loss(rgb_fused, data_samples, self.train_cfg)
        losses_rgb_scaled = {k: v * self.mws_weight for k, v in losses_rgb.items()}
        losses.update(add_prefix(losses_rgb_scaled, 'rgb'))

        losses_thr = self.decode_head.loss(thr_fused, data_samples, self.train_cfg)
        losses_thr_scaled = {k: v * self.mws_weight for k, v in losses_thr.items()}
        losses.update(add_prefix(losses_thr_scaled, 'thr'))

        if self.mask_enabled and self.training:
            import numpy as np
            import torch.nn.functional as F
            masks = []
            for _ in range(B):
                mask1, mask2 = self.mask_generator()
                masks.append(torch.as_tensor(np.stack([mask1, mask2], axis=0)))
            masks = torch.stack(masks, dim=0).float().to(inputs.device)

            rgb_mask = masks[:, 0, :, :, :]
            thr_mask = masks[:, 1, :, :, :]

            input_h, input_w = input_rgb.shape[2], input_rgb.shape[3]
            rgb_mask = F.interpolate(
                rgb_mask.float(), size=(input_h, input_w),
                mode='nearest')
            thr_mask = F.interpolate(
                thr_mask.float(), size=(input_h, input_w),
                mode='nearest')

            masked_rgb_input = input_rgb * (1.0 - rgb_mask)
            masked_thr_input = input_ir * (1.0 - thr_mask)
            masked_rgbt_input = torch.cat([masked_rgb_input, masked_thr_input], dim=1)

            masked_fused_feats, masked_rgb_feats, masked_thr_feats = self.extract_feat(
                masked_rgbt_input)

            if self.with_neck:
                masked_fused = self.neck(masked_fused_feats)
            else:
                masked_fused = masked_fused_feats

            losses_masked = self.decode_head.loss(masked_fused, data_samples, self.train_cfg)
            losses.update(add_prefix(losses_masked, 'masked'))

            with torch.no_grad():
                batch_data_samples = [
                    type(data_samples[0])(metainfo=data_sample.metainfo)
                    for data_sample in data_samples
                ]
                rgbt_cls_scores, _ = self.decode_head.forward(
                    fused_feats, batch_data_samples)
                masked_cls_scores, _ = self.decode_head.forward(
                    masked_fused, batch_data_samples)

                rgb_fused_for_sd = self.neck(masked_rgb_feats) if self.with_neck else masked_rgb_feats
                masked_rgb_cls_scores, _ = self.decode_head.forward(
                    rgb_fused_for_sd, batch_data_samples)

                thr_fused_for_sd = self.neck(masked_thr_feats) if self.with_neck else masked_thr_feats
                masked_thr_cls_scores, _ = self.decode_head.forward(
                    thr_fused_for_sd, batch_data_samples)

            rgbt_logits = rgbt_cls_scores[-1]
            masked_logits = masked_cls_scores[-1]
            masked_rgb_logits = masked_rgb_cls_scores[-1]
            masked_thr_logits = masked_thr_cls_scores[-1]

            loss_self_comp = (rgbt_logits.detach() - masked_logits).abs().mean()
            loss_self_nlocal = (rgbt_logits.detach() - masked_rgb_logits).abs().mean()
            loss_self_nlocal += (rgbt_logits.detach() - masked_thr_logits).abs().mean()

            losses['loss_self_comp'] = self.sdc_weight * loss_self_comp
            losses['loss_self_nlocal'] = self.sdn_weight * loss_self_nlocal

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
        fused_feats, _, _ = self.extract_feat(inputs)
        return self.decode_head.forward(fused_feats)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        fused_feats, _, _ = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
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
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

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
