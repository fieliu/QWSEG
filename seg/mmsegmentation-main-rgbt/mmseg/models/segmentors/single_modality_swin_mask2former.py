import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         SampleList, add_prefix)


@MODELS.register_module()
class SingleModalitySwinMask2Former(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 modality: str = 'rgb',
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            backbone['pretrained'] = pretrained
        assert modality in ('rgb', 'thermal')
        self.modality = modality
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def _select_modality(self, inputs):
        if self.modality == 'rgb':
            return inputs[:, :3]
        else:
            return inputs[:, 3:]

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

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

    def encode_decode(self, inputs, batch_img_metas):
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        mod_input = self._select_modality(inputs)
        x = self.extract_feat(mod_input)
        losses = dict()
        for ds in data_samples:
            if hasattr(ds, 'gt_sem_seg'):
                ds.gt_sem_seg.data = ds.gt_sem_seg.data.squeeze(0)
        loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def predict(self, inputs, data_samples=None):
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

    def _forward(self, inputs, data_samples=None):
        mod_input = self._select_modality(inputs)
        x = self.extract_feat(mod_input)
        return self._get_seg_logits(x, data_samples)

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        return seg_logit

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
                mod_crop = self._select_modality(crop_img)
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(mod_crop, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        return seg_logits

    def whole_inference(self, inputs, batch_img_metas):
        mod_input = self._select_modality(inputs)
        seg_logits = self.encode_decode(mod_input, batch_img_metas)
        return seg_logits

    def postprocess_result(self, seg_logits, data_samples):
        from mmengine.structures import PixelData
        B, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(B)]
        for i in range(B):
            img_meta = data_samples[i].metainfo
            ps = img_meta.get('padding_size', [0] * 4)
            pl, pr, pt, pb = ps
            i_sl = seg_logits[i:i + 1, :, pt:H - pb, pl:W - pr]
            flip = img_meta.get('flip', None)
            if flip:
                fd = img_meta.get('flip_direction', None)
                i_sl = i_sl.flip(dims=(3,) if fd == 'horizontal' else (2,))
            from mmseg.models.utils import resize
            i_sl = resize(i_sl, size=img_meta['ori_shape'], mode='bilinear',
                          align_corners=self.align_corners, warning=False).squeeze(0)
            pred = i_sl.argmax(dim=0, keepdim=True) if C > 1 else (i_sl.sigmoid() > 0.5).to(i_sl)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_sl),
                'pred_sem_seg': PixelData(data=pred)
            })
        return data_samples
