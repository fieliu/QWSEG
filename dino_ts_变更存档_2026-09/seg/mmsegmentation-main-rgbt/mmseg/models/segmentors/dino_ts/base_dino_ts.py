"""Base segmentor for the DINO multi-stage Dense-Teacher / Sparse-Student model.

Wires the shared-ViT backbone (DinoSharedViT) -> Feature2Pyramid neck (anchor
map replicated to 4 pyramid levels at strides 4/8/16/32) -> Mask2Former head
(doc: the decoder reads only the N anchor tokens). Provides:

  - a mode-aware ``extract_feat`` (dense / sparse_soft / sparse_hard),
  - Mask2Former ``loss`` / ``predict`` / ``_forward`` plumbing,
  - slide/whole inference + postprocess (reused verbatim from DINOv3AdapterM2F).

Input protocol (doc 14.1): 6-channel RGB-T (0:3 = RGB, 3:6 = Thermal, thermal
already repeated to 3 channels by LoadRGBTImageFrom4Channel). Whole-modality
missing = that half zeroed, matching the project's eval hooks.

The stage segmentors (Stage 2A/2B/3) subclass this and add training-view
generation, EMA teachers, and distillation losses.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         add_prefix)


@MODELS.register_module()
class DinoTSBase(BaseSegmentor):
    """Shared-ViT anchor/extra backbone + Feature2Pyramid + Mask2Former.

    Args:
        backbone: DinoSharedViT config.
        decode_head: Mask2FormerHead config.
        neck: Feature2Pyramid config (embed_dim = ViT dim). If None, a default
            Feature2Pyramid is built from the backbone embed dim.
        forward_mode: 'dense' | 'sparse_soft' | 'sparse_hard' — the backbone
            token path used for extract_feat. Stage 2A/2B use 'dense'; Stage 3
            switches to sparse during training and 'sparse_hard' for eval.
        target_k: Extra-token budget K for the sparse modes.
    """

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        forward_mode: str = 'dense',
        target_k: Optional[int] = None,
        soft_tau: float = 1.0,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        embed_dim = getattr(self.backbone, 'embed_dims', 768)

        if neck is None:
            neck = dict(type='Feature2Pyramid', embed_dim=embed_dim,
                        rescales=[4, 2, 1, 0.5],
                        norm_cfg=dict(type='BN', requires_grad=True))
        self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.forward_mode = forward_mode
        self.target_k = target_k
        self.soft_tau = soft_tau
        # exposed for hooks/diagnostics (doc 15.3: actual tokens into deep ViT)
        self._last_seq_len = None

    # -- head init ----------------------------------------------------------
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList(
                    [MODELS.build(h) for h in auxiliary_head])
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self) -> bool:
        return getattr(self, 'neck', None) is not None

    @property
    def with_auxiliary_head(self) -> bool:
        return getattr(self, 'auxiliary_head', None) is not None

    # -- feature extraction -------------------------------------------------
    @staticmethod
    def _split(inputs: torch.Tensor):
        """6-channel RGB-T -> (rgb [B,3], thermal [B,3])."""
        return inputs[:, :3], inputs[:, 3:6]

    def _anchor_to_pyramid(self, anchor_map: torch.Tensor):
        """Anchor feature map [B,D,H,W] -> 4 identical inputs -> Feature2Pyramid
        -> (stride 4, 8, 16, 32) pyramid for Mask2Former."""
        pyramid_in = [anchor_map, anchor_map, anchor_map, anchor_map]
        return self.neck(pyramid_in)

    def extract_feat(self, inputs: torch.Tensor, mode: Optional[str] = None,
                     availability: Optional[dict] = None):
        rgb, thermal = self._split(inputs)
        mode = mode or self.forward_mode
        out = self.backbone(rgb, thermal, mode=mode, target_k=self.target_k,
                            soft_tau=self.soft_tau, availability=availability)
        self._last_seq_len = out['seq_len']
        self._last_backbone_out = out
        feats = self._anchor_to_pyramid(out['anchor_map'])
        return feats

    # -- train / predict ----------------------------------------------------
    def _decode_head_forward_train(self, feats, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(feats, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        losses = dict()
        losses.update(self._decode_head_forward_train(feats, data_samples))
        if self.with_auxiliary_head:
            aux = self.auxiliary_head.loss(feats, data_samples, self.train_cfg)
            losses.update(add_prefix(aux, 'aux'))
        return losses

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples=None):
        feats = self.extract_feat(inputs)
        return self.decode_head.forward(feats)

    def encode_decode(self, inputs, batch_img_metas):
        # eval always uses the configured forward_mode (sparse_hard for Stage 3)
        feats = self.extract_feat(inputs)
        return self.decode_head.predict(feats, batch_img_metas, self.test_cfg)

    # -- inference (verbatim from DINOv3AdapterM2F) -------------------------
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
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
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

    def postprocess_result(self, seg_logits, data_samples):
        from mmseg.models.utils import resize
        batch_size, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
        for i in range(batch_size):
            img_meta = data_samples[i].metainfo
            padding_size = img_meta.get(
                'img_padding_size', img_meta.get('padding_size', [0] * 4))
            pl, pr, pt, pb = padding_size
            i_seg_logits = seg_logits[i:i + 1, :, pt:H - pb, pl:W - pr]
            flip = img_meta.get('flip', None)
            if flip:
                flip_direction = img_meta.get('flip_direction', None)
                if flip_direction == 'horizontal':
                    i_seg_logits = i_seg_logits.flip(dims=(3,))
                else:
                    i_seg_logits = i_seg_logits.flip(dims=(2,))
            i_seg_logits = resize(
                i_seg_logits, size=img_meta['ori_shape'], mode='bilinear',
                align_corners=self.align_corners, warning=False).squeeze(0)
            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_seg_logits),
                'pred_sem_seg': PixelData(data=i_seg_pred)})
        return data_samples
