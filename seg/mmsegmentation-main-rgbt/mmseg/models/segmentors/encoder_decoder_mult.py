import torch
import torch.nn as nn
from mmengine.runner import autocast

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from .base import BaseSegmentor


@MODELS.register_module()
class EncoderDecoderMult(BaseSegmentor):
    def __init__(self,
                 backbone=None,
                 decode_head=None,
                 auxiliary_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        assert backbone is not None
        assert decode_head is not None
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = nn.ModuleList(
                    [MODELS.build(auxiliary_head)])

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs, batch_img_metas):
        rgb, thermal, fuse = self.extract_feat(inputs)
        out, out_x, out_fuse = self.decode_head.predict(
            rgb, thermal, fuse, batch_img_metas, self.test_cfg)
        return out_fuse

    def _forward(self, inputs):
        rgb, thermal, fuse = self.extract_feat(inputs)
        out, out_x, out_fuse = self.decode_head.forward(rgb, thermal, fuse)
        return out, out_x, out_fuse

    def forward_dummy(self, inputs):
        rgb, thermal, fuse = self.extract_feat(inputs)
        out, out_x, out_fuse = self.decode_head.forward(rgb, thermal, fuse)
        return out_fuse

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        rgb, thermal, fuse = x
        losses = dict()

        loss_decode = self.decode_head.loss(
            rgb, thermal, fuse, data_samples, self.train_cfg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_auxs = []
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(fuse, data_samples, self.train_cfg)
                loss_auxs.append(loss_aux)
            for i, loss_aux in enumerate(loss_auxs):
                for k, v in loss_aux.items():
                    losses[f'aux_{i}.{k}'] = v

        return losses

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    @property
    def with_auxiliary_head(self):
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None
