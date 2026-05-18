import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from .base import BaseSegmentor


@MODELS.register_module()
class AMDANet(BaseSegmentor):
    def __init__(self,
                 backbone=None,
                 decode_head=None,
                 fuse_head=None,
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
        self._init_fuse_head(fuse_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_fuse_head(self, fuse_head):
        if fuse_head is not None:
            self.fuse_head = MODELS.build(fuse_head)
        else:
            self.fuse_head = None

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs, batch_img_metas):
        out_vision, out_semantic = self.extract_feat(inputs)
        out_semantic = self.decode_head.predict(out_semantic, batch_img_metas, self.test_cfg)
        return out_semantic

    def _forward(self, inputs):
        out_vision, out_semantic = self.extract_feat(inputs)
        seg_logits = self.decode_head.forward(out_semantic)
        return seg_logits

    def forward_dummy(self, inputs):
        seg_logits = self._forward(inputs)
        return seg_logits

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
        out_vision, out_semantic = self.extract_feat(inputs)

        x_rgb = inputs[:, :3, :, :]
        x_modal = inputs[:, 3:, :, :]
        if x_modal.shape[1] == 1:
            x_modal = x_modal.repeat(1, 3, 1, 1)
        original_input = [x_rgb, x_modal]

        losses = dict()

        loss_decode = self.decode_head.loss(out_semantic, data_samples, self.train_cfg)
        losses.update(loss_decode)

        if self.with_fuse_head:
            Fus_img = self.fuse_head.forward(out_vision, original_input)

            vision_cr = []
            semantic_cr = []

            randn_value = torch.randint(1, 3, (1,))
            if randn_value == 1:
                for x_semantic_cr in out_semantic:
                    semantic_drop = (torch.rand_like(x_semantic_cr) > 0.15).float()
                    drop_cr_sem = x_semantic_cr * semantic_drop
                    semantic_cr.append(drop_cr_sem)
                x_vision_r = out_vision
                semantic_r = semantic_cr
            elif randn_value == 2:
                for x_vision_cr in out_vision:
                    vision_drop = (torch.rand_like(x_vision_cr) > 0.15).float()
                    drop_cr_vis = x_vision_cr * vision_drop
                    vision_cr.append(drop_cr_vis)
                x_vision_r = vision_cr
                semantic_r = out_semantic

            Fus_img_r = self.fuse_head.forward(x_vision_r, original_input)

            loss_fuse = self.fuse_head.loss(Fus_img, original_input)
            losses.update(loss_fuse)

            loss_fuse_r = self.fuse_head.loss(Fus_img_r, original_input)
            for k, v in loss_fuse_r.items():
                losses[f'{k}_r'] = v

            loss_decode_r = self.decode_head.loss(semantic_r, data_samples, self.train_cfg)
            for k, v in loss_decode_r.items():
                losses[f'{k}_r'] = v

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
        return self.postprocess_result(seg_logits, data_samples)

    @property
    def with_fuse_head(self):
        return hasattr(self, 'fuse_head') and self.fuse_head is not None

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None
