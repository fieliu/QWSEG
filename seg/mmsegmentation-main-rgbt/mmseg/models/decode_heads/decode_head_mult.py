import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize


class BaseDecodeHeadMult(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_decode_modal=dict(
                     type='M_CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_decode_akd=dict(
                     type='AKDLoss',
                     loss_weight=0.5),
                 loss_decode_head=dict(
                     type='RegionL1',
                     loss_weight=1.0,
                     N_cls=9),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super().__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if isinstance(loss_decode_modal, dict):
            self.loss_decode_modal = MODELS.build(loss_decode_modal)
        elif isinstance(loss_decode_modal, (list, tuple)):
            self.loss_decode_modal = nn.ModuleList()
            for loss in loss_decode_modal:
                self.loss_decode_modal.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode_modal must be a dict or sequence of dict,\
                but got {type(loss_decode_modal)}')

        if isinstance(loss_decode_akd, dict):
            self.loss_decode_akd = MODELS.build(loss_decode_akd)
        elif isinstance(loss_decode_akd, (list, tuple)):
            self.loss_decode_akd = nn.ModuleList()
            for loss in loss_decode_akd:
                self.loss_decode_akd.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode_akd must be a dict or sequence of dict,\
                but got {type(loss_decode_akd)}')

        if isinstance(loss_decode_head, dict):
            self.loss_decode_head = MODELS.build(loss_decode_head)
        elif isinstance(loss_decode_head, (list, tuple)):
            self.loss_decode_head = nn.ModuleList()
            for loss in loss_decode_head:
                self.loss_decode_head.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode_head must be a dict or sequence of dict,\
                but got {type(loss_decode_head)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.conv_seg_x = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout_x = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout_x = None

        self.conv_seg_fuse = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout_fuse = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout_fuse = None

    def extra_repr(self):
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    @abstractmethod
    def forward(self, inputs, inputs_x, inputs_fuse):
        pass

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def cls_seg_x(self, feat):
        if self.dropout_x is not None:
            feat = self.dropout_x(feat)
        output = self.conv_seg_x(feat)
        return output

    def cls_seg_fuse(self, feat):
        if self.dropout_fuse is not None:
            feat = self.dropout_fuse(feat)
        output = self.conv_seg_fuse(feat)
        return output

    def loss(self, inputs: Tuple[Tensor],
             inputs_x: Tuple[Tensor],
             fuse_x: Tuple[Tensor],
             batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        seg_logits, seg_logits_x, seg_logits_fuse = self.forward(inputs, inputs_x, fuse_x)
        seg_logits_ = inputs
        seg_logits_x_ = inputs_x
        seg_logits_fuse_ = fuse_x
        losses = self.loss_by_feat(seg_logits,
                                   seg_logits_x,
                                   seg_logits_fuse,
                                   seg_logits_,
                                   seg_logits_x_,
                                   seg_logits_fuse_,
                                   batch_data_samples)
        return losses

    def predict(self,
                inputs: Tuple[Tensor],
                inputs_x: Tuple[Tensor],
                fuse_x: Tuple[Tensor],
                batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple:
        seg_logits, seg_logits_x, seg_logits_fuse = self.forward(inputs, inputs_x, fuse_x)
        return self.predict_by_feat(seg_logits, seg_logits_x,
                                    seg_logits_fuse, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     seg_logits_x: Tensor,
                     seg_logits_fuse: Tensor,
                     seg_logits_: Tuple[Tensor],
                     seg_logits_x_: Tuple[Tensor],
                     seg_logits_fuse_: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_x = resize(
            input=seg_logits_x,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_fuse = resize(
            input=seg_logits_fuse,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
            seg_weight_x = self.sampler.sample(seg_logits_x, seg_label)
            seg_weight_fuse = self.sampler.sample(seg_logits_fuse, seg_label)
        else:
            seg_weight = None
            seg_weight_x = None
            seg_weight_fuse = None

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        if not isinstance(self.loss_decode_modal, nn.ModuleList):
            losses_decode_modal = [self.loss_decode_modal]
        else:
            losses_decode_modal = self.loss_decode_modal

        if not isinstance(self.loss_decode_akd, nn.ModuleList):
            losses_decode_akd = [self.loss_decode_akd]
        else:
            losses_decode_akd = self.loss_decode_akd

        if not isinstance(self.loss_decode_head, nn.ModuleList):
            losses_decode_head = [self.loss_decode_head]
        else:
            losses_decode_head = self.loss_decode_head

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits_fuse, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode(
                    seg_logits, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode(
                    seg_logits_x, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits_fuse, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode(
                    seg_logits, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode(
                    seg_logits_x, seg_label,
                    weight=seg_weight, ignore_index=self.ignore_index
                )

        for loss_decode_modal in losses_decode_modal:
            if loss_decode_modal.loss_name not in loss:
                loss[loss_decode_modal.loss_name] = loss_decode_modal(
                    seg_logits, seg_logits_fuse,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode_modal(
                    seg_logits_x, seg_logits_fuse,
                    weight=seg_weight, ignore_index=self.ignore_index
                )
            else:
                loss[loss_decode_modal.loss_name] += loss_decode_modal(
                    seg_logits, seg_logits_fuse,
                    weight=seg_weight, ignore_index=self.ignore_index
                ) + loss_decode_modal(
                    seg_logits_x, seg_logits_fuse,
                    weight=seg_weight, ignore_index=self.ignore_index
                )

        for loss_decode_akd in losses_decode_akd:
            if loss_decode_akd.loss_name not in loss:
                loss[loss_decode_akd.loss_name] = loss_decode_akd(
                    seg_logits_fuse_, seg_logits_x_
                ) + loss_decode_akd(
                    seg_logits_fuse_, seg_logits_
                )
            else:
                loss[loss_decode_akd.loss_name] += loss_decode_akd(
                    seg_logits_fuse_, seg_logits_x_
                ) + loss_decode_akd(
                    seg_logits_fuse_, seg_logits_
                )

        for loss_decode_head in losses_decode_head:
            if loss_decode_head.loss_name not in loss:
                loss[loss_decode_head.loss_name] = loss_decode_head(
                    seg_logits, seg_logits_fuse
                ) + loss_decode_head(
                    seg_logits_x, seg_logits_fuse
                )
            else:
                loss[loss_decode_head.loss_name] += loss_decode_head(
                    seg_logits, seg_logits_fuse
                ) + loss_decode_head(
                    seg_logits_x, seg_logits_fuse
                )

        loss['acc_seg'] = accuracy(
            seg_logits_fuse, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        seg_logits_x: Tensor,
                        seg_logits_fuse: Tensor,
                        batch_img_metas: List[dict]) -> Tuple:
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_x = resize(
            input=seg_logits_x,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        seg_logits_fuse = resize(
            input=seg_logits_fuse,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits, seg_logits_x, seg_logits_fuse
