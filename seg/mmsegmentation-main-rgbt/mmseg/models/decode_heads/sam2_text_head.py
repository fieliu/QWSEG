import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize


class BasicConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionC(nn.Module):

    def __init__(self, in_channels: int, conv_block=None):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        c7 = 128
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 64, kernel_size=(1, 7), padding=(0, 3))

        self.final_conv = conv_block(64, 8, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        outputs = branch1x1 + branch7x7 + branch7x7dbl
        outputs = self.final_conv(outputs)
        return outputs


class LearnedUpsamplingModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ContextModule(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))
        self.conv1_1 = BasicConv2d(input_channels, 64, kernel_size=1)
        self.inception = InceptionC(64)
        self.upsample = LearnedUpsamplingModule(8, 256)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.conv1_1(x)
        x = self.inception(x)
        x = self.upsample(x)
        return x


class GlobalFeatureConcat2D(nn.Module):

    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x, context_feature):
        x = torch.cat((x, context_feature), dim=1)
        x = self.project(x)
        return x


class ProjectReadout(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
        )

    def forward(self, x):
        out = self.project(x)
        return out


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)
        return x


@MODELS.register_module()
class SAM2TextHead(BaseModule):

    def __init__(self,
                 in_channels: int = 256,
                 channels_list: Tuple[int, ...] = (256, 256, 256, 256),
                 text_embed_dim: int = 768,
                 num_classes: int = 9,
                 out_scale_factor: int = 4,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index: int = 255,
                 align_corners: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.channels_list = channels_list
        self.text_embed_dim = text_embed_dim
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict, '
                            f'but got {type(loss_decode)}')

        n = len(channels_list)

        self.context_modules = nn.ModuleList()
        self.global_feature_concats = nn.ModuleList()
        self.project_readouts = nn.ModuleList()

        for i in range(n):
            self.context_modules.append(ContextModule(in_channels))
            self.global_feature_concats.append(GlobalFeatureConcat2D(channels=in_channels))
            self.project_readouts.append(ProjectReadout(in_channels))

        self.linear_fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.head = nn.Conv2d(in_channels, text_embed_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

        self.output_conv = Interpolate(
            scale_factor=out_scale_factor, mode='bilinear', align_corners=True)

    def forward(self, inputs, label_feature=None):
        if isinstance(inputs, (list, tuple)):
            fpn_features = inputs
        else:
            fpn_features = [inputs]

        if label_feature is None:
            raise ValueError('label_feature (CLIP text embeddings) is required')

        n = len(fpn_features)
        batch_size = fpn_features[0].shape[0]

        context_features = []
        for i in range(n):
            context_features.append(self.context_modules[i](fpn_features[i]))

        concat_features = []
        for i in range(n):
            concat_features.append(
                self.global_feature_concats[i](fpn_features[i], context_features[i]))

        readout_features = []
        for i in range(n):
            feat = concat_features[i]
            h, w = feat.shape[2], feat.shape[3]
            feat = self.project_readouts[i](
                feat.flatten(2).transpose(1, 2))
            feat = feat.transpose(1, 2).reshape(batch_size, -1, h, w)
            readout_features.append(feat)

        target_h, target_w = readout_features[0].shape[2], readout_features[0].shape[3]
        post_features = []
        for i in range(n):
            feat = readout_features[i]
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode='bilinear', align_corners=False)
            post_features.append(feat)

        fused = self.linear_fuse(
            torch.cat(post_features[::-1], dim=1))

        logit_scale = self.logit_scale.exp()
        image_features = self.head(fused)
        image_features = self.dropout(image_features)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.text_embed_dim)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = label_feature / label_feature.norm(dim=-1, keepdim=True)
        text_features = text_features.to(image_features.dtype)

        logits_per_image = logit_scale * image_features @ text_features.t()
        out = logits_per_image.float().view(
            imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        out = self.output_conv(out)

        return out

    def loss(self, inputs, batch_data_samples, train_cfg=None, **kwargs):
        seg_logits = self.forward(**inputs) if isinstance(inputs, dict) else self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs, batch_img_metas, test_cfg=None, **kwargs):
        seg_logits = self.forward(**inputs) if isinstance(inputs, dict) else self.forward(inputs)
        return seg_logits

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = dict()
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.ndim == 4 and seg_label.shape[1] == 1:
            seg_label = seg_label.squeeze(1)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:] if seg_label.ndim == 4 else seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        if isinstance(self.loss_decode, nn.ModuleList):
            for loss_module in self.loss_decode:
                if loss_module.loss_name == 'loss_ce':
                    losses[loss_module.loss_name] = loss_module(
                        seg_logits, seg_label, ignore_index=self.ignore_index)
                else:
                    losses[loss_module.loss_name] = loss_module(seg_logits, seg_label)
        else:
            losses[self.loss_decode.loss_name] = self.loss_decode(
                seg_logits, seg_label, ignore_index=self.ignore_index)

        losses['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return losses

    @staticmethod
    def _stack_batch_gt(batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
