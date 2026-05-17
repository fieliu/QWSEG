import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..losses import accuracy
from ..utils import resize


class Mlp(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@MODELS.register_module()
class SegDecoderHead(BaseDecodeHead):
    def __init__(self, embed_dim=768, **kwargs):
        kwargs['input_transform'] = 'multiple_select'
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        c1_in, c2_in, c3_in, c4_in = self.in_channels

        self.linear_c4 = Mlp(input_dim=c4_in, embed_dim=embed_dim)
        self.linear_c3 = Mlp(input_dim=c3_in, embed_dim=embed_dim)
        self.linear_c2 = Mlp(input_dim=c2_in, embed_dim=embed_dim)
        self.linear_c1 = Mlp(input_dim=c1_in, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dropout is not None:
            _c = self.dropout(_c)
        x = self.linear_pred(_c)
        return x

    def loss(self, inputs, batch_data_samples, train_cfg=None):
        seg_logits = self.forward(inputs)
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        loss = dict()
        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                seg_logits, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss
