import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head_mult import BaseDecodeHeadMult
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class SegformerHeadMult(BaseDecodeHeadMult):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        self.convs_x = nn.ModuleList()
        self.convs_fuse = nn.ModuleList()

        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(num_inputs):
            self.convs_x.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        for i in range(num_inputs):
            self.convs_fuse.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.fusion_conv_x = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.fusion_conv_fuse = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, inputs_x, inputs_fuse):
        inputs = self._transform_inputs(inputs)
        inputs_x = self._transform_inputs(inputs_x)
        inputs_fuse = self._transform_inputs(inputs_fuse)

        outs = []
        outs_x = []
        outs_fuse = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        for idx in range(len(inputs_x)):
            x1 = inputs_x[idx]
            conv_x = self.convs_x[idx]
            outs_x.append(
                resize(
                    input=conv_x(x1),
                    size=inputs_x[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        for idx in range(len(inputs_fuse)):
            x2 = inputs_fuse[idx]
            conv_fuse = self.convs_fuse[idx]
            outs_fuse.append(
                resize(
                    input=conv_fuse(x2),
                    size=inputs_fuse[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out_x = self.fusion_conv_x(torch.cat(outs_x, dim=1))
        out_fuse = self.fusion_conv_fuse(torch.cat(outs_fuse, dim=1))

        out_x = self.cls_seg_x(out_x)
        out = self.cls_seg(out)
        out_fuse = self.cls_seg_fuse(out_fuse)
        return out, out_x, out_fuse
