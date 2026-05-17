import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmengine.model import BaseModule


class PointConv(nn.Module):
    def __init__(self, in_dim=64, out_dim=64, dilation=1, norm_cfg=None):
        super().__init__()
        kernel_size = 1
        conv_padding = (kernel_size // 2) * dilation
        norm_cfg = norm_cfg or dict(type='BN')
        from mmcv.cnn import build_norm_layer
        self.pconv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=conv_padding, dilation=dilation),
            build_norm_layer(norm_cfg, out_dim)[1],
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.pconv(x)
        return x


@MODELS.register_module()
class FuseDecoderHead(BaseModule):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 embed_dim=16,
                 norm_cfg=dict(type='BN'),
                 align_corners=False,
                 loss_fusion=dict(type='FusionLoss', loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
        self.in_channels = in_channels
        F1_in, F2_in, F3_in, F4_in = self.in_channels
        embedding_dim = embed_dim

        self.PointConv_1_rgb = PointConv(in_dim=F1_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_1_ir = PointConv(in_dim=F1_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_2_rgb = PointConv(in_dim=F2_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_2_ir = PointConv(in_dim=F2_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_3_rgb = PointConv(in_dim=F3_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_3_ir = PointConv(in_dim=F3_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_4_rgb = PointConv(in_dim=F4_in, out_dim=embedding_dim, norm_cfg=norm_cfg)
        self.PointConv_4_ir = PointConv(in_dim=F4_in, out_dim=embedding_dim, norm_cfg=norm_cfg)

        from mmcv.cnn import build_norm_layer
        self.CNN_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 8 + 6, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, embedding_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embedding_dim, out_channels=3, kernel_size=1),
            nn.Sigmoid())

        if isinstance(loss_fusion, dict):
            self.loss_fusion = MODELS.build(loss_fusion)
        elif isinstance(loss_fusion, (list, tuple)):
            self.loss_fusion = nn.ModuleList()
            for loss in loss_fusion:
                self.loss_fusion.append(MODELS.build(loss))

    def forward(self, inputs, original_input):
        F1_rgb, F1_ir, F2_rgb, F2_ir, F3_rgb, F3_ir, F4_rgb, F4_ir = inputs
        input_rgb, input_ir = original_input

        ir_rgb_cat = torch.cat([input_rgb, input_ir], dim=1)

        F1_rgb_c = self.PointConv_1_rgb(F1_rgb)
        F1_rgb_c = F.interpolate(F1_rgb_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F1_ir_c = self.PointConv_1_ir(F1_ir)
        F1_ir_c = F.interpolate(F1_ir_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F2_rgb_c = self.PointConv_2_rgb(F2_rgb)
        F2_rgb_c = F.interpolate(F2_rgb_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F2_ir_c = self.PointConv_2_ir(F2_ir)
        F2_ir_c = F.interpolate(F2_ir_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F3_rgb_c = self.PointConv_3_rgb(F3_rgb)
        F3_rgb_c = F.interpolate(F3_rgb_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F3_ir_c = self.PointConv_3_ir(F3_ir)
        F3_ir_c = F.interpolate(F3_ir_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F4_rgb_c = self.PointConv_4_rgb(F4_rgb)
        F4_rgb_c = F.interpolate(F4_rgb_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F4_ir_c = self.PointConv_4_ir(F4_ir)
        F4_ir_c = F.interpolate(F4_ir_c, size=input_rgb.shape[2:], mode='bilinear', align_corners=self.align_corners)

        F_1_4_cat = torch.cat([F1_rgb_c, F1_ir_c, F2_rgb_c, F2_ir_c, F3_rgb_c, F3_ir_c, F4_rgb_c, F4_ir_c, ir_rgb_cat], dim=1)
        Fuse = self.CNN_fuse(F_1_4_cat)

        return Fuse

    def loss(self, fuse_output, original_input):
        input_rgb, input_ir = original_input
        losses = dict()
        if not isinstance(self.loss_fusion, nn.ModuleList):
            losses_fusion = [self.loss_fusion]
        else:
            losses_fusion = self.loss_fusion

        for loss_fusion in losses_fusion:
            losses[loss_fusion.loss_name] = loss_fusion(input_rgb, input_ir, fuse_output)
        return losses
