import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


class ChannelAttention(BaseModule):

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(BaseModule):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMFusion(BaseModule):

    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.num_stages = len(in_channels_list)
        self.ca_rgb = nn.ModuleList()
        self.sa_rgb = nn.ModuleList()
        self.ca_t = nn.ModuleList()
        self.sa_t = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i, ch in enumerate(in_channels_list):
            self.ca_rgb.append(ChannelAttention(ch))
            self.sa_rgb.append(SpatialAttention())
            self.ca_t.append(ChannelAttention(ch))
            self.sa_t.append(SpatialAttention())
            self.projs.append(nn.Sequential(
                nn.Conv2d(ch * 2, out_channels_list[i], 1, bias=False),
                nn.GroupNorm(32, out_channels_list[i]),
                nn.GELU()))

    def forward(self, x_rgb_list, x_t_list):
        fused_list = []
        for i in range(self.num_stages):
            rgb_feat = self.ca_rgb[i](x_rgb_list[i])
            rgb_feat = self.sa_rgb[i](rgb_feat)
            t_feat = self.ca_t[i](x_t_list[i])
            t_feat = self.sa_t[i](t_feat)
            concat = torch.cat([rgb_feat, t_feat], dim=1)
            fused = self.projs[i](concat)
            fused_list.append(fused)
        return fused_list


def _generate_mask_strategy(B, H, W, mask_ratio=0.5, device='cpu'):
    N = H * W
    num_mask = int(N * mask_ratio)

    r = random.random()
    if r < 1.0 / 3.0:
        rgb_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        t_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:num_mask]
            rgb_mask[b, idx] = True
        return rgb_mask, t_mask, 'rgb_only'
    elif r < 2.0 / 3.0:
        rgb_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        t_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(N, device=device)[:num_mask]
            t_mask[b, idx] = True
        return rgb_mask, t_mask, 't_only'
    else:
        rgb_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        t_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            total_mask = int(N * mask_ratio)
            half_mask = total_mask // 2
            remaining = total_mask - half_mask
            perm = torch.randperm(N, device=device)
            rgb_idx = perm[:half_mask]
            t_idx = perm[half_mask:half_mask + remaining]
            rgb_mask[b, rgb_idx] = True
            t_mask[b, t_idx] = True
        return rgb_mask, t_mask, 'complementary'


def _build_stage_attn_masks(token_mask_1d, hw_shape, window_size,
                            num_heads_list, depths, device):
    B, N = token_mask_1d.shape
    H, W = hw_shape
    assert N == H * W

    stage_masks = []
    cur_mask = token_mask_1d
    cur_h, cur_w = H, W

    for stage_idx in range(len(depths)):
        pad_r = (window_size - cur_w % window_size) % window_size
        pad_b = (window_size - cur_h % window_size) % window_size
        H_pad = cur_h + pad_b
        W_pad = cur_w + pad_r

        mask_2d = cur_mask.float().view(B, cur_h, cur_w)
        if pad_r > 0 or pad_b > 0:
            mask_2d = F.pad(mask_2d, (0, pad_r, 0, pad_b), value=1.0)

        nH_p = H_pad // window_size
        nW_p = W_pad // window_size
        nW = nH_p * nW_p
        ws2 = window_size * window_size

        mask_windows = mask_2d.reshape(B, nH_p, window_size, nW_p,
                                       window_size)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4).reshape(
            B, nW, ws2)

        block_masks = []
        for block_idx in range(depths[stage_idx]):
            shift_size = 0
            if block_idx % 2 == 1:
                shift_size = window_size // 2

            m = mask_windows.clone()
            if shift_size > 0:
                m_2d = mask_2d.clone()
                m_2d = torch.roll(m_2d, shifts=(-shift_size, -shift_size),
                                  dims=(1, 2))
                m = m_2d.reshape(B, nH_p, window_size, nW_p, window_size)
                m = m.permute(0, 1, 3, 2, 4).reshape(B, nW, ws2)

            attn_mask = m.unsqueeze(2) - m.unsqueeze(3)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

            num_heads = num_heads_list[stage_idx]
            attn_mask = attn_mask.unsqueeze(2).expand(
                B, nW, num_heads, -1, -1).reshape(
                B * nW, num_heads, ws2, ws2)
            block_masks.append(attn_mask)

        stage_masks.append(block_masks)

        if stage_idx < len(depths) - 1:
            mask_down = cur_mask.float().view(B, 1, cur_h, cur_w)
            mask_down = F.avg_pool2d(mask_down, kernel_size=2, stride=2)
            cur_mask = (mask_down.view(B, -1) > 0.5)
            cur_h = cur_h // 2
            cur_w = cur_w // 2

    return stage_masks


@MODELS.register_module()
class SwinMulV11MaskMAE(BaseSegmentor):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 mask_ratio: float = 0.5,
                 init_cfg: OptMultiConfig = None):
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

        self.mask_ratio = mask_ratio
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        embed_dims = self.backbone.embed_dims
        depths = self.backbone.depths
        num_heads_list = self.backbone.num_heads
        window_size = self.backbone.window_size

        self.embed_dims_list = [embed_dims * (2 ** i)
                                for i in range(len(depths))]
        self.window_size = window_size
        self.num_heads_list = num_heads_list
        self.depths = depths

        self.fusion = CBAMFusion(self.embed_dims_list, self.embed_dims_list)

        self._last_masks = None

        assert self.with_decode_head

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

    def _forward_backbone_masked(self, img, token_mask_1d):
        x, hw_shape = self.backbone.patch_embed(img)

        if self.backbone.use_abs_pos_embed:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.drop_after_pos(x)

        x[token_mask_1d] = 0.0

        stage_attn_masks = _build_stage_attn_masks(
            token_mask_1d, hw_shape, self.window_size,
            self.num_heads_list, self.depths, x.device)

        outs = []
        cur_mask = token_mask_1d
        cur_h, cur_w = hw_shape

        for i, stage in enumerate(self.backbone.stages):
            for j, block in enumerate(stage.blocks):
                attn_mask = stage_attn_masks[i][j]
                x = block(x, hw_shape, attn_mask=attn_mask)

            x[cur_mask] = 0.0

            if stage.downsample is not None:
                x_down, down_hw_shape = stage.downsample(x, hw_shape)
                mask_down = cur_mask.float().view(
                    x.shape[0], 1, cur_h, cur_w)
                mask_down = F.avg_pool2d(mask_down, kernel_size=2, stride=2)
                cur_mask = (mask_down.view(x.shape[0], -1) > 0.5)
                cur_h, cur_w = down_hw_shape
                hw_shape = down_hw_shape
                x = x_down

            if i in self.backbone.out_indices:
                norm_layer = getattr(self.backbone, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape, x.shape[-1]).permute(
                    0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def extract_feat(self, inputs: torch.Tensor):
        B = inputs.shape[0]
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        H, W = input_rgb.shape[2], input_rgb.shape[3]
        patch_size = self.backbone.patch_embed.patch_size[0] if hasattr(
            self.backbone.patch_embed, 'patch_size') else 4
        H_p = H // patch_size
        W_p = W // patch_size

        if self.training:
            rgb_mask_1d, t_mask_1d, strategy = _generate_mask_strategy(
                B, H_p, W_p, self.mask_ratio, input_rgb.device)
        else:
            rgb_mask_1d = torch.zeros(B, H_p * W_p, dtype=torch.bool,
                                      device=input_rgb.device)
            t_mask_1d = torch.zeros(B, H_p * W_p, dtype=torch.bool,
                                    device=input_rgb.device)
            strategy = 'none'

        self._last_masks = (rgb_mask_1d, t_mask_1d, strategy)

        x_rgb_list = self._forward_backbone_masked(input_rgb, rgb_mask_1d)
        x_t_list = self._forward_backbone_masked(input_ir, t_mask_1d)

        self._last_rgb_feats = x_rgb_list
        self._last_t_feats = x_t_list

        fused = self.fusion(x_rgb_list, x_t_list)

        return fused

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        fused_feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

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

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        fused_feats = self.extract_feat(inputs)

        losses = dict()
        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

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
        fused_feats = self.extract_feat(inputs)
        return self.decode_head.forward(fused_feats)

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
