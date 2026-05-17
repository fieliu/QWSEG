from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig
from ..utils.lora import (LoRALinear, apply_lora_to_model,
                          freeze_non_lora_params, merge_lora_weights)
from .encoder_decoder import EncoderDecoder


class CrossAttentionFusion(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        B, N, D = q.shape
        q_out = self.q_proj(q).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_out = self.k_proj(k).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_out = self.v_proj(v).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(
            q_out, k_out.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(
            attn_weights, v_out).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + q)
        return out


@MODELS.register_module()
class RGBTv1SAMBaseline(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 use_lora: bool = True,
                 lora_rank: int = 4,
                 lora_alpha: float = 4.0,
                 lora_dropout: float = 0.1,
                 lora_target_modules: Optional[List[str]] = None,
                 freeze_backbone: bool = True,
                 fusion_type: str = 'cross_attention',
                 fusion_embed_dim: int = 768,
                 fusion_num_heads: int = 8,
                 fusion_dropout: float = 0.1,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.use_lora = use_lora
        self.fusion_type = fusion_type

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ['qkv', 'proj']
            apply_lora_to_model(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules)
            if freeze_backbone:
                freeze_non_lora_params(self.backbone)
        else:
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        if fusion_type == 'cross_attention':
            num_stages = len(self.backbone.out_indices)
            self.cross_attn_rgb2t = nn.ModuleList([
                CrossAttentionFusion(
                    embed_dim=fusion_embed_dim,
                    num_heads=fusion_num_heads,
                    dropout=fusion_dropout)
                for _ in range(num_stages)
            ])
            self.cross_attn_t2rgb = nn.ModuleList([
                CrossAttentionFusion(
                    embed_dim=fusion_embed_dim,
                    num_heads=fusion_num_heads,
                    dropout=fusion_dropout)
                for _ in range(num_stages)
            ])
            self.fusion_gate = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(fusion_embed_dim * 2, fusion_embed_dim),
                    nn.Sigmoid())
                for _ in range(num_stages)
            ])

    def _prepare_rgbt_input(self, inputs):
        input_rgb = inputs[:, 0:3, :, :]
        input_ir = inputs[:, 3:6, :, :]
        return torch.cat([input_rgb, input_ir], dim=0)

    def extract_feat(self, inputs):
        x_rgbt = self.backbone(inputs)
        fused_feats = []
        for stage_idx, feat in enumerate(x_rgbt):
            b, c, h, w = feat.shape
            B = b // 2
            x_rgb_feat = feat[:B]
            x_ir_feat = feat[B:]

            if self.fusion_type == 'add':
                fused_feat = x_rgb_feat + x_ir_feat
            elif self.fusion_type == 'cross_attention':
                rgb_tokens = x_rgb_feat.flatten(2).transpose(1, 2)
                ir_tokens = x_ir_feat.flatten(2).transpose(1, 2)

                rgb_enhanced = self.cross_attn_rgb2t[stage_idx](
                    rgb_tokens, ir_tokens, ir_tokens)
                ir_enhanced = self.cross_attn_t2rgb[stage_idx](
                    ir_tokens, rgb_tokens, rgb_tokens)

                gate_input = torch.cat([rgb_enhanced, ir_enhanced], dim=-1)
                gate = self.fusion_gate[stage_idx](gate_input)

                fused_tokens = gate * rgb_enhanced + (1 - gate) * ir_enhanced
                fused_feat = fused_tokens.transpose(1, 2).reshape(
                    B, c, h, w)
            else:
                raise ValueError(
                    f'Unknown fusion_type: {self.fusion_type}')

            fused_feats.append(fused_feat)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
        return fused_feats

    def encode_decode(self, inputs, batch_img_metas):
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def _forward(self, inputs, data_samples=None):
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        return self.decode_head.forward(x)

    def merge_lora(self):
        if self.use_lora:
            merge_lora_weights(self.backbone)

    def get_lora_info(self) -> Dict:
        lora_params = 0
        frozen_params = 0
        trainable_params = 0
        total_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += num_params
                trainable_params += num_params
            elif param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params

        return dict(
            lora_params=lora_params,
            frozen_params=frozen_params,
            trainable_params=trainable_params,
            total_params=total_params,
            lora_ratio=lora_params / total_params * 100,
            trainable_ratio=trainable_params / total_params * 100)
