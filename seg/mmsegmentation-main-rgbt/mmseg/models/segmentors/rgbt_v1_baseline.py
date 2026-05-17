from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig
from ..utils.lora import (LoRALinear, apply_lora_to_model,
                          freeze_non_lora_params, merge_lora_weights)
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class RGBTv1Baseline(EncoderDecoder):

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
                 fusion_embed_dims: Optional[List[int]] = None,
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

        if fusion_embed_dims is None:
            fusion_embed_dims = [768, 768, 768, 768]
        self.fusion_mlps = nn.ModuleList()
        for dim in fusion_embed_dims:
            self.fusion_mlps.append(nn.Sequential(
                nn.Conv2d(dim * 2, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)))

    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        B = inputs.shape[0] // 2
        x_rgbt = self.backbone(inputs)

        x_rgb_list = [feat[:B] for feat in x_rgbt]
        x_t_list = [feat[B:] for feat in x_rgbt]

        num_stages = len(x_rgb_list)
        fused_feats = []
        for i in range(num_stages):
            concat_feat = torch.cat([x_rgb_list[i], x_t_list[i]], dim=1)
            fused_feat = self.fusion_mlps[i](concat_feat)
            fused_feats.append(fused_feat)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return fused_feats

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
