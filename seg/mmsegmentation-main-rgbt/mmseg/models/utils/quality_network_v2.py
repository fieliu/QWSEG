import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .quality_network import (QualityAnchorLoss, LevelMarginLoss,
                               SpatialDiversityLoss, HighDegCeilingLoss,
                               LocalDifferenceModule)


@MODELS.register_module()
class QualityNetworkV2(nn.Module):

    def __init__(self,
                 embed_dim=768,
                 proj_dim=128,
                 num_heads=4,
                 num_scales=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.rgb_proj = nn.Linear(embed_dim, proj_dim)
        self.rgb_local_diff = LocalDifferenceModule(proj_dim, num_scales)
        self.rgb_score = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1))

        self.t_proj = nn.Linear(embed_dim, proj_dim)
        self.t_local_diff = LocalDifferenceModule(proj_dim, num_scales)
        self.t_score = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'score' in name and 'weight' in name and param.dim() >= 2:
                if param.shape[0] == 1:
                    nn.init.normal_(param, mean=0.5, std=0.1)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        for module in [self.rgb_score, self.t_score]:
            last_linear = None
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.normal_(last_linear.weight, mean=1.0, std=0.1)
                nn.init.constant_(last_linear.bias, 0.0)

    def _forward_branch(self, tokens, proj, local_diff, score_head):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f'Token count {N} is not a perfect square'

        feat = proj(tokens)
        local_feat = local_diff(feat, (H, W))
        combined = torch.cat([feat, local_feat], dim=-1)
        logits = score_head(combined).squeeze(-1)
        quality = torch.sigmoid(logits)
        return quality

    def forward_rgb(self, tokens):
        return self._forward_branch(
            tokens, self.rgb_proj, self.rgb_local_diff, self.rgb_score)

    def forward_t(self, tokens):
        return self._forward_branch(
            tokens, self.t_proj, self.t_local_diff, self.t_score)

    def forward_thermal(self, tokens):
        return self.forward_t(tokens)

    def forward(self, tokens, modality='rgb'):
        if modality == 'rgb':
            return self.forward_rgb(tokens)
        elif modality in ('t', 'thermal'):
            return self.forward_t(tokens)
        else:
            raise ValueError(f'Unsupported modality: {modality}')
