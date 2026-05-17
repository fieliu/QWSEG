import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class LocalDifferenceModule(nn.Module):

    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.proj = nn.Linear(dim * num_scales, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, hw_shape):
        B, N, D = tokens.shape
        H, W = hw_shape

        x = tokens.transpose(1, 2).reshape(B, D, H, W)

        diff_list = []
        for s in range(self.num_scales):
            kernel_size = 2 ** (s + 1) + 1
            avg = F.avg_pool2d(
                x, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2)
            diff = x - avg
            diff_list.append(diff)

        multi_scale = torch.cat(diff_list, dim=1)
        multi_scale = multi_scale.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(multi_scale)
        out = self.norm(out)
        return out


@MODELS.register_module()
class QualityNetwork(nn.Module):

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

    def forward_thermal(self, tokens):
        return self._forward_branch(
            tokens, self.t_proj, self.t_local_diff, self.t_score)

    def forward(self, tokens1, tokens2=None):
        if tokens2 is not None:
            rgb_quality = self.forward_rgb(tokens1)
            thermal_quality = self.forward_thermal(tokens2)
            return rgb_quality, thermal_quality
        else:
            return self.forward_rgb(tokens1)


class LevelMarginLoss(nn.Module):

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, q_list):
        if len(q_list) < 2:
            return torch.tensor(0.0, device=q_list[0].device)
        loss = torch.tensor(0.0, device=q_list[0].device)
        for i in range(len(q_list) - 1):
            diff = q_list[i] - q_list[i + 1]
            violation = self.margin - diff
            loss = loss + F.relu(violation).mean() if violation.dim() > 0 else loss + F.relu(violation)
        return loss / (len(q_list) - 1)


class QualityAnchorLoss(nn.Module):

    def __init__(self, clean_target=0.8, deg_target=0.15):
        super().__init__()
        self.clean_target = clean_target
        self.deg_target = deg_target

    def forward(self, q_clean, q_deg=None):
        loss_clean = F.mse_loss(q_clean, torch.full_like(q_clean, self.clean_target))

        loss_deg = torch.tensor(0.0, device=q_clean.device)
        if q_deg is not None:
            loss_deg = F.mse_loss(q_deg, torch.full_like(q_deg, self.deg_target))

        return loss_clean, loss_deg


class SpatialDiversityLoss(nn.Module):

    def __init__(self, min_std=0.08):
        super().__init__()
        self.min_std = min_std

    def forward(self, quality_scores):
        per_sample_std = quality_scores.std(dim=-1)
        violation = self.min_std - per_sample_std
        loss = F.relu(violation).pow(2).mean()
        return loss


class HighDegCeilingLoss(nn.Module):

    def __init__(self, ceiling=0.15):
        super().__init__()
        self.ceiling = ceiling

    def forward(self, q_highest_deg):
        violation = q_highest_deg - self.ceiling
        loss = F.relu(violation).pow(2).mean()
        return loss
