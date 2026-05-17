import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class LocalChannelAttention(nn.Module):
    """
    纯局部通道注意力，无全局感知
    - token_proj: 1x1 卷积提取自身特征
    - context_proj: 3x3 DWConv + 1x1 提取邻域上下文
    - attn_gate: 基于自身+上下文生成通道门控，调制自身特征
    - score_head: 输出逐 token 质量标量
    关键：邻域上下文只用于生成调制信号，绝不直接加到自身特征上
    """

    def __init__(self, embed_dim, mid_dim=None):
        super().__init__()
        if mid_dim is None:
            mid_dim = embed_dim // 4
        self.token_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 1),
            nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 3, padding=1, groups=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 1))
        self.attn_gate = nn.Sequential(
            nn.Conv2d(mid_dim * 2, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, embed_dim, 1),
            nn.Sigmoid())
        self.score_head = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 1, 1))

    def forward(self, x_2d):
        token_feat = self.token_proj(x_2d)
        ctx_feat = self.context_proj(x_2d)
        gate_input = torch.cat([token_feat, ctx_feat], dim=1)
        gate = self.attn_gate(gate_input)
        modulated = x_2d * gate
        score_map = self.score_head(modulated)
        return score_map


@MODELS.register_module()
class QualityNetworkV4(nn.Module):
    """
    V4 质量网络：纯局部感知，无全局 Norm，无伪标签
    - 移除 GlobalNorm，避免退化区域污染全图
    - tokens → InputProj → reshape 2D → LocalChannelAttn → sigmoid → scores
    - 训练只用排序损失 + 5级<0.1 + 原图>0.5 + 非退化区域一致性
    """

    def __init__(self,
                 embed_dim=768,
                 proj_dim=128,
                 local_attn_mid=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.rgb_input_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU())
        self.rgb_local_attn = LocalChannelAttention(proj_dim, local_attn_mid)

        self.t_input_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU())
        self.t_local_attn = LocalChannelAttention(proj_dim, local_attn_mid)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for name, attn in [('rgb', self.rgb_local_attn),
                           ('t', self.t_local_attn)]:
            last_conv = attn.score_head[-1]
            nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(last_conv.bias, 0.0)

    def _forward_branch(self, tokens, input_proj, local_attn):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f'Token count {N} is not a perfect square'

        projected = input_proj(tokens)
        x_2d = projected.transpose(1, 2).reshape(B, self.proj_dim, H, W)
        score_map = local_attn(x_2d)
        scores = score_map.reshape(B, 1, -1).squeeze(1)
        quality = torch.sigmoid(scores)
        return quality

    def forward_rgb(self, tokens):
        return self._forward_branch(
            tokens, self.rgb_input_proj, self.rgb_local_attn)

    def forward_thermal(self, tokens):
        return self._forward_branch(
            tokens, self.t_input_proj, self.t_local_attn)

    def forward(self, tokens, modality='rgb'):
        if modality == 'rgb':
            return self.forward_rgb(tokens)
        elif modality in ('t', 'thermal'):
            return self.forward_thermal(tokens)
        else:
            raise ValueError(f'Unsupported modality: {modality}')


class LevelConsistencyLossV4(nn.Module):
    """
    V4 损失函数：纯排序 + 约束，无伪标签
    1. 排序损失：相邻退化级别质量递减（margin=0.1），只在退化区域计算
    2. 原图约束：非退化区域质量 > 0.5
    3. 5级退化约束：退化区域质量 < 0.1
    4. 非退化区域一致性：退化图的非退化区域质量 ≈ 原图对应位置质量
    """

    def __init__(self, min_quality_original=0.5, max_quality_level5=0.1,
                 margin_rank=0.1, consistency_margin=0.05,
                 w_rank=1.0, w_orig=0.1, w_level5=0.1, w_consist=0.5):
        super().__init__()
        self.min_quality_original = min_quality_original
        self.max_quality_level5 = max_quality_level5
        self.margin_rank = margin_rank
        self.consistency_margin = consistency_margin
        self.w_rank = w_rank
        self.w_orig = w_orig
        self.w_level5 = w_level5
        self.w_consist = w_consist

    def forward(self, quality_scores, token_masks):
        """
        Args:
            quality_scores: [B, 6, N], 0=原图, 1-5=退化级别
            token_masks: [B, 6, N], 退化区域=1, 非退化=0
        Returns:
            total_loss, loss_dict
        """
        B, num_levels, N = quality_scores.shape
        device = quality_scores.device

        loss_rank = torch.tensor(0.0, device=device)
        loss_original = torch.tensor(0.0, device=device)
        loss_level5 = torch.tensor(0.0, device=device)
        loss_consist = torch.tensor(0.0, device=device)

        orig_quality = quality_scores[:, 0]
        orig_non_deg = (token_masks[:, 0] == 0).float()

        # 1. 原图约束：非退化区域 > 0.5
        n_clean = orig_non_deg.sum()
        if n_clean > 0:
            loss_original = F.relu(
                self.min_quality_original - orig_quality) * orig_non_deg
            loss_original = loss_original.sum() / (n_clean + 1e-8)

        # 2. 5级退化约束：退化区域 < 0.1
        level5_quality = quality_scores[:, 5]
        level5_deg = (token_masks[:, 5] == 1).float()
        n_deg5 = level5_deg.sum()
        if n_deg5 > 0:
            loss_level5 = F.relu(
                level5_quality - self.max_quality_level5) * level5_deg
            loss_level5 = loss_level5.sum() / (n_deg5 + 1e-8)

        # 3. 排序损失：相邻级别递减（退化区域）
        for i in range(num_levels - 1):
            q_higher = quality_scores[:, i]
            q_lower = quality_scores[:, i + 1]
            mask_i = token_masks[:, i]
            mask_i1 = token_masks[:, i + 1]
            joint_deg = ((mask_i + mask_i1) > 0).float()
            n_joint = joint_deg.sum()
            if n_joint > 0:
                violation = self.margin_rank + q_lower - q_higher
                loss_rank = loss_rank + (
                    F.relu(violation) * joint_deg).sum() / (n_joint + 1e-8)
        loss_rank = loss_rank / max(num_levels - 1, 1)

        # 4. 非退化区域一致性：退化图的非退化区域 ≈ 原图对应位置
        for lvl in range(1, num_levels):
            deg_non_deg = (token_masks[:, lvl] == 0).float()
            n_non_deg = deg_non_deg.sum()
            if n_non_deg > 0:
                diff = (quality_scores[:, lvl] - orig_quality).abs()
                violation = diff - self.consistency_margin
                loss_consist = loss_consist + (
                    F.relu(violation) * deg_non_deg).sum() / (n_non_deg + 1e-8)
        loss_consist = loss_consist / max(num_levels - 1, 1)

        total_loss = (
            self.w_rank * loss_rank
            + self.w_orig * loss_original
            + self.w_level5 * loss_level5
            + self.w_consist * loss_consist
        )

        loss_dict = {
            'total_loss': total_loss,
            'loss_rank': loss_rank,
            'loss_original': loss_original,
            'loss_level5': loss_level5,
            'loss_consist': loss_consist,
        }

        return total_loss, loss_dict
