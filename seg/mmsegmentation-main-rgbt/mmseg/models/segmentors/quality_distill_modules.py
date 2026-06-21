"""Quality + cross-modal compensation modules for the Quality-Distill framework.

These operate on Swin stage features of shape [B, C, H, W] (2D maps), so they
are implemented with 1x1 convs rather than token-sequence ops. This is the
"level A" design: quality-correct the features BEFORE the host's fusion layer,
without touching the (windowed) attention. Attention-bias suppression is a
ViT-only enhancement for later.
"""
import torch
import torch.nn as nn


def f_attn_convex(s, alpha=4.0, gamma=3.0):
    """Convex attention-bias function: bias = -alpha * (1 - s)^gamma.

    s in (0,1) is quality. s=1 -> 0 (no suppression); s=0 -> -alpha (max).
    gamma>1 makes the penalty CONCENTRATE on low quality: medium/high quality
    is barely touched, only clearly-low quality is strongly suppressed. This
    avoids the over-suppression of medium-quality tokens that a steep linear /
    large-alpha threshold function causes. No threshold, smooth everywhere.
    """
    return -alpha * (1.0 - s).clamp(0.0, 1.0) ** gamma


def quality_to_swin_bias_convex(s_2d, window_size, shift_size, alpha=4.0, gamma=3.0):
    """Convex-bias version of the quality->Swin-window-bias converter.

    Mirrors swin_quality_mask2former._quality_score_to_swin_bias (same padding,
    cyclic shift, window partition) but uses f_attn_convex instead of the
    piecewise-linear f_attn. Returns [nW*B, 1, wh*wh, wh*wh].
    """
    import torch.nn.functional as F
    B, _, H, W = s_2d.shape
    bias_map = f_attn_convex(s_2d, alpha=alpha, gamma=gamma)
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        bias_map = F.pad(bias_map, (0, pad_r, 0, pad_b), value=0.0)
    H_pad, W_pad = bias_map.shape[2], bias_map.shape[3]
    if shift_size > 0:
        bias_map = torch.roll(bias_map, shifts=(-shift_size, -shift_size), dims=(2, 3))
    wh = window_size
    nW_h, nW_w = H_pad // wh, W_pad // wh
    bw = bias_map.reshape(B, nW_h, wh, nW_w, wh)
    bw = bw.permute(0, 1, 3, 2, 4).contiguous().reshape(-1, wh * wh)
    return bw.view(-1, 1, 1, wh * wh).expand(-1, 1, wh * wh, wh * wh).contiguous()




class StageQuality(nn.Module):
    """Per-pixel cross-modal quality predictor for one stage.

    Scores each spatial location's quality in (0,1) for EACH modality, using
    BOTH modalities + global context (so it can compare them at the same spot).
    Shared call: pass (self, other) and (other, self) for the two directions.
    """
    def __init__(self, dim):
        super().__init__()
        self.reduce = nn.Conv2d(dim * 2 + dim, dim // 2, 1)
        self.act = nn.GELU()
        self.head = nn.Conv2d(dim // 2, 1, 1)
        # init score head bias HIGH -> initial quality ~sigmoid(2)=0.88 (default
        # ~0.5 is too close to the s->0 collapse point). "Trust all unless proven
        # bad" prior + far from the collapse attractor.
        nn.init.constant_(self.head.bias, 2.0)

    def forward(self, x_self, x_other):
        B, C, H, W = x_self.shape
        glob = x_self.mean(dim=(2, 3), keepdim=True).expand(B, C, H, W)
        feat = torch.cat([x_self, x_other, glob], dim=1)
        logit = self.head(self.act(self.reduce(feat)))
        return torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)  # [B,1,H,W]


class CrossModalCompensation(nn.Module):
    """Replace a modality's low-quality regions with a projection of the OTHER
    modality:  x' = x * s + (1 - s) * Proj(x_other).
    High quality (s->1) keeps the original; low quality (s->0) borrows from the
    co-located clean modality. No information is invented; the still-present bad
    token is down-weighted and compensated by the other modality."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x_self, x_other, s_self):
        return x_self * s_self + (1.0 - s_self) * self.proj(x_other)
