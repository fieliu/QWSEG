"""Cross-attention fusion block for dual-stream RGB-T token sequences.

Each modality attends to the other (query=self, key/value=other), the attended
result is added back as residual, so each stream is enriched by the other and
then continues to propagate. Supports an optional multiplicative key-side
keep-mask derived from a quality score (used by the quality-aware model; the
baseline passes keep_mask=None).

Design: multiplicative key-gate (probability space), NOT additive bias. See
eomt_quality_attn.py for the rationale (softmax exponentially amplifies an
additive bias, destroying proportional suppression for mid-quality tokens).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnFusion(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_self, x_other, keep_mask=None):
        """x_self/x_other: [B, N, C]. keep_mask: [B, N] soft keep-mask D in
        (0,1), applied as a log-space key-side bias (D→log(D)) compatible
        with F.scaled_dot_product_attention (flash attention). The bias
        log(D_j) is added to all queries' scores for key j pre-softmax:
        D=1→log=0 (keep), D=0.01→log=-4.6 (near-full suppress)."""
        B, N, C = x_self.shape
        q = self.q(self.norm_q(x_self))
        kv = self.kv(self.norm_kv(x_other))
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        if keep_mask is not None:
            # log-space key bias: keep_mask [B,Nk] → log [B,1,1,Nk]
            # .expand+contiguous avoids broadcast view stride error in flash attn
            log_bias = torch.log(keep_mask.clamp_min(1e-9))  # [B, Nk]
            attn_bias = log_bias[:, None, None, :].expand(-1, 1, N, -1).contiguous()
        else:
            attn_bias = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return x_self + out


class TokenQualityPredictor(nn.Module):
    """Shared per-token quality predictor with cross-modal context.

    Judges a token's quality (in (0,1)) using its own feature, a global context
    vector, AND the co-located token of the OTHER modality (so it can compare
    the two modalities at the same position). One instance is shared across
    both modalities (call with swapped args for the other direction).
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, dim // 4), nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        # init the score head bias HIGH so the initial quality is ~sigmoid(2)=0.88
        # (default bias~0 -> 0.5, dangerously close to the s->0 collapse point).
        # Prior "trust all tokens unless proven bad" + far from the collapse
        # attractor. Mirrors the old init_high_score / init_keep_bias design.
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(self, x_self, x_other):
        """x_self/x_other: [B, N, C]. Returns quality s in (0,1): [B, N, 1]."""
        xs = self.norm(x_self)
        xo = self.norm(x_other)
        glob = xs.mean(dim=1, keepdim=True).expand_as(xs)
        feat = torch.cat([xs, xo, glob], dim=-1)
        logit = self.mlp(feat)
        return torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)


class TokenCrossModalCompensation(nn.Module):
    """Repair a low-quality token using the co-located token of the OTHER
    modality: x' = x * s + (1 - s) * Proj(x_other). Where quality s is high the
    token passes through; where it is low it is replaced by a projection of the
    other modality. Operates on [B, N, C] token sequences."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x_self, x_other, s_self):
        # s_self: [B, N, 1] in (0,1)
        return x_self * s_self + (1.0 - s_self) * self.proj(x_other)

