"""Quality-aware attention injection for the HuggingFace DINOv3 ViT backbone.

The DINOv3 backbone is an opaque `transformers` module whose self-attention
(a) uses RoPE (applied only to patch tokens, not cls/register prefix tokens)
and (b) applies its `attention_mask` MULTIPLICATIVELY *after* softmax. Neither
lets us cleanly add a pre-softmax key bias, so we wrap each block's attention
module with a forward that re-implements the computation and injects an
ADDITIVE per-key quality bias BEFORE softmax (low-quality key tokens are
suppressed, so they cannot propagate corruption to other tokens).

The bias is read from a per-module attribute `_q_kv_bias` set right before each
stream's forward (RGB and T share the Siamese backbone and run serially, so the
attribute is set twice per block, once per modality).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def f_attn_convex(s, alpha=4.0, gamma=3.0):
    """Convex attention-bias: bias = -alpha*(1-s)^gamma. s in (0,1).

    s=1 -> 0 (high quality untouched); s=0 -> -alpha (max suppression);
    gamma>1 concentrates the penalty on clearly-low quality, leaving
    medium/high quality almost untouched. No threshold, smooth everywhere.
    """
    return -alpha * (1.0 - s).clamp(0.0, 1.0) ** gamma


def quality_score_to_token_bias(s, alpha=4.0, gamma=3.0):
    """[B,N,1] or [B,N] quality -> [B,N] additive key bias (log-free, convex)."""
    if s.dim() == 3:
        s = s.squeeze(-1)
    return f_attn_convex(s, alpha=alpha, gamma=gamma)


def _get_rope_fn():
    """Fetch apply_rotary_pos_emb from the already-imported HF DINOv3 module
    (avoids a hard top-level transformers dependency)."""
    mod = sys.modules.get(
        'transformers.models.dinov3_vit.modeling_dinov3_vit')
    return getattr(mod, 'apply_rotary_pos_emb', None) if mod else None


class QualityAttnWrapper(nn.Module):
    """Wraps one DINOv3ViTAttention module; re-implements its forward with an
    optional pre-softmax additive key bias (`self._q_kv_bias`, shape [B,N]).

    Preserves: q/k/v proj, RoPE on patch tokens only, per-head scaling. Sets
    bias to None after each forward so a missing set() never leaks a stale bias.
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn          # original DINOv3ViTAttention (keeps weights)
        self._q_kv_bias = None     # [B, N] additive key bias, set per-call

    def __getattr__(self, name):
        # delegate missing attrs (num_heads, head_dim, scaling, ...) to the
        # wrapped attention so EoMT's _attn (module.num_heads etc.) still works.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('attn'), name)

    def set_bias(self, bias):
        self._q_kv_bias = bias

    def forward(self, hidden_states, attention_mask=None,
                position_embeddings=None, **kwargs):
        # No quality bias set (e.g. EoMT query-decode stage) -> behave EXACTLY
        # like the original attention, so its native mask handling is preserved.
        if self._q_kv_bias is None:
            return self.attn(hidden_states, attention_mask=attention_mask,
                             position_embeddings=position_embeddings, **kwargs)
        a = self.attn
        B, N, _ = hidden_states.size()
        q = a.q_proj(hidden_states).view(B, N, a.num_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(hidden_states).view(B, N, a.num_heads, a.head_dim).transpose(1, 2)
        v = a.v_proj(hidden_states).view(B, N, a.num_heads, a.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            rope_fn = _get_rope_fn()
            cos, sin = position_embeddings
            if rope_fn is not None:
                q, k = rope_fn(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-1, -2)) * a.scaling  # [B,H,N,N]
        if self._q_kv_bias is not None:
            # additive bias on KEY tokens, broadcast over heads & query positions
            attn = attn + self._q_kv_bias[:, None, None, :]
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and a.dropout > 0:
            attn = F.dropout(attn, p=a.dropout)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1).contiguous()
        out = a.o_proj(out)
        self._q_kv_bias = None  # consume: never leak a stale bias
        return out, attn


def wrap_backbone_attention(backbone):
    """Replace each block's attention module with a QualityAttnWrapper, in place.
    Returns the list of wrappers (block-indexed) for later bias-setting."""
    wrappers = []
    for blk in backbone.blocks:
        if hasattr(blk, 'attn'):
            blk.attn = QualityAttnWrapper(blk.attn)
            wrappers.append(blk.attn)
        elif hasattr(blk, 'attention'):
            blk.attention = QualityAttnWrapper(blk.attention)
            wrappers.append(blk.attention)
        else:
            wrappers.append(None)
    return wrappers


