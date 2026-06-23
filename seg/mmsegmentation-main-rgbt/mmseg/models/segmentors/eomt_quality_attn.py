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
    """[B,N,1] or [B,N] quality -> [B,N] additive key bias (log-free, convex).

    DEPRECATED (soft convex bias, too blunt to fire: gamma=3 gives s=0.5->-0.5).
    Kept for reference; the live mechanism uses quality_mask_to_bias on a hard
    keep-mask instead."""
    if s.dim() == 3:
        s = s.squeeze(-1)
    return f_attn_convex(s, alpha=alpha, gamma=gamma)


def quality_mask_to_bias(D, suppress=-20.0):
    """Hard keep-mask D (STE) -> additive KEY bias [B,N].

    D=1 (keep) -> 0 (key untouched); D=0 (prune) -> `suppress` (-20) so the
    key's softmax weight -> ~0: the token becomes invisible AS A KEY (source-
    blocking: it cannot propagate its corruption to other tokens) while still
    acting as a QUERY (it still observes kept tokens and self-repairs). The STE
    in D routes the downstream (seg) loss gradient back to the quality score s.
    D: [B,N,1] or [B,N]."""
    if D.dim() == 3:
        D = D.squeeze(-1)
    return (1.0 - D) * suppress


def hard_mask_ste(s, tau=0.5):
    """Hard keep-mask from score s via Straight-Through Estimator.

    Forward: D = (s >= tau) hard 0/1 (keep=1, prune=0).
    Backward: the (s >= tau) comparison has no gradient, so the STE term
    `s - s.detach()` carries a unit gradient to s. [B,N,1] -> [B,N,1]."""
    D = (s >= tau).float()
    return D + s - s.detach()


def complementary_fix_tokens(D_rgb, D_t, s_rgb, s_t):
    """Guarantee >=1 modality keeps each position as a visible key.

    Where BOTH modalities prune a position (D<0.5 both), the token would be
    invisible as a key in BOTH streams -> its own signal is replaced by a
    neighbor-average (costly if the shallow predictor was WRONG; the error then
    cascades as the averaged feature is re-scored). Force the higher-s (less-
    bad) modality to KEEP it (D=1) and the other to DROP (D=0).

    STE wrapper: forward = corrected hard mask, backward = gradient through the
    raw (pre-fix) mask -> the predictor still learns. Only triggers in the
    both-prune case; elsewhere D passes through unchanged (already STE)."""
    both_prune = (D_rgb < 0.5) & (D_t < 0.5)
    if not both_prune.any():
        return D_rgb, D_t
    rgb_better = s_rgb >= s_t
    fix_rgb = both_prune & rgb_better      # keep rgb, drop t at these positions
    fix_t = both_prune & (~rgb_better)     # keep t, drop rgb
    D_rgb_f = torch.where(fix_rgb, torch.ones_like(D_rgb), D_rgb)
    D_t_f   = torch.where(fix_rgb, torch.zeros_like(D_t),   D_t)
    D_rgb_f = torch.where(fix_t,  torch.zeros_like(D_rgb_f), D_rgb_f)
    D_t_f   = torch.where(fix_t,  torch.ones_like(D_t),     D_t_f)
    # forward = D_*_f (corrected), backward = unit grad through raw D (-> s)
    D_rgb_out = D_rgb_f.detach() + D_rgb - D_rgb.detach()
    D_t_out   = D_t_f.detach()   + D_t   - D_t.detach()
    return D_rgb_out, D_t_out


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


