"""Quality-aware attention injection for the HuggingFace DINOv3 ViT backbone.

The DINOv3 backbone is an opaque `transformers` module whose self-attention
(a) uses RoPE (applied only to patch tokens, not cls/register prefix tokens)
and (b) applies its `attention_mask` MULTIPLICATIVELY *after* softmax. Neither
lets us cleanly add a pre-softmax key bias, so we wrap each block's attention
module with a forward that re-implements the computation and injects a
MULTIPLICATIVE per-key keep-mask D AFTER softmax (low-quality key tokens are
suppressed, so they cannot propagate corruption to other tokens).

Design: multiplicative key-side gating (probability-space), NOT additive bias.
Rationale: an additive bias `(1-D)*(-20)` before softmax is exponentially
amplified by softmax, so a slightly-degraded token (D=0.8) collapses to a
near-zero weight (0.009) — the same as a fully-pruned token. This destroys
the "proportional suppression" property that motivated the soft mask. A
multiplicative gate `attn = softmax(scores) * D` followed by renormalization
operates in probability space and gives TRUE linear scaling (D=0.8 -> weight
x0.8), preserving the gradient signal for mid-quality tokens. This is the
same probability-space gating used by DynamicViT (NeurIPS'21) after its
Gumbel-Softmax sampling, and by GAU (FLASH, ICML'22).

The keep-mask D is read from a per-module attribute `_q_keep_mask` set right
before each stream's forward (RGB and T share the Siamese backbone and run
serially, so the attribute is set twice per block, once per modality).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Soft keep-mask (replaces the old hard-threshold STE).
#
# Design rationale (informed by DynamicViT NeurIPS'21 and EViT ICLR'22):
#
# 1. DynamicViT uses Gumbel-Softmax to sample a binary keep/prune decision
#    from a learned probability.  The gradient flows through the soft
#    softmax, NOT through a hard comparison.  Key insight: the gradient of
#    sigmoid w.r.t. its logit is D(1-D), which peaks at D=0.5 — the decision
#    boundary.  This *naturally focuses* learning on the tokens whose quality
#    is uncertain, exactly where the predictor needs the most signal.
#
# 2. EViT (ICLR 2022) takes a different route: it uses the [CLS] attention
#    score directly as the importance signal — no separate predictor, no
#    thresholding.  Inattentive tokens are *fused* (weighted-averaged) into
#    a single token rather than hard-pruned, preserving a soft residual.
#    The attention score is inherently soft and differentiable.
#
# 3. The old hard-threshold STE (`D=(s>=tau)+s-s.detach()`) had two problems:
#    (a) unit gradient — s=0.49 and s=0.01 get the SAME gradient, so the
#        predictor cannot learn a continuous quality relationship;
#    (b) it never fires without a deg_ceiling hack pushing degraded tokens
#        past the threshold — the hack fights the model's natural tendency.
#
# The soft mask below fixes both: D = sigmoid((s - tau) / temperature) is
# fully differentiable, its gradient D(1-D)/temp focuses on the boundary,
# and it provides *proportional* suppression (slightly degraded → slightly
# suppressed, heavily degraded → heavily suppressed) without any hack.
# ---------------------------------------------------------------------------

def soft_keep_mask(s, tau=0.5, temperature=0.1):
    """Differentiable soft keep-mask from quality score s.

    D = sigmoid((s - tau) / temperature),  D in (0, 1).

    Unlike the old hard STE (unit gradient everywhere), the gradient is
        dD/ds = D * (1 - D) / temperature
    which peaks at D=0.5 (s=tau) — the decision boundary — and vanishes
    for clearly-kept (D≈1) or clearly-pruned (D≈0) tokens.  This mirrors
    DynamicViT's Gumbel-Softmax gradient behavior: learning focuses on
    uncertain tokens where the predictor needs the most signal.

    temperature controls sharpness:
      - temp=0.1 : sharp (s=0.99→D=0.99, s=0.01→D=0.01) but gradient at
                   boundary is 2.5 (strong, focuses learning)
      - temp=0.5 : moderate (s=0.99→D=0.73, s=0.01→D=0.27) — too blunt,
                   high-quality tokens still get significant suppression
      - temp→0   : approaches hard threshold (but gradient stays smooth)

    Default temp=0.1 gives near-hard behavior at extremes (high-quality
    tokens barely suppressed, low-quality nearly fully suppressed) while
    maintaining a smooth, boundary-focused gradient.  With soft D, the
    multiplicative key-gate gives *proportional* suppression: D=0.99 →
    weight x0.99 (negligible), D=0.5 → weight x0.5 (half), D=0.01 →
    weight x0.01 (near-full).  No deg_ceiling hack needed — the rank loss
    (s_light > s_heavy) naturally pushes s_heavy down, and the soft mask
    translates that into proportional key suppression.

    Args:
        s: [B,N,1] quality score in (0,1).
        tau: decision threshold (default 0.5).
        temperature: sigmoid sharpness (default 0.1; lower = harder).
    Returns:
        D: [B,N,1] soft keep-mask in (0,1).
    """
    return torch.sigmoid((s - tau) / max(temperature, 1e-6))


def quality_mask_to_bias(D, suppress=-20.0):
    """DEPRECATED. Soft keep-mask D -> additive KEY bias [B,N].

    Replaced by multiplicative key-side gating (see QualityAttnWrapper).
    Kept for ablation comparison (additive-bias vs multiplicative-gate).

    D=1 (keep) -> 0 (key untouched); D=0 (prune) -> `suppress` (-20) so the
    key's softmax weight -> ~0.  Problem: softmax exponentially amplifies the
    bias, so D=0.8 -> bias=-4 -> weight 0.009 (same as D=0).  This loses the
    proportional suppression property; use the multiplicative gate instead.

    D: [B,N,1] or [B,N] (soft or hard)."""
    if D.dim() == 3:
        D = D.squeeze(-1)
    return (1.0 - D) * suppress


def complementary_fix_tokens(D_rgb, D_t, s_rgb, s_t, both_prune_thresh=0.1):
    """Soft complementary fix: where BOTH modalities strongly prune a position
    (D < both_prune_thresh in both), boost the higher-s modality's mask.

    With soft masks this is less critical than with hard masks (a soft D=0.1
    still allows ~exp(-18)≈1e-8 attention weight, not exactly zero), but it
    prevents the degenerate case where both streams suppress the same position
    and no clean key is visible.  The fix is also soft: instead of hard-setting
    D=1/D=0, we add a residual boost proportional to the quality gap.

    Unlike the old hard version (which used STE detach tricks), this is
    naturally differentiable — no STE wrapper needed.

    Args:
        D_rgb, D_t: [B,N,1] soft keep-masks in (0,1).
        s_rgb, s_t: [B,N,1] raw quality scores (for deciding which to boost).
        both_prune_thresh: positions with D < this in BOTH modalities are
            considered "both-pruned" and receive the fix (default 0.1).
    Returns:
        D_rgb_f, D_t_f: [B,N,1] corrected soft keep-masks.
    """
    both_prune = (D_rgb < both_prune_thresh) & (D_t < both_prune_thresh)
    if not both_prune.any():
        return D_rgb, D_t
    # Where both are pruned, boost the higher-s modality by a soft residual
    # proportional to the quality gap.  This is differentiable end-to-end.
    gap = (s_rgb - s_t).clamp(-1.0, 1.0)          # +: rgb better, -: t better
    boost_rgb = torch.sigmoid(gap * 5.0)           # ~1 if rgb clearly better
    boost_t = 1.0 - boost_rgb
    # Only apply where both_prune; elsewhere leave D unchanged
    mask = both_prune.float()
    D_rgb_f = D_rgb + mask * boost_rgb * (1.0 - D_rgb)
    D_t_f = D_t + mask * boost_t * (1.0 - D_t)
    return D_rgb_f, D_t_f


# ---------------------------------------------------------------------------
# Deprecated functions (kept for backward-compat / ablation reference)
# ---------------------------------------------------------------------------

def f_attn_convex(s, alpha=4.0, gamma=3.0):
    """DEPRECATED. Convex attention-bias: bias = -alpha*(1-s)^gamma.
    Abandoned because gamma=3 was too blunt (s=0.5 -> -0.5, negligible)."""
    return -alpha * (1.0 - s).clamp(0.0, 1.0) ** gamma


def quality_score_to_token_bias(s, alpha=4.0, gamma=3.0):
    """DEPRECATED. Use quality_mask_to_bias(soft_keep_mask(s)) instead."""
    if s.dim() == 3:
        s = s.squeeze(-1)
    return f_attn_convex(s, alpha=alpha, gamma=gamma)


def hard_mask_ste(s, tau=0.5):
    """DEPRECATED. Hard keep-mask via STE. Replaced by soft_keep_mask.
    Kept for ablation comparison (hard-STE vs soft-sigmoid)."""
    D = (s >= tau).float()
    return D + s - s.detach()


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------

def _get_rope_fn():
    """Fetch apply_rotary_pos_emb from the already-imported HF DINOv3 module
    (avoids a hard top-level transformers dependency).

    Raises AssertionError if the module is imported but the function is not
    found (e.g. transformers version change) — a silent RoPE failure would
    degrade performance without any error."""
    mod = sys.modules.get(
        'transformers.models.dinov3_vit.modeling_dinov3_vit')
    if mod is None:
        return None  # transformers not yet imported; caller handles None
    rope_fn = getattr(mod, 'apply_rotary_pos_emb', None)
    assert rope_fn is not None, (
        "apply_rotary_pos_emb not found in "
        "transformers.models.dinov3_vit.modeling_dinov3_vit — "
        "RoPE would silently fail. Check transformers version compatibility.")
    return rope_fn


class QualityAttnWrapper(nn.Module):
    """Wraps one DINOv3ViTAttention module; re-implements its forward with an
    optional multiplicative key-side keep-mask (`self._q_keep_mask`, [B,N]).

    The mask D is applied AFTER softmax in probability space:
        attn = softmax(scores)              # standard attention
        attn = attn * D[:, None, None, :]   # multiplicative key-gate
        attn = attn / attn.sum(-1, keepdim=True)  # renormalize to sum=1
        out  = attn @ V
    This gives TRUE proportional suppression (D=0.8 -> weight x0.8), unlike
    an additive pre-softmax bias which softmax exponentially amplifies
    (D=0.8 -> bias=-4 -> weight 0.009, same as D=0).  The renormalization
    keeps the weight distribution summing to 1 so the output magnitude is
    stable regardless of how many tokens are suppressed.

    Preserves: q/k/v proj, RoPE on patch tokens only, per-head scaling. Sets
    mask to None after each forward so a missing set() never leaks a stale mask.
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn          # original DINOv3ViTAttention (keeps weights)
        self._q_keep_mask = None   # [B, N] multiplicative key keep-mask, set per-call

    def __getattr__(self, name):
        # delegate missing attrs (num_heads, head_dim, scaling, ...) to the
        # wrapped attention so EoMT's _attn (module.num_heads etc.) still works.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('attn'), name)

    def set_keep_mask(self, keep_mask):
        """Set the per-key keep-mask D in (0,1). Shape [B, N]."""
        self._q_keep_mask = keep_mask

    def forward(self, hidden_states, attention_mask=None,
                position_embeddings=None, **kwargs):
        # No quality mask set (e.g. EoMT query-decode stage) -> behave EXACTLY
        # like the original attention, so its native mask handling is preserved.
        if self._q_keep_mask is None:
            return self.attn(hidden_states, attention_mask=attention_mask,
                             position_embeddings=position_embeddings, **kwargs)
        try:
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

            # Build log-bias from soft keep-mask D for key-side suppression.
            #   softmax(QKᵀ/d + log(D))  ≅  softmax(QKᵀ/d) * D / Σ(D)
            # — functionally nearly identical for source-blocking (low-quality
            # keys get near-zero weight), AND compatible with flash attention
            # (F.scaled_dot_product_attention) which does NOT materialize the
            # [B,H,N,N] attention matrix → massive memory savings. D clamped to
            # 1e-9 so log(0) -> -20.7 (near-full suppression) instead of -inf.
            # Flash attention requires contiguous, properly-strided tensors;
            # .expand() + .contiguous() avoids the broadcast-view stride error.
            log_bias = torch.log(self._q_keep_mask.clamp_min(1e-9))  # [B, N]
            attn_bias = log_bias[:, None, None, :].expand(-1, 1, N, -1).contiguous()

            # Merge with EoMT's native attention_mask (decode-stage masked attn)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn_bias = attn_bias.masked_fill(attention_mask, float('-inf'))
                else:
                    attn_bias = attn_bias + attention_mask

            # PyTorch native scaled_dot_product_attention — leverages flash
            # attention on Ampere+ GPUs (no [B,H,N,N] materialization).
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=a.dropout if self.training else 0.0,
                scale=a.scaling,
            )
            out = out.transpose(1, 2).reshape(B, N, -1).contiguous()
            out = a.o_proj(out)
            return out, None  # must be 2-tuple: EoMT._attn does module(...)[0]
        finally:
            self._q_keep_mask = None  # always consume, even on exception


def wrap_backbone_attention(backbone):
    """Replace each block's attention module with a QualityAttnWrapper, in place.
    Returns the list of wrappers (block-indexed) for later mask-setting."""
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
