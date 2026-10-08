"""Pure-PyTorch building blocks for the DINO multi-stage Dense-Teacher /
Sparse-Student model (docs/DINO_MULTISTAGE_TEACHER_STUDENT.md).

Deliberately free of any mmseg / mmcv / mmdet import so the shapes can be
unit-tested on a CPU-only box without the mm* stack. The mm-dependent glue
(registry, Mask2Former head, data preprocessing) lives in base_dino_ts.py and
the stage segmentors.

Contents:
  - ModalityAdapter:      low-rank residual added to the shared FFN (doc 2.3)
  - AlignmentProjector:   modality-specific LN -> shared 2-layer proj -> L2 (doc 2.4)
  - AnchorExtraFusion:    per-position RGB/T -> one Anchor + one Extra (doc 3.1)
  - UtilityRouter:        per-Extra task utility -> Top-K gather / soft gate (doc 3.2)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adapter: low-rank residual supplementing the shared FFN (doc 2.3)
#   h' = u + SharedFFN(LN(u)) + gamma * Adapter(LN(u))
#   Adapter = LN(D) -> Linear(D, d_adapter) -> GELU -> Linear(d_adapter, D)
#   up-projection zero-initialized so the initial path == plain DINO behavior.
# ---------------------------------------------------------------------------

class ModalityAdapter(nn.Module):
    """Per-modality low-rank residual branch (doc 2.3).

    Applied on the SAME normalized input that feeds the shared FFN, scaled by a
    learnable ``gamma`` (init 1). The up-projection is zero-initialized, so
    the initial residual is zero while its weights receive a gradient.
    Zeroing both gamma and the up-projection would permanently freeze this path.
    """

    def __init__(self, dim: int, d_adapter: int = 64, identity: bool = False):
        super().__init__()
        self.identity = identity
        if identity:
            return
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, d_adapter)
        self.act = nn.GELU()
        self.up = nn.Linear(d_adapter, dim)
        self.gamma = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: [*, D] already LayerNorm'd (LN(u)). Returns the residual to
        add to the block output; identity adapters return exactly zero."""
        if self.identity:
            return x_norm.new_zeros(())  # broadcastable scalar zero
        h = self.up(self.act(self.down(self.norm(x_norm))))
        return self.gamma * h


# ---------------------------------------------------------------------------
# AlignmentProjector: training-only projection to a shared alignment space
# (doc 2.4). ModalitySpecificLN -> SharedTwoLayerProjector -> L2 normalize.
# Removed at inference. One shared projector body; per-modality input LN.
# ---------------------------------------------------------------------------

class AlignmentProjector(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None,
                 out_dim: Optional[int] = None, modalities=('rgb', 'thermal')):
        super().__init__()
        hidden = hidden or dim
        out_dim = out_dim or dim
        self.norms = nn.ModuleDict(
            {m: nn.LayerNorm(dim) for m in modalities})
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """x: [B, N, D]. Returns L2-normalized [B, N, out_dim]."""
        z = self.proj(self.norms[modality](x))
        return F.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# Anchor / Extra fusion (doc 3.1)
#   c_i = concat(r_i, t_i, |r_i - t_i|, r_i * t_i)          -> FusionScorer
#   [a_rgb, a_t] = softmax(FusionScorer(c_i))
#   a_i = a_rgb * V_rgb(r_i) + a_t * V_t(t_i)               -> Anchor
#   e_i = ExtraMLP(concat(r_i - t_i, r_i, t_i, a_i))        -> Extra
# Plus per-Extra additive encodings (doc 3.2): spatial position + extra-type +
# source-modality. Source-modality here is a single learned "extra" code (the
# first version derives one Extra basis from the RGB/T pair).
# ---------------------------------------------------------------------------

class AnchorExtraFusion(nn.Module):
    def __init__(self, dim: int, scorer_hidden: Optional[int] = None):
        super().__init__()
        h = scorer_hidden or dim
        # fusion weights over the two modalities from the 4-way descriptor
        self.fusion_scorer = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, h), nn.GELU(),
            nn.Linear(h, 2))
        # per-modality value projections for the anchor
        self.v_rgb = nn.Linear(dim, dim)
        self.v_t = nn.Linear(dim, dim)
        # extra basis from concat(r-t, r, t, a)
        self.extra_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, h), nn.GELU(),
            nn.Linear(h, dim))
        # Type marker only.  There is deliberately NO learned position embedding
        # here: the shallow and deep blocks both carry RoPE, and Mask2Former adds
        # its own sine positional encoding, so an additive absolute embedding was
        # a fourth, redundant source of position -- and the only fixed-size
        # parameter that stopped the backbone from accepting arbitrary
        # resolutions.  RoPE is relative and therefore cannot separate an anchor
        # from the extra at the same index (they share the same coordinate), so
        # the type marker stays.
        self.extra_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Start as a near-identity RGB path instead of erasing DINO features.

        The former default ``nn.Linear`` initialization replaced both DINO
        streams with random projections before nine pretrained blocks.  The
        segmentation head therefore did not actually receive useful DINOv3
        features at the start of Stage 2.  RGB is the stable anchor initially;
        thermal remains available as the Extra stream and the scorer can learn
        a different mixture from supervision.
        """
        nn.init.eye_(self.v_rgb.weight)
        nn.init.zeros_(self.v_rgb.bias)
        nn.init.eye_(self.v_t.weight)
        nn.init.zeros_(self.v_t.bias)

        scorer_out = self.fusion_scorer[-1]
        nn.init.zeros_(scorer_out.weight)
        with torch.no_grad():
            # softmax([log(9), 0]) = [0.9, 0.1]
            scorer_out.bias.copy_(scorer_out.bias.new_tensor([2.1972246, 0.0]))

        # The residual Extra path starts exactly as the thermal DINO token.
        extra_out = self.extra_mlp[-1]
        nn.init.zeros_(extra_out.weight)
        nn.init.zeros_(extra_out.bias)
        nn.init.zeros_(self.extra_type_embed)

    def forward(self, r: torch.Tensor, t: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """r, t: [B, N, D] RGB / Thermal tokens at the fusion point.

        Returns:
            anchors: [B, N, D]
            extras:  [B, N, D]
            alpha:   [B, N, 2] softmax fusion weights (rgb, t) — exposed for
                     analysis / potential linear-inversion diagnostics.
        """
        desc = torch.cat([r, t, (r - t).abs(), r * t], dim=-1)
        alpha = F.softmax(self.fusion_scorer(desc), dim=-1)  # [B, N, 2]
        a = alpha[..., 0:1] * self.v_rgb(r) + alpha[..., 1:2] * self.v_t(t)
        e = t + self.extra_mlp(torch.cat([r - t, r, t, a], dim=-1))
        N = e.shape[1]
        e = e + self.extra_type_embed
        return a, e, alpha


# ---------------------------------------------------------------------------
# Utility Router (doc 3.2)
#   u_i = Router(concat(a_i, e_i, |a_i - e_i|, a_i * e_i))   task utility
#   sparse_hard: TopK(u, K) + gather -> serialize length N + K (real pruning)
#   sparse_soft: straight-through Gumbel-TopK soft gate (Stage3 early)
# ---------------------------------------------------------------------------

class UtilityRouter(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or (dim // 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, h), nn.GELU(),
            nn.Linear(h, 1))

    def utility(self, anchors: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Per-Extra utility logits. anchors/extras: [B, N, D] -> [B, N]."""
        feat = torch.cat(
            [anchors, extras, (anchors - extras).abs(), anchors * extras],
            dim=-1)
        return self.mlp(feat).squeeze(-1)

    @staticmethod
    def topk_gather(extras: torch.Tensor, u: torch.Tensor, k: int,
                    tau: float = 1.0
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard Top-K selection by utility with a REAL gather (doc 9.2/15.3):
        the returned tensor has sequence length K, not N-with-zeros.

        extras: [B, N, D], u: [B, N], k in [0, N].
        Returns (kept_extras [B, K, D], idx [B, K]).
        """
        B, N, D = extras.shape
        k = max(0, min(int(k), N))
        if k == 0:
            return extras.new_zeros(B, 0, D), u.new_zeros(B, 0, dtype=torch.long)
        idx = u.topk(k, dim=1).indices  # [B, K]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
        kept = torch.gather(extras, 1, idx_exp)
        # Straight-through: the SELECTION stays hard and the sequence really
        # is length K, but the router still receives gradient.
        # u.topk(...).indices is non-differentiable and u is used for
        # nothing else on this path, so without this the router gets
        # grad=None for the whole hard-gather phase and is frozen -- it
        # would only ever be optimised against the soft gate, whose forward
        # keeps the full length N with the pruned tokens merely
        # down-weighted, which is not the regime inference runs in.
        # w / w.detach() is exactly 1.0 in the forward pass (so the kept
        # values are bit-identical to a plain gather) and carries d(w) into
        # u in the backward pass.  The gradient scale is (1 - w) / tau,
        # which is bounded; w is clamped only to keep the division safe.
        u_kept = torch.gather(u, 1, idx)                        # [B, K]
        w = torch.sigmoid(u_kept / max(tau, 1e-6)).clamp_min(1e-6)
        kept = kept * (w / w.detach()).unsqueeze(-1)
        return kept, idx

    @staticmethod
    def soft_gate(extras: torch.Tensor, u: torch.Tensor, k: int,
                  tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Differentiable soft gate for Stage3 early training. Keeps the FULL
        length N (each extra multiplied by a gate in (0,1)); this is the
        non-serialized path used only while the router warms up. Real inference
        must use ``topk_gather``.

        Approximates a Top-K keep via a per-token sigmoid gate whose bias is the
        k-th largest utility (soft threshold). With ``hard=True`` a straight-
        through binary gate at the K/N boundary is used.
        """
        B, N = u.shape
        k = max(0, min(int(k), N))
        if k >= N:
            return extras
        if k == 0:
            return extras * 0.0
        # soft threshold at the k-th largest utility per sample
        thresh = u.topk(k, dim=1).values[:, -1:]  # [B, 1]
        gate = torch.sigmoid((u - thresh) / max(tau, 1e-6))  # [B, N]
        if hard:
            hard_gate = (u >= thresh).float()
            gate = hard_gate + gate - gate.detach()  # straight-through
        return extras * gate.unsqueeze(-1)


# ---------------------------------------------------------------------------
# CPU shape self-test (no mm* deps). Run: python modules.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B, N, D = 2, 30 * 40, 768
    r = torch.randn(B, N, D)
    t = torch.randn(B, N, D)

    adapter = ModalityAdapter(D, d_adapter=64)
    res = adapter(torch.randn(B, N, D))
    assert torch.allclose(res, torch.zeros_like(res)), \
        'zero-init adapter must start as identity (residual == 0)'

    proj = AlignmentProjector(D)
    z = proj(r, 'rgb')
    assert z.shape == (B, N, D)
    assert torch.allclose(z.norm(dim=-1), torch.ones(B, N), atol=1e-4), \
        'projector output must be L2-normalized'

    fusion = AnchorExtraFusion(D)
    a, e, alpha = fusion(r, t)
    assert a.shape == (B, N, D) and e.shape == (B, N, D)
    assert torch.allclose(alpha.sum(-1), torch.ones(B, N), atol=1e-5), \
        'fusion weights must be a softmax over the two modalities'

    router = UtilityRouter(D)
    u = router.utility(a, e)
    assert u.shape == (B, N)
    target_k = int(0.5 * N)
    kept, idx = router.topk_gather(e, u, target_k)
    assert kept.shape == (B, target_k, D), 'hard gather must serialize to K'
    assert idx.shape == (B, target_k)
    # verify gather really picks the highest-utility extras
    top_vals = u.topk(target_k, dim=1).values
    gathered_vals = torch.gather(u, 1, idx)
    assert torch.allclose(top_vals.sort(1).values,
                          gathered_vals.sort(1).values), 'TopK idx mismatch'
    soft = UtilityRouter.soft_gate(e, u, target_k)
    assert soft.shape == (B, N, D), 'soft gate keeps full length N'
    # backprop smoke: everything differentiable
    loss = a.mean() + e.mean() + soft.mean() + u.mean()
    loss.backward()
    print('modules.py self-test OK: '
          f'anchors={tuple(a.shape)} extras={tuple(e.shape)} '
          f'sparse_hard_len={kept.shape[1]} (N={N}, K={target_k})')
