"""Shared-ViT backbone for the DINO multi-stage Dense-Teacher / Sparse-Student
model (docs/DINO_MULTISTAGE_TEACHER_STUDENT.md sections 2, 3, 14).

Design (doc 2.2 "shallow-independent, deep-mixed"):

    RGB  PatchEmbed --\
                       > shared DINO blocks 0..R-1 (params shared, attention
    T    PatchEmbed --/   independent: modalities packed into the batch dim)
                            + per-modality low-rank Adapter (doc 2.3)
                                    |
                          Anchor / Extra fusion (doc 3.1): one Anchor + one
                          Extra per spatial position
                                    |
                          shared DINO blocks R..L-1 over the JOINT
                          [anchors ; extras] sequence (cross-modal global attn)
                                    |
                          decoder reads the N anchors only.

Token budgets (doc 2.1):
    dense:        N anchors + N extras
    sparse_hard:  N anchors + K extras  (real gather, shorter sequence)

Four forward modes (doc 14.3): 'adapt' | 'dense' | 'sparse_soft' | 'sparse_hard'.

RoPE note: DINOv3 blocks are pretrained with rotary position embeddings on a
raster patch grid. The shallow per-modality blocks run on the real H*W grid and
keep RoPE. The deep blocks operate on the joint [anchor ; extra] sequence, which
is NOT a raster grid (extras may be pruned/reordered), so RoPE is disabled there
and position is supplied instead by a learned anchor position embedding plus the
per-Extra spatial/type encodings added at fusion (doc 3.2). The deep blocks are
fine-tuned (Stage 2A) to absorb this, consistent with the design.

This module imports only torch + the project's DINOv3 loader; the mm* registry
decoration is optional (guarded) so the shape logic can be unit-tested on a
CPU-only box without mmseg/mmcv installed.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dino_ts_modules import (ModalityAdapter, AlignmentProjector,
                                  AnchorExtraFusion, UtilityRouter)
except ImportError:  # direct-run CPU test (python dino_shared_vit.py)
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dino_ts_modules import (ModalityAdapter, AlignmentProjector,
                                 AnchorExtraFusion, UtilityRouter)

try:
    from mmseg.registry import MODELS
    _HAS_MMSEG = True
except Exception:  # pragma: no cover - CPU test path without mm*
    MODELS = None
    _HAS_MMSEG = False


# ---------------------------------------------------------------------------
# Block runner: adapted from DINOv3BlockWrapper.forward (dinov3_adapter.py:405)
# but as a free function with a per-call RoPE toggle and an optional per-modality
# adapter residual, operating on an arbitrary-length token sequence.
# ---------------------------------------------------------------------------

def _eager_attention_no_rope(attn, x_norm):
    """Manual multi-head self-attention over a DINOv3ViTAttention module WITHOUT
    RoPE, for the deep joint [anchor ; extra] sequence (not a raster grid).

    Reuses the module's q/k/v/o projections and per-head scaling, so it is
    numerically identical to the HF attention minus the rotary step. Needed
    because HF's forward unconditionally unpacks ``cos, sin = position_embeddings``
    and cannot be called with None.
    """
    B, L, C = x_norm.shape
    Hn = attn.num_heads
    Hd = attn.head_dim
    q = attn.q_proj(x_norm).view(B, L, Hn, Hd).transpose(1, 2)
    k = attn.k_proj(x_norm).view(B, L, Hn, Hd).transpose(1, 2)
    v = attn.v_proj(x_norm).view(B, L, Hn, Hd).transpose(1, 2)
    scale = getattr(attn, 'scaling', Hd ** -0.5)
    aw = torch.matmul(q, k.transpose(-1, -2)) * scale
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(aw, v).transpose(1, 2).reshape(B, L, C).contiguous()
    return attn.o_proj(out)


def run_dino_block(block, x, rope=None, H=None, W=None,
                   adapter: Optional[ModalityAdapter] = None):
    """Run one DINOv3 ViT block on tokens x [B, L, C].

    rope: the backbone's rope module or None.
      - rope given (shallow blocks, tokens on the H*W grid): RoPE (cos, sin) is
        generated for the grid and the block's own attention applies it.
      - rope None (deep blocks over the joint [anchor ; extra] sequence, which
        is NOT a raster grid): attention is computed manually WITHOUT RoPE. We
        do NOT pass position_embeddings=None to the HF attention because it does
        ``cos, sin = position_embeddings`` unconditionally and would crash.
        Position on the joint sequence is instead supplied additively by the
        learned anchor position embedding + per-Extra spatial/type encodings
        (doc 3.2); the deep blocks are fine-tuned (Stage 2A) to absorb this.
    adapter: optional ModalityAdapter whose residual is added to the MLP output
             (doc 2.3: h' = u + SharedFFN(LN(u)) + gamma * Adapter(LN(u))).
    """
    B, L, C = x.shape

    # --- attention sub-block ---
    residual = x
    x_norm = block.norm1(x) if hasattr(block, 'norm1') else x
    attn = block.attn if hasattr(block, 'attn') else block.attention

    if hasattr(attn, 'q_proj'):
        if rope is not None and H is not None:
            # shallow, grid tokens: let the HF attention apply RoPE
            dummy = x.new_zeros(B, 3, H * 16, W * 16)
            pos_emb = rope(dummy)  # (cos, sin)
            out = attn(x_norm, position_embeddings=pos_emb)
            out = out[0] if isinstance(out, (tuple, list)) else out
        else:
            # deep, non-grid joint sequence: manual eager attention, no RoPE
            out = _eager_attention_no_rope(attn, x_norm)
    elif isinstance(attn, nn.MultiheadAttention):
        # CPU shape-test stand-in (_ToyBlock): exercise the real control flow
        out, _ = attn(x_norm, x_norm, x_norm, need_weights=False)
    else:
        out = attn(x_norm)

    if hasattr(block, 'ls1'):
        out = block.ls1(out)
    elif hasattr(block, 'layer_scale1'):
        out = block.layer_scale1(out)
    x = residual + (block.drop_path(out) if hasattr(block, 'drop_path') else out)

    # --- MLP sub-block (+ optional adapter residual) ---
    residual = x
    x_norm = block.norm2(x) if hasattr(block, 'norm2') else x
    mlp_out = block.mlp(x_norm)
    if hasattr(block, 'ls2'):
        mlp_out = block.ls2(mlp_out)
    elif hasattr(block, 'layer_scale2'):
        mlp_out = block.layer_scale2(mlp_out)
    mlp_out = block.drop_path(mlp_out) if hasattr(block, 'drop_path') else mlp_out
    x = residual + mlp_out
    if adapter is not None and not adapter.identity:
        x = x + adapter(x_norm)  # adapter reads the same LN(u) as the FFN
    return x


class ModalityPatchEmbed(nn.Module):
    """Plain Conv2d patch projection, one per modality (doc 2.3, 3.3). Kept
    separate from the HF backbone's own embeddings so RGB and Thermal are fully
    symmetric and prefix (cls/register) tokens are never introduced — the dense
    path needs only patch tokens. Initialized from the pretrained DINO conv when
    available so RGB starts identical to the original patch embedding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        x = self.proj(x)                       # [B, D, H, W]
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x, (H, W)


class _DinoSharedViTImpl(nn.Module):
    """Implementation body; registered subclass below adds the MODELS decorator
    only when mmseg is importable (so CPU tests can construct it directly)."""

    def __init__(
        self,
        backbone_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        backbone_ckpt: Optional[str] = None,
        img_size=(480, 640),
        patch_size: int = 16,
        embed_dims: int = 768,
        depth: int = 12,
        fusion_block: int = 3,          # R: number of shallow (independent) blocks
        d_adapter: int = 64,
        rgb_adapter_identity: bool = False,
        thr_in_channels: int = 3,
        align_out_dim: Optional[int] = None,
        freeze_vit: bool = False,
        local_files_only: bool = True,
        _build_backbone: bool = True,   # False -> skip DINO load (CPU shape test)
        init_cfg=None,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.depth = depth
        self.R = fusion_block
        self.freeze_vit = freeze_vit
        self.modalities = ('rgb', 'thermal')
        # 1/16 grid for the default image size (used to size learned pos embeds)
        self.grid = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        num_tokens = self.grid[0] * self.grid[1]

        # --- shared DINO backbone (blocks + RoPE) ---
        self.rope = None
        conv_w = conv_b = None
        if _build_backbone:
            from mmseg.models.backbones.eomt_core_import import load_dinov3_backbone
            backbone = load_dinov3_backbone(
                backbone_name=backbone_name, backbone_ckpt=backbone_ckpt,
                img_size=self.img_size, patch_size=patch_size,
                local_files_only=local_files_only)
            self.backbone = backbone
            blocks = backbone.blocks if hasattr(backbone, 'blocks') else backbone.layer
            self.blocks = blocks
            self.rope = getattr(backbone, 'rope_embeddings', None)
            self.final_norm = backbone.norm if hasattr(backbone, 'norm') else nn.LayerNorm(embed_dims)
            conv_w, conv_b = self._find_patch_conv_weight(backbone)
            if freeze_vit:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        else:
            # CPU shape-test stand-in: lightweight blocks matching the interface
            self.backbone = None
            self.blocks = nn.ModuleList([_ToyBlock(embed_dims) for _ in range(depth)])
            self.final_norm = nn.LayerNorm(embed_dims)

        # --- modality patch embeds + modality embeddings ---
        self.patch_embed = nn.ModuleDict({
            'rgb': ModalityPatchEmbed(3, embed_dims, patch_size),
            'thermal': ModalityPatchEmbed(thr_in_channels, embed_dims, patch_size),
        })
        if conv_w is not None:
            self._init_patch_embed_from_dino(conv_w, conv_b, thr_in_channels)
        self.modality_embed = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, 1, embed_dims)) for m in self.modalities})
        for p in self.modality_embed.values():
            nn.init.trunc_normal_(p, std=0.02)

        # --- shallow per-modality adapters (doc 2.3): only on blocks 0..R-1 ---
        self.adapters = nn.ModuleDict({
            'rgb': nn.ModuleList([
                ModalityAdapter(embed_dims, d_adapter, identity=rgb_adapter_identity)
                for _ in range(self.R)]),
            'thermal': nn.ModuleList([
                ModalityAdapter(embed_dims, d_adapter) for _ in range(self.R)]),
        })

        # --- fusion + router + alignment projectors ---
        self.fusion = AnchorExtraFusion(embed_dims, num_tokens=num_tokens)
        self.router = UtilityRouter(embed_dims)
        self.anchor_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dims))
        nn.init.trunc_normal_(self.anchor_pos_embed, std=0.02)
        # training-only projectors (doc 2.4): teacher(rgb) frozen/EMA, student trains
        self.student_projector = AlignmentProjector(
            embed_dims, out_dim=align_out_dim, modalities=self.modalities)
        self.teacher_projector = AlignmentProjector(
            embed_dims, out_dim=align_out_dim, modalities=self.modalities)

        self.out_channels = embed_dims
        self.init_cfg = init_cfg

    # -- pretrained patch-conv discovery + copy -----------------------------
    @staticmethod
    def _find_patch_conv_weight(backbone):
        """Locate the pretrained patch-embed conv weight/bias across HF/timm
        DINOv3 layouts. Returns (weight, bias) or (None, None)."""
        candidates = []
        pe = getattr(backbone, 'patch_embed', None)
        if pe is not None:
            candidates += [getattr(pe, 'patch_embeddings', None),
                           getattr(pe, 'projection', None),
                           getattr(pe, 'proj', None), pe]
        for c in candidates:
            if c is None:
                continue
            w = getattr(c, 'weight', None)
            if isinstance(w, torch.Tensor) and w.dim() == 4:
                return w.detach(), getattr(c, 'bias', None)
        return None, None

    def _init_patch_embed_from_dino(self, conv_w, conv_b, thr_in_channels):
        with torch.no_grad():
            if conv_w.shape == self.patch_embed['rgb'].proj.weight.shape:
                self.patch_embed['rgb'].proj.weight.copy_(conv_w)
                if conv_b is not None:
                    self.patch_embed['rgb'].proj.bias.copy_(conv_b)
            # thermal: reuse RGB conv (averaged to thr_in_channels if needed)
            tw = self.patch_embed['thermal'].proj.weight
            if conv_w.shape[1] == 3 and thr_in_channels == 3 and conv_w.shape == tw.shape:
                self.patch_embed['thermal'].proj.weight.copy_(conv_w)
                if conv_b is not None:
                    self.patch_embed['thermal'].proj.bias.copy_(conv_b)
            elif conv_w.shape[1] == 3 and thr_in_channels == 1:
                self.patch_embed['thermal'].proj.weight.copy_(
                    conv_w.mean(dim=1, keepdim=True))
                if conv_b is not None:
                    self.patch_embed['thermal'].proj.bias.copy_(conv_b)

    # -- forward ------------------------------------------------------------
    def _embed(self, x, modality):
        tok, (H, W) = self.patch_embed[modality](x)
        tok = tok + self.modality_embed[modality]
        return tok, (H, W)

    def _run_shallow(self, tok, modality, H, W):
        for l in range(self.R):
            tok = run_dino_block(self.blocks[l], tok, rope=self.rope, H=H, W=W,
                                 adapter=self.adapters[modality][l])
        return tok

    def _run_deep(self, seq):
        for l in range(self.R, self.depth):
            seq = run_dino_block(self.blocks[l], seq, rope=None)
        return seq

    def forward(self, rgb, thermal, mode='dense', target_k=None,
                soft_tau=1.0, availability: Optional[Dict[str, torch.Tensor]] = None):
        """rgb: [B, 3, H, W]; thermal: [B, C_t, H, W].

        Returns a dict; keys depend on mode:
          adapt:  {rgb_tokens, thermal_tokens, grid}      (Stage 1)
          else:   {anchor_map [B,D,H,W], anchors [B,N,D], extras, utility,
                   alpha, grid, seq_len}
        """
        r, (H, W) = self._embed(rgb, 'rgb')
        t, _ = self._embed(thermal, 'thermal')

        # availability: zero a fully-missing modality's tokens (doc 3.3)
        if availability is not None:
            if 'rgb' in availability:
                r = r * availability['rgb'].view(-1, 1, 1).to(r.dtype)
            if 'thermal' in availability:
                t = t * availability['thermal'].view(-1, 1, 1).to(t.dtype)

        r = self._run_shallow(r, 'rgb', H, W)
        t = self._run_shallow(t, 'thermal', H, W)

        if mode == 'adapt':
            return dict(rgb_tokens=r, thermal_tokens=t, grid=(H, W))

        a, e, alpha = self.fusion(r, t)          # [B, N, D] each
        N = a.shape[1]
        a = a + self.anchor_pos_embed[:, :N]

        u = None
        if mode == 'dense':
            seq = torch.cat([a, e], dim=1)       # N + N
        elif mode == 'sparse_soft':
            u = self.router.utility(a, e)
            e_g = self.router.soft_gate(e, u, target_k, tau=soft_tau)
            seq = torch.cat([a, e_g], dim=1)     # N + N (gated, full length)
        elif mode == 'sparse_hard':
            u = self.router.utility(a, e)
            e_k, _ = self.router.topk_gather(e, u, target_k)
            seq = torch.cat([a, e_k], dim=1)     # N + K (real gather)
        else:
            raise ValueError(f'unknown mode: {mode}')

        seq = self._run_deep(seq)
        anchors = self.final_norm(seq[:, :N])
        anchor_map = anchors.transpose(1, 2).reshape(
            anchors.shape[0], self.embed_dims, H, W).contiguous()

        return dict(anchor_map=anchor_map, anchors=anchors, extras=e,
                    utility=u, alpha=alpha, grid=(H, W), seq_len=seq.shape[1])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vit and self.backbone is not None:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        return self


class _ToyBlock(nn.Module):
    """Minimal DINO-block stand-in for CPU shape tests (no HF backbone)."""

    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Linear(dim * 2, dim))

    def forward(self, *a, **k):  # not used; run_dino_block drives sub-parts
        raise NotImplementedError


if _HAS_MMSEG:
    @MODELS.register_module()
    class DinoSharedViT(_DinoSharedViTImpl):
        pass
else:  # pragma: no cover
    DinoSharedViT = _DinoSharedViTImpl


# ---------------------------------------------------------------------------
# CPU shape self-test (no mm* / DINO weights). Run: python dino_shared_vit.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B = 2
    img = (32, 48)           # small grid -> N = 2*3 = 6? use patch 16 -> 2x3
    net = _DinoSharedViTImpl(
        img_size=img, patch_size=16, embed_dims=768, depth=6, fusion_block=2,
        _build_backbone=False)
    N = (img[0] // 16) * (img[1] // 16)
    rgb = torch.randn(B, 3, *img)
    thr = torch.randn(B, 3, *img)

    # adapt mode
    o = net(rgb, thr, mode='adapt')
    assert o['rgb_tokens'].shape == (B, N, 768)
    assert o['thermal_tokens'].shape == (B, N, 768)

    # dense mode: seq len N + N
    o = net(rgb, thr, mode='dense')
    assert o['anchor_map'].shape == (B, 768, img[0] // 16, img[1] // 16)
    assert o['anchors'].shape == (B, N, 768)
    assert o['seq_len'] == 2 * N, o['seq_len']

    # sparse_hard: seq len N + K (real gather)
    K = N // 2
    o = net(rgb, thr, mode='sparse_hard', target_k=K)
    assert o['seq_len'] == N + K, (o['seq_len'], N, K)
    assert o['anchor_map'].shape == (B, 768, img[0] // 16, img[1] // 16)

    # sparse_soft: full length (gated)
    o = net(rgb, thr, mode='sparse_soft', target_k=K)
    assert o['seq_len'] == 2 * N

    # availability: thermal fully missing -> tokens zeroed pre-blocks
    avail = {'rgb': torch.ones(B), 'thermal': torch.zeros(B)}
    o = net(rgb, thr, mode='dense', availability=avail)
    assert o['anchor_map'].shape == (B, 768, img[0] // 16, img[1] // 16)

    # backprop smoke
    net(rgb, thr, mode='dense')['anchor_map'].mean().backward()
    print('dino_shared_vit.py self-test OK: '
          f'N={N}, dense_len={2*N}, sparse_hard_len={N+K} (K={K})')
