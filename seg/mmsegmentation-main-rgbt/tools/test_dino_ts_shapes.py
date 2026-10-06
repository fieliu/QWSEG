"""CPU integration self-test for the DINO Dense-Teacher / Sparse-Student model.

Runs WITHOUT mmseg / mmcv / mmdet (imports only torch + the two mm*-free
modules), so it can validate the architecture's shape logic and gradient flow
on a CPU-only box. The mm-dependent segmentors + Mask2Former head are verified
separately on the GPU machine via tools/train.py smoke runs.

Run:
    python seg/mmsegmentation-main-rgbt/tools/test_dino_ts_shapes.py
"""
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKBONES = os.path.join(_HERE, '..', 'mmseg', 'models', 'backbones')
_DINOTS = os.path.join(_HERE, '..', 'mmseg', 'models', 'segmentors', 'dino_ts')
sys.path.insert(0, os.path.abspath(_BACKBONES))
sys.path.insert(0, os.path.abspath(_DINOTS))

import dino_shared_vit as dv          # noqa: E402
import losses as L                    # noqa: E402


class _ProjectedAttention(nn.Module):
    """Small HF-like attention module for eager/SDPA equivalence testing."""

    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = 0.0
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


def check_sdpa_equivalence():
    attn = _ProjectedAttention().eval()
    eager_x = torch.randn(2, 17, 64, requires_grad=True)
    sdpa_x = eager_x.detach().clone().requires_grad_(True)
    eager = dv._projected_attention(attn, eager_x, backend='eager')
    eager.square().mean().backward()
    eager_grad = eager_x.grad.detach().clone()
    attn.zero_grad(set_to_none=True)
    sdpa = dv._projected_attention(attn, sdpa_x, backend='sdpa')
    sdpa.square().mean().backward()
    torch.testing.assert_close(sdpa, eager, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(sdpa_x.grad, eager_grad, rtol=5e-5, atol=5e-7)


def main():
    torch.manual_seed(0)
    check_sdpa_equivalence()
    # Identity initialization must still allow the adapter to start learning.
    adapter = dv.ModalityAdapter(32, 8)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    x, target = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    assert torch.count_nonzero(adapter(x)) == 0
    (adapter(x) - target).square().mean().backward()
    assert adapter.up.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad()
    (adapter(x) - target).square().mean().backward()
    assert adapter.down.weight.grad.abs().sum() > 0
    B = 2
    img = (48, 64)                     # patch 16 -> grid 3 x 4 -> N = 12
    patch = 16
    N = (img[0] // patch) * (img[1] // patch)
    D = 768

    net = dv._DinoSharedViTImpl(
        img_size=img, patch_size=patch, embed_dims=D, depth=6, fusion_block=2,
        attention_backend='sdpa', with_cp=True, _build_backbone=False)
    rgb = torch.randn(B, 3, *img)
    thr = torch.randn(B, 3, *img)

    # Fusion must preserve the pretrained coordinate system at initialization.
    eye = torch.eye(D)
    torch.testing.assert_close(net.fusion.v_rgb.weight, eye)
    torch.testing.assert_close(net.fusion.v_t.weight, eye)
    torch.testing.assert_close(
        torch.softmax(net.fusion.fusion_scorer[-1].bias, dim=0),
        torch.tensor([0.9, 0.1]))
    assert torch.count_nonzero(net.anchor_pos_embed) == 0
    assert all(torch.count_nonzero(v) == 0 for v in net.modality_embed.values())

    # ---- mode coverage (doc 14.3) ----
    o_adapt = net(rgb, thr, mode='adapt')
    assert o_adapt['rgb_tokens'].shape == (B, N, D)

    o_rgb = net(rgb, thr, mode='rgb_only')
    assert o_rgb['seq_len'] == N
    assert o_rgb['anchor_map'].shape == (B, D, img[0] // patch, img[1] // patch)

    o_dense = net(rgb, thr, mode='dense')
    assert o_dense['seq_len'] == 2 * N, 'dense = N anchors + N extras'
    assert o_dense['anchor_map'].shape == (B, D, img[0] // patch, img[1] // patch)

    K = N // 3
    o_hard = net(rgb, thr, mode='sparse_hard', target_k=K)
    assert o_hard['seq_len'] == N + K, \
        f"sparse_hard must serialize to N+K, got {o_hard['seq_len']} != {N + K}"

    o_soft = net(rgb, thr, mode='sparse_soft', target_k=K)
    assert o_soft['seq_len'] == 2 * N, 'sparse_soft keeps full length (gated)'

    # ---- missing-modality gating (doc 3.3) ----
    avail = {'rgb': torch.ones(B), 'thermal': torch.zeros(B)}
    o_miss = net(rgb, thr, mode='dense', availability=avail)
    assert o_miss['anchor_map'].shape == o_dense['anchor_map'].shape

    # ---- distillation loss wiring on real anchors (doc 9.3/9.4) ----
    a_sparse = o_hard['anchors']
    a_dense = o_dense['anchors'].detach()
    l_comp = L.compression_loss(a_sparse, a_dense)
    l_rob = L.robust_loss(a_sparse, a_dense)
    # Stage-1 alignment on adapt tokens
    z_r = net.teacher_projector(o_adapt['rgb_tokens'], 'rgb')
    z_t = net.student_projector(o_adapt['thermal_tokens'], 'thermal')
    l_patch = L.cross_patch_loss(z_t, z_r)
    l_region = L.cross_region_loss(z_t, z_r, o_adapt['grid'], region=2)
    l_relation = L.cross_relation_loss(
        z_t, z_r, o_adapt['grid'], region=2, temperature=0.2)
    l_anchor = L.anchor_consistency_loss(a_sparse, a_dense)

    for name, v in [('compression', l_comp), ('robust', l_rob),
                    ('cross_patch', l_patch), ('cross_region', l_region),
                    ('cross_relation', l_relation),
                    ('anchor', l_anchor)]:
        assert v.dim() == 0 and v.item() >= 0.0, \
            f'{name} out of range: {v.item()}'

    # Relational matching preserves pairwise geometry even if the modality
    # uses a different feature-coordinate convention.
    permuted = z_r[..., torch.randperm(z_r.shape[-1])]
    assert L.cross_relation_loss(
        permuted, z_r, o_adapt['grid'], region=2).item() < 1e-5

    # ---- gradient flow end-to-end ----
    (o_dense['anchor_map'].mean() + l_comp + l_rob + l_patch
     + l_region + l_relation + l_anchor).backward()

    # ---- token-budget accounting (doc 15.3) ----
    print('=== DINO-TS CPU integration self-test ===')
    print(f'grid = {img[0] // patch} x {img[1] // patch}  ->  N = {N} anchors')
    print(f'dense       seq_len = {o_dense["seq_len"]}   (N + N)')
    print(f'sparse_hard seq_len = {o_hard["seq_len"]}   (N + K, K={K})')
    print(f'compression={l_comp.item():.4f}  robust={l_rob.item():.4f}  '
          f'anchor={l_anchor.item():.4f}')
    print(f'cross_patch={l_patch.item():.4f}  cross_region={l_region.item():.4f}  '
          f'cross_relation={l_relation.item():.4f}')
    print('ALL OK: SDPA equivalence, checkpointing, 4 modes, missing-gating, '
          'distillation losses, backprop.')


if __name__ == '__main__':
    main()
