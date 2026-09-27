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

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKBONES = os.path.join(_HERE, '..', 'mmseg', 'models', 'backbones')
_DINOTS = os.path.join(_HERE, '..', 'mmseg', 'models', 'segmentors', 'dino_ts')
sys.path.insert(0, os.path.abspath(_BACKBONES))
sys.path.insert(0, os.path.abspath(_DINOTS))

import dino_shared_vit as dv          # noqa: E402
import losses as L                    # noqa: E402


def main():
    torch.manual_seed(0)
    B = 2
    img = (48, 64)                     # patch 16 -> grid 3 x 4 -> N = 12
    patch = 16
    N = (img[0] // patch) * (img[1] // patch)
    D = 768

    net = dv._DinoSharedViTImpl(
        img_size=img, patch_size=patch, embed_dims=D, depth=6, fusion_block=2,
        _build_backbone=False)
    rgb = torch.randn(B, 3, *img)
    thr = torch.randn(B, 3, *img)

    # ---- mode coverage (doc 14.3) ----
    o_adapt = net(rgb, thr, mode='adapt')
    assert o_adapt['rgb_tokens'].shape == (B, N, D)

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
    l_anchor = L.anchor_consistency_loss(a_sparse, a_dense)

    for name, v in [('compression', l_comp), ('robust', l_rob),
                    ('cross_patch', l_patch), ('cross_region', l_region),
                    ('anchor', l_anchor)]:
        assert v.dim() == 0 and 0.0 <= v.item() <= 2.0 + 1e-4, \
            f'{name} out of range: {v.item()}'

    # ---- gradient flow end-to-end ----
    (o_dense['anchor_map'].mean() + l_comp + l_rob + l_patch
     + l_region + l_anchor).backward()

    # ---- token-budget accounting (doc 15.3) ----
    print('=== DINO-TS CPU integration self-test ===')
    print(f'grid = {img[0] // patch} x {img[1] // patch}  ->  N = {N} anchors')
    print(f'dense       seq_len = {o_dense["seq_len"]}   (N + N)')
    print(f'sparse_hard seq_len = {o_hard["seq_len"]}   (N + K, K={K})')
    print(f'compression={l_comp.item():.4f}  robust={l_rob.item():.4f}  '
          f'anchor={l_anchor.item():.4f}')
    print(f'cross_patch={l_patch.item():.4f}  cross_region={l_region.item():.4f}')
    print('ALL OK: 4 modes, missing-gating, distillation losses, backprop.')


if __name__ == '__main__':
    main()
