"""Loss functions for the DINO multi-stage Dense-Teacher / Sparse-Student model
(docs/DINO_MULTISTAGE_TEACHER_STUDENT.md sections 6, 8, 9).

Pure PyTorch (no mm* imports) so they are CPU-unit-testable. All feature
arguments are L2-normalizable token/anchor tensors; the segmentors are
responsible for producing them (via AlignmentProjector for Stage 1, raw anchors
for Stage 2B/3).

Conventions:
  - "patch"/"anchor" tensors are [B, N, D]; a valid_mask is [B, N] in {0,1}.
  - Teacher-side targets must already be detached by the caller where required;
    the cosine losses also stop-grad the target argument defensively.
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _masked_mean(per_elem: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """per_elem: [B, N]; mask: [B, N] in {0,1} or None. Returns a scalar."""
    if mask is None:
        return per_elem.mean()
    denom = mask.sum().clamp_min(1.0)
    return (per_elem * mask).sum() / denom


def cosine_distance(z_a: torch.Tensor, z_b: torch.Tensor,
                    valid_mask: Optional[torch.Tensor] = None,
                    stopgrad_b: bool = True) -> torch.Tensor:
    """mean_i (1 - cos(z_a_i, z_b_i)) over valid positions.

    z_a, z_b: [B, N, D]. If they are not already unit-norm they are normalized
    here so the result is a true cosine distance in [0, 2].
    """
    if stopgrad_b:
        z_b = z_b.detach()
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    per = 1.0 - (z_a * z_b).sum(-1)  # [B, N]
    return _masked_mean(per, valid_mask)


# --- Stage 1: cross-modal alignment (doc 6.4) -------------------------------

def cross_patch_loss(z_thermal: torch.Tensor, z_rgb_teacher: torch.Tensor,
                     valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_cross_patch = mean_i (1 - cos(g_student(h_t_i), sg(g_teacher(h_rgb_i)))).

    Both inputs are AlignmentProjector outputs (already L2). Teacher stop-grad.
    """
    return cosine_distance(z_thermal, z_rgb_teacher, valid_mask, stopgrad_b=True)


def cross_region_loss(z_thermal: torch.Tensor, z_rgb_teacher: torch.Tensor,
                      grid_hw, region: int = 2) -> torch.Tensor:
    """Region-pooled variant for slightly-misaligned pairs (doc 6.4).

    Pools projector features over region x region windows before the cosine
    distance. z_*: [B, N, D] with N = H * W; grid_hw = (H, W).
    """
    H, W = grid_hw
    B, N, D = z_thermal.shape
    assert N == H * W, f'token count {N} != H*W {H * W}'

    def pool(z):
        z2d = z.transpose(1, 2).reshape(B, D, H, W)
        z2d = F.avg_pool2d(z2d, kernel_size=region, stride=region,
                           ceil_mode=True)
        return z2d.flatten(2).transpose(1, 2)  # [B, N', D]

    return cosine_distance(pool(z_thermal), pool(z_rgb_teacher),
                           valid_mask=None, stopgrad_b=True)


def cross_relation_loss(z_thermal: torch.Tensor,
                        z_rgb_teacher: torch.Tensor,
                        grid_hw,
                        region: int = 2,
                        temperature: float = 0.2) -> torch.Tensor:
    """Match within-modality region relations without copying RGB features.

    Each pooled region is represented by its cosine similarities to every
    other region in the same image.  The RGB relation distribution is a frozen
    target and the thermal distribution is trained with row-wise KL.  The
    diagonal is excluded because self-similarity is always one and carries no
    cross-modal information.

    Unlike direct token matching, this objective is invariant to a shared
    rotation of the thermal feature coordinates.  Stage 1 therefore combines
    it with a *weak* ``cross_region_loss`` to establish a common coordinate
    system while retaining modality-specific content.
    """
    if temperature <= 0:
        raise ValueError(f'temperature must be positive, got {temperature}')

    H, W = grid_hw
    B, N, D = z_thermal.shape
    if z_rgb_teacher.shape != z_thermal.shape:
        raise ValueError(
            'thermal and RGB projector outputs must have identical shapes, '
            f'got {tuple(z_thermal.shape)} and {tuple(z_rgb_teacher.shape)}')
    if N != H * W:
        raise ValueError(f'token count {N} != H*W {H * W}')

    def pool_and_normalize(z):
        z2d = z.transpose(1, 2).reshape(B, D, H, W)
        z2d = F.avg_pool2d(
            z2d, kernel_size=region, stride=region, ceil_mode=True)
        # Similarity softmax is more stable in fp32 under AMP.
        return F.normalize(z2d.flatten(2).transpose(1, 2).float(), dim=-1)

    thermal = pool_and_normalize(z_thermal)
    rgb = pool_and_normalize(z_rgb_teacher.detach())
    num_regions = thermal.shape[1]
    if num_regions < 2:
        return z_thermal.sum() * 0.0

    thermal_similarity = torch.bmm(thermal, thermal.transpose(1, 2))
    rgb_similarity = torch.bmm(rgb, rgb.transpose(1, 2))
    diagonal = torch.eye(
        num_regions, device=thermal.device, dtype=torch.bool).unsqueeze(0)
    # A finite sentinel keeps the zero-probability diagonal safe in KL under
    # all backends (0 * inf can otherwise become NaN on some kernels).
    # Keep this finite *after* temperature scaling.  ``-finfo.max / tau``
    # overflows to -inf for tau < 1 and can produce 0 * inf = NaN in KL.
    diagonal_value = -1.0e4
    thermal_similarity = thermal_similarity.masked_fill(
        diagonal, diagonal_value)
    rgb_similarity = rgb_similarity.masked_fill(diagonal, diagonal_value)

    teacher_relation = F.softmax(
        rgb_similarity / temperature, dim=-1).detach()
    student_relation = F.log_softmax(
        thermal_similarity / temperature, dim=-1)
    # Mean KL per query region; independent of batch size and token count.
    return F.kl_div(
        student_relation, teacher_relation,
        reduction='none').sum(-1).mean().clamp_min(0.0)


# --- Stage 2B: anchor consistency + optional global DINO (doc 8.3, 8.4) -----

def anchor_consistency_loss(a_online: torch.Tensor, a_ema: torch.Tensor,
                            valid_mask: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
    """L_anchor = mean_i (1 - cos(norm(a_online_i(x_strong)),
                                   sg(norm(a_ema_i(x_weak))))).  (doc 8.3)"""
    return cosine_distance(a_online, a_ema, valid_mask, stopgrad_b=True)


def global_dino_loss(p_online_logits: torch.Tensor, p_ema_logits: torch.Tensor,
                     center: torch.Tensor, teacher_temp: float = 0.04,
                     student_temp: float = 0.1) -> torch.Tensor:
    """Optional DINO-style global CE (doc 8.4): CrossEntropy(sharpen(center(
    p_ema(x_weak))), p_online(x_strong)). center is an EMA buffer maintained by
    the caller (subtracted from teacher logits before sharpening).

    p_*_logits: [B, K] global projection logits. Returns a scalar.
    """
    t = F.softmax((p_ema_logits - center) / teacher_temp, dim=-1).detach()
    s = F.log_softmax(p_online_logits / student_temp, dim=-1)
    return -(t * s).sum(-1).mean()


# --- Stage 3: dense -> sparse distillation (doc 9.3, 9.4, 9.5) --------------

def compression_loss(a_sparse: torch.Tensor, a_dense: torch.Tensor,
                     valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_compression = mean_i (1 - cos(a_sparse_i(C(x)), sg(a_dense_i(C(x))))).

    Isolates the error caused purely by token compression: teacher and student
    receive the SAME degraded input (doc 9.3).
    """
    return cosine_distance(a_sparse, a_dense, valid_mask, stopgrad_b=True)


def robust_loss(a_sparse_deg: torch.Tensor, a_dense_clean: torch.Tensor,
                weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_robust = mean_i (1 - cos(a_sparse_i(C(x)), sg(a_dense_i(x)))).  (doc 9.4)

    Student on degraded input should match dense teacher on CLEAN input. An
    optional per-position weight_map [B, N] can down-weight regions the caller
    knows are unrecoverable (from a known degradation mask / teacher feature
    stability) — never from teacher predictions (no pseudo-labels).
    """
    return cosine_distance(a_sparse_deg, a_dense_clean, weight_map,
                           stopgrad_b=True)


def logit_distill_loss(logits_sparse: torch.Tensor, logits_dense: torch.Tensor,
                       temperature: float = 2.0) -> torch.Tensor:
    """L_logit = T^2 * KL(softmax(logits_dense/T), softmax(logits_sparse/T)).

    Labeled-data-only knowledge distillation on segmentation logits (doc 9.5).
    logits_*: [B, C, H, W]. Teacher detached; aligned to student resolution.
    """
    T = temperature
    if logits_dense.shape[-2:] != logits_sparse.shape[-2:]:
        logits_dense = F.interpolate(logits_dense, size=logits_sparse.shape[-2:],
                                     mode='bilinear', align_corners=False)
    t = F.softmax(logits_dense.detach() / T, dim=1)
    s = F.log_softmax(logits_sparse / T, dim=1)
    kl = (t * (t.clamp_min(1e-8).log() - s)).sum(1)  # [B, H, W]
    return (T * T) * kl.mean()


# ---------------------------------------------------------------------------
# CPU self-test (no mm* deps). Run: python losses.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)
    B, N, D, C = 2, 30 * 40, 768, 9
    H, W = 30, 40

    z_t = torch.randn(B, N, D, requires_grad=True)
    z_r = torch.randn(B, N, D)
    mask = (torch.rand(B, N) > 0.3).float()

    l_patch = cross_patch_loss(z_t, z_r, mask)
    l_region = cross_region_loss(z_t, z_r, (H, W), region=2)
    l_relation = cross_relation_loss(z_t, z_r, (H, W), region=2)
    l_anchor = anchor_consistency_loss(z_t, z_r, mask)
    l_comp = compression_loss(z_t, z_r, mask)
    l_rob = robust_loss(z_t, z_r, mask)

    for name, v in [('cross_patch', l_patch), ('cross_region', l_region),
                    ('cross_relation', l_relation),
                    ('anchor', l_anchor), ('compression', l_comp),
                    ('robust', l_rob)]:
        assert v.dim() == 0, f'{name} must be scalar'
        assert v.item() >= 0.0, f'{name} loss must be non-negative: {v.item()}'

    # perfect match -> ~0 distance
    same = torch.randn(B, N, D)
    assert anchor_consistency_loss(same, same.clone()).item() < 1e-5

    # global DINO
    pe = torch.randn(B, 256)
    po = torch.randn(B, 256, requires_grad=True)
    center = torch.zeros(256)
    l_glob = global_dino_loss(po, pe, center)
    assert l_glob.dim() == 0

    # logit distill
    ls = torch.randn(B, C, H, W, requires_grad=True)
    ld = torch.randn(B, C, H, W)
    l_logit = logit_distill_loss(ls, ld, temperature=2.0)
    assert l_logit.dim() == 0 and l_logit.item() >= 0

    (l_patch + l_region + l_relation + l_anchor + l_comp + l_rob
     + l_glob + l_logit).backward()
    print('losses.py self-test OK: all scalars, losses non-negative, '
          'perfect-match ~0, backprop clean')
