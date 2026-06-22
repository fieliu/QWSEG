"""Shared DETRDistill-style query-matched distillation loss for Mask2Former /
EoMT query outputs (used by both the Swin QualityDistillStudent and the DINOv3
EoMTRGBTQuality).

Why query-level (not per-pixel einsum map): Mask2Former/EoMT emit per-query
(class_logits, mask_logits). The common `einsum(sigmoid(mask), softmax(cls))`
per-pixel map, when normalized by /sum over classes for a KL loss, DEGENERATES
in background regions (no query fires there -> all foreground probs ~0 ->
sum~0 -> normalization explodes into noise), which trains the background wrong
and collapses validation. Distilling at the QUERY level avoids that entirely.

Method (DETRDistill, ICCV 2023): treat the teacher's confident-foreground
queries as pseudo-GT, Hungarian-match student queries to them, and distill each
matched pair: temperature-KD on class logits + BCE on mask logits. Permutation
(arbitrary query order) is resolved by the matching; padding is irrelevant since
losses are computed per query, not per padded pixel.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def _hungarian_match(s_cls, s_mask, t_cls_fg, t_mask_fg, no_obj):
    """Match student queries (s) to teacher pseudo-GT foreground queries (t_fg)
    for ONE image. Cost = class NLL (student pred of teacher's label) + mask BCE.
    s_cls [Q,C+1], s_mask [Q,h,w]; t_cls_fg [K,C+1], t_mask_fg [K,h,w].
    Returns (src_idx, tgt_idx) LongTensors of matched pairs (len = K)."""
    from scipy.optimize import linear_sum_assignment
    Q = s_cls.shape[0]
    K = t_cls_fg.shape[0]
    if K == 0:
        dev = s_cls.device
        return (torch.zeros(0, dtype=torch.long, device=dev),
                torch.zeros(0, dtype=torch.long, device=dev))
    t_lab = t_cls_fg.argmax(-1)                    # [K] teacher's class per fg query
    s_prob = s_cls.softmax(-1)                     # [Q,C+1]
    cost_cls = -s_prob[:, t_lab]                   # [Q,K] -p(student=teacher_label)
    # mask cost: BCE between student mask logits and teacher mask prob, averaged
    s_m = s_mask.flatten(1)                        # [Q, hw]
    t_m = t_mask_fg.sigmoid().flatten(1)           # [K, hw]
    # pairwise BCE-with-logits: mean over pixels
    # bce(x,y) = max(x,0) - x*y + log(1+exp(-|x|)); compute [Q,K] via broadcasting
    x = s_m[:, None, :]                            # [Q,1,hw]
    y = t_m[None, :, :]                            # [1,K,hw]
    bce = x.clamp_min(0) - x * y + torch.log1p(torch.exp(-x.abs()))
    cost_mask = bce.mean(-1)                        # [Q,K]
    cost = (cost_cls + cost_mask).cpu().numpy()
    src, tgt = linear_sum_assignment(cost)
    dev = s_cls.device
    return (torch.as_tensor(src, dtype=torch.long, device=dev),
            torch.as_tensor(tgt, dtype=torch.long, device=dev))


def _query_distill_loss(s_cls, s_mask, t_cls, t_mask, T=2.0):
    """DETRDistill query-matched distillation.
    s_cls/t_cls [B,Q,C+1], s_mask/t_mask [B,Q,h,w] (raw logits).
    Teacher pseudo-GT = queries whose argmax over C+1 is NOT no-object.
    Matched pairs: class temperature-KD (T, scaled by T^2) + mask BCE.
    Returns scalar loss (mean over images with >=1 match)."""
    B, Q, Cp1 = s_cls.shape
    no_obj = Cp1 - 1
    # align teacher mask resolution to student
    if t_mask.shape[-2:] != s_mask.shape[-2:]:
        t_mask = F.interpolate(t_mask, size=s_mask.shape[-2:],
                               mode="bilinear", align_corners=False)
    total = 0.0
    n = 0
    for b in range(B):
        t_lab = t_cls[b].argmax(-1)                 # [Q]
        keep = t_lab != no_obj                      # teacher confident-fg queries
        if keep.sum() == 0:
            continue
        t_cls_fg = t_cls[b][keep]                   # [K,C+1]
        t_mask_fg = t_mask[b][keep]                 # [K,h,w]
        src, tgt = _hungarian_match(
            s_cls[b], s_mask[b], t_cls_fg, t_mask_fg, no_obj)
        if src.numel() == 0:
            continue
        # class temperature-KD on matched pairs
        s_logp = F.log_softmax(s_cls[b, src] / T, dim=-1)
        t_p = F.softmax(t_cls_fg[tgt] / T, dim=-1)
        loss_cls = (T * T) * F.kl_div(s_logp, t_p, reduction="batchmean")
        # mask BCE: student mask logits vs teacher mask prob
        loss_mask = F.binary_cross_entropy_with_logits(
            s_mask[b, src], t_mask_fg[tgt].sigmoid())
        total = total + loss_cls + loss_mask
        n += 1
    if n == 0:
        return s_cls.sum() * 0.0                    # keep graph, zero loss
    return total / n
