"""Shared utilities for V9 model family: QualityPredictor, Gumbel-Softmax,
complementary mask fix, quality degradation, contrastive loss."""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.datasets.transforms.quality_degradation import (
    apply_quality_degradation_rgb,
    apply_quality_degradation_t,
    _QUALITY_RGB_DEG_TYPES,
    _QUALITY_T_DEG_TYPES,
)


# ---------------------------------------------------------------------------
# QualityPredictor
# ---------------------------------------------------------------------------

class QualityPredictor(nn.Module):
    """Per-stage quality predictor with masked global context.

    Input:  x [B, C, H, W] feature map, mask [B, 1, H, W] hard mask (detached).
    Output: gate_logits [B, 2, H, W], q_weight [B, 1, H, W] in [1e-7, 1-1e-7].
    """

    def __init__(self, in_channels):
        super().__init__()
        hidden = min(128, max(64, in_channels // 2))
        self.hidden = hidden
        self.local_conv1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.local_conv2 = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.fuse_conv1 = nn.Conv2d(hidden * 2, hidden, 1, bias=False)
        self.fuse_conv2 = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.gate_head = nn.Conv2d(hidden, 2, 1, bias=True)
        self.weight_head = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, x, mask):
        local = F.gelu(self.local_conv1(x))
        local = F.gelu(self.local_conv2(local))

        mask_d = mask.detach()
        mask_sum = mask_d.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        global_feat = (local * mask_d).sum(dim=(2, 3), keepdim=True) / mask_sum
        global_feat = global_feat.expand(-1, -1, x.shape[2], x.shape[3])

        fused = torch.cat([local, global_feat], dim=1)
        fused = F.gelu(self.fuse_conv1(fused))
        fused = F.gelu(self.fuse_conv2(fused))

        gate_logits = self.gate_head(fused)
        q_weight = torch.sigmoid(self.weight_head(fused)).clamp(1e-7, 1 - 1e-7)
        return gate_logits, q_weight


# backward-compat alias
TokenPrunePredictor = QualityPredictor


# ---------------------------------------------------------------------------
# Gumbel-Softmax hard gating
# ---------------------------------------------------------------------------

def gumbel_softmax_hard(gate_logits, tau=1.0, training=True):
    """Differentiable hard mask via Gumbel-Softmax + Straight-Through Estimator.

    Returns (D_raw, y_soft_keep) where D_raw is hard 0/1 in forward but has
    gradient flowing through y_soft_keep in backward.
    """
    B, C2, H, W = gate_logits.shape
    gate_logits_2d = gate_logits.permute(0, 2, 3, 1).reshape(-1, 2).clamp(-10, 10)
    if training:
        y_soft = F.gumbel_softmax(gate_logits_2d, tau=tau, hard=False)
    else:
        y_soft = F.softmax(gate_logits_2d / max(tau, 0.01), dim=-1).clamp(1e-7, 1 - 1e-7)
    hard_idx = y_soft.argmax(dim=-1, keepdim=True)
    hard_onehot = torch.zeros_like(y_soft).scatter_(1, hard_idx, 1.0)
    y_hard = (hard_onehot - y_soft).detach() + y_soft if training else hard_onehot
    y_hard = y_hard.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    y_soft = y_soft.reshape(B, H, W, 2).permute(0, 3, 1, 2)
    D_raw = 1.0 - y_hard[:, 1:2, :, :]
    y_soft_keep = y_soft[:, 0:1, :, :]
    D_raw = D_raw + y_soft_keep - y_soft_keep.detach()
    return D_raw, y_soft_keep


# ---------------------------------------------------------------------------
# Complementary mask fix for common branch
# ---------------------------------------------------------------------------

def complementary_fix(D_rgb_raw, D_t_raw, q_rgb_weight, q_t_weight):
    """Ensure at least one modality keeps its token at every spatial position."""
    both_zero = (D_rgb_raw < 0.5) & (D_t_raw < 0.5)
    if not both_zero.any():
        return D_rgb_raw.clone(), D_t_raw.clone()
    rgb_better = q_rgb_weight >= q_t_weight
    D_rgb_fixed = D_rgb_raw.clone()
    D_t_fixed = D_t_raw.clone()
    fix_rgb = both_zero & rgb_better
    fix_t = both_zero & ~rgb_better
    D_rgb_fixed = torch.where(fix_rgb, torch.ones_like(D_rgb_fixed), D_rgb_fixed)
    D_t_fixed = torch.where(fix_rgb, torch.zeros_like(D_t_fixed), D_t_fixed)
    D_rgb_fixed = torch.where(fix_t, torch.zeros_like(D_rgb_fixed), D_rgb_fixed)
    D_t_fixed = torch.where(fix_t, torch.ones_like(D_t_fixed), D_t_fixed)
    # STE wrapper: forward=hard, backward=raw
    D_rgb = D_rgb_fixed + D_rgb_raw - D_rgb_raw.detach()
    D_t = D_t_fixed + D_t_raw - D_t_raw.detach()
    return D_rgb, D_t


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def downsample_mask(D, H_k, W_k):
    if D.shape[2] == H_k and D.shape[3] == W_k:
        return D
    return F.adaptive_max_pool2d(D.float(), (H_k, W_k))


# ---------------------------------------------------------------------------
# Quality degradation (progressive curriculum)
# ---------------------------------------------------------------------------

def _denorm_to_01(norm_tensor, mean, std):
    raw = norm_tensor * std + mean
    return (raw / 255.0).clamp(0, 1)


def _renorm_from_01(tensor_01, mean, std):
    raw = tensor_01 * 255.0
    return (raw - mean) / std


def _apply_degradation(img_tensor, modality, mean, std,
                       deg_type=None, level=None,
                       is_local=False, local_mask=None):
    B, C, H, W = img_tensor.shape
    img_01 = _denorm_to_01(img_tensor, mean, std)
    if deg_type is None:
        deg_type = random.choice(
            _QUALITY_RGB_DEG_TYPES if modality == 'rgb' else _QUALITY_T_DEG_TYPES)
    if level is None:
        level = random.randint(2, 5)
    if modality == 'rgb':
        deg_img_01 = apply_quality_degradation_rgb(img_01, deg_type, level)
    else:
        deg_img_01 = apply_quality_degradation_t(img_01, deg_type, level)
    if is_local and local_mask is not None:
        if local_mask.shape[2:] != (H, W):
            local_mask = F.interpolate(local_mask.float(), size=(H, W), mode='nearest')
        local_mask_01 = local_mask.expand(B, C, H, W)
        deg_img_01 = img_01 * (1 - local_mask_01) + deg_img_01 * local_mask_01
    return _renorm_from_01(deg_img_01, mean, std)


def _generate_local_mask(B, H, W, num_regions=3, device='cpu', level=2):
    mask = torch.zeros(B, 1, H, W, device=device)
    coverage = {2: 0.20, 3: 0.35, 4: 0.50, 5: 0.70}
    target_area = coverage.get(level, 0.20) * H * W / num_regions
    for b in range(B):
        for _ in range(num_regions):
            aspect = random.uniform(0.5, 2.0)
            area = target_area * random.uniform(0.7, 1.3)
            rh = max(min(int(math.sqrt(area / aspect)), H), 8)
            rw = max(min(int(math.sqrt(area * aspect)), W), 8)
            y1 = random.randint(0, max(H - rh, 0))
            x1 = random.randint(0, max(W - rw, 0))
            mask[b, 0, y1:y1 + rh, x1:x1 + rw] = 1.0
    return mask


def _lerp(a, b, t):
    return a + (b - a) * t


def get_degradation_schedule(r):
    """Return probability dict for each degradation type at training progress r."""
    if r < 0.1:
        t = r / 0.1
        p_local, p_global, p_missing = 1.0, 0.0, 0.0
        local_levels = {2: _lerp(0.7, 0.7, t), 3: _lerp(0.3, 0.3, t), 4: 0.0, 5: 0.0}
        global_levels = {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    elif r < 0.3:
        t = (r - 0.1) / 0.2
        p_local, p_global, p_missing = _lerp(1.0, 0.8, t), _lerp(0.0, 0.2, t), 0.0
        local_levels = {2: _lerp(0.7, 0.1, t), 3: _lerp(0.3, 0.5, t), 4: _lerp(0.0, 0.4, t), 5: 0.0}
        global_levels = {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    elif r < 0.5:
        t = (r - 0.3) / 0.2
        p_local, p_global, p_missing = _lerp(0.8, 0.6, t), 0.2, _lerp(0.0, 0.2, t)
        local_levels = {2: 0.0, 3: _lerp(0.5, 0.3, t), 4: 0.4, 5: _lerp(0.0, 0.3, t)}
        global_levels = {2: _lerp(1.0, 0.5, t), 3: _lerp(0.0, 0.5, t), 4: 0.0, 5: 0.0}
    elif r < 0.7:
        t = (r - 0.5) / 0.2
        p_local, p_global, p_missing = _lerp(0.6, 0.3, t), _lerp(0.2, 0.25, t), _lerp(0.2, 0.45, t)
        local_levels = {2: 0.0, 3: 0.2, 4: 0.4, 5: 0.4}
        global_levels = {2: 0.3, 3: 0.4, 4: 0.3, 5: 0.0}
    elif r < 0.9:
        p_local, p_global, p_missing = 0.25, 0.25, 0.5
        local_levels = {2: 0.0, 3: 0.1, 4: 0.4, 5: 0.5}
        global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
    else:
        p_local, p_global, p_missing = 0.25, 0.25, 0.5
        local_levels = {2: 0.0, 3: 0.1, 4: 0.4, 5: 0.5}
        global_levels = {2: 0.2, 3: 0.4, 4: 0.3, 5: 0.1}
    return dict(p_local=p_local, p_global=p_global, p_missing=p_missing,
                local_levels=local_levels, global_levels=global_levels)


def sample_level(level_dist):
    levels, probs = [], []
    for lv in [2, 3, 4, 5]:
        p = level_dist.get(lv, 0.0)
        if p > 0:
            levels.append(lv); probs.append(p)
    if not levels:
        return 2
    probs = [p / sum(probs) for p in probs]
    return random.choices(levels, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Cross-modal contrastive loss (used by both MiT and Swin V9)
# ---------------------------------------------------------------------------

def compute_cross_modal_contrastive_loss(feat_rgb, feat_t, labels, D_rgb, D_t,
                                          q_rgb=None, q_t=None,
                                          tau_c=0.07, num_samples=512,
                                          ignore_label=255, pad_mask=None):
    """Category-aware InfoNCE between RGB and thermal features.

    Gate:       D_rgb>0.5 AND D_t>0.5 AND label≠ignore_label AND pad_mask.
    Positives:  same spatial position, rgb ↦ t.
    Negatives:  any valid pixel whose class differs. Same-class ≠ self are
                neutral (masked to -inf in the denominator).
    Weight:     w = min(qr, qt)*(1-|qr-qt|), normalised to sum to 1.
    """
    B, C, H, W = feat_rgb.shape
    device = feat_rgb.device

    # ---- spatial alignment ----
    def _align(t, size, mode='nearest', out_dtype=None):
        if t is None:
            return None
        if t.shape[-2:] != size:
            t = F.interpolate(t.float().unsqueeze(0) if t.dim() == 2 else t.float(),
                              size=size, mode=mode)
            t = t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t
        if out_dtype is not None:
            t = t.to(out_dtype)
        return t

    target = (H, W)
    labels_rs = _align(labels, target, out_dtype=torch.long).squeeze(1) if labels.dim() == 3 else \
        _align(labels.unsqueeze(1), target, out_dtype=torch.long).squeeze(1)
    D_rgb = _align(D_rgb, target)
    D_t = _align(D_t, target)
    if pad_mask is not None:
        pad_mask = _align(pad_mask, target, out_dtype=torch.bool)

    # ---- valid pixel mask ----
    D_dual = (D_rgb.squeeze(1) > 0.5) & (D_t.squeeze(1) > 0.5)
    valid_label = (labels_rs != ignore_label)
    if pad_mask is not None:
        valid_label = valid_label & pad_mask
    valid_mask = D_dual & valid_label
    valid_flat = valid_mask.reshape(-1)

    valid_idx = valid_flat.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    # ---- subsample ----
    if num_samples > 0 and num_samples < valid_idx.numel():
        perm = torch.randperm(valid_idx.numel(), device=device)[:num_samples]
        valid_idx = valid_idx[perm]
    N = valid_idx.numel()

    # ---- extract features & labels ----
    f_rgb_flat = feat_rgb.permute(0, 2, 3, 1).reshape(B * H * W, C)
    f_t_flat = feat_t.permute(0, 2, 3, 1).reshape(B * H * W, C)
    l_flat = labels_rs.reshape(-1)

    f_rgb_sel = f_rgb_flat[valid_idx]  # (N, C)
    f_t_sel = f_t_flat[valid_idx]      # (N, C)
    l_sel = l_flat[valid_idx]          # (N,)

    # ---- L2 normalise ----
    f_rgb_n = F.normalize(f_rgb_sel, dim=-1)
    f_t_n = F.normalize(f_t_sel, dim=-1)

    # ---- similarity matrix (rgb query, t keys) ----
    sim = (f_rgb_n @ f_t_n.T) / tau_c  # (N, N)

    # ---- category-aware negative mask ----
    same_class = (l_sel.unsqueeze(1) == l_sel.unsqueeze(0))  # (N, N)
    is_self = torch.eye(N, dtype=torch.bool, device=device)
    neg_mask = (~is_self) & (~same_class)                     # diff class, not self

    # ---- InfoNCE with logsumexp (masked) ----
    pos_scores = sim.diag()  # (N,)
    sim_masked = sim.masked_fill(~neg_mask, float('-inf'))
    # prepend positive score column
    sim_ext = torch.cat([pos_scores.unsqueeze(1), sim_masked], dim=1)  # (N, N+1)
    log_denom = torch.logsumexp(sim_ext, dim=1)                       # (N,)
    loss_per_sample = -pos_scores + log_denom                         # (N,)

    # ---- quality-consistency weight ----
    if q_rgb is not None and q_t is not None:
        q_rgb = _align(q_rgb, target)
        q_t = _align(q_t, target)
        qr = q_rgb.squeeze(1).reshape(-1)[valid_idx]
        qt = q_t.squeeze(1).reshape(-1)[valid_idx]
        w = torch.min(qr, qt) * (1.0 - torch.abs(qr - qt))
        w = w / (w.sum() + 1e-8)
    else:
        w = torch.ones(N, device=device) / N

    return (loss_per_sample * w).sum()
