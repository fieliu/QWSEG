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
    """Per-stage quality predictor with global context injection.

    Input:  x [B, C, H, W] feature map, mask [B, 1, H, W] hard mask (detached).
    Output: s [B, 1, H, W] continuous quality score in [1e-7, 1-1e-7].
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.ctx_proj = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        hidden = min(128, max(64, in_channels // 2))
        self.hidden = hidden
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.score_head = nn.Conv2d(hidden, 1, 1, bias=True)
        nn.init.constant_(self.score_head.bias, 0.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.score_head.bias, 0.0)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        glob = F.adaptive_avg_pool2d(x.detach(), 1).expand(-1, -1, H, W)
        x = self.ctx_proj(torch.cat([x.detach(), glob], dim=1))
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x).permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = F.gelu(x)
        s = torch.sigmoid(self.score_head(x)).clamp(1e-7, 1 - 1e-7)
        return s

# backward-compat alias
TokenPrunePredictor = QualityPredictor


# ---------------------------------------------------------------------------
# Piecewise modulation functions for continuous quality score
# ---------------------------------------------------------------------------

def f_attn(s, tau=0.3, alpha=10.0):
    """Attention bias function: 0 for high-quality, negative bias for low-quality.

    s > tau: 0.0 (no bias)
    s <= tau: -alpha * (tau - s) / tau (continuous negative bias, range [0, -alpha])
    """
    return torch.where(s > tau, torch.zeros_like(s), -alpha * (tau - s) / tau)


def f_fuse(s, tau=0.3, epsilon=1e-3, beta=6.0):
    """Fusion weight function: quality-modulated weight.

    s > tau: s (direct quality score as weight)
    s <= tau: epsilon + (tau - epsilon) * (s / tau) ** beta (rapidly decaying weight, continuous at tau)
    """
    return torch.where(s > tau, s, epsilon + (tau - epsilon) * (s / tau) ** beta)


def f_hard_mask(s, tau_hard=0.2):
    """Optional hard zeroing mask for extremely low-quality tokens.

    s < tau_hard: 0.0 (completely zeroed)
    s >= tau_hard: 1.0 (kept)
    Must use .detach() to block gradient.
    """
    return (s >= tau_hard).float().detach()


def cascade_quality_suppress(s_current, cumulative_prev, target_h, target_w,
                              clamp_min=0.0):
    """Cross-stage quality score cascading suppression.

    Ensures that once a shallow stage detects a low-quality region, deeper
    stages cannot "recover" it.  The cumulative product of all previous
    stage scores (max-pooled to match resolution) is multiplied element-wise
    with the current stage score.

    Args:
        s_current: [B, 1, H_k, W_k] current stage quality score
        cumulative_prev: [B, 1, H_prev, W_prev] cumulative product of all
            previous stage scores (or None for stage 0)
        target_h, target_w: spatial size of s_current
        clamp_min: lower bound for pooled_prev during early training
            (e.g. 0.1 prevents shallow errors from being unrecoverable)

    Returns:
        s_adjusted: [B, 1, H_k, W_k] adjusted quality score
        new_cumulative: [B, 1, H_k, W_k] updated cumulative product
    """
    if cumulative_prev is None:
        return s_current, s_current

    if cumulative_prev.shape[2:] != (target_h, target_w):
        pooled_prev = F.adaptive_max_pool2d(cumulative_prev, (target_h, target_w))
    else:
        pooled_prev = cumulative_prev

    if clamp_min > 0:
        pooled_prev = pooled_prev.clamp(min=clamp_min)

    s_adjusted = s_current * pooled_prev
    new_cumulative = s_adjusted
    return s_adjusted, new_cumulative


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
    """Return probability dict for each degradation type at training progress r.

    Progressive curriculum:
    - Phase 1 (r<0.05):  local-only, mild, no missing/global
    - Phase 2 (0.05-0.15):  local + global, mild-moderate, no missing
    - Phase 3 early (0.15-0.3):  introduce missing modality gradually
    - Phase 3 late (0.3+):  full mix with 40% missing
    """
    if r < 0.05:          # epoch  0- 9  (200 total) — Phase 1
        t = r / 0.05
        p_local, p_global, p_missing = 1.0, 0.0, 0.0
        local_levels  = {2: _lerp(0.8, 0.5, t), 3: _lerp(0.2, 0.5, t), 4: 0.0, 5: 0.0}
        global_levels = {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    elif r < 0.10:         # epoch 10-19 — Phase 2 early
        t = (r - 0.05) / 0.05
        p_local, p_global, p_missing = _lerp(1.0, 0.7, t), _lerp(0.0, 0.3, t), 0.0
        local_levels  = {2: _lerp(0.5, 0.2, t), 3: _lerp(0.5, 0.4, t), 4: _lerp(0.0, 0.4, t), 5: 0.0}
        global_levels = {2: _lerp(1.0, 0.7, t), 3: _lerp(0.0, 0.3, t), 4: 0.0, 5: 0.0}
    elif r < 0.15:         # epoch 20-29 — Phase 2 late
        t = (r - 0.10) / 0.05
        p_local, p_global, p_missing = _lerp(0.7, 0.5, t), _lerp(0.3, 0.4, t), _lerp(0.0, 0.1, t)
        local_levels  = {2: 0.1, 3: _lerp(0.4, 0.3, t), 4: _lerp(0.4, 0.5, t), 5: _lerp(0.1, 0.2, t)}
        global_levels = {2: _lerp(0.6, 0.3, t), 3: _lerp(0.3, 0.5, t), 4: _lerp(0.1, 0.2, t), 5: 0.0}
    elif r < 0.25:         # epoch 30-49 — Phase 3 early
        t = (r - 0.15) / 0.10
        p_local, p_global, p_missing = _lerp(0.5, 0.35, t), _lerp(0.4, 0.35, t), _lerp(0.1, 0.3, t)
        local_levels  = {2: 0.0, 3: _lerp(0.3, 0.2, t), 4: _lerp(0.5, 0.4, t), 5: _lerp(0.2, 0.4, t)}
        global_levels = {2: 0.2, 3: _lerp(0.5, 0.3, t), 4: _lerp(0.3, 0.4, t), 5: _lerp(0.0, 0.3, t)}
    elif r < 0.40:         # epoch 50-79 — Phase 3 mid
        t = (r - 0.25) / 0.15
        p_local, p_global, p_missing = _lerp(0.35, 0.3, t), 0.35, _lerp(0.3, 0.35, t)
        local_levels  = {2: 0.0, 3: 0.1, 4: _lerp(0.4, 0.3, t), 5: _lerp(0.5, 0.6, t)}
        global_levels = {2: 0.1, 3: _lerp(0.3, 0.3, t), 4: _lerp(0.4, 0.3, t), 5: _lerp(0.2, 0.4, t)}
    else:                   # epoch 80+ — fully trained
        p_local, p_global, p_missing = 0.3, 0.3, 0.4
        local_levels  = {2: 0.0, 3: 0.1, 4: 0.3, 5: 0.6}
        global_levels = {2: 0.1, 3: 0.3, 4: 0.3, 5: 0.3}
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
        orig_dim = t.dim()
        if t.shape[-2:] != size:
            if t.dim() == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 3:
                t = t.unsqueeze(1)
            t = F.interpolate(t.float(), size=size, mode=mode)
        if orig_dim == 2:
            t = t.squeeze(0).squeeze(0)
        elif orig_dim == 3:
            t = t.squeeze(1)
        if out_dtype is not None:
            t = t.to(out_dtype)
        return t

    target = (H, W)
    labels_rs = _align(labels, target, out_dtype=torch.long)
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
    # Clamp similarity to avoid overflow in logsumexp (fp16 safety)
    sim = sim.clamp(-50, 50)
    pos_scores = sim.diag()  # (N,)
    sim_masked = sim.masked_fill(~neg_mask, float('-inf'))
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


# ---------------------------------------------------------------------------
# Explicit quality-score supervision (anchor + ordinal ranking)
# ---------------------------------------------------------------------------

def _pool_level_map(level_map, target_h, target_w):
    """Down-sample a per-pixel degradation-level map to a stage resolution.

    Returns:
        lvl: [B,1,h,w] max-pooled level (token takes the HIGHEST level it covers)
        cov: [B,1,h,w] degraded-pixel fraction inside the token (avg-pool of the
             level>1 binary mask) — used for the local-boundary ignore band.
    """
    lm = level_map.float()
    if lm.shape[2:] != (target_h, target_w):
        lvl = F.adaptive_max_pool2d(lm, (target_h, target_w))
        cov = F.adaptive_avg_pool2d((lm > 1.5).float(), (target_h, target_w))
    else:
        lvl = lm
        cov = (lm > 1.5).float()
    return lvl, cov


def _pool_pad(pad_mask, h, w):
    pm = pad_mask.float()
    if pm.shape[2:] != (h, w):
        pm = F.adaptive_max_pool2d(pm, (h, w))
    return pm


def compute_quality_anchor_loss(s, level_map, pad_mask=None,
                                clean_target=0.8, deg5_target=0.2,
                                cov_lo=0.1, cov_hi=0.9):
    """Single-sided hinge anchors on a raw quality map.

    s:          [B,1,h,w] raw quality score (pre-cascade, post-sigmoid).
    level_map:  [B,1,H,W] per-pixel degradation level (clean=1, deg=2..5).
    Anchors:
        clean tokens (level==1): relu(clean_target - s)  → push up, no ceiling.
        level-5 tokens:          relu(s - deg5_target)    → push down, no floor.
    Mid levels (2/3/4) get NO anchor — shaped only by the ranking loss.
    Local-boundary tokens (cov in (cov_lo, cov_hi)) are ignored.
    """
    h, w = s.shape[2:]
    lvl, cov = _pool_level_map(level_map, h, w)
    clean_mask = (lvl < 1.5).float()
    deg5_mask = (lvl > 4.5).float()
    keep = 1.0 - ((cov > cov_lo) & (cov < cov_hi)).float()
    if pad_mask is not None:
        keep = keep * _pool_pad(pad_mask, h, w)
    loss_clean = F.relu(clean_target - s) * clean_mask * keep
    loss_deg5 = F.relu(s - deg5_target) * deg5_mask * keep
    denom = (clean_mask * keep).sum() + (deg5_mask * keep).sum() + 1e-6
    return (loss_clean.sum() + loss_deg5.sum()) / denom


def compute_quality_rank_loss(s_clean, s_deg, level_clean, level_deg,
                              pad_mask=None, step=0.2, cov_lo=0.1, cov_hi=0.9):
    """Ordinal ranking between the clean and degraded forward passes.

    Where the degraded pass has a higher level than the clean pass, enforce
        s_clean - s_deg >= step * (L_deg - L_clean)
    via relu(step*(L_deg-L_clean) - (s_clean - s_deg)). With clean L=1 this is
    the 0.2-per-level staircase (L2⇒gap≥0.2 … L5⇒gap≥0.8). Unchanged positions
    contribute nothing; local-boundary tokens on either pass are ignored.
    """
    h, w = s_clean.shape[2:]
    lvl_c, cov_c = _pool_level_map(level_clean, h, w)
    lvl_d, cov_d = _pool_level_map(level_deg, h, w)
    margin = step * (lvl_d - lvl_c)
    keep = (margin > 1e-6).float() * (1.0 - (
        ((cov_c > cov_lo) & (cov_c < cov_hi)) |
        ((cov_d > cov_lo) & (cov_d < cov_hi))).float())
    if pad_mask is not None:
        keep = keep * _pool_pad(pad_mask, h, w)
    loss = F.relu(margin - (s_clean - s_deg)) * keep
    return loss.sum() / (keep.sum() + 1e-6)
