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
        nn.init.constant_(self.score_head.bias, 4.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.score_head.bias, 4.0)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        glob = F.adaptive_avg_pool2d(x, 1).expand(-1, -1, H, W)
        x = self.ctx_proj(torch.cat([x, glob], dim=1))
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
    """Attention bias function with continuous gradient.

    s >> tau: 0.0 (no bias, tiny gradient pulling s toward tau if overshoot)
    s > tau:  smooth quadratic ramp near tau for gradient continuity
    s <= tau: -alpha * (tau - s) / tau (strong negative bias)
    Gradient is non-zero everywhere so QP can always learn.
    """
    ramp_width = 0.1
    low_bias = -alpha * (tau - s) / tau
    high_ramp = -alpha * ramp_width / tau * ((tau + ramp_width - s) / ramp_width).pow(2)
    return torch.where(
        s > tau + ramp_width,
        torch.zeros_like(s),
        torch.where(s > tau, high_ramp, low_bias))


def f_fuse(s, tau=0.3, epsilon=1e-6, beta=3.0):
    """Fusion weight function: quality-modulated weight.

    s > tau: s (direct quality score as weight)
    s <= tau: epsilon + (tau - epsilon) * (s / tau) ** beta (rapidly decaying weight, continuous at tau)
    """
    return torch.where(s > tau, s, epsilon + (tau - epsilon) * (s / tau) ** beta)


def ste_hard_mask(s, tau=0.3):
    """STE quality mask: s > tau → 1, s <= tau → 0.
    Forward: binary gate. Backward: identity gradient (STE).
    """
    hard = (s > tau).float()
    return hard + s - s.detach()


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
        pooled_prev = F.adaptive_avg_pool2d(cumulative_prev, (target_h, target_w))
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
    return F.adaptive_avg_pool2d(D.float(), (H_k, W_k))


def apply_mask_to_gt(data_samples, mask_2d):
    """Apply a spatial mask to GT labels, setting masked-out pixels to ignore_label.

    Args:
        data_samples: list of SegDataSample with gt_sem_seg.data of shape
                      [1, H, W] or [H, W]
        mask_2d: [B, H_gt, W_gt] float tensor, values 0/1.
                 mask_2d < 0.5 → set label to 255 (ignore).

    Returns:
        list of new SegDataSample with masked GT. Original samples are not
        modified in-place.
    """
    import copy
    from mmseg.structures import SegDataSample
    from mmengine.structures import PixelData

    new_samples = []
    for i, ds in enumerate(data_samples):
        new_ds = SegDataSample(metainfo=copy.deepcopy(ds.metainfo))
        old_gt = ds.gt_sem_seg.data
        gt = old_gt.clone()
        m = mask_2d[i]
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if gt.dim() == 3 and gt.shape[0] == 1:
            gt_squeezed = gt.squeeze(0)
        else:
            gt_squeezed = gt
        if m.shape != gt_squeezed.shape:
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0).float(),
                size=gt_squeezed.shape, mode='nearest').squeeze(0).squeeze(0)
        gt_squeezed = torch.where(m < 0.5, torch.full_like(gt_squeezed, 255), gt_squeezed)
        if old_gt.dim() == 3 and old_gt.shape[0] == 1:
            new_ds.gt_sem_seg = PixelData(data=gt_squeezed.unsqueeze(0))
        else:
            new_ds.gt_sem_seg = PixelData(data=gt_squeezed)
        new_samples.append(new_ds)
    return new_samples


def compute_missing_loss(ds_r, ds_t, dspr, dspt, miss_rgb, miss_t, num_stages=4):
    """MSE loss pushing quality scores toward 0 on known-missing regions.

    Args:
        ds_r, ds_t: lists of per-stage quality scores for common branch
                    [B, 1, H_i, W_i]
        dspr, dspt: lists of per-stage quality scores for private branches
        miss_rgb: [B, 1, H, W] mask for RGB missing (1 = missing)
        miss_t:  [B, 1, H, W] mask for thermal missing
        num_stages: number of stages to consider

    Returns:
        scalar loss
    """
    device = miss_rgb.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for score_list, miss_mask in [
        (ds_r, miss_rgb),
        (ds_t, miss_t),
        (dspr, miss_rgb),
        (dspt, miss_t),
    ]:
        for i in range(min(num_stages, len(score_list))):
            s = score_list[i]
            if s is None:
                continue
            H_s, W_s = s.shape[2], s.shape[3]
            if miss_mask.shape[2:] != (H_s, W_s):
                m = F.adaptive_avg_pool2d(miss_mask.float(), (H_s, W_s))
            else:
                m = miss_mask.float()
            m_binary = (m > 0.5).float()
            if m_binary.sum() > 0:
                total_loss += ((s * m_binary).pow(2)).sum() / (m_binary.sum() + 1e-6)
                count += 1

    if count > 0:
        total_loss = total_loss / count

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss


def quality_guided_loss(s_clean_list, s_deg_list, deg_level,
                        num_stages=4,
                        w_low=1.0, w_high=1.0, w_rank=1.0,
                        low_thresh=0.1, high_thresh=0.7, rank_margin=0.1):
    total_loss = torch.tensor(0.0, device=deg_level.device)
    count = 0

    for i in range(min(num_stages, len(s_clean_list), len(s_deg_list))):
        s_c = s_clean_list[i].float()
        s_d = s_deg_list[i].float()
        H_s, W_s = s_d.shape[2], s_d.shape[3]

        if s_c.shape[2:] != s_d.shape[2:]:
            s_c = F.interpolate(s_c, size=(H_s, W_s), mode='bilinear', align_corners=False)

        lvl = F.adaptive_avg_pool2d(deg_level.float(), (H_s, W_s))
        lvl = lvl.round().clamp(0, 5).long()

        is_missing = (lvl == 5).float()
        is_clean = (lvl == 0).float()

        if is_missing.sum() > 0:
            loss_low = (is_missing * F.relu(s_d - low_thresh).pow(2)).sum() / (is_missing.sum() + 1e-6)
            total_loss = total_loss + w_low * loss_low
            count += 1

        if is_clean.sum() > 0:
            loss_high_c = (is_clean * F.relu(high_thresh - s_c).pow(2)).sum() / (is_clean.sum() + 1e-6)
            total_loss = total_loss + w_high * loss_high_c
            count += 1
            loss_high_d = (is_clean * F.relu(high_thresh - s_d).pow(2)).sum() / (is_clean.sum() + 1e-6)
            total_loss = total_loss + w_high * loss_high_d
            count += 1

        if is_clean.sum() > 0:
            loss_rank = (is_clean * F.relu(s_d - s_c.detach() + rank_margin).pow(2)).sum() / (is_clean.sum() + 1e-6)
            total_loss = total_loss + w_rank * loss_rank
            count += 1

    if count > 0:
        total_loss = total_loss / count

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.0, device=deg_level.device, requires_grad=True)

    return total_loss


# ---------------------------------------------------------------------------
# Quality degradation (progressive curriculum)
# ---------------------------------------------------------------------------

def _denorm_to_01(norm_tensor, mean, std):
    raw = norm_tensor * std + mean
    return (raw / 255.0).clamp(0, 1)


def _renorm_from_01(tensor_01, mean, std):
    raw = tensor_01 * 255.0
    return (raw - mean) / std


def _apply_missing_degradation(img_tensor, mean, std):
    B, C, H, W = img_tensor.shape
    img_01 = _denorm_to_01(img_tensor, mean, std)
    img_01 = torch.zeros_like(img_01)
    return _renorm_from_01(img_01, mean, std)


def _apply_local_missing(img_tensor, mean, std, local_mask):
    B, C, H, W = img_tensor.shape
    img_01 = _denorm_to_01(img_tensor, mean, std)
    if local_mask.shape[2:] != (H, W):
        local_mask = F.interpolate(local_mask.float(), size=(H, W), mode='nearest')
    local_mask_01 = local_mask.expand(B, C, H, W)
    img_01 = img_01 * (1 - local_mask_01)
    return _renorm_from_01(img_01, mean, std)


def _generate_single_rect_mask(B, H, W, area_ratio, device='cpu'):
    mask = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        area = H * W * area_ratio
        aspect = 0.5 + torch.rand(1, device=device).item() * 1.5  # [0.5, 2.0]
        rh = max(min(int(math.sqrt(area / aspect)), H), 4)
        rw = max(min(int(math.sqrt(area * aspect)), W), 4)
        y1 = torch.randint(0, max(H - rh, 0) + 1, (1,), device=device).item()
        x1 = torch.randint(0, max(W - rw, 0) + 1, (1,), device=device).item()
        mask[b, 0, y1:y1 + rh, x1:x1 + rw] = 1.0
    return mask


def _lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def get_missing_schedule(r):
    """Return degradation schedule for missing-only training.

    Progressive curriculum:
    - r < 0.01:     no degradation (2-epoch clean-only warmup at 200E)
    - 0.01 - 0.10:  local missing only, area 5% -> 30%
    - 0.10+:        local missing 30% -> 60%, global missing 20% -> 60%

    Returns:
        dict with keys:
            p_local: probability of local missing
            p_global: probability of global missing (entire modality zeroed)
            local_area: area ratio for local missing rectangle
    """
    if r < 0.01:
        return dict(p_local=0.0, p_global=0.0, local_area=0.0)
    elif r < 0.10:
        t = (r - 0.01) / 0.09
        local_area = _lerp(0.05, 0.30, t)
        return dict(p_local=1.0, p_global=0.0, local_area=local_area)
    else:
        t = min((r - 0.10) / 0.90, 1.0)
        local_area = _lerp(0.30, 0.60, t)
        p_global = _lerp(0.20, 0.60, t)
        p_local = 1.0 - p_global
        return dict(p_local=p_local, p_global=p_global, local_area=local_area)


get_degradation_schedule = get_missing_schedule
_apply_degradation = _apply_missing_degradation
_generate_local_mask = _generate_single_rect_mask

def sample_level(level_dist):
    return 5


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
