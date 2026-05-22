# QWSEG: Quality-Aware Piecewise Soft Modulation for RGB-Thermal Segmentation

## Architecture Overview

QWSEG is a quality-aware multi-modal segmentation framework that disentangles RGB-Thermal features into **common (通用)** and **private (私有)** components, with **piecewise soft modulation** guiding fusion at every stage.

### Core Idea

Instead of hard binary gating (keep/prune), QWSEG predicts continuous quality scores `s ∈ [0,1]` and applies three complementary piecewise functions — hard zeroing, attention bias, and fusion weight decay — to progressively control low-quality token influence.

### Core Pipeline

```
Input RGB ──┐                              ┌── QualityPredictor (RGB) × 4 stages ── s_rgb
             │                              │
             ├── Common Backbone (Swin/MiT) ├── Quality-Weighted Common Fusion ── zc_fused
             │   (weight-shared, RGB+T      │
             │    concatenated along batch)  └── QualityPredictor (T) × 4 stages ── s_t
             │
Input T ────┘

Input RGB ── Private Branch RGB ── QualityPredictor × 4 ── zp_rgb ── f_fuse modulate + zf ── LayerNorm ── rgb_enhanced ──┐
                                                                                                                            ├── DualGateEnhancedFusion ── final_fused ── Decode Head
Input T ──── Private Branch T ──── QualityPredictor × 4 ── zp_t ──── f_fuse modulate + zf ── LayerNorm ── t_enhanced ────┘
```

### Stage-by-Stage Detail

#### 1. QualityPredictor

A lightweight per-stage module that predicts per-pixel quality scores from feature maps.

- Input: Feature map `x ∈ [B, C, H, W]` (detached from gradient for quality prediction)
- Output: Quality score `s ∈ [B, 1, H, W]`, values in `[1e-7, 1-1e-7]`
- Architecture: `LayerNorm(C) → Conv1×1(C→hidden) → GELU → LayerNorm(hidden) → Conv1×1(hidden→hidden) → GELU → Conv1×1(hidden→1, bias=0.0) → Sigmoid → clamp`
- `hidden = min(128, max(64, C//2))`
- Bias initialization = 0.0 → `sigmoid(0.0) = 0.5`, quality scores start at midpoint, allowing bidirectional learning
- Global feature injection: local features concatenated with global average pooling features, projected back via 1×1 conv `ctx_proj`
- All conv layers: Kaiming Normal (`fan_in`, adapted for GELU)
- Total: 16 predictors (4 stages × 4 sets: common RGB, common T, private RGB, private T)

#### 2. Cross-Stage Cascading Suppression

Deeper stages may "forget" degradation regions identified by shallower stages (context aggregation can make corrupted regions appear normal). Cross-stage propagation ensures low-quality regions once detected are never "recovered":

```
For stage k ≥ 1:
  cumulative_prev = s_0 × s_1 × ... × s_{k-1}  (gradients propagated)
  pooled_prev = AdaptiveMaxPool2d(cumulative_prev, (H_k, W_k))
  if clamp_min > 0:
      pooled_prev = pooled_prev.clamp(min=clamp_min)
  s_k_adjusted = s_k × pooled_prev
  new_cumulative = s_k_adjusted

Stage 0: s_0_adjusted = s_0, cumulative = s_0
```

- **Max pooling** (not average): preserves low scores from shallow stages
- **Differentiable**: deeper `s_k` gets gradient via `pooled_prev`; if `pooled_prev=0`, gradient is 0 (already zeroed, no adjustment needed)
- **Gradient propagation**: `cumulative` not detached, allowing gradients to flow from deep to shallow predictors
- **`clamp_min`**: Phase 3 first 5 epochs apply `clamp_min=0.1` to prevent irreversible zeroing; later reduced to 0

#### 3. Three-Level Progressive Quality Modulation

Given quality score `s ∈ [0,1]`, three piecewise functions work cooperatively:

| Function | Formula | Granularity | Role |
|----------|---------|-------------|------|
| `f_hard_mask` | `s < 0.2 → 0; s ≥ 0.2 → 1` (detached) | Coarsest | Safety net: completely remove hopeless tokens |
| `f_attn` | `s > 0.3 → 0; s ≤ 0.3 → -10×(0.3-s)/0.3` | Intermediate | Suppress low-quality tokens in attention |
| `f_fuse` | `s > 0.3 → s; s ≤ 0.3 → ε+(0.3-ε)×(s/0.3)^6` | Finest | Decay fusion weight for low-quality tokens |

**Synergy table:**

| Quality Range | f_hard_mask | f_attn | f_fuse | Effect |
|---------------|-------------|--------|--------|--------|
| s < 0.2 | 0 (zeroed) | Very strong negative bias | ≈ε | Completely removed |
| 0.2 ≤ s < 0.3 | 1 (kept) | Negative bias (strong→weak) | 0.001→0.3 | Attention suppressed, fusion minimal |
| s ≥ 0.3 | 1 (kept) | 0 (no bias) | s (0.3→1.0) | Normal participation |

#### 4. Common Branch (Quality-Aware Backbone)

Both modalities are concatenated along the batch dimension and processed by a shared backbone (Swin-T or MiT-B2). At each stage:

1. **Quality Prediction**: QualityPredictor predicts `s_rgb` and `s_t` from detached features, then applies cross-stage cascading suppression (`s_k_adjusted = s_k × maxpool(cumulative_prev)`)
2. **Attention Bias Injection**: Relative quality scores are used for attention bias generation
   - `s_avg = (s_rgb + s_t) / 2 + 1e-8`, `rel_s = s / s_avg`
   - When both modalities are low quality, `rel_s ≈ 1`, bias = 0 (no erroneous suppression)
   - `f_attn(rel_s)` is added to attention scores, suppressing low-quality tokens
   - Fusion weights `f_fuse` still use absolute scores `s_rgb` and `s_t`
   - **Swin**: Bias converted to per-window format via `_quality_score_to_swin_bias`, aligned with cyclic shift for SW-MSA
   - **MiT**: Bias downsampled to match K resolution via `adaptive_avg_pool2d`
3. **Hard Zeroing After Norm**: After stage norm_layer, features are multiplied by `f_hard_mask(s)` to ensure clean output
4. **Hard Zeroing Before Downsample**: Before PatchMerging, features are multiplied by `f_hard_mask(s)` to prevent low-quality information spread

**Critical ordering**: `norm → mask` (not `mask → norm`), because LayerNorm maps zero vectors to learnable bias β, causing "resurrection".

Output: `zc_rgb[i]`, `zc_t[i]` — quality-aware common features per stage

#### 5. Quality-Weighted Common Fusion

At each stage, RGB and Thermal common features are fused with quality weighting:

```
w_rgb = f_fuse(s_rgb, τ=0.3, ε=1e-3, β=6.0)
w_t   = f_fuse(s_t,   τ=0.3, ε=1e-3, β=6.0)
w_sum = w_rgb + w_t + 1e-8
zc_fused = (w_rgb / w_sum) × zc_rgb + (w_t / w_sum) × zc_t
```

Then LayerNorm-normalized (no MultiScaleRefine).

Output: `zf[i]` — common fused features

#### 6. Private-Common Feature Fusion (Simplified)

Each stage's private features are directly added to common fused features:

```
Step 1: Soft weight modulation
  w_priv = f_fuse(s_priv, τ, ε, β)
  priv_mod = zp × w_priv

Step 2: Feature addition
  out = zf + priv_mod

Step 3: LayerNorm normalization
  out = LayerNorm(out)
```

Private features are soft-weighted by `f_fuse` and directly added to common features, then LayerNorm-normalized. No MLP or MultiScaleRefine.

Output: `rgb_enhanced[i]`, `t_enhanced[i]`

#### 7. DualGateEnhancedFusion (Final Fusion)

Combines three feature streams per stage:

```
F_rgb, F_t, F_common → spatial alignment via bilinear interpolation
  │
  Concat(F_rgb, F_t) along channels
  │
  ├─ Channel Gate: Conv1×1 → ReLU → Conv1×1 → Sigmoid → split → ch_rgb, ch_t
  ├─ Spatial Gate:  Conv3×3 → Sigmoid → split → sp_rgb, sp_t
  │
  F_fused = ch_rgb × sp_rgb × F_rgb + ch_t × sp_t × F_t
  │
  Pre-Norm residual: LayerNorm(F_fused) → Conv1×1 → GELU → + F_common
  Export normalization: LayerNorm
```

Output: `final_fused[i]` — the ultimate multi-scale features for segmentation

#### 8. Segmentation Heads

- **Main decoder**: Mask2FormerHead (Swin variant) or SegformerHead (MiT variant) on `final_fused`
- **Common auxiliary**: SegformerHead on `zf`
- **RGB private auxiliary**: SegformerHead on `rgb_enhanced`
- **T private auxiliary**: SegformerHead on `t_enhanced`

All auxiliary losses weighted by `aux_loss_weight` (0.3).

---

## Training Strategy

### Three-Phase Training

| Phase | Epochs | Quality Score | Predictor State | force_all_keep | distill / inv |
|-------|--------|---------------|-----------------|----------------|---------------|
| 1 | 0 ~ phase1-1 | All 1 | All frozen | True | ❌ |
| 2 | phase1 ~ phase1+phase2-1 | All 1 | All frozen | True | ❌ |
| 3 | phase1+phase2 ~ end | From predictor | All trainable | False | ✅ |

- **Phase 1**: Foundation learning with all features. Predictors frozen, modulation degenerates to "keep all".
- **Phase 2**: Continued foundation learning. Same as Phase 1 — `force_all_keep=True`, predictors remain frozen. No anchoring loss.
- **Phase 3**: Full adaptive modulation. `force_all_keep=False`, predictors unfrozen. Distillation and invariant losses activated. First 5 epochs apply `clamp_min=0.1` for cross-stage propagation safety.

### Progressive Degradation Curriculum

| Training Progress | Local Deg | Global Deg | Missing Modality | Levels |
|-------------------|-----------|------------|------------------|--------|
| 0–5% | 100% | 0% | 0% | Mild (L2-3) |
| 5–10% | 100→70% | 0→30% | 0% | Mild-Moderate (L2-4) |
| 10–15% | 70→50% | 30→40% | 0→10% | Moderate (L3-5) |
| 15–25% | 50→35% | 40→35% | 10→30% | Moderate-Hard (L3-5) |
| 25–40% | 35→30% | 35% | 30→35% | Hard (L4-5) |
| 40%+ | 30% | 30% | 40% | Hard (L4-5) |

Local degradation includes: global degradation types applied locally, and missing degradation (local regions zeroed out directly).

---

## Loss Functions

```
L_total = L_seg + L_aux + L_align + L_retention + L_deg + L_deg_aux
          + L_distill (Phase 3) + L_inv (Phase 3) + L_priv_align (Phase 3)
```

| Loss | Description | Weight |
|------|-------------|--------|
| `L_seg` | CE + Dice on main decoder (clean + degraded) | 1.0 |
| `L_aux` | CE + Dice on auxiliary heads (clean + degraded) | 0.3 |
| `L_align` | Category-aware InfoNCE contrastive loss | 0.1 |
| `L_retention` | Retention rate regularization (r_min=0.4, r_max=0.98) | 2.0 |
| `L_distill` | KL distillation (clean → degraded), Phase 3 only | 0.3 |
| `L_inv` | Smooth L1 invariant loss (quality-gated), Phase 3 only | 0.03 |
| `L_priv_align` | MSE private-common quality distillation, Phase 3 only | 0.5 |

---

## Model Variants

| Model | Backbone | Main Decoder | Quality Bias | Config |
|-------|----------|-------------|-------------|--------|
| `QualityGatedSwinMask2Former` | Swin-T × 3 | Mask2FormerHead | Window-level (cyclic-shift aligned) | `swinmul_quality_mask2former_swin-t_*.py` |
| `QualityGatedMiTMamba` | MiT-B2 × 3 | SegformerHead | SR-attention K-level (avg-pooled) | `mitmul_quality_mamba_mit-b2_*.py` |

> **Note**: `QualityGatedMiTMamba` class name contains "Mamba" as a legacy naming convention. The current version uses QualityModulatedFusion, not Mamba modules.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.3 | Soft modulation boundary for f_attn and f_fuse |
| `tau_hard` | 0.2 | Hard zeroing threshold |
| `alpha` | 10.0 | Attention bias strength |
| `fuse_beta` | 6.0 | Fusion weight decay exponent |
| `fuse_epsilon` | 1e-3 | Fusion weight minimum |
| `retention_min` | 0.4 | Retention rate lower bound |
| `retention_max` | 0.98 | Retention rate upper bound |
| `loss_distill_weight` | 0.3 | Distillation loss weight |
| `loss_invariant_weight` | 0.03 | Invariant loss weight |
| `aux_loss_weight` | 0.3 | Auxiliary decoder loss weight |

---

## Key Files

| File | Content |
|------|---------|
| `mmseg/models/segmentors/v9_utils.py` | QualityPredictor, f_attn/f_fuse/f_hard_mask, cascade_quality_suppress, degradation schedule, contrastive loss |
| `mmseg/models/segmentors/swin_quality_mask2former.py` | Swin model + Swin-specific quality attention bias + fusion modules |
| `mmseg/models/segmentors/mit_quality_mamba.py` | MiT model + MiT-specific quality attention bias + fusion modules |
| `mmseg/datasets/transforms/quality_degradation.py` | Degradation type implementations |
| `mmseg/engine/hooks/train_vis_hook.py` | Training visualization hook |
