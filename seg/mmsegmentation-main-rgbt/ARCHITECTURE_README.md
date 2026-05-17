# QWSEG: Quality-Aware RGB-Thermal Segmentation

## Architecture Overview

QWSEG is a quality-aware multi-modal segmentation framework that disentangles RGB-Thermal features into **common (通用)** and **private (私有)** components, with quality-guided fusion at every stage.

### Core Pipeline

```
Input RGB ──┐                              ┌── Quality Pyramid Net (RGB) ── q_rgb_maps
             │                              │
             ├── Common Backbone (MiT/Swin) ├── Quality-Weighted Common Fusion ── zc_fused
             │                              │
Input T ────┘                              └── Quality Pyramid Net (T) ──── q_t_maps

Input RGB ── Private Branch RGB ── zp_rgb ── CrossAttn Enhance ── rgb_enhanced ──┐
                                                                                  ├── Quality-Weighted Final Fusion ── final_fused ── Decode Head
Input T ──── Private Branch T ──── zp_t ─── CrossAttn Enhance ── t_enhanced ────┘
```

### Stage-by-Stage Detail

#### 1. Quality Pyramid Network

A lightweight network that predicts per-pixel quality scores for each modality at multiple scales.

- Input: RGB image / Thermal image
- Output: Multi-scale quality maps `q_rgb_maps[i]`, `q_t_maps[i]` (i = 0..3)
- Each quality map has shape `(B, 1, H_i, W_i)`, values in `[0, 1]`
- Pretrained on quality classification, optionally frozen for initial epochs

#### 2. Common Branch (Quality-Aware Backbone)

Both modalities are concatenated along the batch dimension and processed by a shared backbone (MiT-B2 or Swin-T). At each stage:

1. **Quality Mask Generation** (`_get_keep_mask_1d`):
   - For each pixel position, if quality < threshold (default 0.3), the token is masked
   - **Dual-low protection**: When both modalities are low quality at the same position, the modality with higher quality is **kept** (mask=1.0), the other is soft-masked (mask=0.01)
   - This ensures no position is completely zeroed out

2. **Token Masking**: Features are multiplied by the keep_mask before entering the transformer block
3. **Residual Zeroing**: After each attention sub-block, low-quality tokens' residuals are zeroed to prevent contamination

Output: `zc_rgb_list[i]`, `zc_t_list[i]` — quality-aware common features per stage

#### 3. Quality-Weighted Common Fusion

At each stage, RGB and Thermal common features are fused:

```
zc_fused = (q_rgb / (q_rgb + q_t + eps)) * zc_rgb + (q_t / (q_rgb + q_t + eps)) * zc_t
```

- **Both-low handling**: When `q_rgb < threshold AND q_t < threshold`, directly use the feature from the higher-quality modality (no weighted sum, avoids NaN)
- Result: `zc_fused_list[i]` — fully high-quality common features

#### 4. Private Branch with Cross-Attention Enhancement

Each modality has its own private branch (same architecture as backbone, separate weights). Private features are enhanced via cross-attention with common features:

**CrossAttentionEnhance**:
```
Query  = Private Feature (zp_rgb or zp_t)     — carries modality-specific details
Key    = Common Fused Feature (zc_fused)       — provides clean cross-modal context
Value  = Common Fused Feature (zc_fused)       — stable, high-quality semantics
```

Steps:
1. Compute cross-attention: `attn_out = softmax(Q @ K^T / sqrt(d)) @ V`
2. Residual connection: `enhanced = zp + proj(attn_out)`
3. **Quality zeroing**: Multiply by `keep_mask_1d` to ensure low-quality tokens remain zero
4. LayerNorm + quality zeroing again

This design ensures:
- Private features gain complementary context from the other modality via common features
- Low-quality tokens cannot "steal" common information to masquerade as valid features
- The common branch (Key/Value) is not affected by private branch gradients through attention

#### 5. Quality-Weighted Final Fusion

Enhanced private features from both modalities are fused with quality-aware weighting:

```
# Normalized quality weights
w_rgb = q_rgb / (q_rgb + q_t + eps)
w_t   = q_t   / (q_rgb + q_t + eps)

# Quality-weighted sum
fused = w_rgb * rgb_enhanced + w_t * t_enhanced

# Both-low fallback: use higher-quality modality directly
if both_low:
    fused = rgb_enhanced if q_rgb >= q_t else t_enhanced
```

Then:
1. **Channel-Spatial Attention**: Apply channel attention + spatial attention to each modality's enhanced features, add to fused
2. **MLP Enhancement**: Two-layer MLP with GroupNorm and GELU, residual connection

Output: `final_fused_list[i]` — the ultimate multi-scale features for segmentation

#### 6. Segmentation Head

The final fused features are passed through a neck (optional) and decode head (SegFormer head or Mask2Former) to produce segmentation predictions.

---

## Loss Functions

### Main Losses (full gradient)

| Loss | Description | Weight |
|------|-------------|--------|
| `loss.ce` | Cross-entropy on main decode head | 1.0 |
| `loss.align` | Quality-weighted alignment between common features | `loss_align_weight` (0.5) |
| `loss.invariant` | Unidirectional MSE: clean features (detached) → degraded features | `loss_invariant_weight` (1.0) |
| `loss.distill_final` | KL distillation from clean to degraded final predictions | `loss_distill_weight` (1.0) |
| `loss.distill_common` | KL distillation from clean to degraded common predictions | `loss_distill_weight` (1.0) |

### Auxiliary Losses (gradient-isolated, quality-scaled)

| Loss | Description | Effective Weight |
|------|-------------|------------------|
| `loss.common.*` | Common auxiliary head | `aux_loss_weight` (0.3) |
| `loss.rgb_private.*` | RGB private auxiliary head | `0.3 * q_rgb_scale` |
| `loss.t_private.*` | T private auxiliary head | `0.3 * q_t_scale` |

**Gradient isolation**: Private auxiliary losses use `.detach()` on private features before computing loss, preventing gradients from flowing back to the common branch.

**Quality scaling**: `q_rgb_scale = mean(q_rgb_clean[0] >= 0.5).clamp(min=0.1)` — auxiliary loss weight is proportional to the fraction of high-quality pixels.

### Gradient Flow Summary

```
Main decode head       ← full gradient → Common + Private + CrossAttn + FinalFusion ✅
Common auxiliary head  ← 0.3× gradient → Common branch ✅
RGB private auxiliary  ← 0.3×q_scale×gradient → Private RGB + CrossAttn only ✅ (common detached)
T private auxiliary    ← 0.3×q_scale×gradient → Private T + CrossAttn only ✅ (common detached)
loss_invariant         ← one-way gradient → Only degraded branch ✅ (clean detached)
loss_distill_final     ← full gradient → Degraded common + private + fusion ✅
loss_distill_common    ← full gradient → Degraded common ✅
```

---

## Model Variants

| Model | Backbone | Quality Net | Degradation Training | Config Prefix |
|-------|----------|-------------|---------------------|---------------|
| `MiTMulV12DQualityDisentangle` | MiT-B2 | ✅ | ✅ | `mitmul_v12d_quality_disentangle` |
| `MiTMulV12QualityDisentangleNoDeg` | MiT-B2 | ✅ | ❌ | `mitmul_v12_nodeg_quality_disentangle` |
| `MiTMulV12DisentangleOnly` | MiT-B2 | ❌ | ❌ | `mitmul_v12_disentangle_only` |
| `SwinMulV12DQualityDisentangle` | Swin-T | ✅ | ✅ | `swinmul_v12d_quality_disentangle` |
| `SwinMulV12QualityDisentangleNoDeg` | Swin-T | ✅ | ❌ | `swinmul_v12_nodeg_quality_disentangle` |
| `SwinMulV12DisentangleOnly` | Swin-T | ❌ | ❌ | `swinmul_v12_disentangle_only` |

---

## Training Commands

### SegFormer Head (MiT-B2)

```bash
cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt

# 1. Full model: Quality-aware + Degradation training
python tools/train.py \
    configs/segformer/mitmul_v12d_quality_disentangle_mit-b2_1xb2-50E_mfnet-240x320.py \
    --amp

# 2. Ablation: Quality-aware, no degradation
python tools/train.py \
    configs/segformer/mitmul_v12_nodeg_quality_disentangle_mit-b2_1xb2-50E_mfnet-240x320.py \
    --amp

# 3. Ablation: Disentangle only, no quality, no degradation
python tools/train.py \
    configs/segformer/mitmul_v12_disentangle_only_mit-b2_1xb2-50E_mfnet-240x320.py \
    --amp
```

### SegFormer Head (Swin-T)

```bash
# 4. Full model: Swin-T + Quality-aware + Degradation
python tools/train.py \
    configs/segformer/swinmul_v12d_quality_disentangle_swin-t_1xb2-50E_mfnet-240x320.py \
    --amp
```

### Mask2Former Head (Swin-T, MFNet dataset)

```bash
# 5. Full model with Mask2Former
python tools/train.py \
    configs/mask2former/swinmul_v12d_quality_disentangle_swin-t_1xb2-80K_mfnet-480x640.py \
    --amp

# 6. No degradation variant
python tools/train.py \
    configs/mask2former/swinmul_v12_nodeg_quality_disentangle_swin-t_1xb2-80K_mfnet-480x640.py \
    --amp

# 7. Disentangle only
python tools/train.py \
    configs/mask2former/swinmul_v12_disentangle_only_swin-t_1xb2-80K_mfnet-480x640.py \
    --amp
```

### Mask2Former Head (Swin-T, FMB dataset)

```bash
# 8. Full model on FMB dataset
python tools/train.py \
    configs/mask2former/swinmul_v12d_quality_disentangle_swin-t_1xb2-80K_fmb-480x640.py \
    --amp

# 9. No degradation on FMB
python tools/train.py \
    configs/mask2former/swinmul_v12_nodeg_quality_disentangle_swin-t_1xb2-80K_fmb-480x640.py \
    --amp
```

### With Visualization

Add `--visualize --vis-interval 2 --vis-num-samples 2` to any command for training visualization:

```bash
python tools/train.py \
    configs/segformer/mitmul_v12d_quality_disentangle_mit-b2_1xb2-50E_mfnet-240x320.py \
    --amp --visualize --vis-interval 2 --vis-num-samples 2
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quality_threshold` | 0.3 | Threshold for quality masking |
| `loss_align_weight` | 0.5 | Weight for alignment loss |
| `loss_invariant_weight` | 1.0 | Weight for invariant loss |
| `loss_distill_weight` | 1.0 | Weight for distillation losses |
| `aux_loss_weight` | 0.3 | Weight for auxiliary decode heads |
| `missing_ratio` | 0.3 | Ratio of completely missing modality |
| `global_deg_ratio` | 0.3 | Ratio of globally degraded samples |
| `local_deg_ratio` | 0.4 | Ratio of locally degraded samples |
| `quality_freeze_epochs` | 0 | Epochs to freeze quality net |

---

## Architecture Change Log

### v2.0 — Cross-Attention Fusion (Current)

Replaced STARS gate fusion with cross-attention enhancement + quality-weighted final fusion:

| Component | Old (STARS) | New (CrossAttn) |
|-----------|-------------|-----------------|
| Private-Common Fusion | STARSFusionBlock (gate) | CrossAttentionEnhance (Q=private, KV=common) |
| Final Fusion | FinalFusionBlock (concat+proj) | QualityWeightedFinalFusion (quality-weighted sum + CS-attention + MLP) |
| Quality zeroing after fusion | ❌ | ✅ keep_mask applied after cross-attention residual |
| Quality-weighted final fusion | ❌ | ✅ per-pixel quality weights |
| Both-low handling | N/A | ✅ keep higher-quality modality |
| MLP enhancement | ❌ | ✅ 2-layer MLP with residual |

### v1.0 — Initial Release

- STARS gate fusion for private-common interaction
- Concat + projection for final fusion
- Unidirectional invariant loss
- Gradient isolation for auxiliary losses
