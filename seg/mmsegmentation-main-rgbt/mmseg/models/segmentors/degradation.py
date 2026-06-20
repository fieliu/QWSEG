"""In-training degradation generator for RGB-T MISSING-modality robustness.

Missing-only variant (Paradigm One). Degrades ONE modality at a time (the other
stays fully usable, honoring "at least one modality usable per position") via:
  - 'missing'       : whole modality zeroed
  - 'local_missing' : a random region zeroed (local dropout)
Returns degraded tensors + per-modality masks (1 = missing/zeroed at that pixel).

Missing is binary by nature (zeroed or not), so there is no severity grading and
quality supervision is a clean 0/1 target. Region size for local_missing covers
a range of scales for coverage.

`make_paired` (used by EoMTRGBTQuality, use_quality=True) is the CONTINUOUS,
multi-level variant: it reuses the project's 5-level degradation bank
(quality_degradation.py, levels 1=clean .. 5=worst) to produce two severity
versions of the SAME image (same modality, same degradation type, same smooth
spatial mask, only the level differs). This gives a relative-ordering signal
(light level < heavy level) that lets the quality predictor learn a CONTINUOUS,
spatially-structured score instead of collapsing to a binary detector.
"""
import random
import torch

# region area fractions for local_missing (covers small..large holes)
_LOCAL_AREA = [0.10, 0.25, 0.40, 0.60, 0.80]


class DegradationGenerator:
    def __init__(
        self,
        kinds=("missing", "local_missing"),
        kind_probs=(0.5, 0.5),
        degrade_prob=0.8,          # prob that a sample is degraded at all
        curriculum=False,
        total_epochs=200,
        **kwargs,                  # tolerate legacy keys (severity_range, etc.)
    ):
        self.kinds = list(kinds)
        self.kind_probs = list(kind_probs)
        self.degrade_prob = degrade_prob
        self.curriculum = curriculum
        self.total_epochs = total_epochs

    @torch.no_grad()
    def __call__(self, rgb, ir, epoch=0):
        """Returns degraded_rgb, degraded_ir, mask_rgb, mask_ir.

        mask_* : [B, 1, H, W] float, 1 where that modality is MISSING (zeroed).
        """
        B, _, H, W = rgb.shape
        dev = rgb.device
        drgb = rgb.clone()
        dir_ = ir.clone()
        mask_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=rgb.dtype)
        mask_ir = torch.zeros(B, 1, H, W, device=dev, dtype=ir.dtype)

        for b in range(B):
            if random.random() > self.degrade_prob:
                continue  # this sample stays clean
            mod = random.choice(["rgb", "ir"])  # degrade one modality only
            kind = random.choices(self.kinds, weights=self.kind_probs, k=1)[0]

            img = drgb[b:b + 1] if mod == "rgb" else dir_[b:b + 1]
            m = mask_rgb[b:b + 1] if mod == "rgb" else mask_ir[b:b + 1]

            if kind == "missing":
                img.zero_()
                m.fill_(1.0)
            else:  # local_missing: zero a random region
                area = random.choice(_LOCAL_AREA)
                rh = max(1, int(H * (area ** 0.5)))
                rw = max(1, int(W * (area ** 0.5)))
                y0 = random.randint(0, max(0, H - rh))
                x0 = random.randint(0, max(0, W - rw))
                img[:, :, y0:y0 + rh, x0:x0 + rw] = 0.0
                m[:, :, y0:y0 + rh, x0:x0 + rw] = 1.0

            if mod == "rgb":
                drgb[b:b + 1] = img
                mask_rgb[b:b + 1] = m
            else:
                dir_[b:b + 1] = img
                mask_ir[b:b + 1] = m

        return drgb, dir_, mask_rgb, mask_ir

    @torch.no_grad()
    def make_paired(self, rgb, ir, mean, std, epoch=0):
        """Continuous multi-level PAIRED degradation for quality ranking (E2).

        For each sample: pick ONE modality + ONE degradation type, apply it
        UNIFORMLY (no spatial mask -- severity is the discrete level) at TWO
        levels (light lvl_lo < heavy lvl_hi). The OTHER modality stays clean in
        both versions (honoring "at least one modality usable"). 'missing' is
        forced to heavy level 5 (the only level it zeroes at), light a clean
        level 1-4 -> a strong clean-vs-zeroed ranking pair.

        Args:
            rgb, ir: [B,3,H,W] NORMALIZED tensors (data_preprocessor output).
            mean, std: 6-channel mean/std lists/tensors (RGB=0:3, T=3:6); used to
                denorm to [0,1] for the degradation bank, then renorm back.
        Returns:
            light_rgb, light_ir, heavy_rgb, heavy_ir: [B,3,H,W] normalized.
            region_rgb, region_ir: [B,1,H,W] in {0,1}; 1 over a degraded modality
                (whole map, since degradation is uniform). Gates the rank loss to
                the degraded modality only (the clean modality is identical in
                both versions -> no ranking signal).
        """
        from mmseg.datasets.transforms.quality_degradation import (
            apply_quality_degradation_rgb, apply_quality_degradation_t,
            _QUALITY_RGB_DEG_TYPES, _QUALITY_T_DEG_TYPES)

        B, _, H, W = rgb.shape
        dev = rgb.device
        m = torch.as_tensor(mean, device=dev, dtype=rgb.dtype)
        s = torch.as_tensor(std, device=dev, dtype=rgb.dtype)
        m_rgb, m_t = m[:3].view(1, 3, 1, 1), m[3:].view(1, 3, 1, 1)
        s_rgb, s_t = s[:3].view(1, 3, 1, 1), s[3:].view(1, 3, 1, 1)

        def denorm(x, mm, ss):   # normalized -> [0,1]
            return ((x * ss + mm) / 255.0).clamp(0.0, 1.0)

        def renorm(x01, mm, ss):  # [0,1] -> normalized
            return (x01 * 255.0 - mm) / ss

        light_rgb, light_ir = rgb.clone(), ir.clone()
        heavy_rgb, heavy_ir = rgb.clone(), ir.clone()
        region_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=rgb.dtype)
        region_ir = torch.zeros(B, 1, H, W, device=dev, dtype=ir.dtype)

        for b in range(B):
            if random.random() > self.degrade_prob:
                continue  # this sample stays clean in BOTH versions
            mod = random.choice(["rgb", "ir"])
            types = _QUALITY_RGB_DEG_TYPES if mod == "rgb" else _QUALITY_T_DEG_TYPES
            deg_type = random.choice(types)

            # two distinct levels, light < heavy. 'missing' only zeroes at L5.
            if deg_type == "missing":
                lvl_hi = 5
                lvl_lo = random.randint(1, 4)
            else:
                lo, hi = sorted(random.sample([1, 2, 3, 4, 5], 2))
                lvl_lo, lvl_hi = lo, hi

            apply = (apply_quality_degradation_rgb if mod == "rgb"
                     else apply_quality_degradation_t)
            mm, ss = (m_rgb, s_rgb) if mod == "rgb" else (m_t, s_t)
            src = (rgb if mod == "rgb" else ir)[b:b + 1]
            src01 = denorm(src, mm, ss)
            light01 = apply(src01.clone(), deg_type, lvl_lo, spatial_mask=None)
            heavy01 = apply(src01.clone(), deg_type, lvl_hi, spatial_mask=None)
            light_n = renorm(light01, mm, ss)
            heavy_n = renorm(heavy01, mm, ss)

            if mod == "rgb":
                light_rgb[b:b + 1] = light_n
                heavy_rgb[b:b + 1] = heavy_n
                region_rgb[b:b + 1] = 1.0
            else:
                light_ir[b:b + 1] = light_n
                heavy_ir[b:b + 1] = heavy_n
                region_ir[b:b + 1] = 1.0

        return (light_rgb, light_ir, heavy_rgb, heavy_ir,
                region_rgb, region_ir)
