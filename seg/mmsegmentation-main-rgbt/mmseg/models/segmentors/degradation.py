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
        self.level_probs = kwargs.get('level_probs', [0.1, 0.15, 0.2, 0.25, 0.3])
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
        at TWO independently-sampled levels (light lvl_lo <= heavy lvl_hi)
        on the SAME spatial region. The OTHER modality stays clean in both
        versions (honoring "at least one modality usable").

        Key design:
          - ALL samples are degraded (no whole-sample clean). Clean tokens
            come from OUTSIDE the local degraded region, providing the
            upper-anchor for quality supervision naturally.
          - Light/heavy levels are sampled INDEPENDENTLY from _LEVEL_PROBS,
            then assigned by magnitude. This produces diverse severity gaps
            (L1vsL5, L2vsL4, L3vsL5, ...) instead of a fixed gap.
          - _LEVEL_PROBS is tilted towards HEAVY levels (L5=0.30 > L1=0.10)
            to strengthen robustness under severe degradation.
          - Light and heavy SHARE the same spatial mask (sm). The area is
            sampled from _LEVEL_AREA[lvl_hi] (coupled to the HEAVY level),
            so both versions degrade the exact same region -> rank loss can
            compare s_light vs s_heavy at the same positions.

        Degradation types:
          - gaussian_noise / motion_blur / stripe_noise: continuous severity
            (level controls noise sigma / blur kernel), same region.
          - missing: discrete (ratio=1.0 in region, 0 outside). Severity =
            area size (L2=small region .. L5=large region). light/heavy
            differ by area size (level-area coupling).

        Args:
            rgb, ir: [B,3,H,W] NORMALIZED tensors (data_preprocessor output).
            mean, std: 6-channel mean/std lists/tensors (RGB=0:3, T=3:6); used to
                denorm to [0,1] for the degradation bank, then renorm back.
        Returns:
            light_rgb, light_ir, heavy_rgb, heavy_ir: [B,3,H,W] normalized.
            level_light_rgb, level_light_ir: [B,1,H,W] long; L1=clean
                (outside region OR lvl_lo=1 inside), L2-L5=degradation level
                (inside region, =lvl_lo). Label for the LIGHT version.
            level_heavy_rgb, level_heavy_ir: [B,1,H,W] long; same but =lvl_hi
                inside region. Label for the HEAVY version.
            Used directly by quality supervision:
              - anchor: lvl=1 -> s>=clean_floor; lvl=5 -> s<=deg_ceiling
              - rank: s_light - s_heavy >= margin * level_gap (in region)
            rank_mask: [B] float in {0,1}; 1 if lvl_hi > lvl_lo (valid rank
                pair), 0 if equal (skip rank loss for this sample).
            level_gap: [B] float; lvl_hi - lvl_lo (adaptive margin for rank
                loss: larger gap -> larger required margin).
        """
        from mmseg.datasets.transforms.quality_degradation import (
            apply_quality_degradation_rgb, apply_quality_degradation_t)

        # Core degradation types (3 per modality): continuous (noise/blur) +
        # modality-specific (motion_blur/stripe_noise) + discrete (missing).
        # Rationale: enough variety for the quality predictor to learn
        # continuous quality scores, without the 8-type complexity that
        # dilutes per-type training signal. All LOCAL (spatial mask).
        _RGB_CORE_TYPES = ['gaussian_noise', 'motion_blur', 'missing']
        _T_CORE_TYPES = ['gaussian_noise', 'stripe_noise', 'missing']

        # Level-area coupling: higher level -> larger degraded area.
        # Physical rationale: severe degradation (heavy noise/blur/missing)
        # typically affects a larger spatial extent. Level 1 = clean (area 0).
        _LEVEL_AREA = {
            1: [0.0],                         # clean, no degradation
            2: [0.1, 0.15, 0.2],              # mild, small area
            3: [0.2, 0.3, 0.4],               # moderate
            4: [0.3, 0.5, 0.7],               # heavy, larger area
            5: [0.5, 0.7, 0.9, 1.0],          # severe, large area to full
        }
        # Level probabilities: TILTED TOWARDS HEAVY degradation to strengthen
        # robustness under severe conditions. L5 (worst) gets the highest
        # probability. L1 (clean) kept low (0.10) since clean tokens are
        # already abundant OUTSIDE the degraded region (local degradation).
        _LEVEL_PROBS = [0.10, 0.15, 0.20, 0.25, 0.30]

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
        # Level masks: L1=clean (outside region, or L1 applied inside),
        # L2-L5=degradation level (inside region). The ENTIRE image defaults
        # to L1 (no degradation = L1 = original image). Inside the region,
        # light version has lvl_lo, heavy version has lvl_hi.
        # Two masks returned: level_light (for light version) and level_heavy
        # (for heavy version), since light and heavy may have different levels
        # in the region.
        # Used directly by quality supervision:
        #   - anchor: lvl=1 -> s>=clean_floor; lvl=5 -> s<=deg_ceiling
        #   - rank:    s_light - s_heavy >= margin * (lvl_hi - lvl_lo)
        level_light_rgb = torch.ones(B, 1, H, W, device=dev, dtype=torch.long)
        level_light_ir = torch.ones(B, 1, H, W, device=dev, dtype=torch.long)
        level_heavy_rgb = torch.ones(B, 1, H, W, device=dev, dtype=torch.long)
        level_heavy_ir = torch.ones(B, 1, H, W, device=dev, dtype=torch.long)
        # Level pair per sample: rank loss only applies when lvl_hi > lvl_lo.
        lvl_lo_arr = torch.zeros(B, dtype=torch.long, device=dev)
        lvl_hi_arr = torch.zeros(B, dtype=torch.long, device=dev)

        for b in range(B):
            # ALL samples are degraded (no whole-sample clean). Clean tokens
            # come from OUTSIDE the local degraded region, providing the
            # upper-anchor for quality supervision naturally.
            mod = random.choice(["rgb", "ir"])
            types = _RGB_CORE_TYPES if mod == "rgb" else _T_CORE_TYPES
            deg_type = random.choice(types)

            # Independent level sampling: draw TWO levels independently from
            # _LEVEL_PROBS, then assign the smaller to 'light' and the larger
            # to 'heavy'. This naturally produces paired samples where:
            #   - Some pairs are equal (both same level) -> NO rank signal,
            #     rank loss is SKIPPED for these (see eomt_rgbt_quality.py:
            #     rank loss only applies when lvl_hi > lvl_lo).
            #   - Most pairs span a range of severity gaps (L1vsL5, L2vsL4,
            #     L3vsL5, etc.) -> rich continuous ranking signal.
            # Area is coupled to the HEAVY level (the more severe one drives
            # the spatial extent).
            #
            # SPECIAL CASE for 'missing': missing is BINARY (ratio=1.0, region
            # zeroed). There is no "mild missing" vs "severe missing" by ratio
            # — zeroed is zeroed. So:
            #   - heavy level is FIXED at L5 (missing = most severe).
            #   - light level is FIXED at L1 (clean — no "light missing").
            #   - The sampled "level" only selects the REGION SIZE (from
            #     _LEVEL_AREA), giving small/medium/large/full missing holes.
            #   - rank pair: clean (L1) vs zeroed (L5) — the only valid
            #     ordering for a binary degradation.
            if deg_type == "missing":
                lvl_lo = 1                       # clean (no light missing)
                lvl_hi = 5                       # missing is always L5 (worst)
                # Sample region size level (2-5) independently for variety.
                area_lvl = random.choices([2, 3, 4, 5],
                                          weights=_LEVEL_PROBS[1:], k=1)[0]
                area = random.choice(_LEVEL_AREA[area_lvl])
            else:
                lvl_a = random.choices([1, 2, 3, 4, 5],
                                       weights=_LEVEL_PROBS, k=1)[0]
                lvl_b = random.choices([1, 2, 3, 4, 5],
                                       weights=_LEVEL_PROBS, k=1)[0]
                lvl_lo, lvl_hi = min(lvl_a, lvl_b), max(lvl_a, lvl_b)
                # When lvl_lo == lvl_hi, both versions are identical -> no
                # rank signal. We do NOT force them apart (that would inject
                # a false ordering). The rank loss handles this by skipping
                # equal pairs (rank_mask=0).
                area = random.choice(_LEVEL_AREA[lvl_hi])

            # Build spatial mask (single rectangle, random aspect ratio).
            if area <= 0:
                sm = torch.zeros(1, 1, H, W, device=dev, dtype=rgb.dtype)
            else:
                aspect = random.uniform(0.5, 2.0)  # avoid always-square
                target_area = H * W * area
                rh = max(1, int((target_area * aspect) ** 0.5))
                rw = max(1, int(target_area / rh))
                rh, rw = min(rh, H), min(rw, W)
                y0 = random.randint(0, max(0, H - rh))
                x0 = random.randint(0, max(0, W - rw))
                sm = torch.zeros(1, 1, H, W, device=dev, dtype=rgb.dtype)
                sm[:, :, y0:y0 + rh, x0:x0 + rw] = 1.0

            apply = (apply_quality_degradation_rgb if mod == "rgb"
                     else apply_quality_degradation_t)
            mm, ss = (m_rgb, s_rgb) if mod == "rgb" else (m_t, s_t)
            src = (rgb if mod == "rgb" else ir)[b:b + 1]
            src01 = denorm(src, mm, ss)
            light01 = apply(src01.clone(), deg_type, lvl_lo, spatial_mask=sm)
            heavy01 = apply(src01.clone(), deg_type, lvl_hi, spatial_mask=sm)
            light_n = renorm(light01, mm, ss)
            heavy_n = renorm(heavy01, mm, ss)

            # Build level masks: L1 everywhere (clean default), then set the
            # region to the ACTUAL level applied. light region=lvl_lo,
            # heavy region=lvl_hi. Outside region stays L1 for both.
            # When lvl_lo=1, light region is also L1 (= clean, no degradation).
            lvl_mask_light = torch.where(
                sm.squeeze(0) > 0.5,
                torch.full_like(sm.squeeze(0), lvl_lo, dtype=torch.long),
                torch.ones_like(sm.squeeze(0), dtype=torch.long))  # [1,H,W]
            lvl_mask_heavy = torch.where(
                sm.squeeze(0) > 0.5,
                torch.full_like(sm.squeeze(0), lvl_hi, dtype=torch.long),
                torch.ones_like(sm.squeeze(0), dtype=torch.long))
            if mod == "rgb":
                light_rgb[b:b + 1] = light_n
                heavy_rgb[b:b + 1] = heavy_n
                level_light_rgb[b:b + 1] = lvl_mask_light
                level_heavy_rgb[b:b + 1] = lvl_mask_heavy
            else:
                light_ir[b:b + 1] = light_n
                heavy_ir[b:b + 1] = heavy_n
                level_light_ir[b:b + 1] = lvl_mask_light
                level_heavy_ir[b:b + 1] = lvl_mask_heavy

            # Record level pair for rank loss: margin scales with level gap.
            lvl_lo_arr[b] = lvl_lo
            lvl_hi_arr[b] = lvl_hi

        # rank_mask[b] = 1 if this sample has a valid rank pair (lvl_hi > lvl_lo)
        rank_mask = (lvl_hi_arr > lvl_lo_arr).float()
        # level_gap[b] = lvl_hi - lvl_lo (for adaptive margin in rank loss)
        level_gap = (lvl_hi_arr - lvl_lo_arr).float()
        return (light_rgb, light_ir, heavy_rgb, heavy_ir,
                level_light_rgb, level_light_ir,
                level_heavy_rgb, level_heavy_ir,
                rank_mask, level_gap)
