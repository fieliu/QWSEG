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

import numpy as np
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
        """Continuous multi-level PAIRED degradation for quality ranking.

        对齐 RGBT-C 标准 (rgbt_c/corruptions.py), 直接调用 rgbt_c 库的退化函数.
        训练时在 [0,1] 空间施加退化 (与测试集生成一致), 然后反归一化回模型空间.

        流程 (每个样本):
          1. 随机选一个模态 (rgb / ir)
          2. 从该模态的退化类型列表中随机选一种 (全部 13 种, 见 _RGB_TYPES/_T_TYPES)
          3. 从 0-5 级中独立采样两个级别 (lvl_lo <= lvl_hi):
               0 级 = 干净 (无退化)
               1-5 级 = rgbt_c 的 severity 1-5
             级别概率权重 _LEVEL_PROBS = [10, 10, 15, 20, 25, 30]
             (0 级占 ~9%, 5 级占 ~27%, 高级别比例更大)
          4. 根据 lvl_hi 采样局部区域 (level-area coupling)
          5. 在同一区域上施加 light(lvl_lo) 和 heavy(lvl_hi) 两个版本
          6. 区域外保持干净 (level=0)

        Args:
            rgb, ir: [B,3,H,W] NORMALIZED tensors (data_preprocessor output).
            mean, std: 6-channel mean/std (RGB=0:3, T=3:6).
        Returns:
            light_rgb, light_ir, heavy_rgb, heavy_ir: [B,3,H,W] normalized.
            level_light_rgb/ir: [B,1,H,W] long; 0=clean(outside region or
                lvl_lo=0), 1-5=degradation level (inside region, =lvl_lo).
            level_heavy_rgb/ir: [B,1,H,W] long; same but =lvl_hi inside.
            rank_mask: [B] float; 1 if lvl_hi > lvl_lo (valid rank pair).
            level_gap: [B] float; lvl_hi - lvl_lo.
        """
        import sys
        import os
        # 添加 QWSEG 根目录到 path, 以便 import rgbt_c
        # degradation.py 在 mmseg/models/segmentors/ 下, 需上溯 5 级到 QWSEG 根:
        #   segmentors -> models -> mmseg -> mmsegmentation-main-rgbt -> seg -> QWSEG
        qwseg_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
        if qwseg_root not in sys.path:
            sys.path.insert(0, qwseg_root)
        from rgbt_c import get_corruption, RGB_CORRUPTIONS, T_CORRUPTIONS

        # 全部 13 种退化类型 (与 rgbt_c/corruptions.py 一致)
        _RGB_TYPES = RGB_CORRUPTIONS  # 7 种: gaussian_noise, shot_noise,
                                       #       motion_blur, defocus_blur,
                                       #       fog, low_light, rgb_missing
        _T_TYPES = T_CORRUPTIONS       # 6 种: t_gaussian_noise, stripe_noise,
                                       #       t_motion_blur, t_defocus_blur,
                                       #       t_missing, t_quantization

        # 级别概率权重: 0级(干净)=10, 1-5级逐渐增加, 5级最多=30
        # 归一化后: 0级~9%, 1级~9%, 2级~14%, 3级~18%, 4级~23%, 5级~27%
        _LEVEL_CHOICES = [0, 1, 2, 3, 4, 5]
        _LEVEL_PROBS = [10, 10, 15, 20, 25, 30]

        # 区域大小: 固定范围采样, 与级别解耦.
        # 级别控制退化强度 (0-5), 区域控制退化空间范围, 两者独立.
        # 固定 [0.3, 0.8] (参考 CutMix [Yun et al. ICCV 2019] 标准范围,
        # 覆盖中度遮挡主要区间 30-70% [COCO-Occ, 目标检测遮挡定义]):
        #   - 下限 0.3: 与 CutMix 默认 min 一致, 退化区域有足够语义内容
        #   - 上限 0.8: 与 CutMix 默认 max 一致, 保留 30% 干净上下文供融合补偿
        _AREA_RANGE = (0.3, 0.8)

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

        def apply_rgbt_c_corruption(img01_np, corruption_name, severity):
            """调用 rgbt_c 库施加退化.
            Args:
                img01_np: [H,W,C] float32 in [0,1] -> 转 uint8 [0,255]
                corruption_name: rgbt_c 注册的退化名称
                severity: 1-5 (0 = 不退化, 直接返回)
            Returns:
                [H,W,C] float32 in [0,1]
            """
            if severity == 0:
                return img01_np
            # rgbt_c 接受 uint8 [0,255]
            img_uint8 = np.clip(img01_np * 255.0, 0, 255).astype(np.uint8)
            corr = get_corruption(corruption_name)
            out_uint8 = corr(img_uint8, severity=severity)
            return out_uint8.astype(np.float32) / 255.0

        light_rgb, light_ir = rgb.clone(), ir.clone()
        heavy_rgb, heavy_ir = rgb.clone(), ir.clone()
        # Level masks: 0=clean (outside region or lvl=0),
        # 1-5=degradation level (inside region)
        level_light_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        level_light_ir = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        level_heavy_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        level_heavy_ir = torch.zeros(B, 1, H, W, device=dev, dtype=torch.long)
        lvl_lo_arr = torch.zeros(B, dtype=torch.long, device=dev)
        lvl_hi_arr = torch.zeros(B, dtype=torch.long, device=dev)

        for b in range(B):
            # 1. 选模态
            mod = random.choice(["rgb", "ir"])
            types = _RGB_TYPES if mod == "rgb" else _T_TYPES
            deg_type = random.choice(types)

            # 2. 独立采样两个级别 (0-5), 小的给 light, 大的给 heavy
            lvl_a = random.choices(_LEVEL_CHOICES, weights=_LEVEL_PROBS, k=1)[0]
            lvl_b = random.choices(_LEVEL_CHOICES, weights=_LEVEL_PROBS, k=1)[0]
            lvl_lo, lvl_hi = min(lvl_a, lvl_b), max(lvl_a, lvl_b)

            # 3. missing 退化特殊处理: missing 是二值的 (置零), 无级别梯度
            #    heavy 固定为 5 级 (全区域置零), light 固定为 0 级 (干净)
            #    区域大小从固定范围采样 (与普通退化一致)
            if deg_type in ('missing', 'rgb_missing', 't_missing'):
                lvl_lo = 0   # clean
                lvl_hi = 5   # full missing
            # 区域大小: 从固定范围 [0.3, 0.8] 均匀采样, 与级别解耦
            # (0 级=干净时区域无意义, 退化不会施加)
            area = random.uniform(*_AREA_RANGE)

            # 4. 构建空间 mask (单个矩形, 随机长宽比)
            if lvl_lo == 0 and lvl_hi == 0:
                # 两个级别都是 0 (干净), 无需退化区域
                sm = torch.zeros(1, 1, H, W, device=dev, dtype=rgb.dtype)
            else:
                aspect = random.uniform(0.5, 2.0)
                target_area = H * W * area
                rh = max(1, int((target_area * aspect) ** 0.5))
                rw = max(1, int(target_area / rh))
                rh, rw = min(rh, H), min(rw, W)
                y0 = random.randint(0, max(0, H - rh))
                x0 = random.randint(0, max(0, W - rw))
                sm = torch.zeros(1, 1, H, W, device=dev, dtype=rgb.dtype)
                sm[:, :, y0:y0 + rh, x0:x0 + rw] = 1.0

            # 5. 施加退化 (在 [0,1] 空间, 调用 rgbt_c 库)
            mm, ss = (m_rgb, s_rgb) if mod == "rgb" else (m_t, s_t)
            src = (rgb if mod == "rgb" else ir)[b:b + 1]
            src01 = denorm(src, mm, ss)  # [1,3,H,W] in [0,1]

            # 转为 numpy [H,W,C] 施加退化, 再转回 tensor
            src_np = src01.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]

            # light 版本 (lvl_lo)
            if lvl_lo == 0:
                light_np = src_np.copy()
            else:
                light_full = apply_rgbt_c_corruption(src_np, deg_type, lvl_lo)
                # 仅在 mask 区域施加退化, 区域外保持干净
                sm_np = sm.squeeze(0).squeeze(0).cpu().numpy()  # [H,W]
                light_np = np.where(
                    sm_np[:, :, np.newaxis] > 0.5, light_full, src_np)

            # heavy 版本 (lvl_hi)
            if lvl_hi == 0:
                heavy_np = src_np.copy()
            else:
                heavy_full = apply_rgbt_c_corruption(src_np, deg_type, lvl_hi)
                sm_np = sm.squeeze(0).squeeze(0).cpu().numpy()
                heavy_np = np.where(
                    sm_np[:, :, np.newaxis] > 0.5, heavy_full, src_np)

            # 转回 tensor [1,3,H,W] 并 renorm
            light_t = torch.from_numpy(light_np).permute(2, 0, 1).unsqueeze(0).to(dev)
            heavy_t = torch.from_numpy(heavy_np).permute(2, 0, 1).unsqueeze(0).to(dev)
            light_n = renorm(light_t, mm, ss)
            heavy_n = renorm(heavy_t, mm, ss)

            # 6. 构建 level masks
            # 区域内: light=lvl_lo, heavy=lvl_hi; 区域外: 0 (clean)
            lvl_mask_light = torch.where(
                sm.squeeze(0) > 0.5,
                torch.full_like(sm.squeeze(0), lvl_lo, dtype=torch.long),
                torch.zeros_like(sm.squeeze(0), dtype=torch.long))
            lvl_mask_heavy = torch.where(
                sm.squeeze(0) > 0.5,
                torch.full_like(sm.squeeze(0), lvl_hi, dtype=torch.long),
                torch.zeros_like(sm.squeeze(0), dtype=torch.long))

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

            lvl_lo_arr[b] = lvl_lo
            lvl_hi_arr[b] = lvl_hi

        rank_mask = (lvl_hi_arr > lvl_lo_arr).float()
        level_gap = (lvl_hi_arr - lvl_lo_arr).float()
        return (light_rgb, light_ir, heavy_rgb, heavy_ir,
                level_light_rgb, level_light_ir,
                level_heavy_rgb, level_heavy_ir,
                rank_mask, level_gap)
