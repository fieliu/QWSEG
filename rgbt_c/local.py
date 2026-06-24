"""LocalCorruption: 局部退化包装器 (训练期在线退化使用).

设计依据: docs/RGBT-C_Benchmark.md §7
测试期只使用全局退化, 局部退化仅用于训练增强.
"""
import numpy as np


class LocalCorruption:
    """局部退化包装器: 在图像随机矩形区域内施加退化.

    Args:
        corruption: 被包装的全局退化实例 (Corruption)
        area_ratio_range: 单矩形面积占图像比例范围 (min, max)
        num_rects: 矩形数量范围 (min, max), 随机采样
        seed: 随机种子 (None 表示不固定)
    """

    def __init__(self, corruption,
                 area_ratio_range=(0.1, 0.4),
                 num_rects=(1, 3),
                 seed=None):
        self.corruption = corruption
        self.area_ratio_range = area_ratio_range
        self.num_rects = num_rects
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def _rand(self, low, high):
        if self.rng is not None:
            return self.rng.uniform(low, high)
        return np.random.uniform(low, high)

    def _randint(self, low, high):
        if self.rng is not None:
            return self.rng.randint(low, high + 1)
        return np.random.randint(low, high + 1)

    def _sample_rect_mask(self, h, w):
        """在 (h, w) 图像上随机生成 1-N 个矩形 mask."""
        mask = np.zeros((h, w), dtype=bool)
        n = self._randint(self.num_rects[0], self.num_rects[1])
        img_area = h * w
        for _ in range(n):
            # 目标面积
            target_ratio = self._rand(*self.area_ratio_range)
            target_area = img_area * target_ratio
            # 长宽比随机 (0.5 - 2.0)
            aspect = self._rand(0.5, 2.0)
            rect_h = int(np.sqrt(target_area / aspect))
            rect_w = int(np.sqrt(target_area * aspect))
            rect_h = min(rect_h, h)
            rect_w = min(rect_w, w)
            # 随机位置
            top = self._randint(0, h - rect_h)
            left = self._randint(0, w - rect_w)
            mask[top:top + rect_h, left:left + rect_w] = True
        return mask

    def __call__(self, img, severity=1):
        """在随机矩形区域施加退化, 其余区域保持原始.

        Args:
            img: uint8 ndarray (H, W, C)
            severity: 1-5
        Returns:
            uint8 ndarray, same shape
        """
        # 1. 全图施加退化
        img_corrupted = self.corruption(img, severity)
        # 2. 采样矩形 mask
        h, w = img.shape[:2]
        mask = self._sample_rect_mask(h, w)
        # 3. 仅在 mask 区域替换
        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]
        return np.where(mask, img_corrupted, img)

    def __repr__(self):
        return (f'LocalCorruption({self.corruption.name}, '
                f'area_ratio={self.area_ratio_range}, '
                f'num_rects={self.num_rects})')
