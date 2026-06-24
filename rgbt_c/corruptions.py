"""RGBT-C 标准退化实现 (12 种全局退化).

设计依据: docs/RGBT-C_Benchmark.md v1.0
参数严格对齐 ImageNet-C, T 模态按物理特性调整.

接口约定:
    corruption = RGBGaussianNoise()
    img_corrupted = corruption(img, severity=3)   # severity in [1, 5]

输入 img: uint8 ndarray, shape (H, W, C)
    RGB: C=3, range [0, 255]
    T:   C=1 or C=3, range [0, 255]
输出:   uint8 ndarray, same shape as input
"""
import numpy as np
import cv2
from io import BytesIO
from PIL import Image as PILImage

from .utils import (
    disk, plasma_fractal, motion_blur_kernel, apply_blur_per_channel,
    to_float01, to_uint8, ensure_3ch, restore_ch,
)


# /////////////// 基类 ///////////////

class Corruption:
    """退化基类. 所有退化必须实现此接口."""
    name = 'base'
    modality = 'both'
    n_severity = 5

    def __call__(self, img: np.ndarray, severity: int) -> np.ndarray:
        raise NotImplementedError

    def get_params(self, severity: int) -> dict:
        return {}

    def _validate(self, img, severity):
        assert 1 <= severity <= self.n_severity, \
            f'severity must be in [1, {self.n_severity}], got {severity}'
        assert img.dtype == np.uint8, \
            f'img must be uint8, got {img.dtype}'


# /////////////// 注册表 ///////////////

CORRUPTION_REGISTRY = {}


def register_corruption(cls):
    """装饰器: 注册退化类."""
    CORRUPTION_REGISTRY[cls.name] = cls
    return cls


def get_corruption(name: str) -> Corruption:
    """按名称获取退化实例."""
    if name not in CORRUPTION_REGISTRY:
        raise KeyError(
            f'Corruption "{name}" not registered. '
            f'Available: {list(CORRUPTION_REGISTRY.keys())}')
    return CORRUPTION_REGISTRY[name]()


def list_corruptions(modality=None):
    """列出所有注册的退化名称."""
    if modality is None:
        return list(CORRUPTION_REGISTRY.keys())
    return [n for n, c in CORRUPTION_REGISTRY.items()
            if c.modality in (modality, 'both')]


# /////////////// RGB 模态退化 (6 种) ///////////////

@register_corruption
class RGBGaussianNoise(Corruption):
    """RGB 高斯噪声 (共通). 对齐 ImageNet-C gaussian_noise."""
    name = 'gaussian_noise'
    modality = 'rgb'
    _sigmas = [0.08, 0.12, 0.18, 0.26, 0.38]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        c = self._sigmas[severity - 1]
        x = to_float01(img)
        x = x + np.random.normal(size=x.shape, scale=c)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        return {'sigma': self._sigmas[severity - 1]}


@register_corruption
class RGBShotNoise(Corruption):
    """RGB 散粒噪声 (RGB 特有). 对齐 ImageNet-C shot_noise."""
    name = 'shot_noise'
    modality = 'rgb'
    _rates = [60, 25, 12, 5, 3]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        c = self._rates[severity - 1]
        x = to_float01(img)
        x = np.random.poisson(x * c) / float(c)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        return {'rate': self._rates[severity - 1]}


@register_corruption
class RGBMotionBlur(Corruption):
    """RGB 运动模糊 (共通). 对齐 ImageNet-C motion_blur 参数.
    ImageNet-C 使用 wand (ImageMagick) MotionBlur, 此处用 OpenCV 复现.
    """
    name = 'motion_blur'
    modality = 'rgb'
    _params = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        radius, sigma = self._params[severity - 1]
        angle = np.random.uniform(-45, 45)
        kernel = motion_blur_kernel(length=radius, angle=angle, sigma=sigma)
        x = to_float01(img)
        x = apply_blur_per_channel(x, kernel)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        radius, sigma = self._params[severity - 1]
        return {'radius': radius, 'sigma': sigma, 'angle': 'random[-45,45]'}


@register_corruption
class RGBDefocusBlur(Corruption):
    """RGB 离焦模糊 (共通). 对齐 ImageNet-C defocus_blur."""
    name = 'defocus_blur'
    modality = 'rgb'
    _params = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        radius, alias_blur = self._params[severity - 1]
        kernel = disk(radius=radius, alias_blur=alias_blur)
        x = to_float01(img)
        x = apply_blur_per_channel(x, kernel)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        r, a = self._params[severity - 1]
        return {'radius': r, 'alias_blur': a}


@register_corruption
class RGBFog(Corruption):
    """RGB 雾 (RGB 特有). 对齐 ImageNet-C fog."""
    name = 'fog'
    modality = 'rgb'
    _params = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        c0, c1 = self._params[severity - 1]
        x = to_float01(img)
        max_val = x.max()
        h, w = x.shape[:2]
        # plasma_fractal 至少需要 max(h, w) 大小的图, 然后裁剪
        mapsize = 1
        while mapsize < max(h, w):
            mapsize *= 2
        fog_layer = plasma_fractal(mapsize=mapsize, wibbledecay=c1)[:h, :w]
        if x.ndim == 3:
            fog_layer = fog_layer[..., np.newaxis]
        x = x + c0 * fog_layer
        x = x * max_val / (max_val + c0)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        c0, c1 = self._params[severity - 1]
        return {'fog_coef': c0, 'wibbledecay': c1}


@register_corruption
class RGBLowLight(Corruption):
    """RGB 低光照 (RGB 特有).
    参考 ImageNet-C brightness 退化 (HSV 空间调整 V 通道), 方向相反 (减暗).
    ImageNet-C brightness: x[:,:,2] = clip(x[:,:,2] + c, 0, 1)  (增亮)
    Low Light:             x[:,:,2] = clip(x[:,:,2] - c, 0, 1)  (减暗)
    """
    name = 'low_light'
    modality = 'rgb'
    # 与 ImageNet-C brightness 相同的参数, 但方向相反
    _params = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        c = self._params[severity - 1]
        x = to_float01(img)
        # BGR -> HSV (cv2 默认 BGR 输入)
        x_hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        # V 通道减暗 (模拟低光照)
        x_hsv[:, :, 2] = np.clip(x_hsv[:, :, 2] - c, 0, 1)
        x = cv2.cvtColor(x_hsv, cv2.COLOR_HSV2BGR)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        return {'darkness': self._params[severity - 1]}


@register_corruption
class RGBMissing(Corruption):
    """RGB 模态失效 (共通). 整模态置零 (传感器故障/镜头遮挡)."""
    name = 'rgb_missing'
    modality = 'rgb'

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        # severity 不分级, 5 级均为完全缺失
        return np.zeros_like(img)

    def get_params(self, severity):
        return {'mode': 'zero', 'note': 'all severities identical'}


# /////////////// T 模态退化 (6 种) ///////////////

@register_corruption
class TGaussianNoise(Corruption):
    """T 高斯噪声 (共通). 与 RGB Gaussian Noise 相同参数."""
    name = 't_gaussian_noise'
    modality = 't'
    _sigmas = [0.08, 0.12, 0.18, 0.26, 0.38]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        c = self._sigmas[severity - 1]
        target_ch = img.shape[2] if img.ndim == 3 else 1
        x = to_float01(img)
        x = ensure_3ch(x)
        x = x + np.random.normal(size=x.shape, scale=c)
        x = restore_ch(x, target_ch)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        return {'sigma': self._sigmas[severity - 1]}


@register_corruption
class TStripeNoise(Corruption):
    """T 条纹噪声 (T 特有, 自定义).
    模拟红外焦平面列向非均匀性 (固定模式噪声).
    """
    name = 'stripe_noise'
    modality = 't'
    _params = [
        (0.02, 0.005, 0.01),   # sev1
        (0.05, 0.01,  0.02),   # sev2
        (0.10, 0.02,  0.04),   # sev3
        (0.18, 0.04,  0.07),   # sev4
        (0.30, 0.06,  0.12),   # sev5
    ]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        col_amp, row_amp, col_off_std = self._params[severity - 1]
        target_ch = img.shape[2] if img.ndim == 3 else 1
        x = to_float01(img)
        x = ensure_3ch(x)
        h, w, _ = x.shape
        # 列方向条纹 (强) + 行方向条纹 (弱)
        col_offset = np.random.normal(0, col_off_std, size=(1, w))      # (1, W)
        col_gain = 1.0 + np.random.normal(0, col_amp, size=(1, w))      # (1, W)
        row_gain = 1.0 + np.random.normal(0, row_amp, size=(h, 1))      # (H, 1)
        # 广播到 (H, W, C): col_gain (1,W)->(1,W,1), row_gain (H,1)->(H,1,1)
        x = x * col_gain[:, :, np.newaxis] * row_gain[:, :, np.newaxis]
        x = x + col_offset[:, :, np.newaxis]
        x = restore_ch(x, target_ch)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        ca, ra, co = self._params[severity - 1]
        return {'column_amp': ca, 'row_amp': ra, 'column_offset_std': co}


@register_corruption
class TMotionBlur(Corruption):
    """T 运动模糊 (共通). 与 RGB Motion Blur 相同参数."""
    name = 't_motion_blur'
    modality = 't'
    _params = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        radius, sigma = self._params[severity - 1]
        angle = np.random.uniform(-45, 45)
        kernel = motion_blur_kernel(length=radius, angle=angle, sigma=sigma)
        target_ch = img.shape[2] if img.ndim == 3 else 1
        x = to_float01(img)
        x = ensure_3ch(x)
        x = apply_blur_per_channel(x, kernel)
        x = restore_ch(x, target_ch)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        r, s = self._params[severity - 1]
        return {'radius': r, 'sigma': s, 'angle': 'random[-45,45]'}


@register_corruption
class TDefocusBlur(Corruption):
    """T 离焦模糊 (共通, 含热离焦).
    红外材料折射率温度系数大, 且 T 图像对比度低, 需更大 radius 才能产生
    与 RGB 相当的视觉模糊效果.
    """
    name = 't_defocus_blur'
    modality = 't'
    _params = [
        (6,  0.1),    # sev1
        (10, 0.5),    # sev2
        (14, 0.5),    # sev3
        (18, 0.5),    # sev4
        (24, 0.5),    # sev5: 比 RGB sev5 (10) 大一倍多
    ]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        radius, alias_blur = self._params[severity - 1]
        kernel = disk(radius=radius, alias_blur=alias_blur)
        target_ch = img.shape[2] if img.ndim == 3 else 1
        x = to_float01(img)
        x = ensure_3ch(x)
        x = apply_blur_per_channel(x, kernel)
        x = restore_ch(x, target_ch)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        r, a = self._params[severity - 1]
        return {'radius': r, 'alias_blur': a}


@register_corruption
class TMissing(Corruption):
    """T 模态失效 (共通). 整模态置零 (传感器故障)."""
    name = 't_missing'
    modality = 't'

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        # severity 不分级, 5 级均为完全缺失
        return np.zeros_like(img)

    def get_params(self, severity):
        return {'mode': 'zero', 'note': 'all severities identical'}


@register_corruption
class TQuantizationNoise(Corruption):
    """T 量化噪声 (T 特有).
    模拟 14bit 热成像 → 8bit 量化损失 + 抖动.
    参考: Teutsch et al. CVPRW 2020, 热红外 14bit→8bit 色调映射量化误差.
    注: T 图像动态范围窄, 需更少 bits 才能产生可见量化效果.
    """
    name = 't_quantization'
    modality = 't'
    _params = [
        (6, 0.004),   # sev1: 6bit (64 级)   - 轻微
        (5, 0.006),   # sev2: 5bit (32 级)   - 可见
        (4, 0.008),   # sev3: 4bit (16 级)   - 明显
        (3, 0.010),   # sev4: 3bit (8 级)    - 严重
        (2, 0.006),   # sev5: 2bit (4 级)    - 很严重 (减小抖动避免过度)
    ]

    def __call__(self, img, severity=1):
        self._validate(img, severity)
        bits, dither = self._params[severity - 1]
        target_ch = img.shape[2] if img.ndim == 3 else 1
        x = to_float01(img)
        x = ensure_3ch(x)
        x = x + np.random.uniform(-dither, dither, size=x.shape)
        levels = 2 ** bits
        x = np.round(x * (levels - 1)) / (levels - 1)
        x = restore_ch(x, target_ch)
        return to_uint8(x).astype(np.uint8)

    def get_params(self, severity):
        b, d = self._params[severity - 1]
        return {'bits': b, 'dither_amp': d}


# /////////////// 退化分组 (按模态) ///////////////

RGB_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'motion_blur',
    'defocus_blur', 'fog', 'low_light', 'rgb_missing',
]

T_CORRUPTIONS = [
    't_gaussian_noise', 'stripe_noise', 't_motion_blur',
    't_defocus_blur', 't_missing', 't_quantization',
]

ALL_CORRUPTIONS = RGB_CORRUPTIONS + T_CORRUPTIONS
