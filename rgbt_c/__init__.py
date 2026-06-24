"""RGBT-C 标准退化工具库.

设计依据: docs/RGBT-C_Benchmark.md v1.0

使用示例:
    from rgbt_c import get_corruption, list_corruptions, LocalCorruption

    # 全局退化 (测试期预生成 / 训练期在线退化)
    corruption = get_corruption('gaussian_noise')
    img_corrupted = corruption(img, severity=3)

    # 局部退化 (训练期在线退化)
    from rgbt_c import LocalCorruption
    local_corr = LocalCorruption(corruption, area_ratio_range=(0.1, 0.4))
    img_local = local_corr(img, severity=3)
"""
from .corruptions import (
    Corruption,
    CORRUPTION_REGISTRY,
    register_corruption,
    get_corruption,
    list_corruptions,
    RGB_CORRUPTIONS,
    T_CORRUPTIONS,
    ALL_CORRUPTIONS,
    # RGB 模态退化类
    RGBGaussianNoise,
    RGBShotNoise,
    RGBMotionBlur,
    RGBDefocusBlur,
    RGBFog,
    RGBLowLight,
    # T 模态退化类
    TGaussianNoise,
    TStripeNoise,
    TMotionBlur,
    TDefocusBlur,
    TMissing,
    TQuantizationNoise,
)
from .local import LocalCorruption

__version__ = '1.0.0'

__all__ = [
    'Corruption', 'CORRUPTION_REGISTRY', 'register_corruption',
    'get_corruption', 'list_corruptions',
    'RGB_CORRUPTIONS', 'T_CORRUPTIONS', 'ALL_CORRUPTIONS',
    'LocalCorruption',
    # RGB
    'RGBGaussianNoise', 'RGBShotNoise', 'RGBMotionBlur',
    'RGBDefocusBlur', 'RGBFog', 'RGBLowLight',
    # T
    'TGaussianNoise', 'TStripeNoise', 'TMotionBlur',
    'TDefocusBlur', 'TMissing', 'TQuantizationNoise',
]
