"""Project-defined RGB-T corruption library; see docs/RGBT-C_Benchmark.md."""
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

__version__ = '2.0.0'

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

from .protocol import apply_pair, operation, recipe, stable_seed
