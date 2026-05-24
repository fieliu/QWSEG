import math
import random

import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS


class BaseRGBTDegradation(BaseTransform):

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            results['img'] = self.apply_degradation(results['img'])
        return results


@TRANSFORMS.register_module()
class CleanDegradation(BaseRGBTDegradation):
    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        return img


@TRANSFORMS.register_module()
class RGBMissingDegradation(BaseRGBTDegradation):
    def __init__(self, fill_value: int = 0):
        self.fill_value = fill_value

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        img[:, :, :3] = self.fill_value
        return img


@TRANSFORMS.register_module()
class ThermalMissingDegradation(BaseRGBTDegradation):
    def __init__(self, fill_value: int = 0):
        self.fill_value = fill_value

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        img[:, :, 3:] = self.fill_value
        return img


@TRANSFORMS.register_module()
class LocalRGBMissingDegradation(BaseRGBTDegradation):
    def __init__(self, area_ratio: float = 0.5, seed: int = None):
        self.area_ratio = area_ratio
        self.seed = seed

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        H, W = img.shape[:2]
        y1, x1, rh, rw = self._random_rect(H, W, self.area_ratio)
        img[y1:y1 + rh, x1:x1 + rw, :3] = 0
        return img

    def _random_rect(self, H, W, area_ratio):
        area = H * W * area_ratio
        aspect = random.uniform(0.5, 2.0)
        rw = max(min(int(math.sqrt(area * aspect)), W), 1)
        rh = max(min(int(math.sqrt(area / aspect)), H), 1)
        y1 = random.randint(0, max(H - rh, 0))
        x1 = random.randint(0, max(W - rw, 0))
        return y1, x1, rh, rw


@TRANSFORMS.register_module()
class LocalThermalMissingDegradation(BaseRGBTDegradation):
    def __init__(self, area_ratio: float = 0.5, seed: int = None):
        self.area_ratio = area_ratio
        self.seed = seed

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        H, W = img.shape[:2]
        y1, x1, rh, rw = self._random_rect(H, W, self.area_ratio)
        img[y1:y1 + rh, x1:x1 + rw, 3:] = 0
        return img

    def _random_rect(self, H, W, area_ratio):
        area = H * W * area_ratio
        aspect = random.uniform(0.5, 2.0)
        rw = max(min(int(math.sqrt(area * aspect)), W), 1)
        rh = max(min(int(math.sqrt(area / aspect)), H), 1)
        y1 = random.randint(0, max(H - rh, 0))
        x1 = random.randint(0, max(W - rw, 0))
        return y1, x1, rh, rw


# ---------------------------------------------------------------------------
# Legacy helpers (kept for rgbt_augmentation.py compatibility)
# ---------------------------------------------------------------------------

def _make_motion_kernel(kernel_size, angle):
    kernel = np.zeros((kernel_size, kernel_size))
    mid = kernel_size // 2
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for i in range(kernel_size):
        offset = i - mid
        x = int(round(mid + offset * cos_a))
        y = int(round(mid + offset * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    kernel /= (kernel.sum() + 1e-8)
    return kernel


def _apply_gaussian_noise(img_channels, sigma):
    noise = np.random.randn(*img_channels.shape) * sigma
    return np.clip(img_channels + noise, 0, 1)


def _apply_salt_pepper(img_channels, density):
    H, W = img_channels.shape[:2]
    mask = np.random.rand(H, W)
    result = img_channels.copy()
    result[mask < density / 2] = 0.0
    result[mask > 1 - density / 2] = 1.0
    return result


def _apply_stripe_noise(img_channels, intensity):
    H, W = img_channels.shape[:2]
    direction = random.choice(['horizontal', 'vertical'])
    if direction == 'horizontal':
        stripe = np.random.randn(1, W, img_channels.shape[2]) * intensity
    else:
        stripe = np.random.randn(H, 1, img_channels.shape[2]) * intensity
    return np.clip(img_channels + stripe, 0, 1)


def _apply_thermal_contrast(thermal, alpha):
    mean_val = thermal.mean(axis=(0, 1), keepdims=True)
    min_val = thermal.min(axis=(0, 1), keepdims=True)
    max_val = thermal.max(axis=(0, 1), keepdims=True)
    center = mean_val
    half_range = (max_val - min_val) * alpha / 2
    new_min = np.clip(center - half_range, 0, 1)
    new_max = np.clip(center + half_range, 0, 1)
    normalized = (thermal - min_val) / (max_val - min_val + 1e-8)
    return np.clip(new_min + normalized * (new_max - new_min), 0, 1)


def _generate_perlin_noise_2d(shape, res=(4, 4)):
    h, w = shape
    rh, rw = res
    grid_h = np.arange(rh + 1).reshape(-1, 1, 1) + 0.5
    grid_w = np.arange(rw + 1).reshape(1, -1, 1) + 0.5
    angles = np.random.rand(rh + 1, rw + 1) * 2 * np.pi
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    y_coords = np.linspace(0, rh, h)
    x_coords = np.linspace(0, rw, w)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1
    y0 = np.clip(y0, 0, rh)
    y1 = np.clip(y1, 0, rh)
    x0 = np.clip(x0, 0, rw)
    x1 = np.clip(x1, 0, rw)
    fy = yy - y0
    fx = xx - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)

    def _dot_grad(gy, gx, dy, dx):
        g = gradients[gy, gx]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = _dot_grad(y0, x0, fy, fx)
    n10 = _dot_grad(y1, x0, fy - 1, fx)
    n01 = _dot_grad(y0, x1, fy, fx - 1)
    n11 = _dot_grad(y1, x1, fy - 1, fx - 1)
    nx0 = n00 * (1 - fy) + n10 * fy
    nx1 = n01 * (1 - fy) + n11 * fy
    noise = nx0 * (1 - fx) + nx1 * fx
    return noise


def _sample_area_ratio(mean=2.0, std=2.0, low=0.5, high=5.0):
    while True:
        val = np.random.normal(mean, std)
        if low <= val <= high:
            return val / 100.0


def _apply_thermal_halo(thermal, centers, sigma, peak_gain):
    H, W = thermal.shape[:2]
    halo = np.zeros((H, W), dtype=np.float32)
    Y, X = np.ogrid[:H, :W]
    for center in centers:
        cy, cx = center
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        halo += peak_gain * np.exp(-dist_sq / (2 * sigma ** 2))
    thermal = thermal.astype(np.float32) + halo[..., np.newaxis]
    return np.clip(thermal, 0, 255).astype(np.uint8)


_GLOBAL_DEG_PARAMS = {
    'motion_blur': {
        'low': {'kernel_size': 7},
        'medium': {'kernel_size': 20},
        'high': {'kernel_size': 38},
    },
    'gaussian_noise': {
        'low': {'sigma_rgb': 25 / 255.0, 'sigma_t': 30 / 255.0},
        'medium': {'sigma_rgb': 50 / 255.0, 'sigma_t': 55 / 255.0},
        'high': {'sigma_rgb': 80 / 255.0, 'sigma_t': 85 / 255.0},
    },
    'salt_pepper': {
        'low': {'density_rgb': 0.02, 'density_t': 0.02},
        'medium': {'density_rgb': 0.06, 'density_t': 0.06},
        'high': {'density_rgb': 0.12, 'density_t': 0.12},
    },
    'stripe_noise': {
        'low': {'intensity': 0.03},
        'medium': {'intensity': 0.08},
        'high': {'intensity': 0.15},
    },
    'thermal_contrast': {
        'low': {'alpha': 0.7},
        'medium': {'alpha': 0.5},
        'high': {'alpha': 0.3},
    },
}

_SYNC_GLOBAL_TYPES = {'motion_blur'}
_RGB_ONLY_GLOBAL_TYPES = set()
_T_ONLY_GLOBAL_TYPES = {'stripe_noise', 'thermal_contrast'}
_BOTH_GLOBAL_TYPES = {'gaussian_noise', 'salt_pepper'}

_LOCAL_DEG_PARAMS = {
    'local_overexposure': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05, 'gain': 60},
        'medium': {'num_rects': (2, 5), 'total_area': 0.20, 'gain': 120},
        'high': {'num_rects': (5, 7), 'total_area': 0.40, 'gain': 200},
    },
    'local_lowlight': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05, 'gamma': 1.5},
        'medium': {'num_rects': (2, 5), 'total_area': 0.20, 'gamma': 2.5},
        'high': {'num_rects': (5, 7), 'total_area': 0.40, 'gamma': 4.0},
    },
    'water_stain': {
        'low': {'num_drops': (1, 3), 'total_area': 0.10,
                'blur_k': 3, 'brightness_rgb': 0.85, 'brightness_t': 0.5,
                'is_stain': False},
        'medium': {'num_drops': (3, 6), 'total_area': 0.30,
                   'blur_k': 5, 'brightness_rgb': 0.7, 'brightness_t': 0.3,
                   'is_stain': False},
        'high': {'num_drops': (5, 10), 'total_area': 0.40,
                 'blur_k': 7, 'brightness_rgb': 0.55, 'brightness_t': 0.15,
                 'is_stain': False},
    },
    'stain': {
        'low': {'num_drops': (1, 3), 'total_area': 0.05,
                'opacity': (0.3, 0.5), 'rgb_value': [40, 40, 40],
                't_ratio': 0.7, 'is_stain': True},
        'medium': {'num_drops': (3, 6), 'total_area': 0.10,
                   'opacity': (0.5, 0.8), 'rgb_value': [20, 20, 20],
                   't_ratio': 0.4, 'is_stain': True},
        'high': {'num_drops': (5, 10), 'total_area': 0.20,
                 'opacity': (0.8, 1.0), 'rgb_value': [0, 0, 0],
                 't_ratio': 0.1, 'is_stain': True},
    },
    'local_bad_block': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05},
        'medium': {'num_rects': (3, 5), 'total_area': 0.20},
        'high': {'num_rects': (5, 7), 'total_area': 0.40},
    },
    'local_gaussian_noise': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05,
                'sigma_rgb': 25 / 255.0, 'sigma_t': 30 / 255.0},
        'medium': {'num_rects': (2, 5), 'total_area': 0.20,
                   'sigma_rgb': 50 / 255.0, 'sigma_t': 55 / 255.0},
        'high': {'num_rects': (5, 7), 'total_area': 0.40,
                 'sigma_rgb': 80 / 255.0, 'sigma_t': 85 / 255.0},
    },
    'thermal_halo': {
        'low': {'num_centers': (1, 2), 'sigma_range': (4, 6),
                'peak_gain_range': (30, 50)},
        'medium': {'num_centers': (2, 4), 'sigma_range': (8, 12),
                   'peak_gain_range': (60, 90)},
        'high': {'num_centers': (3, 6), 'sigma_range': (15, 20),
                 'peak_gain_range': (100, 140)},
    },
}

_SYNC_LOCAL_TYPES = {'water_stain'}
_RGB_ONLY_LOCAL_TYPES = {'local_overexposure', 'local_lowlight'}
_T_ONLY_LOCAL_TYPES = {'thermal_halo'}
_BOTH_LOCAL_TYPES = {'stain', 'local_bad_block', 'local_gaussian_noise'}
