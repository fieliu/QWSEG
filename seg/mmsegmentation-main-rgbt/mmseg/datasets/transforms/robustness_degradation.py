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


def _apply_motion_blur(channel, kernel_size, angle):
    kernel = _make_motion_kernel(kernel_size, angle)
    return cv2.filter2D(channel, -1, kernel)


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


@TRANSFORMS.register_module()
class GlobalDegradation(BaseRGBTDegradation):

    def __init__(self, deg_type: str = 'random', intensity: str = 'medium',
                 dual_prob: float = 0.1, seed: int = None):
        assert intensity in ['low', 'medium', 'high']
        assert deg_type in ['random', 'motion_blur', 'gaussian_noise',
                            'salt_pepper', 'stripe_noise', 'thermal_contrast']
        self.deg_type = deg_type
        self.intensity = intensity
        self.dual_prob = dual_prob
        self.seed = seed

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        deg_type = self.deg_type
        if deg_type == 'random':
            deg_type = random.choice(list(_GLOBAL_DEG_PARAMS.keys()))

        params = _GLOBAL_DEG_PARAMS[deg_type][self.intensity]

        if deg_type in _SYNC_GLOBAL_TYPES:
            self._apply_sync(img, deg_type, params)
        elif deg_type in _T_ONLY_GLOBAL_TYPES:
            self._apply_thermal_only(img, deg_type, params)
        elif deg_type in _RGB_ONLY_GLOBAL_TYPES:
            self._apply_rgb_only(img, deg_type, params)
        else:
            if random.random() < self.dual_prob:
                self._apply_dual(img, deg_type, params)
            else:
                modality = random.choice(['rgb', 'thermal'])
                if modality == 'rgb':
                    self._apply_rgb_only(img, deg_type, params)
                else:
                    self._apply_thermal_only(img, deg_type, params)

        return img

    def _apply_sync(self, img, deg_type, params):
        if deg_type == 'motion_blur':
            ksize = params['kernel_size']
            angle = random.uniform(0, math.pi)
            kernel = _make_motion_kernel(ksize, angle)
            for c in range(6):
                img[:, :, c] = np.clip(
                    cv2.filter2D(img[:, :, c].astype(np.float32), -1, kernel),
                    0, 255).astype(np.uint8)

    def _apply_rgb_only(self, img, deg_type, params):
        img_norm = img / 255.0
        if deg_type == 'gaussian_noise':
            sigma = params['sigma_rgb']
            img_norm[:, :, :3] = _apply_gaussian_noise(
                img_norm[:, :, :3], sigma)
        elif deg_type == 'salt_pepper':
            density = params['density_rgb']
            img_norm[:, :, :3] = _apply_salt_pepper(
                img_norm[:, :, :3], density)
        img[:] = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)

    def _apply_thermal_only(self, img, deg_type, params):
        img_norm = img / 255.0
        if deg_type == 'gaussian_noise':
            sigma = params['sigma_t']
            img_norm[:, :, 3:] = _apply_gaussian_noise(
                img_norm[:, :, 3:], sigma)
        elif deg_type == 'salt_pepper':
            density = params['density_t']
            img_norm[:, :, 3:] = _apply_salt_pepper(
                img_norm[:, :, 3:], density)
        elif deg_type == 'stripe_noise':
            img_norm[:, :, 3:] = _apply_stripe_noise(
                img_norm[:, :, 3:], params['intensity'])
        elif deg_type == 'thermal_contrast':
            img_norm[:, :, 3:] = _apply_thermal_contrast(
                img_norm[:, :, 3:], params['alpha'])
        img[:] = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)

    def _apply_dual(self, img, deg_type, params):
        self._apply_rgb_only(img, deg_type, params)
        self._apply_thermal_only(img, deg_type, params)


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


_LOCAL_DEG_PARAMS = {
    'local_overexposure': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05,
                'gain': 60},
        'medium': {'num_rects': (2, 5), 'total_area': 0.20,
                   'gain': 120},
        'high': {'num_rects': (5, 7), 'total_area': 0.40,
                 'gain': 200},
    },
    'local_lowlight': {
        'low': {'num_rects': (1, 2), 'total_area': 0.05,
                'gamma': 1.5},
        'medium': {'num_rects': (2, 5), 'total_area': 0.20,
                   'gamma': 2.5},
        'high': {'num_rects': (5, 7), 'total_area': 0.40,
                 'gamma': 4.0},
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


@TRANSFORMS.register_module()
class LocalDegradation(BaseRGBTDegradation):

    def __init__(self, deg_type: str = 'random', intensity: str = 'medium',
                 dual_prob: float = 0.1, seed: int = None):
        assert intensity in ['low', 'medium', 'high']
        assert deg_type in ['random', 'local_overexposure', 'local_lowlight',
                            'water_stain', 'stain', 'local_bad_block',
                            'local_gaussian_noise', 'thermal_halo']
        self.deg_type = deg_type
        self.intensity = intensity
        self.dual_prob = dual_prob
        self.seed = seed

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        deg_type = self.deg_type
        if deg_type == 'random':
            deg_type = random.choice(list(_LOCAL_DEG_PARAMS.keys()))

        params = _LOCAL_DEG_PARAMS[deg_type][self.intensity]

        if deg_type in _SYNC_LOCAL_TYPES:
            self._apply_sync(img, deg_type, params)
        elif deg_type in _T_ONLY_LOCAL_TYPES:
            self._apply_thermal_only(img, deg_type, params)
        elif deg_type in _RGB_ONLY_LOCAL_TYPES:
            self._apply_rgb_only(img, deg_type, params)
        else:
            if random.random() < self.dual_prob:
                self._apply_dual(img, deg_type, params)
            else:
                modality = random.choice(['rgb', 'thermal'])
                if modality == 'rgb':
                    self._apply_rgb_only(img, deg_type, params)
                else:
                    self._apply_thermal_only(img, deg_type, params)

        return img

    def _sample_rects_with_area(self, H, W, num_rects_range, total_area):
        num = random.randint(*num_rects_range)
        rects = []
        remaining_area = total_area
        for i in range(num):
            if i == num - 1:
                area_ratio = remaining_area
            else:
                max_next = remaining_area / (num - i)
                area_ratio = min(_sample_area_ratio(), max_next)
                remaining_area -= area_ratio
            area_ratio = max(area_ratio, 0.005)
            y, x, rh, rw = self._random_rect(H, W, area_ratio)
            rects.append((y, x, rh, rw, area_ratio))
        return rects

    def _random_rect(self, H, W, area_ratio):
        area = H * W * area_ratio
        aspect = random.uniform(0.5, 2.0)
        rw = int(math.sqrt(area * aspect))
        rh = int(math.sqrt(area / aspect))
        rw = max(rw, 1)
        rh = max(rh, 1)
        rw = min(rw, W)
        rh = min(rh, H)
        y = random.randint(0, max(H - rh, 0))
        x = random.randint(0, max(W - rw, 0))
        return y, x, rh, rw

    def _apply_rgb_only(self, img, deg_type, params):
        H, W = img.shape[:2]
        if deg_type == 'local_overexposure':
            rects = self._sample_rects_with_area(
                H, W, params['num_rects'], params['total_area'])
            gain = params['gain']
            for y, x, rh, rw, _ in rects:
                region = img[y:y + rh, x:x + rw, :3].astype(np.float32)
                region = np.clip(region + gain, 0, 255)
                img[y:y + rh, x:x + rw, :3] = region.astype(np.uint8)
        elif deg_type == 'local_lowlight':
            rects = self._sample_rects_with_area(
                H, W, params['num_rects'], params['total_area'])
            gamma = params['gamma']
            for y, x, rh, rw, _ in rects:
                region = img[y:y + rh, x:x + rw, :3].astype(np.float32) / 255.0
                region = np.clip(np.power(region + 1e-8, gamma), 0, 1)
                img[y:y + rh, x:x + rw, :3] = (region * 255).astype(np.uint8)
        elif deg_type in ('water_stain', 'stain', 'local_bad_block',
                          'local_gaussian_noise'):
            self._apply_local_to_channels(img, slice(None, 3), deg_type, params)

    def _apply_thermal_only(self, img, deg_type, params):
        if deg_type == 'thermal_halo':
            self._apply_thermal_halo(img, params)
        else:
            self._apply_local_to_channels(img, slice(3, None), deg_type, params)

    def _apply_thermal_halo(self, img, params):
        H, W = img.shape[:2]
        num_centers = random.randint(*params['num_centers'])
        sigma = random.uniform(*params['sigma_range'])
        peak_gain = random.uniform(*params['peak_gain_range'])

        thermal = img[:, :, 3:]
        thermal_float = thermal.astype(np.float32)
        threshold = 200.0
        high_temp_mask = thermal_float.max(axis=-1) > threshold

        centers = []
        if high_temp_mask.any():
            ys, xs = np.where(high_temp_mask)
            indices = np.random.choice(len(ys), min(num_centers, len(ys)),
                                       replace=False)
            centers = [(ys[i], xs[i]) for i in indices]

        while len(centers) < num_centers:
            cy = random.randint(0, H - 1)
            cx = random.randint(0, W - 1)
            centers.append((cy, cx))

        img[:, :, 3:] = _apply_thermal_halo(img[:, :, 3:], centers, sigma,
                                             peak_gain)

    def _apply_local_to_channels(self, img, ch_slice, deg_type, params):
        H, W = img.shape[:2]
        if deg_type in ('local_overexposure', 'local_lowlight', 'thermal_halo'):
            return

        if deg_type == 'local_bad_block':
            rects = self._sample_rects_with_area(
                H, W, params['num_rects'], params['total_area'])
            for y, x, rh, rw, _ in rects:
                img[y:y + rh, x:x + rw, ch_slice] = 0
            return

        if deg_type == 'local_gaussian_noise':
            rects = self._sample_rects_with_area(
                H, W, params['num_rects'], params['total_area'])
            is_rgb = (ch_slice == slice(None, 3))
            sigma = params['sigma_rgb'] if is_rgb else params['sigma_t']
            for y, x, rh, rw, _ in rects:
                region = img[y:y + rh, x:x + rw, ch_slice].astype(
                    np.float32) / 255.0
                noise = np.random.randn(*region.shape) * sigma
                region = np.clip(region + noise, 0, 1)
                img[y:y + rh, x:x + rw, ch_slice] = (
                    region * 255).astype(np.uint8)
            return

        if deg_type == 'water_stain':
            self._apply_water_stain(img, ch_slice, params)
        elif deg_type == 'stain':
            self._apply_stain(img, ch_slice, params)

    def _generate_water_drop_mask(self, H, W, area_ratio):
        res_h = max(3, int(math.sqrt(area_ratio) * 8))
        res_w = max(3, int(math.sqrt(area_ratio) * 8))
        noise = _generate_perlin_noise_2d((H, W), res=(res_h, res_w))
        noise2 = _generate_perlin_noise_2d((H, W), res=(res_h + 2, res_w + 2))
        combined = 0.6 * noise + 0.4 * noise2
        threshold = np.percentile(combined, 100 * (1 - area_ratio * 2))
        mask = (combined > threshold).astype(np.float32)
        if mask.sum() < 1:
            return None
        dist = cv2.distanceTransform((mask * 255).astype(np.uint8),
                                     cv2.DIST_L2, 5)
        dist_norm = dist / (dist.max() + 1e-8)
        edge_mask = (dist_norm < 0.3).astype(np.float32)
        smooth_mask = mask * (1 - edge_mask * 0.5)
        return smooth_mask

    def _apply_water_stain(self, img, ch_slice, params):
        H, W = img.shape[:2]
        num_drops = random.randint(*params['num_drops'])
        blur_k = params['blur_k']
        brightness_rgb = params['brightness_rgb']
        brightness_t = params['brightness_t']
        total_area = params['total_area']

        remaining_area = total_area
        for i in range(num_drops):
            if i == num_drops - 1:
                area_ratio = remaining_area
            else:
                max_next = remaining_area / (num_drops - i)
                area_ratio = min(_sample_area_ratio(), max_next)
                remaining_area -= area_ratio
            area_ratio = max(area_ratio, 0.005)

            mask = self._generate_water_drop_mask(H, W, area_ratio)
            if mask is None:
                continue

            mask_uint8 = (mask * 255).astype(np.uint8)
            kernel_size = blur_k * 2 + 1
            mask_uint8 = cv2.GaussianBlur(
                mask_uint8, (kernel_size, kernel_size), 0)
            mask_blended = (mask_uint8 / 255.0)[..., np.newaxis]

            is_rgb = (ch_slice == slice(None, 3))
            if is_rgb:
                channels = img[:, :, :3].astype(np.float32)
                blurred = np.zeros_like(channels)
                for c in range(3):
                    blurred[:, :, c] = cv2.GaussianBlur(
                        channels[:, :, c], (kernel_size, kernel_size), 0)
                degraded = blurred * brightness_rgb
                channels = channels * (1 - mask_blended) + degraded * mask_blended
                img[:, :, :3] = np.clip(channels, 0, 255).astype(np.uint8)
            else:
                channels = img[:, :, 3:].astype(np.float32)
                degraded = channels * brightness_t
                channels = channels * (1 - mask_blended) + degraded * mask_blended
                img[:, :, 3:] = np.clip(channels, 0, 255).astype(np.uint8)

    def _apply_stain(self, img, ch_slice, params):
        H, W = img.shape[:2]
        num_drops = random.randint(*params['num_drops'])
        opacity_range = params['opacity']
        rgb_value = np.array(params['rgb_value'], dtype=np.float32)
        t_ratio = params['t_ratio']
        total_area = params['total_area']

        remaining_area = total_area
        for i in range(num_drops):
            if i == num_drops - 1:
                area_ratio = remaining_area
            else:
                max_next = remaining_area / (num_drops - i)
                area_ratio = min(_sample_area_ratio(), max_next)
                remaining_area -= area_ratio
            area_ratio = max(area_ratio, 0.005)

            mask = self._generate_water_drop_mask(H, W, area_ratio)
            if mask is None:
                continue

            alpha = random.uniform(*opacity_range)
            mask_alpha = mask * alpha

            is_rgb = (ch_slice == slice(None, 3))
            if is_rgb:
                channels = img[:, :, :3].astype(np.float32)
                for c in range(3):
                    channels[:, :, c] = (
                        channels[:, :, c] * (1 - mask_alpha) +
                        rgb_value[c] * mask_alpha)
                img[:, :, :3] = np.clip(channels, 0, 255).astype(np.uint8)
            else:
                channels = img[:, :, 3:].astype(np.float32)
                t_mean = channels.mean()
                t_fill = t_mean * t_ratio
                mask_3c = np.stack([mask_alpha] * 3, axis=-1)
                channels = channels * (1 - mask_3c) + t_fill * mask_3c
                img[:, :, 3:] = np.clip(channels, 0, 255).astype(np.uint8)

    def _apply_sync(self, img, deg_type, params):
        if deg_type == 'water_stain':
            H, W = img.shape[:2]
            num_drops = random.randint(*params['num_drops'])
            blur_k = params['blur_k']
            brightness_rgb = params['brightness_rgb']
            brightness_t = params['brightness_t']
            total_area = params['total_area']

            remaining_area = total_area
            for i in range(num_drops):
                if i == num_drops - 1:
                    area_ratio = remaining_area
                else:
                    max_next = remaining_area / (num_drops - i)
                    area_ratio = min(_sample_area_ratio(), max_next)
                    remaining_area -= area_ratio
                area_ratio = max(area_ratio, 0.005)

                mask = self._generate_water_drop_mask(H, W, area_ratio)
                if mask is None:
                    continue

                mask_uint8 = (mask * 255).astype(np.uint8)
                kernel_size = blur_k * 2 + 1
                mask_uint8 = cv2.GaussianBlur(
                    mask_uint8, (kernel_size, kernel_size), 0)
                mask_blended = (mask_uint8 / 255.0)[..., np.newaxis]

                rgb_ch = img[:, :, :3].astype(np.float32)
                blurred_rgb = np.zeros_like(rgb_ch)
                for c in range(3):
                    blurred_rgb[:, :, c] = cv2.GaussianBlur(
                        rgb_ch[:, :, c], (kernel_size, kernel_size), 0)
                degraded_rgb = blurred_rgb * brightness_rgb
                rgb_ch = rgb_ch * (1 - mask_blended) + degraded_rgb * mask_blended
                img[:, :, :3] = np.clip(rgb_ch, 0, 255).astype(np.uint8)

                t_ch = img[:, :, 3:].astype(np.float32)
                degraded_t = t_ch * brightness_t
                t_ch = t_ch * (1 - mask_blended) + degraded_t * mask_blended
                img[:, :, 3:] = np.clip(t_ch, 0, 255).astype(np.uint8)

    def _apply_dual(self, img, deg_type, params):
        self._apply_rgb_only(img, deg_type, params)
        self._apply_thermal_only(img, deg_type, params)


@TRANSFORMS.register_module()
class MultiDegradation(BaseRGBTDegradation):
    def __init__(self, degradations: list):
        self.degradation_instances = []
        for deg_cfg in degradations:
            deg_cfg = deg_cfg.copy()
            deg_type = deg_cfg.pop('type')
            cls = TRANSFORMS.get(deg_type)
            if cls is None:
                raise ValueError(
                    f'Degradation type "{deg_type}" not found in TRANSFORMS')
            self.degradation_instances.append(cls(**deg_cfg))

    def apply_degradation(self, img: np.ndarray) -> np.ndarray:
        for deg in self.degradation_instances:
            img = deg.apply_degradation(img)
        return img
