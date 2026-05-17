import math
import random

import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
from .robustness_degradation import (
    _GLOBAL_DEG_PARAMS, _LOCAL_DEG_PARAMS,
    _SYNC_GLOBAL_TYPES, _RGB_ONLY_GLOBAL_TYPES, _T_ONLY_GLOBAL_TYPES,
    _BOTH_GLOBAL_TYPES,
    _SYNC_LOCAL_TYPES, _RGB_ONLY_LOCAL_TYPES, _T_ONLY_LOCAL_TYPES,
    _BOTH_LOCAL_TYPES,
    _make_motion_kernel, _apply_gaussian_noise, _apply_salt_pepper,
    _apply_stripe_noise, _apply_thermal_contrast,
    _generate_perlin_noise_2d, _sample_area_ratio, _apply_thermal_halo)


class BaseRGBTAugmentation(BaseTransform):

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            results['img'] = self.apply_augmentation(results['img'])
        return results


@TRANSFORMS.register_module()
class RGBTNoiseDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 noise_type='gaussian',
                 noise_level_range=(5, 30),
                 apply_to='random',
                 prob=0.5):
        assert noise_type in ['gaussian', 'salt_pepper']
        assert apply_to in ['rgb', 'thermal', 'random', 'both']
        self.noise_type = noise_type
        self.noise_level_range = noise_level_range
        self.apply_to = apply_to
        self.prob = prob

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()
        H, W, C = img.shape

        if self.apply_to == 'random':
            target = random.choice(['rgb', 'thermal'])
        elif self.apply_to == 'both':
            target = 'both'
        else:
            target = self.apply_to

        level = random.uniform(*self.noise_level_range)

        if self.noise_type == 'gaussian':
            noise = np.random.randn(H, W) * level
            if target in ('rgb', 'both'):
                for c in range(3):
                    img[:, :, c] = np.clip(
                        img[:, :, c].astype(np.float32) + noise, 0,
                        255).astype(np.uint8)
            if target in ('thermal', 'both'):
                for c in range(3, 6):
                    img[:, :, c] = np.clip(
                        img[:, :, c].astype(np.float32) + noise, 0,
                        255).astype(np.uint8)
        elif self.noise_type == 'salt_pepper':
            density = level / 100.0
            mask = np.random.rand(H, W)
            if target in ('rgb', 'both'):
                for c in range(3):
                    channel = img[:, :, c].astype(np.float32) / 255.0
                    channel[mask < density / 2] = 0.0
                    channel[mask > 1 - density / 2] = 1.0
                    img[:, :, c] = (np.clip(channel, 0, 1) * 255).astype(
                        np.uint8)
            if target in ('thermal', 'both'):
                for c in range(3, 6):
                    channel = img[:, :, c].astype(np.float32) / 255.0
                    channel[mask < density / 2] = 0.0
                    channel[mask > 1 - density / 2] = 1.0
                    img[:, :, c] = (np.clip(channel, 0, 1) * 255).astype(
                        np.uint8)

        return img


@TRANSFORMS.register_module()
class RGBTBlurDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 kernel_size_range=(3, 7),
                 blur_type='gaussian',
                 apply_to='random',
                 prob=0.5):
        assert blur_type in ['gaussian', 'motion']
        assert apply_to in ['rgb', 'thermal', 'random', 'both']
        self.kernel_size_range = kernel_size_range
        self.blur_type = blur_type
        self.apply_to = apply_to
        self.prob = prob

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()

        if self.apply_to == 'random':
            target = random.choice(['rgb', 'thermal'])
        elif self.apply_to == 'both':
            target = 'both'
        else:
            target = self.apply_to

        ksize = random.choice(
            range(self.kernel_size_range[0],
                  self.kernel_size_range[1] + 1, 2))

        if self.blur_type == 'gaussian':
            sigma = random.uniform(0.5, 3.0)
            if target in ('rgb', 'both'):
                for c in range(3):
                    img[:, :, c] = cv2.GaussianBlur(
                        img[:, :, c], (ksize, ksize), sigma)
            if target in ('thermal', 'both'):
                for c in range(3, 6):
                    img[:, :, c] = cv2.GaussianBlur(
                        img[:, :, c], (ksize, ksize), sigma)
        elif self.blur_type == 'motion':
            angle = random.uniform(0, 180)
            M = cv2.getRotationMatrix2D(
                (ksize // 2, ksize // 2), angle, 1)
            kernel = np.zeros((ksize, ksize))
            kernel[ksize // 2, :] = 1.0
            kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
            kernel = kernel / kernel.sum()
            if target in ('rgb', 'both'):
                for c in range(3):
                    img[:, :, c] = cv2.filter2D(
                        img[:, :, c], -1, kernel).astype(np.uint8)
            if target in ('thermal', 'both'):
                for c in range(3, 6):
                    img[:, :, c] = cv2.filter2D(
                        img[:, :, c], -1, kernel).astype(np.uint8)

        return img


@TRANSFORMS.register_module()
class RGBTMissingDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 missing_ratio_range=(0.1, 0.5),
                 apply_to='random',
                 prob=0.5,
                 fill_value=0):
        assert apply_to in ['rgb', 'thermal', 'random', 'both']
        self.missing_ratio_range = missing_ratio_range
        self.apply_to = apply_to
        self.prob = prob
        self.fill_value = fill_value

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()
        H, W, C = img.shape

        if self.apply_to == 'random':
            target = random.choice(['rgb', 'thermal'])
        elif self.apply_to == 'both':
            target = 'both'
        else:
            target = self.apply_to

        ratio = random.uniform(*self.missing_ratio_range)

        if target in ('rgb', 'both'):
            mask = np.random.rand(H, W) > ratio
            for c in range(3):
                img[:, :, c] = (img[:, :, c] * mask).astype(np.uint8)
        if target in ('thermal', 'both'):
            mask = np.random.rand(H, W) > ratio
            for c in range(3, 6):
                img[:, :, c] = (img[:, :, c] * mask).astype(np.uint8)

        return img


@TRANSFORMS.register_module()
class RGBTLowLightDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 brightness_range=(0.2, 0.6),
                 add_noise=True,
                 noise_sigma_range=(0.01, 0.05),
                 prob=0.3):
        self.brightness_range = brightness_range
        self.add_noise = add_noise
        self.noise_sigma_range = noise_sigma_range
        self.prob = prob

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()

        scale = random.uniform(*self.brightness_range)
        rgb = img[:, :, :3].astype(np.float32) / 255.0 * scale
        if self.add_noise:
            sigma = random.uniform(*self.noise_sigma_range)
            noise = np.random.randn(*rgb.shape) * sigma
            rgb = np.clip(rgb + noise, 0, 1)
        img[:, :, :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        return img


@TRANSFORMS.register_module()
class RGBTOverexposureDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 overexposure_range=(1.5, 3.0),
                 prob=0.2):
        self.overexposure_range = overexposure_range
        self.prob = prob

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()

        scale = random.uniform(*self.overexposure_range)
        rgb = img[:, :, :3].astype(np.float32) / 255.0 * scale
        img[:, :, :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        return img


@TRANSFORMS.register_module()
class RGBTPatchDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 patch_ratio_range=(0.05, 0.2),
                 num_patches_range=(1, 3),
                 apply_to='random',
                 prob=0.3,
                 fill_value=0):
        assert apply_to in ['rgb', 'thermal', 'random', 'both']
        self.patch_ratio_range = patch_ratio_range
        self.num_patches_range = num_patches_range
        self.apply_to = apply_to
        self.prob = prob
        self.fill_value = fill_value

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return img
        img = img.copy()
        H, W, C = img.shape

        if self.apply_to == 'random':
            target = random.choice(['rgb', 'thermal'])
        elif self.apply_to == 'both':
            target = 'both'
        else:
            target = self.apply_to

        num_patches = random.randint(*self.num_patches_range)
        for _ in range(num_patches):
            ratio = random.uniform(*self.patch_ratio_range)
            ph = int(H * ratio)
            pw = int(W * ratio)
            y = random.randint(0, H - ph)
            x = random.randint(0, W - pw)
            if target in ('rgb', 'both'):
                img[y:y + ph, x:x + pw, :3] = self.fill_value
            if target in ('thermal', 'both'):
                img[y:y + ph, x:x + pw, 3:] = self.fill_value

        return img


@TRANSFORMS.register_module()
class RGBTCombinedDegradation(BaseRGBTAugmentation):

    def __init__(self,
                 noise_prob=0.3,
                 blur_prob=0.3,
                 missing_prob=0.2,
                 lowlight_prob=0.2,
                 overexposure_prob=0.1,
                 patch_prob=0.1,
                 max_degradations=3):
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.missing_prob = missing_prob
        self.lowlight_prob = lowlight_prob
        self.overexposure_prob = overexposure_prob
        self.patch_prob = patch_prob
        self.max_degradations = max_degradations

    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        degradations = []
        if random.random() < self.noise_prob:
            degradations.append('noise')
        if random.random() < self.blur_prob:
            degradations.append('blur')
        if random.random() < self.missing_prob:
            degradations.append('missing')
        if random.random() < self.lowlight_prob:
            degradations.append('lowlight')
        if random.random() < self.overexposure_prob:
            degradations.append('overexposure')
        if random.random() < self.patch_prob:
            degradations.append('patch')

        if len(degradations) > self.max_degradations:
            degradations = random.sample(degradations,
                                         self.max_degradations)

        for deg in degradations:
            if deg == 'noise':
                level = random.uniform(5, 30)
                target = random.choice(['rgb', 'thermal'])
                H, W, _ = img.shape
                noise = np.random.randn(H, W) * level
                img = img.copy()
                if target == 'rgb':
                    for c in range(3):
                        img[:, :, c] = np.clip(
                            img[:, :, c].astype(np.float32) + noise,
                            0, 255).astype(np.uint8)
                else:
                    for c in range(3, 6):
                        img[:, :, c] = np.clip(
                            img[:, :, c].astype(np.float32) + noise,
                            0, 255).astype(np.uint8)
            elif deg == 'blur':
                ksize = random.choice([3, 5, 7])
                sigma = random.uniform(0.5, 3.0)
                target = random.choice(['rgb', 'thermal'])
                img = img.copy()
                if target == 'rgb':
                    for c in range(3):
                        img[:, :, c] = cv2.GaussianBlur(
                            img[:, :, c], (ksize, ksize), sigma)
                else:
                    for c in range(3, 6):
                        img[:, :, c] = cv2.GaussianBlur(
                            img[:, :, c], (ksize, ksize), sigma)
            elif deg == 'missing':
                ratio = random.uniform(0.1, 0.5)
                target = random.choice(['rgb', 'thermal'])
                img = img.copy()
                H, W, _ = img.shape
                mask = np.random.rand(H, W) > ratio
                if target == 'rgb':
                    for c in range(3):
                        img[:, :, c] = (img[:, :, c] * mask).astype(
                            np.uint8)
                else:
                    for c in range(3, 6):
                        img[:, :, c] = (img[:, :, c] * mask).astype(
                            np.uint8)
            elif deg == 'lowlight':
                scale = random.uniform(0.2, 0.6)
                img = img.copy()
                rgb = img[:, :, :3].astype(np.float32) / 255.0 * scale
                img[:, :, :3] = (np.clip(rgb, 0, 1) * 255).astype(
                    np.uint8)
            elif deg == 'overexposure':
                scale = random.uniform(1.5, 3.0)
                img = img.copy()
                rgb = img[:, :, :3].astype(np.float32) / 255.0 * scale
                img[:, :, :3] = (np.clip(rgb, 0, 1) * 255).astype(
                    np.uint8)
            elif deg == 'patch':
                ratio = random.uniform(0.05, 0.2)
                target = random.choice(['rgb', 'thermal'])
                img = img.copy()
                H, W, _ = img.shape
                ph = int(H * ratio)
                pw = int(W * ratio)
                y = random.randint(0, H - ph)
                x = random.randint(0, W - pw)
                if target == 'rgb':
                    img[y:y + ph, x:x + pw, :3] = 0
                else:
                    img[y:y + ph, x:x + pw, 3:] = 0

        return img


@TRANSFORMS.register_module()
class RGBTModalDegradation(BaseTransform):

    def __init__(self,
                 clean_prob=0.3,
                 missing_prob=0.3,
                 global_deg_prob=0.2,
                 local_deg_prob=0.2,
                 missing_fill_value=0,
                 dual_prob=0.1,
                 dual_modal_prob=None,
                 global_missing_prob=0.3):
        total = clean_prob + missing_prob + global_deg_prob + local_deg_prob
        assert abs(total - 1.0) < 1e-6, \
            f'Probabilities must sum to 1.0, got {total}'
        self.clean_prob = clean_prob
        self.missing_prob = missing_prob
        self.global_deg_prob = global_deg_prob
        self.local_deg_prob = local_deg_prob
        self.missing_fill_value = missing_fill_value
        if dual_modal_prob is not None:
            self.dual_prob = dual_modal_prob
        else:
            self.dual_prob = dual_prob
        self.global_missing_prob = global_missing_prob

    def _apply_single_missing(self, img, target=None):
        if target is None:
            target = random.choice(['rgb', 'thermal'])
        img = img.copy()
        if target == 'rgb':
            img[:, :, :3] = self.missing_fill_value
        else:
            img[:, :, 3:] = self.missing_fill_value
        rgb_deg = 'missing' if target == 'rgb' else 'clean'
        thermal_deg = 'missing' if target == 'thermal' else 'clean'
        return img, rgb_deg, thermal_deg

    def _apply_global_degradation(self, img):
        img = img.copy()
        deg_type = random.choice(list(_GLOBAL_DEG_PARAMS.keys()))
        intensity = random.choice(['low', 'medium', 'high'])
        params = _GLOBAL_DEG_PARAMS[deg_type][intensity]

        rgb_deg = 'clean'
        thermal_deg = 'clean'

        if deg_type in _SYNC_GLOBAL_TYPES:
            self._apply_global_sync(img, deg_type, params)
            rgb_deg = 'degraded'
            thermal_deg = 'degraded'
        elif deg_type in _T_ONLY_GLOBAL_TYPES:
            self._apply_global_thermal(img, deg_type, params)
            thermal_deg = 'degraded'
        elif deg_type in _RGB_ONLY_GLOBAL_TYPES:
            self._apply_global_rgb(img, deg_type, params)
            rgb_deg = 'degraded'
        else:
            if random.random() < self.dual_prob:
                self._apply_global_rgb(img, deg_type, params)
                self._apply_global_thermal(img, deg_type, params)
                rgb_deg = 'degraded'
                thermal_deg = 'degraded'
            else:
                modality = random.choice(['rgb', 'thermal'])
                if modality == 'rgb':
                    self._apply_global_rgb(img, deg_type, params)
                    rgb_deg = 'degraded'
                else:
                    self._apply_global_thermal(img, deg_type, params)
                    thermal_deg = 'degraded'

        if random.random() < self.global_missing_prob:
            missing_target = random.choice(['rgb', 'thermal'])
            if missing_target == 'rgb':
                img[:, :, :3] = self.missing_fill_value
                rgb_deg = 'missing'
            else:
                img[:, :, 3:] = self.missing_fill_value
                thermal_deg = 'missing'

        return img, rgb_deg, thermal_deg

    def _apply_global_sync(self, img, deg_type, params):
        if deg_type == 'motion_blur':
            ksize = params['kernel_size']
            angle = random.uniform(0, math.pi)
            kernel = _make_motion_kernel(ksize, angle)
            for c in range(6):
                img[:, :, c] = np.clip(
                    cv2.filter2D(img[:, :, c].astype(np.float32), -1, kernel),
                    0, 255).astype(np.uint8)

    def _apply_global_rgb(self, img, deg_type, params):
        img_norm = img / 255.0
        if deg_type == 'gaussian_noise':
            img_norm[:, :, :3] = _apply_gaussian_noise(
                img_norm[:, :, :3], params['sigma_rgb'])
        elif deg_type == 'salt_pepper':
            img_norm[:, :, :3] = _apply_salt_pepper(
                img_norm[:, :, :3], params['density_rgb'])
        img[:] = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)

    def _apply_global_thermal(self, img, deg_type, params):
        img_norm = img / 255.0
        if deg_type == 'gaussian_noise':
            img_norm[:, :, 3:] = _apply_gaussian_noise(
                img_norm[:, :, 3:], params['sigma_t'])
        elif deg_type == 'salt_pepper':
            img_norm[:, :, 3:] = _apply_salt_pepper(
                img_norm[:, :, 3:], params['density_t'])
        elif deg_type == 'stripe_noise':
            img_norm[:, :, 3:] = _apply_stripe_noise(
                img_norm[:, :, 3:], params['intensity'])
        elif deg_type == 'thermal_contrast':
            img_norm[:, :, 3:] = _apply_thermal_contrast(
                img_norm[:, :, 3:], params['alpha'])
        img[:] = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)

    def _apply_local_degradation(self, img):
        img = img.copy()
        deg_type = random.choice(list(_LOCAL_DEG_PARAMS.keys()))
        intensity = random.choice(['low', 'medium', 'high'])
        params = _LOCAL_DEG_PARAMS[deg_type][intensity]

        rgb_deg = 'clean'
        thermal_deg = 'clean'

        if deg_type in _SYNC_LOCAL_TYPES:
            self._apply_local_sync(img, deg_type, params)
            rgb_deg = 'degraded'
            thermal_deg = 'degraded'
        elif deg_type in _T_ONLY_LOCAL_TYPES:
            self._apply_local_thermal(img, deg_type, params)
            thermal_deg = 'degraded'
        elif deg_type in _RGB_ONLY_LOCAL_TYPES:
            self._apply_local_rgb(img, deg_type, params)
            rgb_deg = 'degraded'
        else:
            if random.random() < self.dual_prob:
                self._apply_local_rgb(img, deg_type, params)
                self._apply_local_thermal(img, deg_type, params)
                rgb_deg = 'degraded'
                thermal_deg = 'degraded'
            else:
                modality = random.choice(['rgb', 'thermal'])
                if modality == 'rgb':
                    self._apply_local_rgb(img, deg_type, params)
                    rgb_deg = 'degraded'
                else:
                    self._apply_local_thermal(img, deg_type, params)
                    thermal_deg = 'degraded'
        return img, rgb_deg, thermal_deg

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

    def _apply_local_rgb(self, img, deg_type, params):
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
        else:
            self._apply_local_to_channels(img, slice(None, 3), deg_type, params)

    def _apply_local_thermal(self, img, deg_type, params):
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

    def _apply_local_sync(self, img, deg_type, params):
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

    def transform(self, results: dict) -> dict:
        if 'img' not in results:
            return results

        r = random.random()
        if r < self.clean_prob:
            results['rgb_degradation'] = 'clean'
            results['thermal_degradation'] = 'clean'
        elif r < self.clean_prob + self.missing_prob:
            img, rgb_deg, thermal_deg = self._apply_single_missing(
                results['img'])
            results['img'] = img
            results['rgb_degradation'] = rgb_deg
            results['thermal_degradation'] = thermal_deg
        elif r < self.clean_prob + self.missing_prob + self.global_deg_prob:
            img, rgb_deg, thermal_deg = self._apply_global_degradation(
                results['img'])
            results['img'] = img
            results['rgb_degradation'] = rgb_deg
            results['thermal_degradation'] = thermal_deg
        else:
            img, rgb_deg, thermal_deg = self._apply_local_degradation(
                results['img'])
            results['img'] = img
            results['rgb_degradation'] = rgb_deg
            results['thermal_degradation'] = thermal_deg

        return results
