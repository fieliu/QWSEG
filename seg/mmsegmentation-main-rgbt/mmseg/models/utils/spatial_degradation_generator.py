import random

import torch
import torch.nn.functional as F

from mmseg.datasets.transforms.quality_degradation import (
    _QUALITY_LEVEL_CONFIGS, _QUALITY_RGB_DEG_TYPES, _QUALITY_T_DEG_TYPES,
    apply_quality_degradation_rgb, apply_quality_degradation_t)


class SpatialDegradationGenerator:

    def __init__(self,
                 num_regions_range=(5, 10),
                 region_size_range=(32, 80),
                 num_levels=5,
                 num_stages=4):
        self.num_regions_range = num_regions_range
        self.region_size_range = region_size_range
        self.num_levels = num_levels
        self.num_stages = num_stages

        self._img_binary_mask = None
        self._img_mask_full = None
        self._deg_type = None
        self._modality = None

    def _generate_region_mask_image(self, H, W, num_regions, device):
        mask = torch.zeros(1, 1, H, W, device=device)
        for _ in range(num_regions):
            size = random.randint(*self.region_size_range)
            y0 = random.randint(0, max(0, H - size))
            x0 = random.randint(0, max(0, W - size))
            mask[:, :, y0:y0 + size, x0:x0 + size] = 1.0
        return mask

    def prepare_degradation(self, img_tensor, modality='rgb'):
        B, C, H, W = img_tensor.shape

        num_regions = random.randint(*self.num_regions_range)
        img_mask = self._generate_region_mask_image(
            H, W, num_regions, img_tensor.device)
        img_mask = img_mask.expand(B, -1, -1, -1).clone()

        self._img_binary_mask = img_mask
        self._img_mask_full = img_mask

        if modality == 'rgb':
            self._deg_type = random.choice(_QUALITY_RGB_DEG_TYPES)
        else:
            self._deg_type = random.choice(_QUALITY_T_DEG_TYPES)
        self._modality = modality

    def apply_level(self, img_tensor, level, mean=None, std=None):
        if level == 0:
            if mean is not None and std is not None:
                return (img_tensor * 255.0 - mean) / std
            return img_tensor
        if self._modality == 'rgb':
            deg_img = apply_quality_degradation_rgb(
                img_tensor, self._deg_type, level)
        else:
            deg_img = apply_quality_degradation_t(
                img_tensor, self._deg_type, level)
        result = img_tensor * (1 - self._img_mask_full) + deg_img * self._img_mask_full
        if mean is not None and std is not None:
            return (result * 255.0 - mean) / std
        return result

    def apply_all_levels_batched(self, img_tensor, mean=None, std=None):
        B = img_tensor.shape[0]
        if mean is not None and std is not None:
            level0 = (img_tensor * 255.0 - mean) / std
        else:
            level0 = img_tensor
        all_norm = [level0]
        for level in range(1, self.num_levels):
            if self._modality == 'rgb':
                deg_img = apply_quality_degradation_rgb(
                    img_tensor, self._deg_type, level)
            else:
                deg_img = apply_quality_degradation_t(
                    img_tensor, self._deg_type, level)
            result = img_tensor * (1 - self._img_mask_full) + deg_img * self._img_mask_full
            if mean is not None and std is not None:
                result = (result * 255.0 - mean) / std
            all_norm.append(result)
        batched = torch.cat(all_norm, dim=0)
        return batched, B

    def get_level_mask_list(self):
        zero_mask = torch.zeros_like(self._img_binary_mask)
        mask_list = [zero_mask]
        for lvl in range(1, self.num_levels):
            mask_list.append(self._img_binary_mask.clone())
        return mask_list

    def generate_multi_level(self, img_tensor, modality='rgb', deg_type=None):
        self.prepare_degradation(img_tensor, modality)

        results = []
        for level in range(self.num_levels):
            if level == 0:
                deg_img = img_tensor
            else:
                deg_img = self.apply_level(img_tensor, level)
            results.append((deg_img, self._img_binary_mask))

        return results, self._deg_type

    def generate_degraded_image(self, img_tensor, modality='rgb',
                                deg_type=None, level=None):
        if level is None:
            level = random.randint(1, self.num_levels - 1)
        if deg_type is None:
            if modality == 'rgb':
                deg_type = random.choice(_QUALITY_RGB_DEG_TYPES)
            else:
                deg_type = random.choice(_QUALITY_T_DEG_TYPES)

        B, C, H, W = img_tensor.shape
        num_regions = random.randint(*self.num_regions_range)
        img_mask = self._generate_region_mask_image(
            H, W, num_regions, img_tensor.device)
        img_mask = img_mask.expand(B, -1, -1, -1).clone()
        img_mask_full = img_mask.expand(B, C, H, W).clone()

        if modality == 'rgb':
            deg_img = apply_quality_degradation_rgb(
                img_tensor, deg_type, level)
        else:
            deg_img = apply_quality_degradation_t(
                img_tensor, deg_type, level)

        result = img_tensor * (1 - img_mask_full) + deg_img * img_mask_full
        return result, img_mask

    def compute_local_sharpness(self, img_tensor):
        B, C, H, W = img_tensor.shape
        gray = img_tensor.mean(dim=1, keepdim=True)
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32, device=img_tensor.device
        ).view(1, 1, 3, 3)
        laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
        sharpness = laplacian.abs()
        patch_size = 4 * (2 ** (self.num_stages - 1))
        H_tok = H // patch_size
        W_tok = W // patch_size
        sharpness_tokens = F.adaptive_avg_pool2d(
            sharpness, (H_tok, W_tok))
        sharpness_tokens = sharpness_tokens.reshape(B, -1)
        min_val = sharpness_tokens.min(dim=-1, keepdim=True)[0]
        max_val = sharpness_tokens.max(dim=-1, keepdim=True)[0]
        sharpness_norm = (sharpness_tokens - min_val) / (
            max_val - min_val + 1e-8)
        return sharpness_norm

    def compute_pseudo_labels(self, orig_img, deg_img, token_mask):
        sharp_orig = self.compute_local_sharpness(orig_img)
        sharp_deg = self.compute_local_sharpness(deg_img)
        sharp_diff = sharp_orig - sharp_deg
        sharp_diff = sharp_diff.clamp(min=0)
        mask_flat = token_mask.reshape(orig_img.shape[0], -1)
        pseudo = torch.ones_like(sharp_orig) * 0.9
        pseudo = pseudo - sharp_diff * mask_flat
        pseudo = pseudo.clamp(min=0.05, max=0.95)
        return pseudo
