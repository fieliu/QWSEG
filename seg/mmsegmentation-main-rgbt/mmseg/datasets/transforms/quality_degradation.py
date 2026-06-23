import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _generate_spatial_variation_mask(H, W, device):
    """生成空间变化的掩码，用于让退化在不同区域有不同强度"""
    # 基础随机噪声
    mask = torch.rand(1, 1, H, W, device=device)
    # 用高斯模糊让掩码平滑
    kernel_size = H // 4
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = F.avg_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
    mask = F.avg_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
    # 归一化到 [0, 1]
    mask_min = mask.min()
    mask_max = mask.max()
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
    return mask


_QUALITY_RGB_DEG_TYPES = [
    'motion_blur',
    'gaussian_noise',
    'salt_pepper',
    'overexposure',
    'lowlight',
    'stain',
    'bad_block',
    'missing',
]

_QUALITY_T_DEG_TYPES = [
    'gaussian_noise',
    'salt_pepper',
    'stripe_noise',
    'thermal_contrast',
    'thermal_halo',
    'stain',
    'bad_block',
    'missing',
]

_QUALITY_LEVEL_CONFIGS = {
    'motion_blur': {
        1: {'kernel_size': 0},
        2: {'kernel_size': 5},
        3: {'kernel_size': 11},
        4: {'kernel_size': 21},
        5: {'kernel_size': 35},
    },
    'gaussian_noise': {
        1: {'sigma': 0.0},
        2: {'sigma': 0.02},
        3: {'sigma': 0.05},
        4: {'sigma': 0.10},
        5: {'sigma': 0.20},
    },
    'salt_pepper': {
        1: {'density': 0.0},
        2: {'density': 0.01},
        3: {'density': 0.03},
        4: {'density': 0.07},
        5: {'density': 0.15},
    },
    'overexposure': {
        1: {'gain': 0.0},
        2: {'gain': 30},
        3: {'gain': 70},
        4: {'gain': 130},
        5: {'gain': 200},
    },
    'lowlight': {
        1: {'gamma': 1.0},
        2: {'gamma': 1.5},
        3: {'gamma': 2.5},
        4: {'gamma': 3.5},
        5: {'gamma': 5.0},
    },
    'stripe_noise': {
        1: {'intensity': 0.0},
        2: {'intensity': 0.02},
        3: {'intensity': 0.05},
        4: {'intensity': 0.10},
        5: {'intensity': 0.20},
    },
    'thermal_contrast': {
        1: {'alpha': 1.0},
        2: {'alpha': 0.7},
        3: {'alpha': 0.5},
        4: {'alpha': 0.3},
        5: {'alpha': 0.1},
    },
    'thermal_halo': {
        1: {'peak_gain': 0},
        2: {'peak_gain': 20},
        3: {'peak_gain': 50},
        4: {'peak_gain': 90},
        5: {'peak_gain': 140},
    },
    'stain': {
        1: {'opacity': 0.0},
        2: {'opacity': 0.15},
        3: {'opacity': 0.35},
        4: {'opacity': 0.60},
        5: {'opacity': 0.90},
    },
    'bad_block': {
        1: {'ratio': 0.0},
        2: {'ratio': 0.05},
        3: {'ratio': 0.15},
        4: {'ratio': 0.30},
        5: {'ratio': 0.50},
    },
    'missing': {
        1: {'ratio': 0.0},    # clean (no missing)
        2: {'ratio': 1.0},    # full missing (severity = area size, not ratio)
        3: {'ratio': 1.0},
        4: {'ratio': 1.0},
        5: {'ratio': 1.0},
    },
}


def _make_motion_kernel(kernel_size, angle, device='cpu'):
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    mid = kernel_size // 2
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for i in range(kernel_size):
        offset = i - mid
        x = int(round(mid + offset * cos_a))
        y = int(round(mid + offset * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def apply_quality_degradation_rgb(img_tensor, deg_type, level, spatial_mask=None):
    B, C, H, W = img_tensor.shape
    assert C == 3

    configs = _QUALITY_LEVEL_CONFIGS[deg_type]
    level = max(1, min(5, level))
    params = configs[level]

    uniform = spatial_mask is None

    if not uniform:
        if spatial_mask.dim() == 2:
            spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
        spatial_mask = spatial_mask.expand(B, C, H, W)

    if deg_type == 'motion_blur':
        ksize = params['kernel_size']
        if ksize <= 1:
            return img_tensor
        angle = random.uniform(0, math.pi)
        kernel_t = _make_motion_kernel(ksize, angle, device=img_tensor.device)
        kernel_t = kernel_t.unsqueeze(0).unsqueeze(0).expand(3, 1, ksize, ksize)
        pad = ksize // 2
        padded = F.pad(img_tensor, [pad, pad, pad, pad], mode='reflect')
        blurred = F.conv2d(padded, kernel_t, groups=3)
        if uniform:
            img_tensor = blurred
        else:
            alpha = spatial_mask
            img_tensor = img_tensor * (1 - alpha) + blurred * alpha

    elif deg_type == 'gaussian_noise':
        sigma = params['sigma']
        if sigma <= 0:
            return img_tensor
        noise = torch.randn_like(img_tensor) * sigma
        if not uniform:
            noise = noise * spatial_mask
        img_tensor = torch.clamp(img_tensor + noise, 0, 1)

    elif deg_type == 'salt_pepper':
        density = params['density']
        if density <= 0:
            return img_tensor
        if uniform:
            density_map = density
        else:
            density_map = density * spatial_mask[:, 0:1]
        mask = torch.rand_like(img_tensor)
        pepper_mask = (mask < density_map / 2).float()
        salt_mask = (mask > 1 - density_map / 2).float()
        img_tensor = img_tensor * (1 - pepper_mask) * (1 - salt_mask) + salt_mask

    elif deg_type == 'overexposure':
        gain = params['gain']
        if gain <= 0:
            return img_tensor
        if uniform:
            gain_map = gain
        else:
            gain_map = gain * spatial_mask
        img_norm = img_tensor * 255.0
        img_norm = torch.clamp(img_norm + gain_map, 0, 255)
        img_tensor = img_norm / 255.0

    elif deg_type == 'lowlight':
        gamma = params['gamma']
        if gamma <= 1.0:
            return img_tensor
        if uniform:
            gamma_map = gamma
        else:
            gamma_map = 1.0 + (gamma - 1.0) * spatial_mask
        img_tensor = torch.clamp(img_tensor, min=1e-8)
        img_tensor = torch.pow(img_tensor, gamma_map)

    elif deg_type == 'stain':
        opacity = params['opacity']
        if opacity <= 0:
            return img_tensor
        if uniform:
            opacity_map = opacity
        else:
            opacity_map = opacity * spatial_mask
        stain_color = torch.tensor([0.1, 0.1, 0.1], device=img_tensor.device)
        stain_color = stain_color.view(1, 3, 1, 1)
        img_tensor = img_tensor * (1 - opacity_map) + stain_color * opacity_map

    elif deg_type == 'bad_block':
        ratio = params['ratio']
        if ratio <= 0:
            return img_tensor
        if uniform:
            ratio_map = ratio
        else:
            ratio_map = ratio * spatial_mask[:, 0:1]
        mask = torch.rand_like(img_tensor)
        img_tensor = img_tensor * (mask >= ratio_map).float()

    elif deg_type == 'missing':
        ratio = params['ratio']
        if ratio <= 0:
            return img_tensor  # L1: clean, no change
        # L2-L5: ratio=1.0, zero out the entire spatial_mask region.
        # Severity is encoded by AREA size (level-area coupling in
        # degradation.py), not by ratio. Missing is binary by nature.
        if not uniform:
            img_tensor = img_tensor * (1 - spatial_mask.expand(B, C, H, W))
        else:
            img_tensor = torch.zeros_like(img_tensor)

    return img_tensor


def apply_quality_degradation_t(img_tensor, deg_type, level, spatial_mask=None):
    B, C, H, W = img_tensor.shape
    assert C == 3

    configs = _QUALITY_LEVEL_CONFIGS[deg_type]
    level = max(1, min(5, level))
    params = configs[level]

    uniform = spatial_mask is None

    if not uniform:
        if spatial_mask.dim() == 2:
            spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
        spatial_mask = spatial_mask.expand(B, C, H, W)

    if deg_type == 'gaussian_noise':
        sigma = params['sigma']
        if sigma <= 0:
            return img_tensor
        noise = torch.randn_like(img_tensor) * sigma
        if not uniform:
            noise = noise * spatial_mask
        img_tensor = torch.clamp(img_tensor + noise, 0, 1)

    elif deg_type == 'salt_pepper':
        density = params['density']
        if density <= 0:
            return img_tensor
        if uniform:
            density_map = density
        else:
            density_map = density * spatial_mask[:, 0:1]
        mask = torch.rand_like(img_tensor)
        pepper_mask = (mask < density_map / 2).float()
        salt_mask = (mask > 1 - density_map / 2).float()
        img_tensor = img_tensor * (1 - pepper_mask) * (1 - salt_mask) + salt_mask

    elif deg_type == 'stripe_noise':
        intensity = params['intensity']
        if intensity <= 0:
            return img_tensor
        direction = random.choice(['horizontal', 'vertical'])
        if direction == 'horizontal':
            stripe = torch.randn(B, C, 1, W, device=img_tensor.device) * intensity
            stripe = stripe.expand(B, C, H, W)
        else:
            stripe = torch.randn(B, C, H, 1, device=img_tensor.device) * intensity
            stripe = stripe.expand(B, C, H, W)
        if uniform:
            img_tensor = torch.clamp(img_tensor + stripe, 0, 1)
        else:
            img_tensor = torch.clamp(img_tensor + stripe * spatial_mask, 0, 1)

    elif deg_type == 'thermal_contrast':
        alpha = params['alpha']
        if alpha >= 1.0:
            return img_tensor
        if uniform:
            alpha_map = alpha
        else:
            alpha_map = alpha + (1.0 - alpha) * (1.0 - spatial_mask)
        mean_val = img_tensor.mean(dim=[2, 3], keepdim=True)
        min_val = img_tensor.amin(dim=[2, 3], keepdim=True)
        max_val = img_tensor.amax(dim=[2, 3], keepdim=True)
        center = mean_val
        half_range = (max_val - min_val) * alpha_map / 2
        new_min = torch.clamp(center - half_range, 0, 1)
        new_max = torch.clamp(center + half_range, 0, 1)
        normalized = (img_tensor - min_val) / (max_val - min_val + 1e-8)
        img_tensor = torch.clamp(new_min + normalized * (new_max - new_min), 0, 1)

    elif deg_type == 'thermal_halo':
        peak_gain = params['peak_gain']
        if peak_gain <= 0:
            return img_tensor
        Y = torch.arange(H, device=img_tensor.device, dtype=torch.float32).view(1, 1, H, 1)
        X = torch.arange(W, device=img_tensor.device, dtype=torch.float32).view(1, 1, 1, W)
        num_centers = random.randint(2, 5)
        sigma = random.uniform(8, 20)
        halo = torch.zeros(1, 1, H, W, device=img_tensor.device)
        for _ in range(num_centers):
            cy = random.randint(0, H - 1)
            cx = random.randint(0, W - 1)
            dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
            halo = halo + peak_gain / 255.0 * torch.exp(-dist_sq / (2 * sigma ** 2))
        halo_t = halo.expand(B, C, H, W)
        if uniform:
            img_tensor = torch.clamp(img_tensor + halo_t, 0, 1)
        else:
            img_tensor = torch.clamp(img_tensor + halo_t * spatial_mask, 0, 1)

    elif deg_type == 'stain':
        opacity = params['opacity']
        if opacity <= 0:
            return img_tensor
        if uniform:
            opacity_map = opacity
        else:
            opacity_map = opacity * spatial_mask
        stain_color = torch.tensor([0.05, 0.05, 0.05], device=img_tensor.device)
        stain_color = stain_color.view(1, 3, 1, 1)
        img_tensor = img_tensor * (1 - opacity_map) + stain_color * opacity_map

    elif deg_type == 'bad_block':
        ratio = params['ratio']
        if ratio <= 0:
            return img_tensor
        if uniform:
            ratio_map = ratio
        else:
            ratio_map = ratio * spatial_mask[:, 0:1]
        mask = torch.rand_like(img_tensor)
        img_tensor = img_tensor * (mask >= ratio_map).float()

    elif deg_type == 'missing':
        ratio = params['ratio']
        if ratio <= 0:
            return img_tensor  # L1: clean, no change
        # L2-L5: ratio=1.0, zero out the entire spatial_mask region.
        if not uniform:
            img_tensor = img_tensor * (1 - spatial_mask.expand(B, C, H, W))
        else:
            img_tensor = torch.zeros_like(img_tensor)

    return img_tensor


def generate_quality_degradation_levels(img_tensor, deg_type, modality, num_levels=5):
    deg_images = []
    B, C, H, W = img_tensor.shape
    
    # 生成一个空间掩码，所有级别使用同一个掩码
    base_mask = _generate_spatial_variation_mask(H, W, img_tensor.device)
    # 保证最小退化强度：掩码范围 [0.3, 1.0]
    # 即使 base_mask=0 的位置也有 30% 的退化，避免零退化
    min_mask = 0.3
    spatial_mask = min_mask + (1.0 - min_mask) * base_mask
    
    for level in range(1, num_levels + 1):
        if level == 1:
            deg_images.append(img_tensor.clone())
        else:
            # 不再乘 level_scale！params 本身已经随 level 增大
            # 掩码只负责空间变化，不负责级别缩放
            if modality == 'rgb':
                deg_img = apply_quality_degradation_rgb(img_tensor, deg_type, level, spatial_mask=spatial_mask)
            else:
                deg_img = apply_quality_degradation_t(img_tensor, deg_type, level, spatial_mask=spatial_mask)
            deg_images.append(deg_img)
    return deg_images
