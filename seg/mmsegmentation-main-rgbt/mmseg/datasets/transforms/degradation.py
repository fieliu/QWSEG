import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_RGB_LEVEL_CONFIGS = {
    'low_light': {
        2: {'brightness_scale': 0.5},
        3: {'brightness_scale': 0.3},
        4: {'brightness_scale': 0.15, 'noise_sigma': 0.02},
        5: {'brightness_scale': 0.06, 'noise_sigma': 0.05},
    },
    'overexposure': {
        2: {'gain': 1.5},
        3: {'gain': 2.5},
        4: {'gain': 4.0},
        5: {'gain': 8.0},
    },
    'motion_blur': {
        2: {'kernel_size': 5},
        3: {'kernel_size': 9},
        4: {'kernel_size': 15},
        5: {'kernel_size': 23},
    },
}

_THERMAL_LEVEL_CONFIGS = {
    'thermal_contrast': {
        2: {'dynamic_range_scale': 0.7},
        3: {'dynamic_range_scale': 0.4},
        4: {'dynamic_range_scale': 0.15},
        5: {'dynamic_range_scale': 0.05},
    },
    'stripe_noise': {
        2: {'intensity': 0.03},
        3: {'intensity': 0.06},
        4: {'intensity': 0.12},
        5: {'intensity': 0.25},
    },
    'thermal_noise': {
        2: {'sigma': 0.03},
        3: {'sigma': 0.06},
        4: {'sigma': 0.12},
        5: {'sigma': 0.20},
    },
    'thermal_saturation': {
        2: {'saturation_ratio': 0.01},
        3: {'saturation_ratio': 0.05},
        4: {'saturation_ratio': 0.15},
        5: {'saturation_ratio': 0.30},
    },
}


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


def _apply_rgb_degradation(img_tensor, deg_type, level):
    B, C, H, W = img_tensor.shape
    # 支持两种情况：3通道（只有RGB）或6通道（RGB+IR）
    is_3_channel = (C == 3)
    rgb_slice = slice(None, None) if is_3_channel else slice(None, 3)

    if deg_type == 'low_light':
        configs = _RGB_LEVEL_CONFIGS['low_light']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        params = configs[level]
        brightness_scale = params['brightness_scale']
        noise_sigma = params.get('noise_sigma', 0.0)

        img_tensor = img_tensor.clone()
        img_tensor[:, rgb_slice] = img_tensor[:, rgb_slice] * brightness_scale
        if noise_sigma > 0:
            noise = torch.randn_like(img_tensor[:, rgb_slice]) * noise_sigma
            img_tensor[:, rgb_slice] = torch.clamp(img_tensor[:, rgb_slice] + noise, 0, 1)

    elif deg_type == 'overexposure':
        configs = _RGB_LEVEL_CONFIGS['overexposure']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        gain = configs[level]['gain']
        img_tensor = img_tensor.clone()
        img_tensor[:, rgb_slice] = torch.clamp(img_tensor[:, rgb_slice] * gain, 0, 1)

    elif deg_type == 'motion_blur':
        configs = _RGB_LEVEL_CONFIGS['motion_blur']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        kernel_size = configs[level]['kernel_size']
        angle = random.uniform(0, math.pi)
        kernel = _make_motion_kernel(kernel_size, angle)
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        kernel_tensor = kernel_tensor.to(img_tensor.device)

        img_tensor = img_tensor.clone()
        pad = kernel_size // 2
        num_channels = 3 if is_3_channel else 3
        for c in range(num_channels):
            channel = img_tensor[:, c:c + 1] if is_3_channel else img_tensor[:, c:c + 1]
            channel = F.pad(channel, [pad, pad, pad, pad], mode='reflect')
            channel = F.conv2d(channel, kernel_tensor)
            img_tensor[:, c:c + 1] = channel

    elif deg_type == 'modality_missing':
        img_tensor = img_tensor.clone()
        img_tensor[:, rgb_slice] = 0.0

    return img_tensor


def _apply_thermal_degradation(img_tensor, deg_type, level):
    B, C, H, W = img_tensor.shape
    # 支持两种情况：3通道（只有IR）或6通道（RGB+IR）
    is_3_channel = (C == 3)
    thermal_slice = slice(None, None) if is_3_channel else slice(3, None)
    thermal_offset = 0 if is_3_channel else 3

    if deg_type == 'thermal_contrast':
        configs = _THERMAL_LEVEL_CONFIGS['thermal_contrast']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        dynamic_range_scale = configs[level]['dynamic_range_scale']

        img_tensor = img_tensor.clone()
        thermal = img_tensor[:, thermal_slice]
        mean_val = thermal.mean(dim=[2, 3], keepdim=True)
        min_val = thermal.amin(dim=[2, 3], keepdim=True)
        max_val = thermal.amax(dim=[2, 3], keepdim=True)
        center = mean_val
        half_range = (max_val - min_val) * dynamic_range_scale / 2
        new_min = torch.clamp(center - half_range, 0, 1)
        new_max = torch.clamp(center + half_range, 0, 1)
        normalized = (thermal - min_val) / (max_val - min_val + 1e-8)
        thermal = new_min + normalized * (new_max - new_min)
        img_tensor[:, thermal_slice] = torch.clamp(thermal, 0, 1)

    elif deg_type == 'stripe_noise':
        configs = _THERMAL_LEVEL_CONFIGS['stripe_noise']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        intensity = configs[level]['intensity']

        img_tensor = img_tensor.clone()
        direction = random.choice(['horizontal', 'vertical'])
        if direction == 'horizontal':
            stripe = torch.randn(B, 1, W, 3, device=img_tensor.device) * intensity
            stripe = stripe.expand(B, 1, H, 3)
        else:
            stripe = torch.randn(B, H, 1, 3, device=img_tensor.device) * intensity
            stripe = stripe.expand(B, H, W, 3)
        stripe = stripe.permute(0, 3, 1, 2)
        img_tensor[:, thermal_slice] = torch.clamp(img_tensor[:, thermal_slice] + stripe, 0, 1)

    elif deg_type == 'thermal_noise':
        configs = _THERMAL_LEVEL_CONFIGS['thermal_noise']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        sigma = configs[level]['sigma']

        img_tensor = img_tensor.clone()
        noise = torch.randn_like(img_tensor[:, thermal_slice]) * sigma
        img_tensor[:, thermal_slice] = torch.clamp(img_tensor[:, thermal_slice] + noise, 0, 1)

    elif deg_type == 'thermal_saturation':
        configs = _THERMAL_LEVEL_CONFIGS['thermal_saturation']
        level = min(configs.keys(), key=lambda x: abs(x - level))
        saturation_ratio = configs[level]['saturation_ratio']

        img_tensor = img_tensor.clone()
        num_pixels = H * W
        num_saturated = int(num_pixels * saturation_ratio)
        if num_saturated > 0:
            for c in range(3):
                channel = img_tensor[:, thermal_offset + c]
                flat = channel.reshape(B, -1)
                _, top_indices = flat.topk(num_saturated, dim=1)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, top_indices, 1.0)
                mask = mask.reshape(B, H, W)
                kernel = torch.ones(1, 1, 3, 3, device=img_tensor.device) / 9.0
                mask_float = mask.float().unsqueeze(1)
                blurred_mask = F.conv2d(
                    F.pad(mask_float, [1, 1, 1, 1], mode='reflect'),
                    kernel)
                channel = channel * (1 - blurred_mask.squeeze(1)) + blurred_mask.squeeze(1)
                img_tensor[:, thermal_offset + c] = torch.clamp(channel, 0, 1)

    elif deg_type == 'modality_missing':
        img_tensor = img_tensor.clone()
        img_tensor[:, thermal_slice] = 0.0

    return img_tensor


def apply_degradation(img_tensor, deg_type, level, modality='rgb'):
    if modality == 'rgb':
        return _apply_rgb_degradation(img_tensor, deg_type, level)
    elif modality == 'ir':
        return _apply_thermal_degradation(img_tensor, deg_type, level)
    else:
        raise ValueError(
            f'Unknown modality: {modality}. Supported: rgb, ir')


def apply_multi_region_degradation(img_tensor, region_configs, modality='rgb'):
    img_tensor = img_tensor.clone()
    B, C, H, W = img_tensor.shape

    for region, deg_type, level in region_configs:
        start_h, start_w, region_h, region_w = region

        region_tensor = img_tensor[:, :, start_h:start_h + region_h,
                                   start_w:start_w + region_w]
        deg_region = apply_degradation(region_tensor, deg_type, level, modality)
        img_tensor[:, :, start_h:start_h + region_h,
                   start_w:start_w + region_w] = deg_region

    return img_tensor
