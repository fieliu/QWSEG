import sys
sys.path.insert(0, '/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt')

import os
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from mmseg.datasets.transforms.robustness_degradation import (
    GlobalDegradation, LocalDegradation,
    RGBMissingDegradation, ThermalMissingDegradation
)


DATASET = 'FMB'

if DATASET == 'FMB':
    DATA_ROOT = '/home/lh/code/data/FMB_ALL/FMB'
    DATA_ROOT_T = '/home/lh/code/data/FMB_ALL/FMB_T'
    VAL_DIR = 'images/validation'
    NUM_CLASSES = 15
    CLASS_NAMES = ['Background', 'Road', 'Sidewalk', 'Building',
                   'Traffic Light', 'Traffic Sign', 'Vegetation', 'Sky',
                   'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pole']
elif DATASET == 'MFNet':
    DATA_ROOT = '/home/lh/code/data/MFNet'
    DATA_ROOT_T = None
    VAL_DIR = 'images'
    NUM_CLASSES = 9
    CLASS_NAMES = ['unlabeled', 'car', 'person', 'bike', 'curve',
                   'car_stop', 'guardrail', 'color_cone', 'bump']

def load_rgbt_image_fmb(img_name):
    rgb_path = os.path.join(DATA_ROOT, VAL_DIR, img_name)
    t_path = os.path.join(DATA_ROOT_T, VAL_DIR, img_name)
    rgb = cv2.imread(rgb_path)
    t = cv2.imread(t_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
    img = np.concatenate((rgb, t), axis=2)
    return img

def load_rgbt_image_mfnet(img_name):
    img_path = os.path.join(DATA_ROOT, VAL_DIR, img_name)
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raw = cv2.imread(img_path)
    if len(raw.shape) == 2:
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        t = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 4:
        rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        t_gray = raw[:, :, 3]
        t = cv2.cvtColor(t_gray, cv2.COLOR_GRAY2RGB)
    elif raw.shape[2] == 3:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        t = rgb.copy()
    else:
        raise ValueError(f'Unexpected image shape: {raw.shape}')
    img = np.concatenate((rgb, t), axis=2)
    return img

if DATASET == 'FMB':
    val_images = sorted(os.listdir(os.path.join(DATA_ROOT, VAL_DIR)))
    load_fn = load_rgbt_image_fmb
else:
    with open(os.path.join(DATA_ROOT, 'val.txt'), 'r') as f:
        val_images = [line.strip().split()[0] for line in f if line.strip()]
    val_images = [v if v.endswith('.png') else v + '.png' for v in val_images]
    load_fn = load_rgbt_image_mfnet

random.seed(42)
selected = random.sample(val_images, min(4, len(val_images)))
print(f'Dataset: {DATASET}, Selected images: {selected}')

images = [load_fn(name) for name in selected]

def resize_long_edge(img, target=250):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def split_rgbt(img_6ch):
    return img_6ch[:, :, :3], img_6ch[:, :, 3:]

def apply_deg_to_img(img_6ch, deg_cls, deg_type, intensity, dual_prob=1.0, seed=42):
    deg = deg_cls(deg_type=deg_type, intensity=intensity,
                  dual_prob=dual_prob, seed=seed)
    return deg.apply_degradation(img_6ch.copy())


GLOBAL_DEG_TYPES = [
    ('motion_blur', '运动模糊', 'both'),
    ('gaussian_noise', '高斯噪声', 'both'),
    ('salt_pepper', '椒盐噪声', 'both'),
    ('stripe_noise', '条纹噪声', 'thermal'),
    ('thermal_contrast', '热对比度降低', 'thermal'),
]

INTENSITIES = ['low', 'medium', 'high']
INTENSITY_CN = {'low': '低', 'medium': '中', 'high': '高'}

img_idx = 0
img = images[img_idx]

n_rows = len(GLOBAL_DEG_TYPES) + 1
n_cols = len(INTENSITIES) * 2

fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.5))
gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.25)

for col_idx, intensity in enumerate(INTENSITIES):
    rgb_orig, t_orig = split_rgbt(img)
    rgb_small = resize_long_edge(rgb_orig, 250)
    t_small = resize_long_edge(t_orig, 250)

    ax_rgb = fig.add_subplot(gs[0, col_idx * 2])
    ax_rgb.imshow(rgb_small)
    ax_rgb.set_title(f'RGB ({INTENSITY_CN[intensity]})', fontsize=9)
    ax_rgb.axis('off')

    ax_t = fig.add_subplot(gs[0, col_idx * 2 + 1])
    ax_t.imshow(t_small)
    ax_t.set_title(f'T ({INTENSITY_CN[intensity]})', fontsize=9)
    ax_t.axis('off')

for row_idx, (deg_type, deg_cn, modality) in enumerate(GLOBAL_DEG_TYPES):
    for col_idx, intensity in enumerate(INTENSITIES):
        if modality == 'both':
            degraded = apply_deg_to_img(img, GlobalDegradation, deg_type,
                                        intensity, dual_prob=1.0, seed=42)
        elif modality == 'thermal':
            degraded = apply_deg_to_img(img, GlobalDegradation, deg_type,
                                        intensity, dual_prob=0.0, seed=42)
        else:
            degraded = apply_deg_to_img(img, GlobalDegradation, deg_type,
                                        intensity, dual_prob=1.0, seed=42)

        rgb_deg, t_deg = split_rgbt(degraded)
        rgb_small = resize_long_edge(rgb_deg, 250)
        t_small = resize_long_edge(t_deg, 250)

        ax_rgb = fig.add_subplot(gs[row_idx + 1, col_idx * 2])
        ax_rgb.imshow(rgb_small)
        if col_idx == 0:
            ax_rgb.set_ylabel(deg_cn, fontsize=10, rotation=0,
                              labelpad=60, va='center')
        ax_rgb.axis('off')

        ax_t = fig.add_subplot(gs[row_idx + 1, col_idx * 2 + 1])
        ax_t.imshow(t_small)
        ax_t.axis('off')

fig.suptitle(f'全局退化可视化 - {DATASET} (图像: {selected[img_idx]})',
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(f'global_degradation_vis_{DATASET}.png', dpi=150, bbox_inches='tight')
plt.show()


LOCAL_DEG_TYPES = [
    ('local_overexposure', '局部过曝', 'rgb'),
    ('local_lowlight', '局部过暗', 'rgb'),
    ('water_stain', '水滴', 'both'),
    ('stain', '污渍', 'both'),
    ('local_bad_block', '传感器坏块', 'both'),
    ('local_gaussian_noise', '局部高斯噪声', 'both'),
]

img_idx = 1
img = images[img_idx]

n_rows = len(LOCAL_DEG_TYPES) + 1
n_cols = len(INTENSITIES) * 2

fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.5))
gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.25)

for col_idx, intensity in enumerate(INTENSITIES):
    rgb_orig, t_orig = split_rgbt(img)
    rgb_small = resize_long_edge(rgb_orig, 250)
    t_small = resize_long_edge(t_orig, 250)

    ax_rgb = fig.add_subplot(gs[0, col_idx * 2])
    ax_rgb.imshow(rgb_small)
    ax_rgb.set_title(f'RGB ({INTENSITY_CN[intensity]})', fontsize=9)
    ax_rgb.axis('off')

    ax_t = fig.add_subplot(gs[0, col_idx * 2 + 1])
    ax_t.imshow(t_small)
    ax_t.set_title(f'T ({INTENSITY_CN[intensity]})', fontsize=9)
    ax_t.axis('off')

for row_idx, (deg_type, deg_cn, modality) in enumerate(LOCAL_DEG_TYPES):
    for col_idx, intensity in enumerate(INTENSITIES):
        if modality == 'both':
            degraded = apply_deg_to_img(img, LocalDegradation, deg_type,
                                        intensity, dual_prob=1.0, seed=42)
        elif modality == 'rgb':
            degraded = apply_deg_to_img(img, LocalDegradation, deg_type,
                                        intensity, dual_prob=1.0, seed=42)
        else:
            degraded = apply_deg_to_img(img, LocalDegradation, deg_type,
                                        intensity, dual_prob=0.0, seed=42)

        rgb_deg, t_deg = split_rgbt(degraded)
        rgb_small = resize_long_edge(rgb_deg, 250)
        t_small = resize_long_edge(t_deg, 250)

        ax_rgb = fig.add_subplot(gs[row_idx + 1, col_idx * 2])
        ax_rgb.imshow(rgb_small)
        if col_idx == 0:
            ax_rgb.set_ylabel(deg_cn, fontsize=10, rotation=0,
                              labelpad=60, va='center')
        ax_rgb.axis('off')

        ax_t = fig.add_subplot(gs[row_idx + 1, col_idx * 2 + 1])
        ax_t.imshow(t_small)
        ax_t.axis('off')

fig.suptitle(f'局部退化可视化 - {DATASET} (图像: {selected[img_idx]})',
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(f'local_degradation_vis_{DATASET}.png', dpi=150, bbox_inches='tight')
plt.show()