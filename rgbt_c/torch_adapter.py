"""CPU reference augmentation bridge for normalized 6-channel RGB-T tensors.

The NumPy reference kernels deliberately match offline PNG generation. This
implementation transfers images to CPU; profile it before large-scale runs.
"""
import copy

import numpy as np
import torch

from .protocol import (TRAIN_CORRUPTIONS, apply_pair, operation, recipe,
                       resolve_corruptions, sample_recipe)


class RGBTDegrader:
    def __init__(self, mean, std, corruptions=None, degrade_prob=.8,
                 modality_probs=(.45, .45, .10), scope_probs=(.5, .25, .25),
                 severity_range=(1, 5), area_range=(.1, .6), missing_prob=.1,
                 weak_severity=0):
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)
        if len(self.mean) != 6 or len(self.std) != 6 or min(self.std) <= 0:
            raise ValueError('Expected six means and six positive std values')
        names = resolve_corruptions(corruptions or list(TRAIN_CORRUPTIONS))
        if any(c.endswith('_missing') for c in names):
            raise ValueError('Use missing_prob for missing modalities')
        for values in (modality_probs, scope_probs):
            if len(values) != 3 or min(values) < 0 or not np.isclose(sum(values), 1):
                raise ValueError('Probabilities must be three nonnegative values summing to 1')
        if not 0 <= degrade_prob <= 1 or not 0 <= missing_prob <= 1:
            raise ValueError('Invalid degradation probability')
        if not 1 <= severity_range[0] <= severity_range[1] <= 5:
            raise ValueError('Invalid severity range')
        if not 0 < area_range[0] <= area_range[1] <= 1:
            raise ValueError('Invalid area range')
        if weak_severity not in (0, 1):
            raise ValueError('Teacher weak_severity must be 0 (clean) or 1')
        self.policy = dict(corruptions=names, degrade_prob=degrade_prob,
                           modality_probs=modality_probs, scope_probs=scope_probs,
                           severity_range=severity_range, area_range=area_range,
                           missing_prob=missing_prob)
        self.weak_severity = weak_severity

    def sample(self, count, seed=None):
        # Torch's process/rank-aware RNG is checkpointed by the training runtime.
        if seed is None:
            seed = int(torch.randint(0, 2**32, (1,), dtype=torch.int64).item())
        rng = np.random.default_rng(seed)
        return [sample_recipe(rng, **self.policy) for _ in range(count)]

    @torch.no_grad()
    def apply(self, rgb, thermal, recipes, valid_shapes=None):
        if rgb.shape != thermal.shape or rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected paired Bx3xHxW normalized tensors')
        if len(recipes) != rgb.shape[0] or not rgb.is_floating_point():
            raise ValueError('Recipe count mismatch or unnormalized integer input')
        # MMSeg pads normalized tensors with zero, which does not denormalize
        # to a grayscale triplet. Corrupt only the original image rectangle;
        # preserve right/bottom padding and mark its corruption strength zero.
        if valid_shapes is not None:
            if len(valid_shapes) != rgb.shape[0]:
                raise ValueError('Valid-shape count mismatch')
            shapes = [tuple(int(x) for x in shape[:2]) for shape in valid_shapes]
            height, width = rgb.shape[-2:]
            if any(not (0 < h <= height and 0 < w <= width) for h, w in shapes):
                raise ValueError('Valid shape outside padded tensor')
            if any(shape != (height, width) for shape in shapes):
                r_out, t_out = rgb.clone(), thermal.clone()
                r_mask = rgb.new_zeros((rgb.shape[0], 1, height, width))
                t_mask = r_mask.clone()
                for i, (h, w) in enumerate(shapes):
                    r, t, mr, mt = self.apply(
                        rgb[i:i+1, :, :h, :w], thermal[i:i+1, :, :h, :w],
                        recipes[i:i+1])
                    r_out[i:i+1, :, :h, :w] = r
                    t_out[i:i+1, :, :h, :w] = t
                    r_mask[i:i+1, :, :h, :w] = mr
                    t_mask[i:i+1, :, :h, :w] = mt
                return r_out, t_out, r_mask, t_mask
        images = torch.cat([rgb, thermal], dim=1).float()
        mean = images.new_tensor(self.mean)[None, :, None, None]
        std = images.new_tensor(self.std)[None, :, None, None]
        raw = ((images * std + mean).round().clamp(0, 255)
               .to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        out = []
        masks = []
        for i, spec in enumerate(recipes):
            if np.max(np.abs(raw[i, :, :, 3:].astype(np.int16)
                             - raw[i, :, :, 3:4].astype(np.int16))) > 1:
                raise ValueError('Thermal channels must be copies of one grayscale image')
            r, t, mr, mt = apply_pair(raw[i, :, :, :3], raw[i, :, :, 3:4], spec, True)
            out.append(np.concatenate([r, np.repeat(t, 3, axis=2)], axis=2))
            masks.append(np.concatenate([mr, mt], axis=2))
        pixels = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(images)
        normalized = ((pixels - mean) / std).to(rgb.dtype)
        levels = torch.from_numpy(np.stack(masks)).permute(0, 3, 1, 2).to(images)
        # Avoid a quantization round-trip on samples whose recipe is clean.
        for i, spec in enumerate(recipes):
            if not spec['operations']:
                normalized[i] = images[i].to(rgb.dtype)
        return normalized[:, :3], normalized[:, 3:], levels[:, :1], levels[:, 1:]

    def __call__(self, rgb, thermal, epoch=0, valid_shapes=None):
        return self.apply(rgb, thermal, self.sample(rgb.shape[0]), valid_shapes)

    @torch.no_grad()
    def missing(self, inputs, valid_shapes=None):
        specs = []
        for _ in range(inputs.shape[0]):
            mod = int(torch.randint(0, 2, (1,)).item())
            specs.append(recipe([operation(('rgb_missing', 't_missing')[mod], 1, 0)]))
        r, t, _, _ = self.apply(inputs[:, :3], inputs[:, 3:], specs, valid_shapes)
        # Missing is represented solely by raw black pixels, at train and test.
        return torch.cat([r, t], dim=1)

    def make_paired(self, rgb, thermal, mean=None, std=None, epoch=0,
                    valid_shapes=None):
        if mean is not None:
            actual = tuple(float(x) for x in mean)
            if not np.allclose(actual, self.mean) or not np.allclose(tuple(float(x) for x in std), self.std):
                raise ValueError('Preprocessor normalization differs from degrader')
        strong_specs = self.sample(rgb.shape[0])
        weak_specs = copy.deepcopy(strong_specs)
        for spec in weak_specs:
            if self.weak_severity == 0:
                spec['operations'] = []
            else:
                spec['operations'] = [dict(op, severity=1) for op in spec['operations']
                                      if not op['name'].endswith('_missing')]
        lr, lt, lmr, lmt = self.apply(rgb, thermal, weak_specs, valid_shapes)
        hr, ht, hmr, hmt = self.apply(rgb, thermal, strong_specs, valid_shapes)
        gap = torch.maximum(hmr-lmr, hmt-lmt).flatten(1).amax(1)
        return lr, lt, hr, ht, lmr, lmt, hmr, hmt, (gap > 0).float(), gap
