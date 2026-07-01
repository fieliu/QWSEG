"""Hook for evaluating under PARTIAL (local) degradation during training.

Unlike MissingModalityEvalHook (global whole-modality zeroing), this applies
LOCAL rectangular degradation to a small subset of the validation set —
exactly the scenario where the quality mechanism's per-token gating matters.
Reports mIoU grouped by degraded modality and severity level.

Cost: ~50 images × 1 degradation each = ~1 min per eval cycle.
"""

import random
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.logging import print_log
from prettytable import PrettyTable

from mmseg.registry import HOOKS
from mmseg.models.segmentors.degradation import DegradationGenerator


# -- IoU helpers (identical to missing_modality_eval_hook.py) --

def _intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def _compute_miou(area_intersect, area_union):
    iou = area_intersect / (area_union + 1e-10)
    valid = area_union > 0
    return (iou[valid].mean().item() * 100) if valid.any() else 0.0


# Representative corruption types (2-3 per modality, covering different
# degradation families). Not the full 13 — these are for training-time
# lightweight monitoring; full RGBT-C evaluation is done offline.
_DEGRADE_MENU = {
    'rgb': [
        'gaussian_noise',   # sensor noise
        'fog',              # weather
        'low_light',        # illumination
    ],
    't': [
        't_gaussian_noise', # sensor noise
        'stripe_noise',     # IR-specific
        't_defocus_blur',   # optics
    ],
}


@HOOKS.register_module()
class PartialDegradeEvalHook(Hook):
    """Lightweight partial-degradation evaluation during training.

    Every `interval` epochs, samples `num_samples` images from the validation
    set, applies ONE local degradation per image (random modality, corruption
    type, severity 1-5, rectangle area ~0.2-0.5), and reports mIoU grouped
    by degraded modality and by severity.

    Args:
        interval (int): Evaluate every N epochs. Default: 5.
        num_samples (int): Number of validation images to evaluate.
            Default: 50.
        area_range (tuple): (min, max) fraction of image area for the
            degradation rectangle. Default: (0.15, 0.4).
        seed (int): Fixed seed for subset sampling (deterministic across
            epochs). Default: 42.
    """
    priority = 'LOW'

    def __init__(self, interval=5, num_samples=50,
                 area_range=(0.15, 0.4), seed=42):
        super().__init__()
        self.interval = interval
        self.num_samples = num_samples
        self.area_range = area_range
        self.seed = seed
        self._subset_indices = None  # fixed subset, sampled once

    def after_val_epoch(self, runner, metrics=None):
        epoch = runner.epoch
        if epoch % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        dataloader = runner.val_dataloader
        dataset = dataloader.dataset

        # class info
        dataset_meta = getattr(dataset, 'metainfo', {})
        class_names = dataset_meta.get('classes', [])
        if not class_names:
            runner.logger.warning(
                'PartialDegradeEvalHook: no class names, skipping')
            return
        num_classes = len(class_names)
        ignore_index = 255

        # Fixed subset: sample once, reuse across epochs for comparability
        if self._subset_indices is None:
            rng = random.Random(self.seed)
            total = len(dataset)
            n = min(self.num_samples, total)
            self._subset_indices = sorted(rng.sample(range(total), n))
            print_log(
                f'PartialDegradeEvalHook: fixed subset of {n}/{total} '
                f'validation images (seed={self.seed})', logger=runner.logger)

        # Get denorm params from model's data_preprocessor
        pp = model.data_preprocessor
        if hasattr(pp, 'mean') and hasattr(pp, 'std'):
            mean = pp.mean.flatten()[:3]
            std = pp.std.flatten()[:3]
        else:
            mean = torch.tensor([123.675, 116.28, 103.53],
                                device=next(model.parameters()).device)
            std = torch.tensor([58.395, 57.12, 57.375],
                               device=next(model.parameters()).device)

        # -- evaluate each corruption type --
        model.eval()
        with torch.no_grad():
            for modality, corr_types in _DEGRADE_MENU.items():
                for corr in corr_types:
                    self._eval_one_corruption(
                        runner, model, dataset, class_names, num_classes,
                        ignore_index, mean, std, modality, corr, epoch)

    def _eval_one_corruption(self, runner, model, dataset, class_names,
                              num_classes, ignore_index, mean, std,
                              modality, corr_type, epoch):
        """Evaluate the subset under ONE specific corruption type, across
        multiple severities. Each image gets a random severity + random
        rectangle."""
        area_intersect = torch.zeros((5, num_classes), dtype=torch.float64)
        area_union     = torch.zeros((5, num_classes), dtype=torch.float64)

        rng = random.Random(self.seed + epoch)  # vary degradation each epoch

        for idx in self._subset_indices:
            sample = dataset[idx]
            # mmseg val dataset returns (img, data_sample) or just img
            if isinstance(sample, (tuple, list)):
                img = sample[0]
            else:
                img = sample['inputs'] if isinstance(sample, dict) else sample

            # Handle dict-style data_sample
            if isinstance(img, dict):
                continue  # skip non-tensor

            if not isinstance(img, torch.Tensor):
                continue
            if img.ndim == 3:
                img = img.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
            img = img.to(next(model.parameters()).device)

            # Random severity 1-5
            sev = rng.randint(1, 5)
            sev_idx = sev - 1  # 0-based index

            # Apply local degradation
            degraded = self._apply_degradation(
                img, modality, corr_type, sev, mean, std, rng)

            # Forward
            processed = model.data_preprocessor(
                {'inputs': [degraded[0]],
                 'data_samples': [sample[1]] if isinstance(sample, (tuple, list)) and len(sample) > 1 else []},
                False)
            inputs = processed['inputs']
            if isinstance(inputs, (list, tuple)):
                inputs = torch.stack(inputs)

            # predict
            results = model.predict(inputs, processed.get('data_samples', None))
            if results is None:
                continue

            for ds in results:
                pred_label = ds.pred_sem_seg.data.squeeze()
                if isinstance(sample, (tuple, list)) and len(sample) > 1:
                    label = sample[1].gt_sem_seg.data.squeeze().to(pred_label)
                else:
                    continue

                ai, au, _, _ = _intersect_and_union(
                    pred_label, label, num_classes, ignore_index)
                area_intersect[sev_idx] += ai.cpu().float()
                area_union[sev_idx]     += au.cpu().float()

        # Report
        header = f'{modality}/{corr_type}'
        print_log(f'\n{header}:', logger=runner.logger)
        table = PrettyTable()
        table.field_names = ['Severity', 'mIoU']
        overall = 0.0; count = 0
        for sev_idx, sev in enumerate([1, 2, 3, 4, 5]):
            miou = _compute_miou(area_intersect[sev_idx], area_union[sev_idx])
            table.add_row([sev, f'{miou:.2f}'])
            if area_union[sev_idx].sum() > 0:
                overall += miou
                count += 1
        if count > 0:
            overall /= count
            table.add_row(['avg', f'{overall:.2f}'])

        print_log('\n' + table.get_string(), logger=runner.logger)
        runner.logger.info(
            f'{header} avg(sev1-5) mIoU: {overall:.2f}')

    def _apply_degradation(self, img, modality, corr_type, severity,
                           mean, std, rng):
        """Apply LOCAL partial degradation to one modality of a 6ch RGB-T
        image. Uses the same rgbt_c library as training (via DegradationGenerator
        helper) for consistency."""
        B, C, H, W = img.shape
        device = img.device

        # Denorm to [0,1]
        rgb = img[:, :3]
        t   = img[:, 3:6]
        rgb_01 = ((rgb * std[:, None, None].to(device) + mean[:, None, None].to(device)) / 255.0).clamp(0, 1)
        t_01   = ((t   * std[:, None, None].to(device) + mean[:, None, None].to(device)) / 255.0).clamp(0, 1)

        # Random rectangle area
        area_frac = rng.uniform(*self.area_range)
        aspect = rng.uniform(0.5, 2.0)
        rh = int(min(H, max(16, (area_frac * H * W / aspect) ** 0.5)))
        rw = int(min(W, max(16, rh * aspect)))
        y1 = rng.randint(0, max(1, H - rh))
        x1 = rng.randint(0, max(1, W - rw))

        # Create local mask [B,1,H,W]
        mask = torch.zeros(B, 1, H, W, device=device)
        mask[:, :, y1:y1+rh, x1:x1+rw] = 1.0

        # Degrade specific channels using DegradationGenerator (via denorm→rgbt_c)
        degraded_01 = self._call_rgbt_c(
            rgb_01 if modality == 'rgb' else t_01,
            modality, corr_type, severity, mask, device)

        # Renorm back
        degraded_norm = (degraded_01 * 255.0 - mean[:, None, None].to(device)) / std[:, None, None].to(device)

        # Reconstruct 6ch
        if modality == 'rgb':
            out = torch.cat([degraded_norm, t], dim=1)
        else:
            out = torch.cat([rgb, degraded_norm], dim=1)
        return out

    def _call_rgbt_c(self, img_01, modality, corr_type, severity, mask, device):
        """Call the rgbt_c library to degrade img_01 [B,3,H,W] locally.
        get_corruption(name) returns a callable; apply with severity= kwarg,
        matching degradation.py's pattern."""
        import os, sys
        qwseg_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
        if qwseg_root not in sys.path:
            sys.path.insert(0, qwseg_root)
        from rgbt_c import get_corruption

        B, C, H, W = img_01.shape
        # Convert to numpy uint8 (rgbt_c works with numpy, per degradation.py)
        img_np = (img_01 * 255.0).clamp(0, 255).cpu().numpy().astype('uint8')
        degraded_np = img_np.copy()
        mask_np = mask.cpu().numpy().astype('uint8')

        corr = get_corruption(corr_type)
        if corr is not None:
            for b in range(B):
                img_b = degraded_np[b:b+1]  # [1,3,H,W]
                # severity is a kwarg to the callable, per degradation.py
                degraded_b = corr(img_b, severity=severity)
                # Only replace masked region
                m = mask_np[b:b+1]  # [1,1,H,W]
                degraded_np[b:b+1] = degraded_b * m + img_np[b:b+1] * (1 - m)

        degraded = torch.from_numpy(degraded_np.astype('float32')).to(device)
        return degraded / 255.0
