"""Hook for evaluating under PARTIAL (local) degradation during training.

Reuses the model's OWN DegradationGenerator (same code path as training).
Cost: ~50 images x 1 degradation = ~30s per eval cycle.
"""

import random
import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from prettytable import PrettyTable

from mmseg.registry import HOOKS


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


# Lightweight corruption menu — not the full 13, just for trend monitoring.
_DEGRADE_MENU = {
    'rgb': ['gaussian_noise', 'fog', 'low_light'],
    't':   ['t_gaussian_noise', 'stripe_noise', 't_defocus_blur'],
}


@HOOKS.register_module()
class PartialDegradeEvalHook(Hook):
    """Lightweight partial-degradation evaluation during training.

    Reuses the model's DegradationGenerator — EXACT same code path as training.
    Every `interval` epochs, runs the validation dataloader through the normal
    pipeline, applies a single on-the-fly degradation per image, and reports
    mIoU grouped by degraded modality and severity.

    Args:
        interval (int): Evaluate every N epochs. Default: 5.
        num_samples (int): Number of val images to evaluate. Default: 50.
        seed (int): Fixed seed for deterministic subset. Default: 42.
    """
    priority = 'LOW'

    def __init__(self, interval=5, num_samples=50, seed=42):
        super().__init__()
        self.interval = interval
        self.num_samples = num_samples
        self.seed = seed
        self._subset_indices = None

    def after_val_epoch(self, runner, metrics=None):
        epoch = runner.epoch
        if epoch % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Must have a DegradationGenerator
        if not hasattr(model, 'degrader'):
            runner.logger.warning(
                'PartialDegradeEvalHook: model has no degrader, skipping')
            return

        dataloader = runner.val_dataloader
        dataset = dataloader.dataset

        # class info
        dataset_meta = getattr(dataset, 'metainfo', {})
        class_names = dataset_meta.get('classes', [])
        if not class_names:
            return
        num_classes = len(class_names)
        ignore_index = 255

        # Fixed subset
        if self._subset_indices is None:
            rng = random.Random(self.seed)
            total = len(dataset)
            n = min(self.num_samples, total)
            self._subset_indices = sorted(rng.sample(range(total), n))
            print_log(
                f'PartialDegradeEvalHook: fixed subset {n}/{total} '
                f'images (seed={self.seed})', logger=runner.logger)

        # Denorm params from model's data_preprocessor
        pp = model.data_preprocessor
        if hasattr(pp, 'mean') and hasattr(pp, 'std'):
            mean = pp.mean.flatten()
            std = pp.std.flatten()
        else:
            mean = torch.tensor([123.675, 116.28, 103.53, 123.675, 116.28, 103.53])
            std = torch.tensor([58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

        model.eval()
        with torch.no_grad():
            for modality, corr_types in _DEGRADE_MENU.items():
                for corr in corr_types:
                    self._eval_one(
                        runner, model, dataset, class_names, num_classes,
                        ignore_index, mean, std, modality, corr, epoch)

    def _eval_one(self, runner, model, dataset, class_names,
                   num_classes, ignore_index, mean, std, modality, corr, epoch):
        area_intersect = torch.zeros((5, num_classes), dtype=torch.float64)
        area_union     = torch.zeros((5, num_classes), dtype=torch.float64)
        device = next(model.parameters()).device
        rng = random.Random(self.seed + epoch)

        for idx in self._subset_indices:
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                img = sample[0]
                ds  = sample[1]
            elif isinstance(sample, dict):
                img = sample.get('inputs', sample)
                ds  = sample.get('data_samples', None)
            else:
                continue

            if not isinstance(img, torch.Tensor):
                continue
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(device)

            # === SAME pipeline as training: degrader + forward ===
            # Split 6ch into rgb/t, apply degradation using model's own degrader
            rgb, t = img[:, :3], img[:, 3:]
            epoch_val = getattr(model, 'current_epoch', epoch)

            # Use make_paired to get light/heavy versions (same as training)
            # We only need ONE degraded version for eval — take the heavy one.
            try:
                (l_rgb, l_ir, h_rgb, h_ir,
                 lvl_light_rgb, lvl_light_ir,
                 lvl_heavy_rgb, lvl_heavy_ir,
                 rank_mask, level_gap) = model.degrader.make_paired(
                    rgb, t, mean, std, epoch=epoch_val)
            except Exception:
                # degrader not available or make_paired failed — skip
                continue

            # Use heavy version as the degraded input for this eval
            degraded = torch.cat([h_rgb, h_ir], dim=1)  # [B,6,H,W]

            # Forward through standard pipeline (same as training val)
            batch = {'inputs': [degraded[0]], 'data_samples': [ds] if ds is not None else []}
            processed = model.data_preprocessor(batch, False)
            inputs = processed['inputs']
            if isinstance(inputs, (list, tuple)):
                inputs = torch.stack(inputs)

            try:
                results = model.predict(inputs, processed.get('data_samples', []))
            except Exception:
                continue
            if results is None:
                continue

            # Get severity from level map (1-5 in degraded region)
            if modality == 'rgb':
                sev_map = lvl_heavy_rgb  # [B,1,H,W] long
            else:
                sev_map = lvl_heavy_ir

            sev = int(sev_map[sev_map > 0].float().mean().round().clamp(1, 5).item()) \
                if (sev_map > 0).any() else 3
            sev_idx = sev - 1

            # Get GT label
            if ds is not None and hasattr(ds, 'gt_sem_seg') and ds.gt_sem_seg is not None:
                label = ds.gt_sem_seg.data.squeeze().to(device)
            else:
                continue

            for r in results:
                pred_label = r.pred_sem_seg.data.squeeze()
                ai, au, _, _ = _intersect_and_union(
                    pred_label, label, num_classes, ignore_index)
                area_intersect[sev_idx] += ai.cpu().float()
                area_union[sev_idx]     += au.cpu().float()

        # Report
        header = f'{modality}/{corr}'
        print_log(f'\n{header}:', logger=runner.logger)
        table = PrettyTable()
        table.field_names = ['Severity', 'mIoU']
        overall = 0.0; count = 0
        for sev_ix, sev in enumerate([1, 2, 3, 4, 5]):
            miou = _compute_miou(area_intersect[sev_ix], area_union[sev_ix])
            table.add_row([sev, f'{miou:.2f}'])
            if area_union[sev_ix].sum() > 0:
                overall += miou; count += 1
        if count > 0:
            overall /= count
            table.add_row(['avg', f'{overall:.2f}'])

        print_log('\n' + table.get_string(), logger=runner.logger)
        runner.logger.info(f'{header} avg(sev1-5) mIoU: {overall:.2f}')
