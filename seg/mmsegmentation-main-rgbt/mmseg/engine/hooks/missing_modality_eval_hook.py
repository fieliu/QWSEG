"""Hook for evaluating RGB/T missing modality scenarios during validation.

After each validation epoch, runs two extra evaluation passes:
1. RGB missing: replace RGB channels with zeros, keep T intact.
2. T missing: replace T channels with zeros, keep RGB intact.

Outputs per-class IoU/Acc tables and summary mIoU/mAcc/aAcc for each
missing scenario, using the same format as the standard IoUMetric.
"""

import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from prettytable import PrettyTable

from mmseg.registry import HOOKS


def _intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and union for a single image."""
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


def _compute_iou_table(area_intersect, area_union, area_label, class_names):
    """Compute per-class IoU/Acc and summary metrics."""
    iou = area_intersect / (area_union + 1e-10)
    acc = area_intersect / (area_label + 1e-10)

    valid = area_union > 0
    mIoU = iou[valid].mean().item() * 100 if valid.any() else 0.0
    mAcc = acc[valid].mean().item() * 100 if valid.any() else 0.0
    aAcc = (area_intersect.sum().item() /
            (area_label.sum().item() + 1e-10)) * 100

    table = PrettyTable()
    table.field_names = ['Class', 'IoU', 'Acc']
    for i, name in enumerate(class_names):
        iou_val = f'{iou[i].item() * 100:.2f}' if area_union[i] > 0 else 'N/A'
        acc_val = f'{acc[i].item() * 100:.2f}' if area_label[i] > 0 else 'N/A'
        table.add_row([name, iou_val, acc_val])

    return mIoU, mAcc, aAcc, table


@HOOKS.register_module()
class MissingModalityEvalHook(Hook):
    """Evaluate segmentation performance under RGB/T missing scenarios.

    After each validation epoch, iterates over the validation set twice
    (RGB-missing and T-missing) and prints per-class IoU/Acc tables.

    Args:
        interval (int): Evaluate every N epochs. Default: 1.
    """

    priority = 'LOW'

    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval

    def after_val_epoch(self, runner, metrics=None):
        epoch = runner.epoch
        if epoch % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        if not hasattr(model, 'predict_with_missing'):
            return

        # Skip during phase 1/2 (predictors frozen, quality scores all 1)
        epoch_val = getattr(model, 'current_epoch', epoch)
        if hasattr(model, '_get_training_phase'):
            ph = model._get_training_phase(epoch_val)
            if ph < 3:
                runner.logger.info(
                    'MissingModalityEvalHook: skipping (phase < 3)')
                return

        dataloader = runner.val_dataloader
        dataset_meta = getattr(dataloader.dataset, 'metainfo', {})
        class_names = dataset_meta.get('classes', [])
        if not class_names:
            runner.logger.warning(
                'MissingModalityEvalHook: no class names found, skipping')
            return
        num_classes = len(class_names)
        ignore_index = 255

        for scenario, mask_rgb, mask_t in [
            ('rgb_missing', True, False),
            ('thermal_missing', False, True),
        ]:
            area_intersect_total = torch.zeros(num_classes,
                                               dtype=torch.float64)
            area_union_total = torch.zeros(num_classes, dtype=torch.float64)
            area_label_total = torch.zeros(num_classes, dtype=torch.float64)

            model.eval()
            with torch.no_grad():
                for data_batch in dataloader:
                    # Use data_preprocessor to get normalized inputs
                    processed = model.data_preprocessor(data_batch, False)
                    inputs = processed['inputs']
                    data_samples = processed['data_samples']

                    if isinstance(inputs, (list, tuple)):
                        inputs = torch.stack(inputs)
                    inputs = inputs.to(next(model.parameters()).device)

                    # predict_with_missing handles postprocess (padding
                    # crop, resize to ori_shape) just like predict()
                    results = model.predict_with_missing(
                        inputs, data_samples,
                        mask_rgb=mask_rgb, mask_t=mask_t)

                    for ds in results:
                        pred_label = ds.pred_sem_seg.data.squeeze()
                        label = ds.gt_sem_seg.data.squeeze().to(pred_label)

                        ai, au, _, al = _intersect_and_union(
                            pred_label, label, num_classes, ignore_index)
                        area_intersect_total += ai.cpu().float()
                        area_union_total += au.cpu().float()
                        area_label_total += al.cpu().float()

            mIoU, mAcc, aAcc, table = _compute_iou_table(
                area_intersect_total, area_union_total,
                area_label_total, class_names)

            # Add summary rows
            table.add_row(['mIoU', f'{mIoU:.2f}', ''])
            table.add_row(['mAcc', '', f'{mAcc:.2f}'])
            table.add_row(['aAcc', '', f'{aAcc:.2f}'])

            print_log(f'\n{scenario}:', logger=runner.logger)
            print_log('\n' + table.get_string(), logger=runner.logger)
            runner.logger.info(
                f'{scenario} — mIoU: {mIoU:.2f}, mAcc: {mAcc:.2f}, '
                f'aAcc: {aAcc:.2f}')
