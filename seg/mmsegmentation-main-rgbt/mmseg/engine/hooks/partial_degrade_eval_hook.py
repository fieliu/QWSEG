"""Deterministic validation monitor using the same recipes as offline tests.

This is a small validation monitor, not the final full benchmark. Errors are
reported, never silently skipped. It does not select checkpoints on test data.
"""
import random
import sys
from pathlib import Path

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

_ROOT = str(Path(__file__).resolve().parents[5])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from rgbt_c.protocol import benchmark_cases, benchmark_recipe
from rgbt_c.torch_adapter import RGBTDegrader


@HOOKS.register_module()
class PartialDegradeEvalHook(Hook):
    priority = 'LOW'

    def __init__(self, interval=5, num_samples=50, seed=42,
                 corruptions=('gaussian_noise', 't_gaussian_noise'),
                 severities=(3,), scope='local'):
        if interval < 1 or num_samples < 1:
            raise ValueError('interval and num_samples must be positive')
        self.interval = interval
        self.num_samples = num_samples
        self.seed = seed
        self.cases = benchmark_cases(corruptions, severities, (scope,))

    @torch.no_grad()
    def after_val_epoch(self, runner, metrics=None):
        if runner.epoch % self.interval or not is_main_process():
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        dataset = runner.val_dataloader.dataset
        num_classes = len(dataset.metainfo['classes'])
        ignore_index = getattr(dataset, 'ignore_index', 255)
        ids = sorted(random.Random(self.seed).sample(
            range(len(dataset)), min(self.num_samples, len(dataset))))
        pp = model.data_preprocessor
        degrader = RGBTDegrader(pp.mean.flatten(), pp.std.flatten())
        was_training = model.training
        model.eval()
        try:
            for case in self.cases:
                intersection = torch.zeros(num_classes, dtype=torch.float64)
                union = torch.zeros_like(intersection)
                for index in ids:
                    sample = dataset[index]
                    if not isinstance(sample, dict) or 'data_samples' not in sample:
                        raise ValueError('Monitor requires a plain segmentation dataset')
                    ds = sample['data_samples']
                    processed = pp(dict(inputs=[sample['inputs']], data_samples=[ds]), False)
                    inputs = processed['inputs']
                    info = dataset.get_data_info(index)
                    # Match offline sample IDs (relative image path without suffix).
                    root = Path(dataset.data_prefix['img_path'])
                    image_id = Path(info['img_path']).relative_to(root).with_suffix('').as_posix()
                    spec = benchmark_recipe(case, image_id, self.seed)
                    rgb, thermal, _, _ = degrader.apply(
                        inputs[:, :3], inputs[:, 3:], [spec],
                        valid_shapes=[ds.img_shape for ds in processed['data_samples']])
                    predictions = model.predict(torch.cat([rgb, thermal], 1),
                                                processed['data_samples'])
                    pred = predictions[0].pred_sem_seg.data.squeeze().cpu().long()
                    target = ds.gt_sem_seg.data.squeeze().cpu().long()
                    if pred.shape != target.shape:
                        raise ValueError('Prediction/GT size mismatch in corruption monitor')
                    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
                    target, pred = target[valid], pred[valid]
                    if ((pred < 0) | (pred >= num_classes)).any():
                        raise ValueError('Predicted class outside label space')
                    hits = torch.bincount(target[pred == target], minlength=num_classes).double()
                    intersection += hits
                    union += (torch.bincount(target, minlength=num_classes)
                              + torch.bincount(pred, minlength=num_classes)).double() - hits
                valid = union > 0
                if not valid.any():
                    raise ValueError('Monitor has no valid labeled pixels')
                miou = float((intersection[valid] / union[valid]).mean() * 100)
                runner.logger.info(
                    f"RGBT-C validation monitor {case['id']}: mIoU={miou:.2f}, "
                    f"n={len(ids)}, seed={self.seed}")
        finally:
            model.train(was_training)
