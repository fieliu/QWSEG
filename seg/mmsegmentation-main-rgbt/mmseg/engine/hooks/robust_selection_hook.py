"""Proxy robustness evaluation for robustness-aware checkpoint selection.

Stage 2 deliberately trades clean accuracy for robustness, but the stock
CheckpointHook selects on clean ``mIoU`` -- i.e. it picks the *least*
robustified checkpoint.  This hook runs a small, deterministic corrupted
benchmark after each validation and injects a composite score into the
validation ``metrics`` dict, so the ordinary CheckpointHook can select on it by
setting ``save_best='robust_score'``.

Why a proxy instead of the full benchmark
-----------------------------------------
The full benchmark (8 training operators x 5 severities + 2 missing cases, at
three spatial scopes) is ~121 cases; over the 393 test images that is roughly
30 minutes per evaluation, which is unaffordable every 5 epochs.

The image count is NOT reduced.  Reducing images would drop the rare classes --
``guardrail`` occurs in only 4 test images, ``bump`` in 17, ``color_cone`` in
61 -- and any class absent from a subset is silently removed from the mean,
turning the 9-class metric into a smaller-class one that is not comparable to
the headline number.  Only the number of *cases* is cut.

Determinism
-----------
``benchmark_recipe(case, image_id, seed)`` derives every operator seed from a
hash of (protocol version, seed, image id, case id, operator name), so a fixed
seed reproduces byte-identical corruptions on every evaluation and on every
rerun.  Nothing is written to disk: corruptions are regenerated on the fly.

Selection score
---------------
``robust_score = clean_weight * clean_mIoU + (1 - clean_weight) * corrupted_mIoU``

Clean mIoU is taken from the validation metrics the main val loop already
computed, so it costs nothing extra and is exactly the headline number.  Giving
clean performance half the weight keeps the "do not visibly sacrifice clean
accuracy" requirement (handoff step 4) inside the objective; a plain mean over
cases would give clean only 1/N of the weight and effectively ignore it.
"""
from pathlib import Path
from typing import Optional, Sequence

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmseg.registry import HOOKS

DEFAULT_CASES = (
    'gaussian_noise/3',      # RGB degradation
    't_gaussian_noise/3',    # thermal degradation
    'rgb_missing/1',         # whole-modality missing
    't_missing/1',
)


@HOOKS.register_module()
class RobustSelectionHook(Hook):
    """Evaluate a fixed corrupted proxy set and inject a selection score.

    Args:
        cases (Sequence[str]): ``benchmark_cases`` ids to evaluate.  The clean
            case is deliberately not listed: clean mIoU comes from the main
            validation metrics.
        seed (int): Seed for ``benchmark_recipe``.  Fixed so that every
            evaluation sees the identical corrupted images.
        clean_weight (float): Weight on clean mIoU in the composite score.
        key (str): Key injected into the validation metrics dict.
        max_samples (int, optional): Limit the number of images.  For smoke
            tests only -- leaving classes out changes the metric's meaning.
    """

    # CheckpointHook is 'VERY_LOW' and reads metrics[save_best] in its own
    # after_val_epoch, so this hook has to run first.
    priority = 'NORMAL'

    def __init__(self,
                 cases: Sequence[str] = DEFAULT_CASES,
                 seed: int = 42,
                 clean_weight: float = 0.5,
                 key: str = 'robust_score',
                 max_samples: Optional[int] = None) -> None:
        if not 0.0 <= clean_weight <= 1.0:
            raise ValueError('clean_weight must be in [0, 1]')
        self.case_ids = tuple(cases)
        self.seed = int(seed)
        self.clean_weight = float(clean_weight)
        self.key = key
        self.max_samples = max_samples
        self._cases = None
        self._degrader = None

    # -- helpers -----------------------------------------------------------
    def _resolve_cases(self):
        if self._cases is not None:
            return self._cases
        from rgbt_c.protocol import benchmark_cases
        pool = {}
        for names in (['seen'], ['missing']):
            for case in benchmark_cases(names, severities=(1, 2, 3, 4, 5),
                                        scopes=('global',)):
                pool[case['id']] = case
        unknown = [c for c in self.case_ids if c not in pool]
        if unknown:
            raise KeyError(f'Unknown benchmark case ids: {unknown}. '
                           f'Available examples: {sorted(pool)[:8]}')
        self._cases = [pool[c] for c in self.case_ids]
        return self._cases

    def _get_degrader(self, preprocessor):
        if self._degrader is None:
            from rgbt_c.torch_adapter import RGBTDegrader
            self._degrader = RGBTDegrader(preprocessor.mean.flatten(),
                                          preprocessor.std.flatten())
        return self._degrader

    @staticmethod
    def _accumulate(inter, union, pred, target, num_classes, ignore_index):
        valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
        target, pred = target[valid], pred[valid]
        if target.numel() == 0:
            return
        if ((pred < 0) | (pred >= num_classes)).any():
            raise ValueError('Predicted class outside the label space')
        hits = torch.bincount(target[pred == target],
                              minlength=num_classes).double()
        inter += hits
        union += (torch.bincount(target, minlength=num_classes).double()
                  + torch.bincount(pred, minlength=num_classes).double() - hits)

    # -- main --------------------------------------------------------------
    @torch.no_grad()
    def after_val_epoch(self, runner, metrics=None) -> None:
        if not is_main_process() or metrics is None:
            return
        if 'mIoU' not in metrics:
            print_log('RobustSelectionHook: no mIoU in the validation metrics; '
                      'skipping.', logger='current')
            return

        model = runner.model
        model = model.module if hasattr(model, 'module') else model
        was_training = model.training
        model.eval()
        try:
            per_case = self._evaluate(runner, model)
        finally:
            model.train(was_training)

        clean_miou = float(metrics['mIoU'])
        corrupted_miou = sum(per_case.values()) / len(per_case)
        score = (self.clean_weight * clean_miou
                 + (1.0 - self.clean_weight) * corrupted_miou)

        metrics[self.key] = score
        metrics['robust/clean_mIoU'] = clean_miou
        metrics['robust/corrupted_mIoU'] = corrupted_miou
        for case_id, value in per_case.items():
            metrics['robust/' + case_id] = value

        detail = '  '.join(f'{k}={v:.2f}' for k, v in per_case.items())
        print_log(f'RobustSelectionHook: clean={clean_miou:.2f} '
                  f'corrupted={corrupted_miou:.2f} -> {self.key}={score:.2f}  '
                  f'| {detail}', logger='current')

    def _evaluate(self, runner, model):
        from rgbt_c.protocol import benchmark_recipe

        cases = self._resolve_cases()
        preprocessor = model.data_preprocessor
        degrader = self._get_degrader(preprocessor)

        dataset = runner.val_dataloader.dataset
        num_classes = len(dataset.metainfo['classes'])
        ignore_index = getattr(dataset, 'ignore_index', 255)
        device = next(model.parameters()).device

        # data_prefix may already have been joined with data_root.
        root = Path(dataset.data_prefix['img_path'])
        if not root.is_absolute():
            root = Path(dataset.data_root) / root

        n = len(dataset)
        if self.max_samples is not None:
            n = min(self.max_samples, n)

        ids = [c['id'] for c in cases]
        inter = {i: torch.zeros(num_classes, dtype=torch.float64) for i in ids}
        union = {i: torch.zeros(num_classes, dtype=torch.float64) for i in ids}

        for index in range(n):
            sample = dataset[index]
            if not isinstance(sample, dict) or 'data_samples' not in sample:
                raise ValueError('RobustSelectionHook needs a plain '
                                 'segmentation dataset')
            ds = sample['data_samples']
            processed = preprocessor(
                dict(inputs=[sample['inputs']], data_samples=[ds]), False)
            inputs = processed['inputs']
            shapes = [d.img_shape for d in processed['data_samples']]
            info = dataset.get_data_info(index)
            image_id = (Path(info['img_path']).relative_to(root)
                        .with_suffix('').as_posix())
            target = ds.gt_sem_seg.data.squeeze().long()

            for case in cases:
                spec = benchmark_recipe(case, image_id, self.seed)
                r, t, _, _ = degrader.apply(inputs[:, :3], inputs[:, 3:],
                                            [spec], valid_shapes=shapes)
                x = torch.cat([r, t], dim=1).to(device)
                preds = model.predict(x, processed['data_samples'])
                pred = preds[0].pred_sem_seg.data.squeeze().cpu().long()
                if pred.shape != target.shape:
                    raise ValueError(
                        'Prediction/GT size mismatch '
                        f'({tuple(pred.shape)} vs {tuple(target.shape)})')
                self._accumulate(inter[case['id']], union[case['id']],
                                 pred, target, num_classes, ignore_index)

        per_case = {}
        for case in cases:
            u = union[case['id']]
            valid = u > 0
            if not valid.any():
                raise ValueError(f'No labelled pixels for case {case["id"]}')
            # With the full test set every class is present, so this mean is
            # over all num_classes entries and matches the headline metric.
            if int(valid.sum()) != num_classes:
                print_log(
                    f'RobustSelectionHook: WARNING case {case["id"]} has only '
                    f'{int(valid.sum())}/{num_classes} classes present; its '
                    'mIoU is not comparable to the 9-class headline metric.',
                    logger='current')
            per_case[case['id']] = float(
                (inter[case['id']][valid] / u[valid]).mean() * 100)
        return per_case
