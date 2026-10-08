"""Full RGBT-C benchmark for a trained DINO-TS checkpoint.

Evaluates the complete case set on the whole test split and writes a JSON
summary.  The recipe seed is FIXED at 42 so that the corrupted images are
byte-identical across checkpoints, across runs, and across machines -- the
same seed is what makes this experiment's benchmark reproducible and what a
later offline test-set generation should reuse.

Case set:
  clean                     the unmodified test images
  seen/{op}/{sev}           the 8 operators x 5 severities the training policy
                            samples from
  heldout/{op}/{sev}        fog, stripe_noise, t_quantization -- deliberately
                            excluded from training, so these measure unseen-
                            corruption generalisation
  missing/{rgb,t}_missing   whole-modality missing

All cases use the global spatial scope.  The image count is never reduced:
guardrail occurs in only 4 MFNet test images, bump in 17 and color_cone in 61,
and any class absent from a subset is dropped from the mean, which would make
the number incomparable to the 9-class headline mIoU.

Usage:
    python tools/benchmark_rgbt_c_full.py <config> <checkpoint> <out.json>
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import DATASETS, MODELS

from rgbt_c.protocol import benchmark_cases, benchmark_recipe
from rgbt_c.torch_adapter import RGBTDegrader

# Fixed for the whole experiment.  Do not change between checkpoints.
RECIPE_SEED = 42

GROUPS = (
    ('seen', dict(names=['seen'], severities=(1, 2, 3, 4, 5),
                  scopes=('global',))),
    ('heldout', dict(names=['heldout'], severities=(1, 2, 3, 4, 5),
                     scopes=('global',))),
    ('missing', dict(names=['missing'], severities=(1,), scopes=('global',))),
)


def build_cases():
    """All benchmark cases, deduplicated by id across groups."""
    pool = {}
    for _, kwargs in GROUPS:
        for case in benchmark_cases(**kwargs):
            pool[case['id']] = case
    return list(pool.values())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=None,
                    help='debug only; changes the class denominator')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope('mmseg')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, strict=False, map_location='cpu')
    model = model.to('cuda:0').eval()

    dataset = DATASETS.build(cfg.val_dataloader['dataset'])
    preprocessor = model.data_preprocessor
    degrader = RGBTDegrader(preprocessor.mean.flatten(),
                            preprocessor.std.flatten())
    num_classes = len(dataset.metainfo['classes'])
    ignore_index = getattr(dataset, 'ignore_index', 255)
    device = next(model.parameters()).device

    root = Path(dataset.data_prefix['img_path'])
    if not root.is_absolute():
        root = Path(dataset.data_root) / root

    cases = build_cases()
    n_img = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f'checkpoint : {args.checkpoint}')
    print(f'dataset    : {n_img} images, {num_classes} classes')
    print(f'cases      : {len(cases)}')
    print(f'seed       : {RECIPE_SEED} (fixed)')
    print(f'total      : {n_img * len(cases)} forward passes', flush=True)

    ids = [c['id'] for c in cases]
    inter = {i: np.zeros(num_classes, dtype=np.float64) for i in ids}
    union = {i: np.zeros(num_classes, dtype=np.float64) for i in ids}
    present = {i: 0 for i in ids}

    t0 = time.time()
    for index in range(n_img):
        sample = dataset[index]
        ds = sample['data_samples']
        processed = preprocessor(
            dict(inputs=[sample['inputs']], data_samples=[ds]), False)
        inputs = processed['inputs']
        shapes = [d.img_shape for d in processed['data_samples']]
        image_id = (Path(dataset.get_data_info(index)['img_path'])
                    .relative_to(root).with_suffix('').as_posix())
        target = ds.gt_sem_seg.data.squeeze().long()

        for case in cases:
            spec = benchmark_recipe(case, image_id, RECIPE_SEED)
            if case['names']:
                r, t, _, _ = degrader.apply(inputs[:, :3], inputs[:, 3:],
                                            [spec], valid_shapes=shapes)
                x = torch.cat([r, t], dim=1).to(device)
            else:
                x = inputs.to(device)
            pred = model.predict(x, processed['data_samples'])[0] \
                .pred_sem_seg.data.squeeze().cpu().long()
            if pred.shape != target.shape:
                raise ValueError(f'Prediction/GT size mismatch for {case["id"]}: '
                                 f'{tuple(pred.shape)} vs {tuple(target.shape)}')
            valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
            tg, pr = target[valid], pred[valid]
            hits = np.bincount(tg[pr == tg].numpy(), minlength=num_classes)
            inter[case['id']] += hits
            union[case['id']] += (np.bincount(tg.numpy(), minlength=num_classes)
                                  + np.bincount(pr.numpy(), minlength=num_classes)
                                  - hits)

        if (index + 1) % 25 == 0 or index + 1 == n_img:
            el = time.time() - t0
            print(f'  {index + 1}/{n_img} images  {el:.0f}s  '
                  f'eta {el / (index + 1) * (n_img - index - 1) / 60:.1f} min',
                  flush=True)

    per_case = {}
    for case in cases:
        u = union[case['id']]
        valid = u > 0
        present[case['id']] = int(valid.sum())
        if int(valid.sum()) != num_classes:
            print(f'  WARNING {case["id"]}: only {int(valid.sum())}/'
                  f'{num_classes} classes present')
        per_case[case['id']] = float((inter[case['id']][valid] / u[valid]).mean() * 100)

    # ---- grouping ----
    def mean_of(pred):
        vals = [v for k, v in per_case.items() if pred(k)]
        return float(np.mean(vals)) if vals else None

    summary = {
        'checkpoint': str(args.checkpoint),
        'config': str(args.config),
        'recipe_seed': RECIPE_SEED,
        'n_images': n_img,
        'num_classes': num_classes,
        'classes': list(dataset.metainfo['classes']),
        'per_case': per_case,
        'per_class_present': present,
        'groups': {
            'clean': per_case.get('clean'),
            'seen_macro': mean_of(lambda k: k.startswith(('gaussian_noise/',
                                                          'shot_noise/',
                                                          'motion_blur/',
                                                          'defocus_blur/',
                                                          'low_light/',
                                                          't_gaussian_noise/',
                                                          't_motion_blur/',
                                                          't_defocus_blur/'))),
            'heldout_macro': mean_of(lambda k: k in (
                'fog', 'stripe_noise', 't_quantization') or
                k.split('/')[0] in ('fog', 'stripe_noise', 't_quantization')),
            'missing_macro': mean_of(lambda k: 'missing' in k),
        },
    }
    for op in sorted({k.split('/')[0] for k in per_case if '/' in k}):
        vals = [v for k, v in per_case.items() if k.startswith(op + '/')]
        summary.setdefault('per_operator', {})[op] = float(np.mean(vals))
    summary['corrupted_macro'] = float(np.mean(
        [v for k, v in per_case.items() if k != 'clean']))

    Path(args.out).write_text(json.dumps(summary, indent=2))

    print('\n===== SUMMARY =====')
    print(f'  clean            : {summary["groups"]["clean"]:.2f}')
    print(f'  corrupted macro  : {summary["corrupted_macro"]:.2f}')
    print(f'  seen macro       : {summary["groups"]["seen_macro"]:.2f}')
    print(f'  heldout macro    : {summary["groups"]["heldout_macro"]:.2f}')
    print(f'  missing macro    : {summary["groups"]["missing_macro"]:.2f}')
    print('\n  per operator:')
    for op, v in sorted(summary['per_operator'].items()):
        tag = '  (heldout)' if op in ('fog', 'stripe_noise', 't_quantization') else ''
        print(f'    {op:20s} {v:6.2f}{tag}')
    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
