"""Evaluate exactly the conditions recorded in an RGBT-C v2 manifest.

mIoU is always reported in percent (0..100). Missing-modality cases are separate
from ordinary corruption averages. mCE needs a matched, explicit baseline.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt_c.io import sha256_file
from rgbt_c.protocol import PROTOCOL_VERSION, MISSING


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_manifest(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['identity']['version'] != PROTOCOL_VERSION:
        raise ValueError('Incomplete or unsupported benchmark; regenerate with v2')
    if sha256_file(root / 'recipes.jsonl') != manifest['recipes_sha256']:
        raise ValueError('Benchmark recipe index checksum mismatch')
    if not (root / manifest['split']).is_file():
        raise FileNotFoundError('Benchmark split missing')
    return manifest


def extract_miou(metrics):
    matches = [float(v) for k, v in metrics.items()
               if k == 'mIoU' or k.endswith('/mIoU')]
    if len(matches) != 1 or not 0 <= matches[0] <= 100:
        raise ValueError(f'Expected one percent mIoU metric, got {metrics}')
    return matches[0]


def summarize(results, cases, baseline=None):
    expected = {c['id'] for c in cases}
    if set(results) != expected:
        raise ValueError(f'Incomplete/mismatched result set: {expected.symmetric_difference(results)}')
    values = {k: float(v) for k, v in results.items()}
    if any(not 0 <= v <= 100 for v in values.values()):
        raise ValueError('mIoU must be finite percent values in [0, 100]')
    if baseline is not None:
        if set(baseline) != expected or any(not 0 <= float(v) <= 100 for v in baseline.values()):
            raise ValueError('Baseline must have the same conditions and percent units')
    per_type = {}
    grouped = {}
    missing = {}
    ce = []
    for case in cases:
        if case['group'] == 'clean':
            continue
        key = case['name']
        if case['group'] == 'missing':
            missing[case['id']] = values[case['id']]
            continue
        per_type.setdefault(key, []).append(values[case['id']])
        grouped.setdefault(case['scope'] + '/' + case['group'], {}).setdefault(key, []).append(values[case['id']])
    def mean(xs):
        return sum(xs) / len(xs) if xs else None
    means = {key: mean(xs) for key, xs in per_type.items()}
    if baseline is not None:
        for name in per_type:
            ids = [c['id'] for c in cases if c['name'] == name]
            denominator = sum(100-float(baseline[key]) for key in ids)
            if denominator <= 0:
                raise ValueError('mCE is undefined for a perfect baseline condition')
            ce.append(sum(100-values[key] for key in ids) / denominator)
    corrupt_mean = mean(list(means.values()))
    return dict(unit='percent', clean_mIoU=values['clean'],
                corruption_mIoU=corrupt_mean,
                clean_minus_corruption=(values['clean']-corrupt_mean) if corrupt_mean is not None else None,
                per_corruption_mIoU=means, missing_mIoU=missing,
                grouped_mIoU={k: mean([mean(xs) for xs in groups.values()])
                              for k, groups in grouped.items()},
                mCE_ratio=mean(ce), n_conditions=len(values))


def configure_dataset(cfg, root, manifest, case):
    dataset = cfg.test_dataloader.dataset
    if 'dataset' in dataset or 'datasets' in dataset:
        raise ValueError('Use an unwrapped, four-channel segmentation test dataset')
    dataset.data_root = str(Path(root).resolve())
    dataset.data_prefix = dict(img_path='images/' + case['id'], seg_map_path='labels')
    dataset.ann_file = manifest['split']
    dataset.img_suffix = '.png'
    dataset.seg_map_suffix = manifest['label_suffix']
    cfg.test_dataloader.sampler = dict(type='DefaultSampler', shuffle=False)
    # Reuse model setup and resizing, but load the benchmark's four-channel PNG.
    pipeline = dataset.pipeline
    if not pipeline or pipeline[0].get('type') not in (
            'LoadRGBTImageFrom4Channel', 'LoadRGBTImageFromFile', 'LoadImageFromFile'):
        raise ValueError('Unsupported image pipeline; adapt it explicitly')
    pipeline[0] = dict(type='LoadRGBTImageFrom4Channel')
    return cfg


def run_benchmark(config, checkpoint, root, manifest, work_dir):
    from mmengine.config import Config
    from mmengine.runner import Runner
    # Models/modules registered by custom_imports are imported by Runner.
    cfg = Config.fromfile(config)
    cfg.load_from = str(Path(checkpoint).resolve())
    cfg.resume = False
    cfg.launcher = 'none'
    cfg.work_dir = str(Path(work_dir).resolve())
    cfg.custom_hooks = []  # Evaluation must not execute training monitor hooks.
    configure_dataset(cfg, root, manifest, manifest['cases'][0])
    runner = Runner.from_cfg(cfg)
    results = {}
    for case in manifest['cases']:
        configure_dataset(cfg, root, manifest, case)
        loader = runner.build_dataloader(copy.deepcopy(cfg.test_dataloader), seed=42)
        loop = runner.test_loop
        loop.dataloader = loader
        loop.evaluator.dataset_meta = loader.dataset.metainfo
        runner.visualizer.dataset_meta = loader.dataset.metainfo
        results[case['id']] = extract_miou(runner.test())
        out = Path(work_dir) / 'conditions' / case['id'] / 'result.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(case=case['id'], mIoU=results[case['id']],
                                       unit='percent')), encoding='utf-8')
        print(f"{case['id']}: {results[case['id']]:.2f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--results-json', help='Re-summarize a full result JSON from this script')
    parser.add_argument('--baseline-results')
    parser.add_argument('--work-dir', default='work_dirs/rgbtc_eval')
    parser.add_argument('--output', default='work_dirs/rgbtc_eval/results.json')
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    manifest = load_manifest(root)
    digest = identity_hash(manifest['identity'])
    # Record both protocol identity and actual generated images/index identity.
    benchmark_id = digest + ':' + manifest['recipes_sha256']
    if args.results_json:
        data = json.loads(Path(args.results_json).read_text(encoding='utf-8'))
        if data['benchmark_id'] != benchmark_id or data['unit'] != 'percent':
            raise ValueError('Results belong to a different benchmark or unit')
        results = data['results']
    elif args.config and args.checkpoint:
        results = run_benchmark(args.config, args.checkpoint, root, manifest, args.work_dir)
        data = dict(config=str(Path(args.config).resolve()),
                    checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_sha256=sha256_file(args.checkpoint))
    else:
        parser.error('Supply --config and --checkpoint, or --results-json')
    baseline = None
    if args.baseline_results:
        reference = json.loads(Path(args.baseline_results).read_text(encoding='utf-8'))
        if reference['benchmark_id'] != benchmark_id or reference['unit'] != 'percent':
            raise ValueError('Baseline does not match this benchmark')
        baseline = reference['results']
    metrics = summarize(results, manifest['cases'], baseline)
    data.update(benchmark_id=benchmark_id, unit='percent', results=results, metrics=metrics)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
