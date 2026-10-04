"""Generate deterministic RGB-T benchmarks, recording recipes and checksums.

Supports MFNet four-channel PNG or separate RGB/gray-T folders. Output always
uses four-channel PNG plus unchanged labels and a normalized split manifest.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt_c.io import (library_digest, pack_4ch, read_image, sha256_file,
                       unpack_4ch, write_png)
from rgbt_c.protocol import (PROTOCOL_VERSION, TRAIN_CORRUPTIONS,
                             HELDOUT_CORRUPTIONS, apply_pair, benchmark_cases,
                             benchmark_recipe, operation, recipe)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--split', default='test.txt')
    parser.add_argument('--images-dir', default='images')
    parser.add_argument('--rgb-dir', help='Use with --thermal-dir for separate images')
    parser.add_argument('--thermal-dir')
    parser.add_argument('--labels-dir', default='labels')
    parser.add_argument('--img-suffix', default='.png')
    parser.add_argument('--thermal-suffix', default='.png')
    parser.add_argument('--label-suffix', default='.png')
    parser.add_argument('--corruptions', nargs='+', default=['all'])
    parser.add_argument('--severities', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--scopes', nargs='+', default=['global'])
    parser.add_argument('--paired', nargs='*', default=[], help='E.g. low_light+t_gaussian_noise')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true', help='Regenerate the SAME protocol')
    return parser.parse_args()


def load_split(src_root, split_file, suffix='.png'):
    names = []
    for line in (Path(src_root) / split_file).read_text(encoding='utf-8-sig').splitlines():
        name = line.strip().replace('\\', '/')
        if not name:
            continue
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or not name:
            raise ValueError(f'Unsafe sample ID: {name}')
        names.append(name)
    if not names or len(set(names)) != len(names):
        raise ValueError('Split is empty or contains duplicate IDs')
    return names


def apply_corruption_to_4ch(img4, corruption_name, severity, seed=42):
    """Compatibility helper using the shared raw-pixel API."""
    rgb, thermal = unpack_4ch(img4)
    return pack_4ch(*apply_pair(rgb, thermal, recipe([operation(corruption_name, severity, seed)])))


def process_one(job):
    name, cfg, cases = job
    cv2.setNumThreads(1)
    src, dst = Path(cfg['src']), Path(cfg['dst'])
    if cfg['rgb_dir']:
        rgb_path = src / cfg['rgb_dir'] / (name + cfg['img_suffix'])
        thermal_path = src / cfg['thermal_dir'] / (name + cfg['thermal_suffix'])
        bgr, thermal = read_image(rgb_path), read_image(thermal_path)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f'Expected three-channel visible image: {rgb_path}')
        if thermal.ndim == 3:
            if thermal.shape[2] != 3 or not np.all(thermal == thermal[:, :, :1]):
                raise ValueError(f'Expected grayscale thermal, not pseudocolor: {thermal_path}')
            thermal = thermal[:, :, 0]
        rgb, thermal = bgr[:, :, ::-1].copy(), thermal[..., None]
        sources = {str(rgb_path.relative_to(src)): sha256_file(rgb_path),
                   str(thermal_path.relative_to(src)): sha256_file(thermal_path)}
    else:
        image_path = src / cfg['images_dir'] / (name + cfg['img_suffix'])
        rgb, thermal = unpack_4ch(read_image(image_path))
        sources = {str(image_path.relative_to(src)): sha256_file(image_path)}
    label_path = src / cfg['labels_dir'] / (name + cfg['label_suffix'])
    label_hash = sha256_file(label_path)
    target_label = dst / 'labels' / (name + cfg['label_suffix'])
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(label_path, target_label)
    records = []
    for case in cases:
        spec = benchmark_recipe(case, name, cfg['seed'])
        result = pack_4ch(*apply_pair(rgb, thermal, spec))
        relative = f"images/{case['id']}/{name}.png"
        output = dst / relative
        write_png(output, result)
        records.append(dict(sample=name, case=case['id'], image=relative,
                            source_sha256=sources, label_sha256=label_hash,
                            output_sha256=sha256_file(output), recipe=spec))
    return records


def main():
    args = parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Source and destination must be separate directory trees')
    if bool(args.rgb_dir) != bool(args.thermal_dir) or args.workers < 1:
        raise ValueError('Supply both paired directories; workers must be positive')
    for value in (args.split, args.images_dir, args.labels_dir, args.rgb_dir, args.thermal_dir):
        if value is not None and not (src / value).resolve().is_relative_to(src):
            raise ValueError(f'Path escapes source directory: {value}')
    names = load_split(src, args.split, args.img_suffix)
    cases = benchmark_cases(args.corruptions, args.severities, args.scopes, args.paired)
    cfg = vars(args).copy()
    cfg.update(src=str(src), dst=str(dst))
    identity = {k: v for k, v in cfg.items() if k not in ('workers', 'overwrite', 'dst')}
    identity.update(version=PROTOCOL_VERSION, library_sha256=library_digest(),
                    split_sha256=sha256_file(src / args.split), cases=cases,
                    numpy=np.__version__, opencv=cv2.__version__)
    manifest_path = dst / 'manifest.json'
    if dst.exists() and any(dst.iterdir()):
        if not args.overwrite or not manifest_path.exists():
            raise FileExistsError('Use a new destination, or --overwrite for the same protocol')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('Protocol/source/version changed; choose a new destination')
    dst.mkdir(parents=True, exist_ok=True)
    manifest = dict(identity=identity, cases=cases, n_samples=len(names),
                    split='split.txt', label_suffix=args.label_suffix,
                    train_corruptions=list(TRAIN_CORRUPTIONS),
                    heldout_corruptions=list(HELDOUT_CORRUPTIONS),
                    status='building', created_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (dst / 'split.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    jobs = ((name, cfg, cases) for name in names)
    print(f'Generating {len(names)} images x {len(cases)} conditions; {PROTOCOL_VERSION}', flush=True)
    def save_batch(records, stream):
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (dst / 'recipes.jsonl').open('w', encoding='utf-8') as stream:
        if args.workers == 1:
            for index, records in enumerate(map(process_one, jobs), 1):
                save_batch(records, stream)
                if index % 50 == 0:
                    print(f'{index}/{len(names)} source images', flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for index, records in enumerate(pool.map(process_one, jobs), 1):
                    save_batch(records, stream)
                    if index % 50 == 0:
                        print(f'{index}/{len(names)} source images', flush=True)
    manifest['status'] = 'complete'
    manifest['recipes_sha256'] = sha256_file(dst / 'recipes.jsonl')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Complete: {dst / "manifest.json"}')


if __name__ == '__main__':
    main()
