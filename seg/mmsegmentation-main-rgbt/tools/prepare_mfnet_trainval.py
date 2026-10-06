#!/usr/bin/env python3
"""Build MFNet trainval.txt and reject accidental train/test leakage."""
import argparse
from pathlib import Path


def read_ids(path: Path):
    ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
           if line.strip()]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f'{path}: {duplicates} duplicate entries')
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=Path,
                        help='MFNet root containing train.txt/val.txt/test.txt')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    train = read_ids(root / 'train.txt')
    val = read_ids(root / 'val.txt')
    test = read_ids(root / 'test.txt')

    overlap_train_val = set(train) & set(val)
    overlap_train_test = set(train) & set(test)
    overlap_val_test = set(val) & set(test)
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            'MFNet splits overlap: '
            f'train/val={len(overlap_train_val)}, '
            f'train/test={len(overlap_train_test)}, '
            f'val/test={len(overlap_val_test)}')

    trainval = train + val
    output = root / 'trainval.txt'
    output.write_text('\n'.join(trainval) + '\n', encoding='utf-8')
    print(f'train={len(train)} val={len(val)} trainval={len(trainval)} '
          f'test={len(test)}')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
