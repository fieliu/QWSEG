"""生成 RGBT-C 测试数据集 (MFNet-C / FMB-C / PST900-C).

设计依据: docs/RGBT-C_Benchmark.md v1.0

用法:
    # 生成 MFNet-C 全部 12 种退化 × 5 级
    python tools/generate_rgbt_c.py \\
        --src /home/lh/code/data/MFNet \\
        --dst /home/lh/code/data/MFNet-C \\
        --split test.txt \\
        --corruptions all \\
        --severities 1 2 3 4 5 \\
        --workers 8

    # 只生成 RGB fog 退化
    python tools/generate_rgbt_c.py \\
        --src /home/lh/code/data/MFNet \\
        --dst /home/lh/code/data/MFNet-C \\
        --corruptions fog \\
        --severities 3

输出目录结构 (见 docs/RGBT-C_Benchmark.md §5.1):
    {dst}/images/{corruption_name}/{severity}/{filename}.png
    {dst}/labels/  (软链接到原 labels)
    {dst}/test.txt (复制原 split 文件)

注意:
    - 输入图像为 4 通道 PNG (RGB + T), 与 MFNet 原始格式一致
    - 单模态退化: 只退化指定模态, 另一模态保持原始
    - RGB 退化作用于通道 0-2, T 退化作用于通道 3
    - 使用固定随机种子保证可复现
"""
import argparse
import os
import sys
import shutil
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt_c import (
    get_corruption, RGB_CORRUPTIONS, T_CORRUPTIONS, ALL_CORRUPTIONS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate RGBT-C corruption benchmark dataset')
    parser.add_argument('--src', type=str, required=True,
                        help='Source dataset root (e.g. /home/lh/code/data/MFNet)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination dataset root (e.g. /home/lh/code/data/MFNet-C)')
    parser.add_argument('--split', type=str, default='test.txt',
                        help='Split file name in src (default: test.txt)')
    parser.add_argument('--images-dir', type=str, default='images',
                        help='Images subdirectory name in src (default: images)')
    parser.add_argument('--labels-dir', type=str, default='labels',
                        help='Labels subdirectory name in src (default: labels)')
    parser.add_argument('--img-suffix', type=str, default='.png',
                        help='Image file suffix (default: .png)')
    parser.add_argument('--corruptions', type=str, nargs='+', default=['all'],
                        help='Corruption names (default: all). '
                             'Use "all" / "rgb" / "t" for groups')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Severity levels 1-5 (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing files')
    return parser.parse_args()


def resolve_corruptions(names):
    """解析退化名称列表."""
    result = []
    for n in names:
        if n == 'all':
            result.extend(ALL_CORRUPTIONS)
        elif n == 'rgb':
            result.extend(RGB_CORRUPTIONS)
        elif n == 't':
            result.extend(T_CORRUPTIONS)
        else:
            result.append(n)
    # 去重保持顺序
    seen = set()
    out = []
    for n in result:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def load_split(src_root, split_file):
    """读取 split 文件, 返回图像名列表 (无扩展名)."""
    split_path = os.path.join(src_root, split_file)
    with open(split_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def apply_corruption_to_4ch(img4, corruption_name, severity, seed=None):
    """对 4 通道图像 (RGB+T) 施加单模态退化.

    Args:
        img4: (H, W, 4) uint8, 通道 0-2=RGB, 通道 3=T
        corruption_name: 退化名称
        severity: 1-5
        seed: 随机种子 (按图像+退化+级别组合, 保证可复现)
    Returns:
        (H, W, 4) uint8
    """
    if seed is not None:
        np.random.seed(seed)

    rgb = img4[:, :, :3]
    t = img4[:, :, 3:4]

    corr = get_corruption(corruption_name)

    if corruption_name in RGB_CORRUPTIONS:
        # RGB 退化, T 保持
        rgb_c = corr(rgb, severity=severity)
        return np.concatenate([rgb_c, t], axis=2)
    elif corruption_name in T_CORRUPTIONS:
        # T 退化, RGB 保持
        t_c = corr(t, severity=severity)
        return np.concatenate([rgb, t_c], axis=2)
    else:
        raise ValueError(f'Unknown corruption: {corruption_name}')


def process_one(args_tuple):
    """处理单张图像 (用于多进程)."""
    (src_img_path, dst_img_path, corruption_name, severity,
     seed, overwrite) = args_tuple

    if os.path.exists(dst_img_path) and not overwrite:
        return f'SKIP (exists): {dst_img_path}'

    # 读取 4 通道图像
    img4 = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
    if img4 is None:
        return f'ERROR (read failed): {src_img_path}'
    if img4.ndim != 3 or img4.shape[2] != 4:
        return f'ERROR (not 4-channel): {src_img_path} shape={img4.shape}'

    # 施加退化
    out = apply_corruption_to_4ch(img4, corruption_name, severity, seed=seed)

    # 保存
    os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
    # 使用 PNG 无损保存 (避免 JPEG 压缩引入额外退化)
    cv2.imwrite(dst_img_path, out)
    return f'OK: {dst_img_path}'


def main():
    args = parse_args()

    # 解析退化列表
    corruptions = resolve_corruptions(args.corruptions)
    print(f'Corruptions ({len(corruptions)}): {corruptions}')
    print(f'Severities: {args.severities}')
    print(f'Total configs: {len(corruptions) * len(args.severities)}')

    # 读取 split
    names = load_split(args.src, args.split)
    print(f'Split file: {args.split}, {len(names)} images')

    # 准备任务列表
    tasks = []
    for name in names:
        src_img_path = os.path.join(args.src, args.images_dir, name + args.img_suffix)
        for corr_name in corruptions:
            for sev in args.severities:
                dst_img_path = os.path.join(
                    args.dst, 'images', corr_name, str(sev),
                    name + args.img_suffix)
                # 种子: 按图像名+退化+级别组合, 保证可复现且不同图像不同
                seed = (args.seed + hash(name) % 100000
                        + hash(corr_name) % 1000 + sev) % (2**32)
                tasks.append((src_img_path, dst_img_path, corr_name, sev,
                              seed, args.overwrite))

    print(f'Total tasks: {len(tasks)}')
    print(f'Workers: {args.workers}')

    # 多进程执行
    ok, skip, fail = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            msg = fut.result()
            if msg.startswith('OK'):
                ok += 1
            elif msg.startswith('SKIP'):
                skip += 1
            else:
                fail += 1
                print(f'  {msg}')
            if (i + 1) % 500 == 0:
                print(f'  Progress: {i+1}/{len(tasks)} (OK={ok} SKIP={skip} FAIL={fail})')

    print(f'\nDone: OK={ok} SKIP={skip} FAIL={fail}')

    # 复制 labels (软链接) 和 split 文件
    src_labels = os.path.join(args.src, args.labels_dir)
    dst_labels = os.path.join(args.dst, args.labels_dir)
    if os.path.exists(src_labels) and not os.path.exists(dst_labels):
        os.symlink(os.path.abspath(src_labels), dst_labels)
        print(f'Linked labels: {dst_labels} -> {src_labels}')

    src_split = os.path.join(args.src, args.split)
    dst_split = os.path.join(args.dst, args.split)
    if os.path.exists(src_split) and not os.path.exists(dst_split):
        shutil.copy(src_split, dst_split)
        print(f'Copied split: {dst_split}')

    # 写 README
    readme_path = os.path.join(args.dst, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write('RGBT-C Benchmark Dataset\n')
        f.write('=' * 50 + '\n')
        f.write(f'Source: {args.src}\n')
        f.write(f'Split: {args.split} ({len(names)} images)\n')
        f.write(f'Corruptions: {corruptions}\n')
        f.write(f'Severities: {args.severities}\n')
        f.write(f'Seed: {args.seed}\n')
        f.write(f'Generated: {os.popen("date").read().strip()}\n')
        f.write('\nStructure:\n')
        f.write('  images/{corruption}/{severity}/{name}.png\n')
        f.write('  labels/  (symlink to source)\n')
        f.write(f'  {args.split}\n')
    print(f'Wrote README: {readme_path}')


if __name__ == '__main__':
    main()
