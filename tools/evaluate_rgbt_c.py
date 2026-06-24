"""RGBT-C 评估脚本: 计算 mIoU / mCE / mRC / mIoU_avg.

设计依据: docs/RGBT-C_Benchmark.md v1.0 §6

用法:
    # 评估单个模型在 MFNet-C 上的表现
    python tools/evaluate_rgbt_c.py \\
        --config configs/eomt/eomt_rgbt_fusion_contrast_dinov3-b_mfnet-480x640.py \\
        --checkpoint work_dirs/eomt_rgbt_fusion/latest.pth \\
        --data-root /home/lh/code/data/MFNet-C \\
        --split test.txt \\
        --corruptions all \\
        --severities 1 2 3 4 5 \\
        --baseline-clean-miou 0.75  \\
        --output results_rgbt_c.json

    # 仅汇总已有结果 (跳过推理)
    python tools/evaluate_rgbt_c.py \\
        --results-dir work_dirs/eomt_rgbt_fusion/rgbtc_results \\
        --baseline-clean-miou 0.75 \\
        --output results_rgbt_c.json

指标说明:
    - mIoU_c,s:  退化 c 级别 s 的 mIoU
    - mCE:       mean Corruption Error (相对 baseline, 越小越鲁棒)
    - mRC:       mean Relative CE (考虑绝对性能, 越小越鲁棒)
    - mIoU_avg:  所有退化×级别的平均 mIoU
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgbt_c import RGB_CORRUPTIONS, T_CORRUPTIONS, ALL_CORRUPTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RGBT-C benchmark (mIoU/mCE/mRC)')
    # 模式 1: 运行推理
    parser.add_argument('--config', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='RGBT-C dataset root')
    parser.add_argument('--split', type=str, default='test.txt',
                        help='Split file name (default: test.txt)')
    parser.add_argument('--corruptions', type=str, nargs='+', default=['all'],
                        help='Corruption names (default: all)')
    parser.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Severity levels (default: 1-5)')
    # 模式 2: 汇总已有结果
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory of per-corruption result JSONs')
    # 通用参数
    parser.add_argument('--baseline-clean-miou', type=float, default=None,
                        help='Baseline clean mIoU (for mCE/mRC). '
                             'If None, use clean mIoU of current model')
    parser.add_argument('--baseline-results', type=str, default=None,
                        help='Path to baseline results JSON (for mCE/mRC)')
    parser.add_argument('--output', type=str, default='results_rgbt_c.json',
                        help='Output JSON file (default: results_rgbt_c.json)')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Work dir to save per-corruption results')
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
    seen = set()
    out = []
    for n in result:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def run_inference_per_corruption(config, checkpoint, data_root, split,
                                  corruption, severity, work_dir):
    """对单个 (corruption, severity) 运行 mmseg 推理并返回 mIoU.

    通过修改 data_root 指向 {data_root}/images/{corruption}/{severity}/
    来加载退化后的测试集.
    """
    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmcv

    # 构造退化数据集路径
    corr_img_dir = os.path.join(data_root, 'images', corruption, str(severity))
    if not os.path.exists(corr_img_dir):
        print(f'  WARNING: {corr_img_dir} not found, skip')
        return None

    # 加载配置
    cfg = Config.fromfile(config)

    # 修改测试数据集路径: 让 data_prefix.img_path 指向退化图像目录
    # 注意: 退化图像文件名与原始一致, 只是路径不同
    # 我们通过自定义 test_dataloader 的 dataset 来实现
    # 简化方案: 临时修改 data_root 和 data_prefix
    test_dataset = cfg.test_dataloader.dataset

    # 保存原始路径, 用于恢复
    orig_data_root = test_dataset.get('data_root', None)
    orig_img_path = test_dataset.get('data_prefix', {}).get('img_path', 'images')

    # 修改为退化图像目录
    test_dataset['data_root'] = data_root
    test_dataset['data_prefix'] = dict(
        img_path=os.path.join('images', corruption, str(severity)),
        seg_map_path='labels',
    )
    test_dataset['ann_file'] = split

    # 保存 per-corruption 结果
    if work_dir:
        corr_work_dir = os.path.join(work_dir, 'rgbtc_results', corruption, str(severity))
    else:
        corr_work_dir = None

    # 运行
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume(checkpoint)
    metrics = runner.test()
    # 提取 mIoU
    miou = None
    for k, v in metrics.items():
        if 'mIoU' in v:
            miou = v['mIoU']
            break
    return miou


def collect_results_from_dir(results_dir):
    """从结果目录收集 per-corruption mIoU.

    期望目录结构:
        {results_dir}/{corruption}/{severity}/result.json
    或:
        {results_dir}/{corruption}_{severity}.json
    """
    results = {}  # {corruption: {severity: mIoU}}
    results_dir = Path(results_dir)

    # 模式 1: {corruption}/{severity}/result.json
    for corr_dir in results_dir.iterdir():
        if not corr_dir.is_dir():
            continue
        corr_name = corr_dir.name
        results[corr_name] = {}
        for sev_dir in corr_dir.iterdir():
            if not sev_dir.is_dir():
                continue
            sev = int(sev_dir.name)
            result_file = sev_dir / 'result.json'
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                miou = None
                for k, v in data.items():
                    if isinstance(v, dict) and 'mIoU' in v:
                        miou = v['mIoU']
                        break
                    elif k == 'mIoU':
                        miou = v
                        break
                if miou is not None:
                    results[corr_name][sev] = miou

    # 模式 2: {corruption}_{severity}.json
    if not any(results.values()):
        for json_file in results_dir.glob('*.json'):
            name = json_file.stem
            # 解析 corruption_severity
            parts = name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            corr_name, sev_str = parts
            try:
                sev = int(sev_str)
            except ValueError:
                continue
            with open(json_file) as f:
                data = json.load(f)
            miou = None
            for k, v in data.items():
                if isinstance(v, dict) and 'mIoU' in v:
                    miou = v['mIoU']
                    break
                elif k == 'mIoU':
                    miou = v
                    break
            if miou is not None:
                if corr_name not in results:
                    results[corr_name] = {}
                results[corr_name][sev] = miou

    return results


def compute_metrics(results, baseline_results=None, baseline_clean_miou=None):
    """计算 mIoU_avg / mCE / mRC.

    Args:
        results: {corruption: {severity: mIoU}}
        baseline_results: 同格式, 用于 mCE/mRC (可选)
        baseline_clean_miou: baseline 在 clean 数据上的 mIoU (可选)
    Returns:
        dict of metrics
    """
    # 收集所有 (corruption, severity) 的 mIoU
    all_miou = []
    per_corruption = {}

    for corr, sev_dict in results.items():
        per_corruption[corr] = {}
        for sev, miou in sev_dict.items():
            all_miou.append(miou)
            per_corruption[corr][sev] = miou

    # mIoU_avg
    mIoU_avg = float(np.mean(all_miou)) if all_miou else 0.0

    # mCE / mRC (需要 baseline)
    mCE = None
    mRC = None
    per_corruption_CE = {}
    per_corruption_RC = {}

    if baseline_results is not None:
        CE_list = []
        RC_list = []
        for corr, sev_dict in results.items():
            if corr not in baseline_results:
                continue
            base_sev_dict = baseline_results[corr]
            # E = 1 - mIoU
            E_c = sum(1 - sev_dict[s] for s in sev_dict if s in base_sev_dict)
            E_base = sum(1 - base_sev_dict[s] for s in sev_dict if s in base_sev_dict)
            if E_base > 0:
                CE_c = E_c / E_base
                RC_c = (E_c - E_base) / E_base
                per_corruption_CE[corr] = CE_c
                per_corruption_RC[corr] = RC_c
                CE_list.append(CE_c)
                RC_list.append(RC_c)
        if CE_list:
            mCE = float(np.mean(CE_list))
            mRC = float(np.mean(RC_list))

    # 按模态分组
    rgb_miou = []
    t_miou = []
    for corr in RGB_CORRUPTIONS:
        if corr in per_corruption:
            rgb_miou.extend(per_corruption[corr].values())
    for corr in T_CORRUPTIONS:
        if corr in per_corruption:
            t_miou.extend(per_corruption[corr].values())

    return {
        'mIoU_avg': mIoU_avg,
        'mCE': mCE,
        'mRC': mRC,
        'mIoU_RGB': float(np.mean(rgb_miou)) if rgb_miou else None,
        'mIoU_T': float(np.mean(t_miou)) if t_miou else None,
        'per_corruption': per_corruption,
        'per_corruption_CE': per_corruption_CE,
        'per_corruption_RC': per_corruption_RC,
        'n_configs': len(all_miou),
    }


def print_report(metrics, model_name='Model'):
    """打印评估报告."""
    print('\n' + '=' * 70)
    print(f'RGBT-C Evaluation Report: {model_name}')
    print('=' * 70)
    print(f'  Configs evaluated: {metrics["n_configs"]}')
    print(f'  mIoU_avg:  {metrics["mIoU_avg"]:.4f}')
    if metrics.get('mIoU_RGB') is not None:
        print(f'  mIoU_RGB:  {metrics["mIoU_RGB"]:.4f}')
    if metrics.get('mIoU_T') is not None:
        print(f'  mIoU_T:    {metrics["mIoU_T"]:.4f}')
    if metrics.get('mCE') is not None:
        print(f'  mCE:       {metrics["mCE"]:.4f}  (<1 = more robust than baseline)')
    if metrics.get('mRC') is not None:
        print(f'  mRC:       {metrics["mRC"]:.4f}  (<0 = more robust than baseline)')

    print('\n  Per-corruption mIoU (avg over severities):')
    print(f'  {"Corruption":<22s} {"mIoU":>8s} {"CE":>8s} {"RC":>8s}')
    print('  ' + '-' * 48)
    for corr in RGB_CORRUPTIONS + T_CORRUPTIONS:
        if corr not in metrics['per_corruption']:
            continue
        miou = np.mean(list(metrics['per_corruption'][corr].values()))
        ce = metrics['per_corruption_CE'].get(corr, None)
        rc = metrics['per_corruption_RC'].get(corr, None)
        ce_str = f'{ce:.4f}' if ce is not None else '-'
        rc_str = f'{rc:.4f}' if rc is not None else '-'
        print(f'  {corr:<22s} {miou:>8.4f} {ce_str:>8s} {rc_str:>8s}')
    print('=' * 70)


def main():
    args = parse_args()

    # 收集结果
    if args.results_dir:
        # 模式 2: 从已有结果汇总
        print(f'Collecting results from: {args.results_dir}')
        results = collect_results_from_dir(args.results_dir)
    elif args.config and args.checkpoint and args.data_root:
        # 模式 1: 运行推理
        corruptions = resolve_corruptions(args.corruptions)
        print(f'Running inference: {len(corruptions)} corruptions × '
              f'{len(args.severities)} severities')
        results = {}
        for corr in corruptions:
            results[corr] = {}
            for sev in args.severities:
                print(f'\n>>> {corr} sev{sev}')
                miou = run_inference_per_corruption(
                    args.config, args.checkpoint, args.data_root,
                    args.split, corr, sev, args.work_dir)
                if miou is not None:
                    results[corr][sev] = miou
                    print(f'    mIoU = {miou:.4f}')
    else:
        print('ERROR: must provide either --results-dir or '
              '(--config + --checkpoint + --data-root)')
        sys.exit(1)

    # 加载 baseline
    baseline_results = None
    if args.baseline_results:
        with open(args.baseline_results) as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data.get('per_corruption', baseline_data)

    # 计算指标
    metrics = compute_metrics(results, baseline_results, args.baseline_clean_miou)

    # 打印报告
    print_report(metrics, model_name=args.config or 'Model')

    # 保存结果
    output = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'data_root': args.data_root,
        'corruptions': resolve_corruptions(args.corruptions),
        'severities': args.severities,
        'metrics': metrics,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
