"""
RGB-T语义分割鲁棒性测试脚本

核心思路:
  1. 模型只加载一次，场景切换只更换数据管道（dataloader）
  2. 支持单场景测试和批量全场景测试
  3. 可选TensorBoard可视化（--visualize）
  4. 生成独立的results.log，包含基础信息、各场景逐类结果和汇总

用法:
  # Clean基线
  python tools/test_robustness.py <config> <checkpoint>

  # 单场景测试
  python tools/test_robustness.py <config> <checkpoint> \
      --degradation RGBMissingDegradation

  # 单场景测试带参数
  python tools/test_robustness.py <config> <checkpoint> \
      --degradation GlobalDegradation \
      --deg-kwargs modality=rgb intensity=low

  # 批量全场景测试
  python tools/test_robustness.py <config> <checkpoint> --batch-all

  # 带可视化的批量测试
  python tools/test_robustness.py <config> <checkpoint> --batch-all --visualize

目录结构:
  eval/robustness/<version>/<timestamp>/
    results.log                   # 独立结果日志（基础信息+逐类表格+汇总）
    robustness_results.json       # JSON格式汇总结果
    eval.log                      # 完整终端日志
    clean/
      metrics.json                # 包含summary和per_class
    emm_rgb_missing/
      metrics.json
    global_low/
      metrics.json
    ...
    vis/                          # TensorBoard日志（--visualize启用时）
      events.out.tfevents...

results.log格式示例:
  ================================================================
  RGB-T Robustness Evaluation Results
  ================================================================

  Time: 2026-05-04 12:46:16
  Command: python tools/test_robustness.py <config> <checkpoint> --batch-all
  Config: /path/to/config.py
  Checkpoint: /path/to/checkpoint.pth
  Dataset: MFNetDataset
  Data root: /path/to/MFNet
  Test split: val
  Total samples: 392

  ================================================================

  ================================================================================
  Scenario: clean
  Degradation: CleanDegradation
  ================================================================================
    Class       IoU       Acc
    --------  --------  --------
    unlabeled     96.70     98.45
    car           70.86     89.22
    ...
    --------  --------  --------
    mIoU          63.73
    mAcc          75.99
    aAcc          97.68

  ================================================================================
  Summary
  ================================================================================
    Scenario                             mIoU      mAcc      aAcc  mIoU Drop  Rel Drop%
    --------------------------------  --------  --------  --------  ----------  ----------
    clean                                63.73     75.99     97.68           -           -
    emm_rgb_missing                      50.44     59.25     96.79       13.29        20.8
    ...

可视化布局（6行x4列）:
  Row 1: RGB原图, T原图, [空], [空]
  Row 2-5: 各阶段特征图（最大激活通道灰度图）
           列: zc_rgb, zc_t, zp_rgb, zp_t
  Row 6: 标签, 预测, [空], [空]
"""

import argparse
import ast
import json
import os
import os.path as osp
import random
from datetime import datetime

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner
from torch.utils.tensorboard import SummaryWriter

from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.utils import register_all_modules

import sys
import os.path as osp
_tools_dir = osp.dirname(osp.abspath(__file__))
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)

from vis_utils import (
    _to_uint8,
    _unwrap_model as _vis_unwrap,
    create_composite_vis,
    detect_model_type,
    extract_features,
)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_deg_kwargs(kwargs_list):
    if kwargs_list is None:
        return {}
    result = {}
    for item in kwargs_list:
        key, value = item.split('=', 1)
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        result[key] = parsed
    return result


def build_pipeline(cfg, deg_type, deg_kwargs):
    base_pipeline = list(cfg.test_pipeline)
    deg_transform = dict(type=deg_type, **deg_kwargs)

    new_pipeline = []
    inserted = False
    for t in base_pipeline:
        new_pipeline.append(t)
        if isinstance(t, dict) and t.get('type') == 'LoadAnnotations' \
                and not inserted:
            new_pipeline.append(deg_transform)
            inserted = True
    if not inserted:
        for i, t in enumerate(new_pipeline):
            if isinstance(t, dict) and t.get('type') == 'PackSegInputs':
                new_pipeline.insert(i, deg_transform)
                break

    return new_pipeline


def build_dataloader(cfg, deg_type, deg_kwargs, use_val=False):
    pipeline = build_pipeline(cfg, deg_type, deg_kwargs)

    dl_cfg = cfg.val_dataloader if use_val else cfg.test_dataloader
    dataset_cfg = dl_cfg.dataset
    dataset_cfg_copy = dict(dataset_cfg)
    dataset_cfg_copy['pipeline'] = pipeline

    dataset = DATASETS.build(dataset_cfg_copy)

    sampler = DefaultSampler(dataset, shuffle=False)

    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader

    num_workers = dl_cfg.get('num_workers', 0)
    batch_size = dl_cfg.get('batch_size', 1)
    persistent_workers = dl_cfg.get('persistent_workers', False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=pseudo_collate,
        pin_memory=True,
    )
    return dataloader


def run_test_loop(model, dataloader, evaluator, data_preprocessor=None,
                  scenario_name=None, scenario_dir=None, num_vis_samples=0):
    model.eval()
    evaluator.dataset_meta = dataloader.dataset.metainfo

    from tqdm import tqdm

    writer = None
    vis_data = []
    vis_saved = False
    if scenario_dir is not None and num_vis_samples > 0:
        os.makedirs(scenario_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=scenario_dir)

    model_type = detect_model_type(model)
    print(f'  Model type: {model_type}')

    for idx, data_batch in enumerate(tqdm(dataloader, desc='Testing',
                                          leave=False)):
        with torch.no_grad():
            results = model.val_step(data_batch)
        data_samples = []
        for r in results:
            ds = {}
            if hasattr(r, 'pred_sem_seg') and r.pred_sem_seg is not None:
                ds['pred_sem_seg'] = {'data': r.pred_sem_seg.data}
            if hasattr(r, 'gt_sem_seg') and r.gt_sem_seg is not None:
                ds['gt_sem_seg'] = {'data': r.gt_sem_seg.data}
            if hasattr(r, 'img_path'):
                ds['img_path'] = r.img_path
            data_samples.append(ds)
        evaluator.process(data_batch=data_batch, data_samples=data_samples)

        if writer is not None and not vis_saved \
                and len(vis_data) < num_vis_samples:
            try:
                raw_inputs = data_batch['inputs']
                raw_data_samples = data_batch['data_samples']
                B = len(raw_inputs)

                proc_inputs = None
                if data_preprocessor is not None:
                    try:
                        proc = data_preprocessor(data_batch, False)
                        proc_inputs = proc['inputs']
                    except Exception as e:
                        print(f'  data_preprocessor error: {e}')

                for b in range(B):
                    if len(vis_data) >= num_vis_samples:
                        break

                    img_tensor = raw_inputs[b]
                    if not isinstance(img_tensor, torch.Tensor):
                        continue

                    img_np = img_tensor.cpu().numpy()
                    if img_np.ndim == 3 and img_np.shape[0] in (3, 6):
                        img_np = img_np.transpose(1, 2, 0)

                    if img_np.shape[2] >= 6:
                        rgb_raw = img_np[:, :, :3]
                        t_raw = img_np[:, :, 3:6]
                    elif img_np.shape[2] >= 3:
                        rgb_raw = img_np[:, :, :3]
                        t_raw = img_np[:, :, :3]
                    else:
                        rgb_raw = img_np
                        t_raw = img_np

                    rgb_vis = _to_uint8(rgb_raw)
                    t_gray = _to_uint8(t_raw)
                    if t_gray.ndim == 2:
                        t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)
                    else:
                        t_gray = cv2.cvtColor(t_gray, cv2.COLOR_RGB2GRAY)
                        t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)

                    feat_lists = None
                    if proc_inputs is not None and b < proc_inputs.shape[0]:
                        single_input = proc_inputs[b:b + 1]
                        try:
                            feat_lists = extract_features(model, single_input)
                        except Exception as e:
                            print(f'  extract_features error: {e}')

                    pred_seg = np.zeros(
                        (img_tensor.shape[-2], img_tensor.shape[-1]),
                        dtype=np.uint8)
                    if b < len(data_samples):
                        pred_data = data_samples[b].get(
                            'pred_sem_seg', {}).get('data', None)
                        if pred_data is not None:
                            pred_seg = pred_data.squeeze().cpu().numpy()

                    label_seg = np.zeros_like(pred_seg)
                    ds = raw_data_samples[b] if b < len(raw_data_samples) \
                        else None
                    if ds is not None and hasattr(ds, 'gt_sem_seg'):
                        gt_data = ds.gt_sem_seg.data
                        if gt_data is not None:
                            label_seg = gt_data.squeeze().cpu().numpy()

                    vis_data.append({
                        'rgb': rgb_vis,
                        'thermal': t_vis,
                        'features': feat_lists,
                        'label': label_seg,
                        'pred': pred_seg,
                    })

                if len(vis_data) >= num_vis_samples:
                    try:
                        palette = None
                        if hasattr(dataloader.dataset, 'metainfo') \
                                and 'palette' in dataloader.dataset.metainfo:
                            palette = dataloader.dataset.metainfo['palette']
                        composite = create_composite_vis(
                            vis_data, short_side=250, palette=palette)
                        writer.add_image(f'{scenario_name}/all_samples',
                                         composite.transpose(2, 0, 1))
                        png_path = osp.join(scenario_dir,
                                            f'{scenario_name}_vis.png')
                        cv2.imwrite(png_path,
                                    cv2.cvtColor(composite,
                                                 cv2.COLOR_RGB2BGR))
                        print(f'  Visualization saved: {png_path} '
                              f'({len(vis_data)} samples)')
                        vis_saved = True
                    except Exception as e:
                        import traceback
                        print(f'  Composite image error: {e}')
                        traceback.print_exc()
            except Exception as e:
                import traceback
                print(f'  Vis data collection error: {e}')
                traceback.print_exc()

    if writer is not None:
        if not vis_saved and len(vis_data) > 0:
            try:
                palette = None
                if hasattr(dataloader.dataset, 'metainfo') \
                        and 'palette' in dataloader.dataset.metainfo:
                    palette = dataloader.dataset.metainfo['palette']
                composite = create_composite_vis(
                    vis_data, short_side=250, palette=palette)
                writer.add_image(f'{scenario_name}/all_samples',
                                 composite.transpose(2, 0, 1))
                png_path = osp.join(scenario_dir,
                                    f'{scenario_name}_vis.png')
                cv2.imwrite(png_path,
                            cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                print(f'  Visualization saved: {png_path} '
                      f'({len(vis_data)} samples)')
            except Exception as e:
                import traceback
                print(f'  Composite image error: {e}')
                traceback.print_exc()
        elif not vis_saved:
            print(f'  WARNING: No vis_data collected for {scenario_name}')
        writer.close()

    per_class_data = None
    if evaluator.results:
        results = tuple(zip(*evaluator.results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_label = sum(results[3])
        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        iou_np = iou.numpy() if hasattr(iou, 'numpy') else np.array(iou)
        acc_np = acc.numpy() if hasattr(acc, 'numpy') else np.array(acc)
        class_names = evaluator.dataset_meta['classes']
        per_class_data = []
        for i, name in enumerate(class_names):
            per_class_data.append({
                'class': name,
                'IoU': float(np.round(iou_np[i] * 100, 2)) if i < len(iou_np) and not np.isnan(iou_np[i]) else 0.0,
                'Acc': float(np.round(acc_np[i] * 100, 2)) if i < len(acc_np) and not np.isnan(acc_np[i]) else 0.0,
            })

    summary = evaluator.evaluate(len(dataloader.dataset))
    return summary, per_class_data


def get_batch_scenarios():
    scenarios = []

    scenarios.append(('clean', 'CleanDegradation', {}))

    scenarios.append(('rgb_missing', 'RGBMissingDegradation', {}))

    scenarios.append(('thermal_missing', 'ThermalMissingDegradation', {}))

    scenarios.append(('local_rgb_missing_50', 'LocalRGBMissingDegradation',
                      {'area_ratio': 0.5}))

    scenarios.append(('local_thermal_missing_50', 'LocalThermalMissingDegradation',
                      {'area_ratio': 0.5}))

    return scenarios


def format_scenario_block(scenario_name, deg_type, deg_kwargs,
                          per_class, summary):
    lines = []
    lines.append('=' * 80)
    lines.append(f'Scenario: {scenario_name}')
    lines.append(f'Degradation: {deg_type}')
    if deg_kwargs:
        lines.append(f'Parameters: {deg_kwargs}')
    lines.append('=' * 80)

    class_name_max = max(len(c['class']) for c in per_class)
    header = f'  {"Class":<{class_name_max}}  {"IoU":>8}  {"Acc":>8}'
    sep = f'  {"-" * class_name_max}  {"--------":>8}  {"--------":>8}'
    lines.append(header)
    lines.append(sep)
    for c in per_class:
        iou_str = f'{c["IoU"]:.2f}' if not np.isnan(c['IoU']) else 'N/A'
        acc_str = f'{c["Acc"]:.2f}' if not np.isnan(c['Acc']) else 'N/A'
        lines.append(f'  {c["class"]:<{class_name_max}}  {iou_str:>8}  {acc_str:>8}')

    lines.append(sep)
    lines.append(f'  {"mIoU":<{class_name_max}}  {summary["mIoU"]:>8.2f}')
    lines.append(f'  {"mAcc":<{class_name_max}}  {summary["mAcc"]:>8.2f}')
    lines.append(f'  {"aAcc":<{class_name_max}}  {summary["aAcc"]:>8.2f}')
    lines.append('')

    return '\n'.join(lines)


def write_results_log(base_work_dir, args, cfg, all_scenario_data):
    log_path = osp.join(base_work_dir, 'results.log')
    lines = []

    lines.append('=' * 80)
    lines.append('RGB-T Robustness Evaluation Results')
    lines.append('=' * 80)
    lines.append('')

    lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Command: python tools/test_robustness.py {args.config} '
                 f'{args.checkpoint}')
    if args.batch_all:
        lines.append('       --batch-all')
    if args.visualize:
        lines.append('       --visualize')
    if args.use_val:
        lines.append('       --use-val')
    if args.version != 'original':
        lines.append(f'       --version {args.version}')
    lines.append(f'Config: {osp.abspath(args.config)}')
    lines.append(f'Checkpoint: {osp.abspath(args.checkpoint)}')

    dataset_name = cfg.get('dataset_type', 'Unknown')
    data_root = cfg.get('data_root', 'Unknown')
    lines.append(f'Dataset: {dataset_name}')
    lines.append(f'Data root: {data_root}')

    dl_cfg = cfg.val_dataloader if args.use_val else cfg.test_dataloader
    split = 'val' if args.use_val else 'test'
    lines.append(f'Test split: {split}')

    try:
        dataset_cfg = dict(dl_cfg.dataset)
        dataset = DATASETS.build(dataset_cfg)
        lines.append(f'Total samples: {len(dataset)}')
    except Exception:
        lines.append('Total samples: Unknown')

    lines.append('')
    lines.append('=' * 80)
    lines.append('')

    for item in all_scenario_data:
        lines.append(format_scenario_block(
            item['scenario_name'], item['deg_type'], item['deg_kwargs'],
            item['per_class'], item['summary']))

    if len(all_scenario_data) > 1:
        clean_summary = None
        for item in all_scenario_data:
            if item['scenario_name'] == 'clean':
                clean_summary = item['summary']
                break

        lines.append('=' * 80)
        lines.append('Summary')
        lines.append('=' * 80)
        lines.append('')

        header = (f'  {"Scenario":<35}  {"mIoU":>8}  {"mAcc":>8}  '
                  f'{"aAcc":>8}  {"mIoU Drop":>10}  {"Rel Drop%":>10}')
        sep = (f'  {"-" * 35}  {"--------":>8}  {"--------":>8}  '
               f'{"--------":>8}  {"----------":>10}  {"----------":>10}')
        lines.append(header)
        lines.append(sep)

        clean_miou = clean_summary['mIoU'] if clean_summary else None

        for item in all_scenario_data:
            s = item['summary']
            name = item['scenario_name']
            miou = s['mIoU']
            macc = s['mAcc']
            aacc = s['aAcc']

            drop_str = '-'
            rel_str = '-'
            if clean_miou is not None and name != 'clean':
                drop = clean_miou - miou
                rel = drop / clean_miou * 100 if clean_miou > 0 else 0
                drop_str = f'{drop:.2f}'
                rel_str = f'{rel:.1f}'

            lines.append(f'  {name:<35}  {miou:>8.2f}  {macc:>8.2f}  '
                         f'{aacc:>8.2f}  {drop_str:>10}  {rel_str:>10}')

        lines.append(sep)
        lines.append('')

        if clean_miou is not None:
            valid = [item for item in all_scenario_data
                     if item['scenario_name'] != 'clean']
            drops = [clean_miou - item['summary']['mIoU'] for item in valid]
            if drops:
                lines.append(f'Clean baseline mIoU: {clean_miou:.2f}')
                lines.append(f'Average mIoU drop: {np.mean(drops):.2f}')
                max_idx = int(np.argmax(drops))
                min_idx = int(np.argmin(drops))
                lines.append(f'Max mIoU drop: {max(drops):.2f} '
                             f'({valid[max_idx]["scenario_name"]})')
                lines.append(f'Min mIoU drop: {min(drops):.2f} '
                             f'({valid[min_idx]["scenario_name"]})')
                lines.append('')

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Results log saved to: {log_path}')
    return log_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='RGB-T Robustness Test')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='directory to save results '
             '(default: eval/robustness/<timestamp>)')
    parser.add_argument(
        '--degradation',
        type=str,
        default='CleanDegradation',
        help='degradation transform class name (default: CleanDegradation)')
    parser.add_argument(
        '--deg-kwargs',
        nargs='+',
        default=None,
        help='keyword arguments for degradation class, '
             'e.g. missing_ratio=0.5')
    parser.add_argument(
        '--batch-all',
        action='store_true',
        help='run all robustness scenarios in batch')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='enable TensorBoard visualization')
    parser.add_argument(
        '--num-vis-samples',
        type=int,
        default=4,
        help='number of samples for visualization (default: 4)')
    parser.add_argument(
        '--version',
        type=str,
        default='original',
        help='model version name for output directory '
             '(default: original)')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed (default: 42)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=None,
        help='override some settings in the used config')
    parser.add_argument(
        '--use-val',
        action='store_true',
        help='use val_dataloader instead of test_dataloader '
             '(for fair comparison with training logs)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    register_all_modules()

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.work_dir is not None:
        output_root = args.work_dir
    else:
        output_root = osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))),
            'eval', 'robustness')

    base_work_dir = osp.join(output_root, args.version, date_str)

    os.makedirs(base_work_dir, exist_ok=True)

    log_path = osp.join(base_work_dir, 'eval.log')
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = TeeLogger(osp.join(base_work_dir, 'eval_error.log'))

    print(f'Log file: {log_path}')
    print(f'Work dir: {base_work_dir}')
    print(f'Args: {vars(args)}')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join(base_work_dir, '_model_load')
    os.makedirs(cfg.work_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f'Loading model (one-time setup)...')
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f"{'=' * 80}")

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    runner.model.eval()

    model = _vis_unwrap(runner.model)
    data_preprocessor = model.data_preprocessor

    print(f'Model loaded successfully.')
    print(f"{'=' * 80}")

    if args.batch_all:
        _run_batch(runner, model, data_preprocessor, cfg, args,
                   base_work_dir)
    else:
        _run_single(runner, model, data_preprocessor, cfg, args,
                    base_work_dir)


def _run_single(runner, model, data_preprocessor, cfg, args, base_work_dir):
    deg_type = args.degradation
    deg_kwargs = parse_deg_kwargs(args.deg_kwargs)

    if TRANSFORMS.get(deg_type) is None:
        available = [k for k in TRANSFORMS.module_dict.keys()
                     if 'Degradation' in k
                     or 'degradation' in k.lower()]
        print(f"Error: Degradation '{deg_type}' not found in TRANSFORMS.")
        print(f'Available degradation classes: {available}')
        return

    scenario_name = deg_type
    if deg_kwargs:
        kwargs_str = '_'.join(f'{k}{v}' for k, v in deg_kwargs.items())
        scenario_name = f'{deg_type}_{kwargs_str}'

    print(f"\n{'=' * 80}")
    print(f'Single Scenario Robustness Test')
    print(f'Scenario: {scenario_name}')
    print(f'Degradation: {deg_type}')
    print(f'Degradation kwargs: {deg_kwargs}')
    print(f'Work dir: {base_work_dir}')
    print(f'Visualize: {args.visualize}')
    print(f"{'=' * 80}")

    from mmseg.evaluation import IoUMetric
    evaluator = IoUMetric(iou_metrics=['mIoU'])

    dataloader = build_dataloader(cfg, deg_type, deg_kwargs,
                                  use_val=args.use_val)
    scenario_dir = osp.join(base_work_dir, scenario_name) \
        if args.visualize else None
    num_vis = args.num_vis_samples if args.visualize else 0

    eval_results = run_test_loop(
        model, dataloader, evaluator, data_preprocessor,
        scenario_name, scenario_dir, num_vis)

    summary, per_class = eval_results
    if per_class is None:
        per_class = []

    scenario_work_dir = osp.join(base_work_dir, scenario_name)
    os.makedirs(scenario_work_dir, exist_ok=True)
    metrics_path = osp.join(scenario_work_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_class': per_class},
                  f, indent=2, ensure_ascii=False, default=str)

    results_path = osp.join(base_work_dir, 'robustness_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({scenario_name: summary}, f, indent=2,
                  ensure_ascii=False, default=str)

    all_scenario_data = [{
        'scenario_name': scenario_name,
        'deg_type': deg_type,
        'deg_kwargs': deg_kwargs,
        'per_class': per_class,
        'summary': summary,
    }]
    write_results_log(base_work_dir, args, cfg, all_scenario_data)

    print(f"\n{'=' * 80}")
    print(f'Scenario: {scenario_name}')
    print(f'mIoU={summary["mIoU"]:.2f}  mAcc={summary["mAcc"]:.2f}  '
          f'aAcc={summary["aAcc"]:.2f}')
    print(f"{'=' * 80}")
    print(f'\nResults saved to: {results_path}')


def _run_batch(runner, model, data_preprocessor, cfg, args, base_work_dir):
    scenarios = get_batch_scenarios()
    all_results = {}
    all_scenario_data = []

    results_json_path = osp.join(base_work_dir, 'robustness_results.json')

    print(f"\n{'=' * 80}")
    print(f'Batch Robustness Evaluation')
    print(f'Total scenarios: {len(scenarios)}')
    print(f'Work dir: {base_work_dir}')
    print(f'Visualize: {args.visualize}')
    print(f"{'=' * 80}")

    num_vis = args.num_vis_samples if args.visualize else 0

    from mmseg.evaluation import IoUMetric

    for idx, (scenario_name, deg_type, deg_kwargs) \
            in enumerate(scenarios):
        print(f'\n[{idx + 1}/{len(scenarios)}] '
              f'Testing: {scenario_name} ({deg_type}, {deg_kwargs})')

        try:
            evaluator = IoUMetric(iou_metrics=['mIoU'])
            dataloader = build_dataloader(cfg, deg_type, deg_kwargs,
                                          use_val=args.use_val)
            
            # Only visualize for clean scenario to save time
            is_clean = scenario_name == 'clean'
            scenario_dir = osp.join(base_work_dir, scenario_name) \
                if args.visualize and is_clean else None
            current_num_vis = num_vis if is_clean else 0
            
            eval_results = run_test_loop(
                model, dataloader, evaluator, data_preprocessor,
                scenario_name, scenario_dir, current_num_vis)

            summary, per_class = eval_results
            if per_class is None:
                per_class = []

            all_results[scenario_name] = summary
            all_scenario_data.append({
                'scenario_name': scenario_name,
                'deg_type': deg_type,
                'deg_kwargs': deg_kwargs,
                'per_class': per_class,
                'summary': summary,
            })

            print(f'  [{idx + 1}/{len(scenarios)}] {scenario_name}: '
                  f'mIoU={summary["mIoU"]:.2f}  mAcc={summary["mAcc"]:.2f}  '
                  f'aAcc={summary["aAcc"]:.2f}')

            scenario_work_dir = osp.join(base_work_dir, scenario_name)
            os.makedirs(scenario_work_dir, exist_ok=True)
            metrics_path = osp.join(scenario_work_dir, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({'summary': summary, 'per_class': per_class},
                          f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f'Error in scenario {scenario_name}: {e}')
            import traceback
            traceback.print_exc()
            all_results[scenario_name] = {'error': str(e)}

        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2,
                      ensure_ascii=False, default=str)

    write_results_log(base_work_dir, args, cfg, all_scenario_data)
    print(f'\nAll results saved to: {results_json_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFATAL ERROR: {e}')
        import traceback
        traceback.print_exc()
