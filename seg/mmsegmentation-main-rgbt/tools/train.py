# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys
from datetime import datetime

_project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='enable training visualization hook')
    parser.add_argument(
        '--vis-interval',
        type=int,
        default=20,
        help='visualization interval in epochs (default: 20)')
    parser.add_argument(
        '--vis-num-samples',
        type=int,
        default=2,
        help='number of samples to visualize per trigger (default: 2)')
    parser.add_argument(
        '--vis-version',
        type=str,
        default='auto',
        help='version name for visualization directory '
             '(default: auto, detected from model type)')
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='version name for output directory structure '
             '(default: auto, detected from config filename)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def detect_version_from_config(config_path):
    basename = osp.splitext(osp.basename(config_path))[0]
    version_map = {
        'v1_baseline': 'v1_baseline',
        'v2_disentangle': 'v2_disentangle',
        'v3_degradation': 'v3_degradation',
        'v4_quality_pruning': 'v4_quality_pruning',
        'v5_quality_joint': 'v5_quality_joint',
        'quality_network': 'quality_network',
        'v1_freeze_backbone': 'v1_freeze_backbone',
        'v1_full_finetune': 'v1_full_finetune',
        'freeze_backbone': 'v1_freeze_backbone',
        'full_finetune': 'v1_full_finetune',
    }
    for keyword, version in version_map.items():
        if keyword in basename:
            dataset = 'unknown'
            size = 'unknown'
            if 'fmb' in basename:
                dataset = 'fmb'
            elif 'pst900' in basename:
                dataset = 'pst900'
            elif 'semanticrt' in basename:
                dataset = 'semanticrt'
            elif 'mfnet' in basename:
                dataset = 'mfnet'
            if '768x768' in basename:
                size = '768x768'
            elif '480x480' in basename:
                size = '480x480'
            parts = [version, dataset, size]
            return '_'.join(parts)
    return basename


def save_weight_paths(weight_dir):
    best_pth = None
    final_pth = None
    if not osp.isdir(weight_dir):
        return
    for f in os.listdir(weight_dir):
        if f.startswith('best'):
            best_pth = osp.join(weight_dir, f)
        elif f.startswith('iter_') or f.startswith('epoch_'):
            all_iters = sorted(
                [x for x in os.listdir(weight_dir)
                 if x.startswith('iter_') or x.startswith('epoch_')])
            if all_iters:
                final_pth = osp.join(weight_dir, all_iters[-1])

    info_path = osp.join(weight_dir, 'weight_paths.txt')
    with open(info_path, 'w') as f:
        if best_pth:
            f.write(f'best: {best_pth}\n')
        if final_pth:
            f.write(f'final: {final_pth}\n')


def save_config_info(run_dir, cfg, config_name, args):
    config_path = osp.join(run_dir, 'config.py')
    cfg.dump(config_path)

    info_path = osp.join(run_dir, 'config.txt')
    model_cfg = cfg.get('model', {})
    model_type = model_cfg.get('type', 'Unknown')

    with open(info_path, 'w') as f:
        f.write(f'Config: {config_name}\n')
        f.write(f'Model: {model_type}\n')
        f.write(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        backbone = model_cfg.get('backbone', model_cfg.get(
            'universal_backbone', {}))
        f.write(f'Backbone: {backbone.get("type", "Unknown")}\n')

        decode_head = model_cfg.get('decode_head', {})
        f.write(f'Decoder: {decode_head.get("type", "Unknown")}\n')

        train_cfg = cfg.get('train_cfg', {})
        max_epochs = train_cfg.get('max_epochs', 'Unknown')
        f.write(f'Epochs: {max_epochs}\n')

        optim = cfg.get('optim_wrapper', {}).get('optimizer', {})
        f.write(f'Optimizer: {optim.get("type", "Unknown")}\n')
        f.write(f'Learning rate: {optim.get("lr", "Unknown")}\n')

        train_dl = cfg.get('train_dataloader', {})
        f.write(f'Batch size: {train_dl.get("batch_size", "Unknown")}\n')
        f.write(f'AMP: {args.amp}\n')

        pretrained = model_cfg.get('pretrained', None)
        init_cfg = model_cfg.get('init_cfg', None)
        if pretrained:
            f.write(f'Pretrained: {pretrained}\n')
        if init_cfg and isinstance(init_cfg, dict):
            f.write(f'Init checkpoint: {init_cfg.get("checkpoint", "N/A")}\n')

        lora = model_cfg.get('use_lora', False)
        f.write(f'LoRA: {lora}\n')
        if lora:
            f.write(f'  rank: {model_cfg.get("lora_rank", "N/A")}\n')
            f.write(f'  alpha: {model_cfg.get("lora_alpha", "N/A")}\n')

        f.write(f'\nFull args:\n')
        for k, v in sorted(vars(args).items()):
            f.write(f'  {k}: {v}\n')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    config_name = osp.splitext(osp.basename(args.config))[0]

    if args.version is not None:
        version_name = args.version
    else:
        version_name = detect_version_from_config(args.config)

    if args.work_dir is not None:
        base_dir = args.work_dir
    elif cfg.get('work_dir', None) is not None:
        base_dir = cfg.work_dir
    else:
        base_dir = osp.join('./work_dirs', version_name)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = osp.join(base_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)

    weight_dir = osp.join(work_dir, 'weight')
    os.makedirs(weight_dir, exist_ok=True)

    cfg.work_dir = work_dir

    if cfg.get('default_hooks', None) is not None:
        ckpt_hook = cfg.default_hooks.get('checkpoint', {})
        if isinstance(ckpt_hook, dict):
            ckpt_hook['out_dir'] = weight_dir

    print_log(f'Config: {config_name}', logger='current')
    print_log(f'Work directory: {work_dir}', logger='current')
    print_log(f'Weight directory: {weight_dir}', logger='current')

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.visualize:
        vis_hook = dict(
            type='TrainVisHook',
            interval=args.vis_interval,
            num_samples=args.vis_num_samples,
        )
        if cfg.get('custom_hooks', None) is None:
            cfg.custom_hooks = [vis_hook]
        else:
            cfg.custom_hooks.append(vis_hook)
        print_log(
            f'TrainVisHook enabled: interval={args.vis_interval} epochs, '
            f'num_samples={args.vis_num_samples}',
            logger='current')

    cfg.resume = args.resume

    save_config_info(work_dir, cfg, config_name, args)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()

    save_weight_paths(weight_dir)
    print_log(f'Training done. Weight paths saved to {weight_dir}/weight_paths.txt', logger='current')


if __name__ == '__main__':
    main()
