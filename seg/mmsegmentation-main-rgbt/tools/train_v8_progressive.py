"""
MiTMulV8QualityPyramid 三阶段渐进式训练脚本

训练流程:
  阶段0: 质量网络自监督预训练 (独立脚本 train_quality_pyramid_net.py)
  阶段1: 冻结质量网络，训练分割模型 (backbone + private + fusion + head)
  阶段2: 解冻质量网络，联合微调 (分割 + 质量排序损失)
  阶段3: 全量联合训练 (分割 + 质量排序 + 退化一致性损失)

使用方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt

  # 完整三阶段训练
  bash tools/train_v8_progressive.sh fmb

  # 仅运行某个阶段
  bash tools/train_v8_progressive.sh fmb 2

  # 指定质量网络预训练权重
  bash tools/train_v8_progressive.sh fmb 1 /path/to/quality_net.pth
"""

import os
import sys
import argparse
import subprocess

DATASETS = {
    'fmb': {
        'config': 'configs/segformer/mitmul_v8_quality_pyramid_mit-b2-b0_1xb2-40K_fmb-480x480.py',
        'num_classes': 15,
    },
    'pst900': {
        'config': 'configs/segformer/mitmul_v8_quality_pyramid_mit-b2-b0_1xb2-40K_pst900-480x480.py',
        'num_classes': 5,
    },
    'mfnet': {
        'config': 'configs/segformer/mitmul_v8_quality_pyramid_mit-b2-b0_1xb2-40K_mfnet-240x320.py',
        'num_classes': 9,
    },
}

PHASE_EPOCHS = {
    1: 40,
    2: 40,
    3: 22,
}

PHASE_LR = {
    1: 6e-5,
    2: 6e-5,
    3: 3e-5,
}


def run_cmd(cmd):
    print(f'\n{"="*80}')
    print(f'CMD: {cmd}')
    print(f'{"="*80}\n')
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'Command failed with return code {result.returncode}')
        sys.exit(result.returncode)


def get_phase_work_dir(base_dir, phase):
    return os.path.join(base_dir, f'phase{phase}')


def modify_config_for_phase(config_path, phase, quality_pretrained,
                            prev_ckpt, work_dir, epochs, lr):
    import mmengine
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)

    cfg.model.train_phase = phase
    cfg.train_cfg.max_epochs = epochs
    cfg.optim_wrapper.optimizer.lr = lr

    if quality_pretrained:
        cfg.model.quality_pretrained = quality_pretrained

    if phase == 1:
        if 'quality_pyramid_net' in cfg.optim_wrapper.paramwise_cfg.custom_keys:
            del cfg.optim_wrapper.paramwise_cfg.custom_keys['quality_pyramid_net']
    elif phase >= 2:
        cfg.optim_wrapper.paramwise_cfg.custom_keys['quality_pyramid_net'] = \
            dict(lr_mult=0.1)

    if phase == 3:
        cfg.model.loss_degradation_consistency_weight = 1.0
    else:
        cfg.model.loss_degradation_consistency_weight = 0.0

    phase_config_path = os.path.join(work_dir, f'phase{phase}_config.py')
    os.makedirs(work_dir, exist_ok=True)
    cfg.dump(phase_config_path)
    return phase_config_path


def train_phase(config_path, phase, quality_pretrained, prev_ckpt,
                base_work_dir, epochs, lr, gpu_id, visualize=False,
                vis_interval=20, vis_num_samples=2):
    work_dir = get_phase_work_dir(base_work_dir, phase)
    os.makedirs(work_dir, exist_ok=True)

    phase_config = modify_config_for_phase(
        config_path, phase, quality_pretrained,
        prev_ckpt, work_dir, epochs, lr)

    load_from = ''
    if prev_ckpt and os.path.exists(prev_ckpt):
        load_from = f'--cfg-options load_from={prev_ckpt}'

    vis_args = ''
    if visualize:
        vis_args = (
            f'--visualize '
            f'--vis-interval {vis_interval} '
            f'--vis-num-samples {vis_num_samples}'
        )

    cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} '
        f'python tools/train.py '
        f'{phase_config} '
        f'--work-dir {work_dir} '
        f'{load_from} '
        f'{vis_args}'
    )
    run_cmd(cmd)

    best_ckpt = os.path.join(work_dir, 'best_mIoU_iter_*.pth')
    last_ckpt = os.path.join(work_dir, 'iter_*.pth')
    return work_dir


def find_best_checkpoint(work_dir):
    import glob
    best_ckpts = glob.glob(os.path.join(work_dir, 'best_mIoU_iter_*.pth'))
    if best_ckpts:
        return max(best_ckpts, key=os.path.getmtime)
    all_ckpts = glob.glob(os.path.join(work_dir, 'iter_*.pth'))
    if all_ckpts:
        return max(all_ckpts, key=os.path.getmtime)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='MiTMulV8 Progressive Training')
    parser.add_argument('dataset', choices=DATASETS.keys(),
                        help='Dataset name')
    parser.add_argument('--phase', type=int, default=None,
                        choices=[1, 2, 3],
                        help='Run only specific phase (default: all phases)')
    parser.add_argument('--quality-pretrained', type=str, default=None,
                        help='Path to pretrained quality network weights')
    parser.add_argument('--work-dir', type=str,
                        default='work_dirs/v8_progressive',
                        help='Base work directory')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--phase1-epochs', type=int, default=40)
    parser.add_argument('--phase2-epochs', type=int, default=40)
    parser.add_argument('--phase3-epochs', type=int, default=22)
    parser.add_argument('--visualize', action='store_true',
                        default=True,
                        help='enable training visualization hook')
    parser.add_argument('--no-visualize', action='store_true',
                        help='disable training visualization hook')
    parser.add_argument('--vis-interval', type=int, default=5,
                        help='visualization interval in epochs')
    parser.add_argument('--vis-num-samples', type=int, default=2,
                        help='number of samples to visualize')
    args = parser.parse_args()
    if args.no_visualize:
        args.visualize = False

    dataset_info = DATASETS[args.dataset]
    config_path = dataset_info['config']

    if not os.path.exists(config_path):
        print(f'Config not found: {config_path}')
        sys.exit(1)

    base_work_dir = f'{args.work_dir}/{args.dataset}'
    os.makedirs(base_work_dir, exist_ok=True)

    phase_epochs = {
        1: args.phase1_epochs,
        2: args.phase2_epochs,
        3: args.phase3_epochs,
    }

    quality_pretrained = args.quality_pretrained
    prev_ckpt = None

    phases_to_run = [1, 2, 3] if args.phase is None else [args.phase]

    if args.phase == 2 and not quality_pretrained and not prev_ckpt:
        phase1_dir = get_phase_work_dir(base_work_dir, 1)
        ckpt = find_best_checkpoint(phase1_dir)
        if ckpt:
            prev_ckpt = ckpt
            print(f'Auto-detected Phase 1 checkpoint: {ckpt}')

    if args.phase == 3:
        phase2_dir = get_phase_work_dir(base_work_dir, 2)
        ckpt = find_best_checkpoint(phase2_dir)
        if ckpt:
            prev_ckpt = ckpt
            print(f'Auto-detected Phase 2 checkpoint: {ckpt}')

    for phase in phases_to_run:
        print(f'\n{"#"*80}')
        print(f'# Phase {phase}')
        if phase == 1:
            print(f'#   Quality network: FROZEN')
            print(f'#   Training: backbone + private + fusion + head')
            print(f'#   Losses: seg + zc_seg + invariant + disentangle')
        elif phase == 2:
            print(f'#   Quality network: UNFROZEN (lr_mult=0.1)')
            print(f'#   Training: all components jointly')
            print(f'#   Losses: seg + zc_seg + invariant + disentangle + quality_rank + quality_consist')
        elif phase == 3:
            print(f'#   Quality network: UNFROZEN (lr_mult=0.1)')
            print(f'#   Training: all components + degradation consistency')
            print(f'#   Losses: all losses including deg_consist')
        print(f'#   Epochs: {phase_epochs[phase]}')
        print(f'{"#"*80}\n')

        work_dir = train_phase(
            config_path=config_path,
            phase=phase,
            quality_pretrained=quality_pretrained if phase == 1 else None,
            prev_ckpt=prev_ckpt,
            base_work_dir=base_work_dir,
            epochs=phase_epochs[phase],
            lr=PHASE_LR[phase],
            gpu_id=args.gpu_id,
            visualize=args.visualize,
            vis_interval=args.vis_interval,
            vis_num_samples=args.vis_num_samples,
        )

        prev_ckpt = find_best_checkpoint(work_dir)
        if prev_ckpt:
            print(f'Phase {phase} best checkpoint: {prev_ckpt}')
        else:
            print(f'Warning: No checkpoint found for Phase {phase}')

        quality_pretrained = None

    print(f'\n{"="*80}')
    print(f'Progressive training complete!')
    print(f'Results in: {base_work_dir}')
    for phase in phases_to_run:
        wd = get_phase_work_dir(base_work_dir, phase)
        ckpt = find_best_checkpoint(wd)
        print(f'  Phase {phase}: {ckpt}')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
