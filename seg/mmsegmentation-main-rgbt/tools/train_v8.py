"""
MiTMulV8QualityPyramid 训练脚本

训练方式:
  退化在数据管道中完成（RGBTModalDegradation），模型内部不做退化处理。
  质量网络无独立损失，完全通过分割损失的反向传播学习。

  数据管道退化比例:
  - 30% 原图
  - 20% 单模态缺失
  - 20% 全局退化（排除模态缺失）
  - 30% 局部退化

  模型损失:
  - 分割损失: decode.loss_ce + decode.loss_dice
  - 通用特征分割辅助: loss_seg_zc
  - 私有特征残差分割: loss_seg_zp_residual
  - 质量加权不变性损失: loss_invariant
  - 解纠缠正交损失: loss_disentangle_s0~s3
  - 模态分类损失: loss_modality
  - 方差最大化: loss_variance

使用方式:
  cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt

  # MFNet训练
  python tools/train_v8.py mfnet --gpu-id 0

  # FMB训练
  python tools/train_v8.py fmb --gpu-id 0

  # 指定质量网络预训练权重（可选）
  python tools/train_v8.py mfnet \
      --quality-pretrained /path/to/best_val_quality_pyramid_net.pth \
      --gpu-id 0

  # 指定工作目录和epoch数
  python tools/train_v8.py mfnet \
      --work-dir work_dirs/v8_mfnet \
      --epochs 200 \
      --gpu-id 0
"""

import os
import sys
import argparse
import subprocess

DATASETS = {
    'fmb': {
        'config': 'configs/segformer/mitmul_v8_quality_pyramid_mit-b2-b0_1xb2-40K_fmb-480x480.py',
    },
    'mfnet': {
        'config': 'configs/segformer/mitmul_v8_quality_pyramid_mit-b2-b0_1xb2-40K_mfnet-240x320.py',
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='MiTMulV8 Training')
    parser.add_argument('dataset', choices=DATASETS.keys(),
                        help='Dataset name')
    parser.add_argument('--quality-pretrained', type=str, default=None,
                        help='Path to pretrained quality network weights')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Work directory (auto-generated if not set)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=6e-5,
                        help='Learning rate')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--load-from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    dataset_info = DATASETS[args.dataset]
    config_path = dataset_info['config']

    if not os.path.exists(config_path):
        print(f'Config not found: {config_path}')
        sys.exit(1)

    if args.work_dir is None:
        args.work_dir = f'work_dirs/v8_{args.dataset}'

    cfg_options = []
    if args.quality_pretrained:
        cfg_options.append(
            f'model.quality_pretrained={args.quality_pretrained}')
    if args.load_from:
        cfg_options.append(f'load_from={args.load_from}')

    cfg_options.append(f'train_cfg.max_epochs={args.epochs}')
    cfg_options.append(f'optim_wrapper.optimizer.lr={args.lr}')

    cfg_options_str = ''
    if cfg_options:
        cfg_options_str = '--cfg-options ' + ' '.join(cfg_options)

    cmd = (
        f'CUDA_VISIBLE_DEVICES={args.gpu_id} '
        f'python tools/train.py '
        f'{config_path} '
        f'--work-dir {args.work_dir} '
        f'{cfg_options_str}'
    )

    print(f'\n{"="*80}')
    print(f'MiTMulV8 Training')
    print(f'  Dataset: {args.dataset}')
    print(f'  Config: {config_path}')
    print(f'  Quality pretrained: {args.quality_pretrained}')
    print(f'  Epochs: {args.epochs}')
    print(f'  LR: {args.lr}')
    print(f'  Work dir: {args.work_dir}')
    print(f'  GPU: {args.gpu_id}')
    print(f'{"="*80}\n')

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'Training failed with return code {result.returncode}')
        sys.exit(result.returncode)

    print(f'\n{"="*80}')
    print(f'Training complete! Results in: {args.work_dir}')
    print(f'{"="*80}')


if __name__ == '__main__':
    main()
