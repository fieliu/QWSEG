"""
测试脚本：验证 MiTMulV6Baseline 预训练权重加载 + 数据集前向测试

运行方式：
cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
python test_baseline_simple.py
"""
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 清除 __pycache__ 缓存，避免旧代码干扰
for root, dirs, files in os.walk(PROJECT_ROOT):
    if '__pycache__' in dirs:
        cache_dir = os.path.join(root, '__pycache__')
        shutil.rmtree(cache_dir, ignore_errors=True)

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope

# ============================================================
# 关键步骤1：在 Config.fromfile() 之前导入 mmseg，
# 触发 mmseg/models/__init__.py 中的所有 @MODELS.register_module()
# 这样 SegDataPreProcessor、MixVisionTransformer、
# SegformerHead、MiTMulV6Baseline 等都会被注册
# ============================================================
try:
    import mmseg.models
    import mmseg.datasets
    print('[OK] mmseg.models 和 mmseg.datasets 导入成功，所有模块已注册')
except Exception as e:
    print(f'[FAIL] 导入 mmseg 失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 关键步骤2：设置 DefaultScope 为 'mmseg'
# BaseModel.__init__ 内部使用 mmengine.registry.MODELS.build()
# 来构建 data_preprocessor，如果不设置 DefaultScope，
# mmengine 的全局 registry 找不到 mmseg 子 registry 中注册的
# SegDataPreProcessor
# ============================================================
DefaultScope.get_instance('mmseg', scope_name='mmseg')
print('[OK] DefaultScope 设置为 mmseg')

from mmseg.registry import MODELS, DATASETS


def download_pretrained(pretrain_path):
    if os.path.exists(pretrain_path):
        print(f'[OK] 预训练权重已存在: {pretrain_path}')
        return True

    pretrain_dir = os.path.dirname(pretrain_path)
    os.makedirs(pretrain_dir, exist_ok=True)

    url = 'https://download.openmmlab.com/mmsegmentation/v0.5/mit/mit_b2.pth'
    print(f'[INFO] 下载预训练权重: {url}')
    print(f'[INFO] 保存到: {pretrain_path}')

    try:
        import urllib.request
        urllib.request.urlretrieve(url, pretrain_path)
        fsize = os.path.getsize(pretrain_path)
        if fsize > 1e6:
            print(f'[OK] 下载成功 ({fsize / 1e6:.1f} MB)')
            return True
        else:
            print(f'[FAIL] 文件过小 ({fsize} bytes)，可能下载不完整')
            os.remove(pretrain_path)
            return False
    except Exception as e:
        print(f'[FAIL] 下载失败: {e}')
        print(f'请手动下载: wget {url} -O {pretrain_path}')
        return False


def check_pretrained_keys(model, ckpt_path):
    print('\n' + '=' * 60)
    print('[1] 预训练权重 Key 匹配检查')
    print('=' * 60)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt_state = ckpt['state_dict']
    elif 'model' in ckpt:
        ckpt_state = ckpt['model']
    else:
        ckpt_state = ckpt
    ckpt_keys = set(ckpt_state.keys())

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    backbone_keys = {k for k in model_keys if k.startswith('backbone.')}
    other_keys = model_keys - backbone_keys

    print(f'\n模型总参数数: {len(model_keys)}')
    print(f'  backbone 参数数: {len(backbone_keys)}')
    print(f'  非backbone 参数数: {len(other_keys)} (fusion_mlps, decode_head 等)')
    print(f'预训练权重数: {len(ckpt_keys)}')

    print(f'\n预训练权重 key 示例 (前5个):')
    for k in sorted(ckpt_keys)[:5]:
        print(f'  {k}: {ckpt_state[k].shape}')

    print(f'\n模型 backbone key 示例 (前5个):')
    for k in sorted(backbone_keys)[:5]:
        print(f'  {k}: {model_state[k].shape}')

    matched = ckpt_keys & backbone_keys
    missing = backbone_keys - ckpt_keys
    extra = ckpt_keys - model_keys

    print(f'\n直接匹配数: {len(matched)}/{len(backbone_keys)}')

    if len(matched) == 0 and len(ckpt_keys) > 0:
        ckpt_sample = sorted(ckpt_keys)[0]
        model_sample = sorted(backbone_keys)[0]
        print(f'\n[WARN] 没有直接匹配的 key!')
        print(f'  预训练 key 格式: {ckpt_sample}')
        print(f'  模型 key 格式: {model_sample}')

        if not ckpt_sample.startswith('backbone.') and model_sample.startswith('backbone.'):
            print(f'\n[INFO] 检测到 key 前缀不匹配，尝试添加 backbone. 前缀...')
            remapped_ckpt_keys = {f'backbone.{k}' for k in ckpt_keys}
            matched = remapped_ckpt_keys & backbone_keys
            missing = backbone_keys - remapped_ckpt_keys
            extra = remapped_ckpt_keys - model_keys
            print(f'  添加前缀后匹配数: {len(matched)}/{len(backbone_keys)}')

    if missing:
        print(f'\n缺失的 backbone keys ({len(missing)}):')
        for k in sorted(missing)[:10]:
            print(f'  - {k}')
        if len(missing) > 10:
            print(f'  ... 共 {len(missing)} 个')
    else:
        print('\n[OK] backbone 全部匹配!')

    shape_match = 0
    shape_mismatch = []
    for k in matched:
        ckpt_key = k
        if ckpt_key not in ckpt_state:
            ckpt_key = k.replace('backbone.', '')
        if ckpt_key in ckpt_state:
            ckpt_shape = ckpt_state[ckpt_key].shape
        else:
            continue
        model_shape = model_state[k].shape
        if ckpt_shape == model_shape:
            shape_match += 1
        else:
            shape_mismatch.append((k, ckpt_shape, model_shape))

    print(f'\n形状匹配数: {shape_match}/{len(matched)}')
    if shape_mismatch:
        print(f'  形状不匹配 ({len(shape_mismatch)}):')
        for k, cs, ms in shape_mismatch[:5]:
            print(f'    - {k}: ckpt={cs}, model={ms}')
    else:
        print('  [OK] 形状全部匹配!')

    return len(matched) == len(backbone_keys) and len(shape_mismatch) == 0


def test_init_weights(cfg):
    print('\n' + '=' * 60)
    print('[2] 通过 MMEngine 框架加载预训练权重')
    print('=' * 60)

    model = MODELS.build(cfg.model)

    print(f'模型类型: {type(model).__name__}')
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith('backbone.'))
    other_params = total_params - backbone_params
    print(f'模型参数总数: {total_params:,}')
    print(f'  backbone 参数: {backbone_params:,}')
    print(f'  其他参数: {other_params:,}')

    pretrain_path = cfg.model.backbone.init_cfg.checkpoint
    if os.path.exists(pretrain_path):
        print(f'\n调用 model.init_weights() ...')
        model.init_weights()
        print('[OK] init_weights() 调用完成')

        backbone_state = {k: v for k, v in model.state_dict().items()
                          if k.startswith('backbone.')}
        ckpt = torch.load(pretrain_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt_state = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt_state = ckpt['model']
        else:
            ckpt_state = ckpt

        match_count = 0
        mismatch_count = 0
        for k in backbone_state:
            ckpt_key = k.replace('backbone.', '') if k.startswith('backbone.') else k
            if ckpt_key in ckpt_state:
                if torch.equal(backbone_state[k].cpu(), ckpt_state[ckpt_key]):
                    match_count += 1
                else:
                    mismatch_count += 1

        print(f'\n权重值验证:')
        print(f'  完全匹配: {match_count}/{len(backbone_state)}')
        if mismatch_count:
            print(f'  值不匹配: {mismatch_count}')
        else:
            print('  [OK] 所有 backbone 权重值与预训练完全一致!')
    else:
        print(f'[SKIP] 预训练文件不存在: {pretrain_path}')

    return model


def test_forward(model, cfg):
    print('\n' + '=' * 60)
    print('[3] 数据集前向推理测试')
    print('=' * 60)

    try:
        dataset = DATASETS.build(cfg.val_dataloader.dataset)
        print(f'验证集样本数: {len(dataset)}')

        sample = dataset[0]
        img = sample['inputs']
        data_samples = [sample['data_samples']]
        print(f'输入图像 shape: {img.shape}')

        img_batch = img.unsqueeze(0)
        if torch.cuda.is_available():
            model = model.cuda()
            img_batch = img_batch.cuda()

        model.eval()
        with torch.no_grad():
            results = model.predict(img_batch, data_samples)
            print(f'[OK] 前向推理成功!')
            if results:
                result = results[0]
                if hasattr(result, 'pred_sem_seg') and result.pred_sem_seg is not None:
                    pred = result.pred_sem_seg.data
                    print(f'  预测 shape: {pred.shape}')
                    print(f'  预测类别: {torch.unique(pred).tolist()}')
    except Exception as e:
        print(f'[FAIL] 前向推理失败: {e}')
        import traceback
        traceback.print_exc()


def test_loss(model, cfg):
    print('\n' + '=' * 60)
    print('[4] 损失计算测试')
    print('=' * 60)

    try:
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        sample = dataset[0]

        img = sample['inputs'].unsqueeze(0)
        data_samples = [sample['data_samples']]

        if torch.cuda.is_available():
            img = img.cuda()

        model.train()
        losses = model.loss(img, data_samples)
        print(f'[OK] 损失计算成功!')
        for k, v in losses.items():
            print(f'  {k}: {v.item():.6f}')
    except Exception as e:
        print(f'[FAIL] 损失计算失败: {e}')
        import traceback
        traceback.print_exc()


def main():
    config_path = os.path.join(
        PROJECT_ROOT,
        'configs/segformer/mitmul_v6_baseline_mit-b2_1xb2-40K_mfnet-480x640.py')

    print('加载配置...')
    cfg = Config.fromfile(config_path)

    pretrain_path = cfg.model.backbone.init_cfg.checkpoint
    print(f'预训练权重路径: {pretrain_path}')

    if not download_pretrained(pretrain_path):
        print('[SKIP] 无法下载预训练权重')
        return

    # 先构建模型检查 key 匹配
    model = MODELS.build(cfg.model)
    check_pretrained_keys(model, pretrain_path)

    # 重新构建模型并加载权重
    model = test_init_weights(cfg)

    # 数据集前向测试
    test_forward(model, cfg)

    # 损失计算测试
    test_loss(model, cfg)

    print('\n' + '=' * 60)
    print('全部测试完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()
