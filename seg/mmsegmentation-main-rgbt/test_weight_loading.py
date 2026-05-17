"""
权重加载测试脚本：验证 MiTMulV6Baseline 和 MiTMulV6Disentangle 的预训练权重加载

运行方式：
cd /home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt
python test_weight_loading.py
"""
import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

for root, dirs, files in os.walk(PROJECT_ROOT):
    if '__pycache__' in dirs:
        cache_dir = os.path.join(root, '__pycache__')
        shutil.rmtree(cache_dir, ignore_errors=True)

import torch
from mmengine.config import Config
from mmengine.registry import DefaultScope

import mmseg.models
import mmseg.datasets
print('[OK] mmseg.models 和 mmseg.datasets 导入成功')

DefaultScope.get_instance('mmseg', scope_name='mmseg')
print('[OK] DefaultScope 设置为 mmseg')

from mmseg.registry import MODELS


def analyze_checkpoint_keys(ckpt_path, label=''):
    """分析预训练权重文件的 key 格式"""
    print(f'\n--- 分析权重文件: {ckpt_path} {label} ---')
    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] 文件不存在')
        return None

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        prefix = 'state_dict'
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
        prefix = 'model'
    elif isinstance(ckpt, dict):
        state = ckpt
        prefix = 'root'
    else:
        print(f'  [SKIP] 无法识别的格式: {type(ckpt)}')
        return None

    keys = sorted(state.keys())
    print(f'  存储位置: {prefix}')
    print(f'  总 key 数: {len(keys)}')

    top_prefixes = {}
    for k in keys:
        prefix_k = k.split('.')[0]
        top_prefixes[prefix_k] = top_prefixes.get(prefix_k, 0) + 1
    print(f'  顶层前缀分布: {dict(sorted(top_prefixes.items()))}')

    backbone_keys = [k for k in keys if k.startswith('backbone.')]
    decode_head_keys = [k for k in keys if k.startswith('decode_head.')]
    other_keys = [k for k in keys if not k.startswith('backbone.') and not k.startswith('decode_head.')]

    print(f'  backbone.*: {len(backbone_keys)}')
    print(f'  decode_head.*: {len(decode_head_keys)}')
    print(f'  其他: {len(other_keys)}')

    if backbone_keys:
        print(f'  backbone key 示例: {backbone_keys[0]}')
    if decode_head_keys:
        print(f'  decode_head key 示例: {decode_head_keys[0]}')
    if other_keys:
        print(f'  其他 key 示例: {other_keys[0]}')

    return state


def compare_keys(model_state, ckpt_state, model_name, prefix_filter=None):
    """比较模型和预训练权重的 key 匹配情况"""
    print(f'\n--- Key 匹配分析: {model_name} ---')

    if prefix_filter:
        model_keys = {k for k in model_state if k.startswith(prefix_filter)}
    else:
        model_keys = set(model_state.keys())
    ckpt_keys = set(ckpt_state.keys())

    matched = model_keys & ckpt_keys
    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys

    print(f'模型参数数: {len(model_keys)}')
    print(f'预训练参数数: {len(ckpt_keys)}')
    print(f'匹配数: {len(matched)}')
    print(f'模型中有但预训练中没有: {len(missing_in_ckpt)}')
    print(f'预训练中有但模型中没有: {len(extra_in_ckpt)}')

    shape_match = 0
    shape_mismatch = []
    for k in matched:
        if model_state[k].shape == ckpt_state[k].shape:
            shape_match += 1
        else:
            shape_mismatch.append((k, ckpt_state[k].shape, model_state[k].shape))

    print(f'形状匹配: {shape_match}/{len(matched)}')
    if shape_mismatch:
        print(f'形状不匹配 ({len(shape_mismatch)}):')
        for k, cs, ms in shape_mismatch[:10]:
            print(f'  {k}: ckpt={cs}, model={ms}')

    if missing_in_ckpt:
        missing_prefixes = {}
        for k in sorted(missing_in_ckpt):
            parts = k.split('.')
            if len(parts) >= 2:
                p = '.'.join(parts[:2])
            else:
                p = parts[0]
            missing_prefixes[p] = missing_prefixes.get(p, 0) + 1
        print(f'缺失 key 前缀分布: {dict(sorted(missing_prefixes.items()))}')

    if extra_in_ckpt:
        extra_prefixes = {}
        for k in sorted(extra_in_ckpt):
            parts = k.split('.')
            if len(parts) >= 2:
                p = '.'.join(parts[:2])
            else:
                p = parts[0]
            extra_prefixes[p] = extra_prefixes.get(p, 0) + 1
        print(f'多余 key 前缀分布: {dict(sorted(extra_prefixes.items()))}')

    return {
        'matched': len(matched),
        'total_model': len(model_keys),
        'shape_match': shape_match,
        'shape_mismatch': len(shape_mismatch),
    }


def test_baseline():
    """测试 MiTMulV6Baseline 的权重加载"""
    print('\n' + '=' * 70)
    print('测试 MiTMulV6Baseline 权重加载')
    print('=' * 70)

    config_path = os.path.join(
        PROJECT_ROOT,
        'configs/segformer/mitmul_v6_baseline_mit-b2_1xb2-40K_mfnet-480x640.py')

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    model_state = model.state_dict()

    print(f'\n模型类型: {type(model).__name__}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数总数: {total_params:,}')

    model_prefixes = {}
    for k in model_state:
        p = k.split('.')[0]
        model_prefixes[p] = model_prefixes.get(p, 0) + 1
    print(f'模型参数前缀分布: {dict(sorted(model_prefixes.items()))}')

    # 测试1: 纯 backbone 权重 (mit_b2.pth)
    mit_b2_path = os.path.join(PROJECT_ROOT, 'pretrain/mit_b2.pth')
    ckpt_b2 = analyze_checkpoint_keys(mit_b2_path, '(纯 backbone)')

    if ckpt_b2 is not None:
        print('\n--- mit_b2.pth vs MiTMulV6Baseline.backbone ---')
        backbone_state = {k: v for k, v in model_state.items() if k.startswith('backbone.')}

        ckpt_keys_no_prefix = {k.replace('backbone.', ''): k for k in ckpt_b2 if k.startswith('backbone.')}
        if not ckpt_keys_no_prefix:
            ckpt_keys_no_prefix = {k: f'backbone.{k}' for k in ckpt_b2}

        remapped_ckpt = {}
        for k, v in ckpt_b2.items():
            if k.startswith('backbone.'):
                remapped_ckpt[k] = v
            else:
                remapped_ckpt[f'backbone.{k}'] = v

        result = compare_keys(backbone_state, remapped_ckpt, 'Baseline.backbone vs mit_b2.pth')

    # 测试2: 完整 SegFormer 模型权重
    segformer_b2_path = os.path.join(
        PROJECT_ROOT, 'pretrain/segformer_mit-b2_512x512_160k_ade20k.pth')
    ckpt_segformer = analyze_checkpoint_keys(segformer_b2_path, '(完整 SegFormer)')

    if ckpt_segformer is not None:
        print('\n--- segformer_mit-b2 vs MiTMulV6Baseline (完整模型) ---')
        result = compare_keys(model_state, ckpt_segformer, 'Baseline vs segformer_mit-b2')

    # 测试3: 实际加载权重
    print('\n--- 实际加载 mit_b2.pth 到 backbone ---')
    if os.path.exists(mit_b2_path):
        model2 = MODELS.build(cfg.model)
        model2.init_weights()
        print('[OK] init_weights() 完成')

        backbone_loaded = {k: v for k, v in model2.state_dict().items() if k.startswith('backbone.')}
        ckpt_b2_state = torch.load(mit_b2_path, map_location='cpu')
        if isinstance(ckpt_b2_state, dict) and 'state_dict' in ckpt_b2_state:
            ckpt_b2_state = ckpt_b2_state['state_dict']
        elif isinstance(ckpt_b2_state, dict) and 'model' in ckpt_b2_state:
            ckpt_b2_state = ckpt_b2_state['model']

        value_match = 0
        value_mismatch = 0
        for k in backbone_loaded:
            ckpt_key = k.replace('backbone.', '')
            if ckpt_key in ckpt_b2_state:
                if torch.equal(backbone_loaded[k].cpu(), ckpt_b2_state[ckpt_key]):
                    value_match += 1
                else:
                    value_mismatch += 1

        print(f'权重值验证: {value_match} 匹配, {value_mismatch} 不匹配 / {len(backbone_loaded)} 总 backbone 参数')
        if value_mismatch == 0 and value_match > 0:
            print('[OK] backbone 权重值与预训练完全一致!')
    else:
        print(f'[SKIP] mit_b2.pth 不存在')

    # 测试4: 前向推理
    print('\n--- 前向推理测试 ---')
    model.eval()
    dummy_input = torch.randn(2, 6, 480, 640)
    with torch.no_grad():
        try:
            output = model.whole_inference(
                dummy_input,
                [dict(ori_shape=(480, 640), img_shape=(480, 640),
                      pad_shape=(480, 640), padding_size=[0, 0, 0, 0])] * 2)
            print(f'[OK] 前向推理成功! 输出 shape: {output.shape}')
        except Exception as e:
            print(f'[FAIL] 前向推理失败: {e}')
            import traceback
            traceback.print_exc()


def test_disentangle():
    """测试 MiTMulV6Disentangle 的权重加载"""
    print('\n' + '=' * 70)
    print('测试 MiTMulV6Disentangle 权重加载')
    print('=' * 70)

    config_path = os.path.join(
        PROJECT_ROOT,
        'configs/segformer/mitmul_v6_disentangle_mit-b2-b0_1xb2-40K_mfnet-480x640.py')

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    model_state = model.state_dict()

    print(f'\n模型类型: {type(model).__name__}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数总数: {total_params:,}')

    model_prefixes = {}
    for k in model_state:
        p = k.split('.')[0]
        model_prefixes[p] = model_prefixes.get(p, 0) + 1
    print(f'模型参数前缀分布: {dict(sorted(model_prefixes.items()))}')

    # 分析各子模块参数
    for module_name in ['backbone', 'private_branch_rgb', 'private_branch_t',
                        'decode_head', 'fusion_mlps', 'channel_attn',
                        'spatial_attn', 'self_attn_universal', 'self_attn_rgb',
                        'self_attn_t', 'private_proj_rgb', 'private_proj_t',
                        'cross_attn_rgb', 'cross_attn_t', 'cross_proj_rgb',
                        'cross_proj_t', 'zc_seg_head_mlp']:
        count = sum(1 for k in model_state if k.startswith(f'{module_name}.'))
        if count > 0:
            params = sum(p.numel() for n, p in model.named_parameters() if n.startswith(f'{module_name}.'))
            print(f'  {module_name}: {count} keys, {params:,} params')

    # 测试1: 通用分支 backbone (MiT-B2) vs mit_b2.pth
    mit_b2_path = os.path.join(PROJECT_ROOT, 'pretrain/mit_b2.pth')
    ckpt_b2 = analyze_checkpoint_keys(mit_b2_path, '(纯 backbone B2)')

    if ckpt_b2 is not None:
        print('\n--- mit_b2.pth vs Disentangle.backbone (通用分支) ---')
        backbone_state = {k: v for k, v in model_state.items() if k.startswith('backbone.')}
        remapped_ckpt = {}
        for k, v in ckpt_b2.items():
            if k.startswith('backbone.'):
                remapped_ckpt[k] = v
            else:
                remapped_ckpt[f'backbone.{k}'] = v
        result = compare_keys(backbone_state, remapped_ckpt, 'Disentangle.backbone vs mit_b2.pth')

    # 测试2: 特有分支 (MiT-B0) vs mit_b0.pth
    mit_b0_path = os.path.join(PROJECT_ROOT, 'pretrain/mit_b0.pth')
    ckpt_b0 = analyze_checkpoint_keys(mit_b0_path, '(纯 backbone B0)')

    if ckpt_b0 is not None:
        print('\n--- mit_b0.pth vs Disentangle.private_branch_rgb ---')
        rgb_state = {k.replace('private_branch_rgb.', '', 1): v
                     for k, v in model_state.items()
                     if k.startswith('private_branch_rgb.')}
        remapped_b0 = {}
        for k, v in ckpt_b0.items():
            if k.startswith('backbone.'):
                remapped_b0[k.replace('backbone.', '', 1)] = v
            else:
                remapped_b0[k] = v
        result = compare_keys(rgb_state, remapped_b0, 'Disentangle.private_branch_rgb vs mit_b0.pth')

        print('\n--- mit_b0.pth vs Disentangle.private_branch_t ---')
        t_state = {k.replace('private_branch_t.', '', 1): v
                   for k, v in model_state.items()
                   if k.startswith('private_branch_t.')}
        result = compare_keys(t_state, remapped_b0, 'Disentangle.private_branch_t vs mit_b0.pth')

    # 测试3: 完整 SegFormer 模型权重 vs Disentangle
    segformer_b2_path = os.path.join(
        PROJECT_ROOT, 'pretrain/segformer_mit-b2_512x512_160k_ade20k.pth')
    ckpt_segformer = analyze_checkpoint_keys(segformer_b2_path, '(完整 SegFormer B2)')

    if ckpt_segformer is not None:
        print('\n--- segformer_mit-b2 vs Disentangle (完整模型) ---')
        result = compare_keys(model_state, ckpt_segformer, 'Disentangle vs segformer_mit-b2')

    # 测试4: 实际加载权重
    print('\n--- 实际加载 mit_b2.pth 到通用分支 backbone ---')
    if os.path.exists(mit_b2_path):
        model2 = MODELS.build(cfg.model)
        model2.init_weights()
        print('[OK] init_weights() 完成')

        backbone_loaded = {k: v for k, v in model2.state_dict().items() if k.startswith('backbone.')}
        ckpt_b2_state = torch.load(mit_b2_path, map_location='cpu')
        if isinstance(ckpt_b2_state, dict) and 'state_dict' in ckpt_b2_state:
            ckpt_b2_state = ckpt_b2_state['state_dict']
        elif isinstance(ckpt_b2_state, dict) and 'model' in ckpt_b2_state:
            ckpt_b2_state = ckpt_b2_state['model']

        value_match = 0
        value_mismatch = 0
        for k in backbone_loaded:
            ckpt_key = k.replace('backbone.', '')
            if ckpt_key in ckpt_b2_state:
                if torch.equal(backbone_loaded[k].cpu(), ckpt_b2_state[ckpt_key]):
                    value_match += 1
                else:
                    value_mismatch += 1

        print(f'通用分支权重值验证: {value_match} 匹配, {value_mismatch} 不匹配 / {len(backbone_loaded)} 总参数')

    # 测试5: 特有分支是否有预训练权重
    print('\n--- 特有分支预训练权重检查 ---')
    if os.path.exists(mit_b0_path):
        print(f'mit_b0.pth 存在，但配置中 private_branch_rgb/t 没有 init_cfg')
        print(f'当前 private_branch_rgb 配置:')
        print(f'  type: {cfg.model.private_branch_rgb.type}')
        print(f'  init_cfg: {cfg.model.private_branch_rgb.get("init_cfg", "None")}')

        print(f'\n建议: 为 private_branch_rgb/t 添加 init_cfg 以加载 mit_b0.pth')
        print(f'  private_branch_rgb=dict(')
        print(f'      type="MixVisionTransformer",')
        print(f'      ...')
        print(f'      init_cfg=dict(type="Pretrained", checkpoint="./pretrain/mit_b0.pth"))')
    else:
        print(f'mit_b0.pth 不存在，特有分支只能随机初始化')

    # 测试6: 前向推理
    print('\n--- 前向推理测试 ---')
    model.eval()
    dummy_input = torch.randn(2, 6, 480, 640)
    with torch.no_grad():
        try:
            output = model.whole_inference(
                dummy_input,
                [dict(ori_shape=(480, 640), img_shape=(480, 640),
                      pad_shape=(480, 640), padding_size=[0, 0, 0, 0])] * 2)
            print(f'[OK] 前向推理成功! 输出 shape: {output.shape}')
        except Exception as e:
            print(f'[FAIL] 前向推理失败: {e}')
            import traceback
            traceback.print_exc()

    # 测试7: 损失计算
    print('\n--- 损失计算测试 ---')
    model.train()
    dummy_input = torch.randn(2, 6, 480, 640)
    dummy_gt = torch.randint(0, 9, (2, 480, 640))
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    data_samples = []
    for i in range(2):
        data_sample = SegDataSample()
        gt_sem_seg = PixelData()
        gt_sem_seg.data = dummy_gt[i]
        data_sample.gt_sem_seg = gt_sem_seg
        data_sample.set_metainfo(dict(
            ori_shape=(480, 640), img_shape=(480, 640),
            pad_shape=(480, 640), padding_size=[0, 0, 0, 0]))
        data_samples.append(data_sample)

    try:
        losses = model.loss(dummy_input, data_samples)
        print(f'[OK] 损失计算成功!')
        for k, v in losses.items():
            print(f'  {k}: {v.item():.6f}')
    except Exception as e:
        print(f'[FAIL] 损失计算失败: {e}')
        import traceback
        traceback.print_exc()


def main():
    test_baseline()
    test_disentangle()

    print('\n' + '=' * 70)
    print('全部测试完成!')
    print('=' * 70)


if __name__ == '__main__':
    main()
