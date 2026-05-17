import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.structures import PixelData

from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class RGBTv1SAM2(BaseSegmentor):

    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 num_classes: int = 9,
                 pretrained: Optional[str] = None,
                 clip_checkpoint: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 label_feature_path: Optional[str] = None,
                 fusion_type: str = 'mlp',
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.num_classes = num_classes
        self.fusion_type = fusion_type

        self.backbone = MODELS.build(backbone)

        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.out_channels = self.decode_head.num_classes

        d_model = self.backbone.d_model
        num_stages = len(self.backbone.out_indices)

        if fusion_type == 'mlp':
            self.fusion_mlps = nn.ModuleList()
            for _ in range(num_stages):
                self.fusion_mlps.append(nn.Sequential(
                    nn.Conv2d(d_model * 2, d_model, 1, bias=False),
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(inplace=True)))
        elif fusion_type == 'add':
            self.fusion_mlps = None
        else:
            raise ValueError(f'Unknown fusion_type: {fusion_type}')

        self.class_names = class_names
        self.clip_checkpoint = clip_checkpoint
        self.label_feature_path = label_feature_path
        self._clip_model = None
        self._label_feature_cache = None

        if pretrained is not None:
            self._load_sam2_pretrained(pretrained)

        self._init_label_feature()

    def _load_sam2_pretrained(self, pretrained_path):
        try:
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            trunk_state = {}
            neck_state = {}
            for k, v in state_dict.items():
                if k.startswith('sam2_image_encoder.trunk.'):
                    new_key = k.replace('sam2_image_encoder.trunk.', '')
                    trunk_state[new_key] = v
                elif k.startswith('sam2_image_encoder.neck.'):
                    new_key = k.replace('sam2_image_encoder.neck.', '')
                    neck_state[new_key] = v
                elif k.startswith('image_encoder.trunk.'):
                    new_key = k.replace('image_encoder.trunk.', '')
                    trunk_state[new_key] = v
                elif k.startswith('image_encoder.neck.'):
                    new_key = k.replace('image_encoder.neck.', '')
                    neck_state[new_key] = v
                elif k.startswith('sam2_image_encoder.'):
                    new_key = k.replace('sam2_image_encoder.', '')
                    if new_key.startswith('trunk.'):
                        trunk_state[new_key.replace('trunk.', '')] = v
                    elif new_key.startswith('neck.'):
                        neck_state[new_key.replace('neck.', '')] = v

            if trunk_state:
                model_state = self.backbone.trunk.state_dict()
                loaded_keys = []
                missing_keys = []
                for ckpt_key, ckpt_val in trunk_state.items():
                    if ckpt_key in model_state:
                        if model_state[ckpt_key].shape == ckpt_val.shape:
                            model_state[ckpt_key] = ckpt_val
                            loaded_keys.append(ckpt_key)
                        else:
                            missing_keys.append(ckpt_key)
                    else:
                        lora_key = ckpt_key.replace('.weight', '.original_linear.weight')
                        lora_key = lora_key.replace('.bias', '.original_linear.bias')
                        if lora_key in model_state:
                            if model_state[lora_key].shape == ckpt_val.shape:
                                model_state[lora_key] = ckpt_val
                                loaded_keys.append(lora_key)
                            else:
                                missing_keys.append(ckpt_key)
                        elif 'attn.qkv.' in ckpt_key:
                            split_keys = self._split_qkv_weight(
                                ckpt_key, ckpt_val, model_state)
                            if split_keys:
                                loaded_keys.extend(split_keys)
                            else:
                                missing_keys.append(ckpt_key)
                        else:
                            missing_keys.append(ckpt_key)

                self.backbone.trunk.load_state_dict(model_state)
                print(f'Loaded trunk: {len(loaded_keys)}/{len(trunk_state)} keys from '
                      f'{pretrained_path}')
                if missing_keys:
                    print(f'Trunk missing/unmatched keys ({len(missing_keys)}): '
                          f'{missing_keys[:5]}...')
            else:
                print(f'No trunk keys found in {pretrained_path}')

            if neck_state:
                neck_model_state = self.backbone.neck.state_dict()
                neck_loaded = []
                neck_missing = []
                for ckpt_key, ckpt_val in neck_state.items():
                    if ckpt_key in neck_model_state:
                        if neck_model_state[ckpt_key].shape == ckpt_val.shape:
                            neck_model_state[ckpt_key] = ckpt_val
                            neck_loaded.append(ckpt_key)
                        else:
                            neck_missing.append(ckpt_key)
                    else:
                        neck_missing.append(ckpt_key)

                self.backbone.neck.load_state_dict(neck_model_state)
                print(f'Loaded neck: {len(neck_loaded)}/{len(neck_state)} keys from '
                      f'{pretrained_path}')
                if neck_missing:
                    print(f'Neck missing/unmatched keys ({len(neck_missing)}): '
                          f'{neck_missing[:5]}...')
            else:
                print(f'No neck keys found in {pretrained_path}, neck uses random init')
        except Exception as e:
            print(f'Failed to load SAM2 pretrained weights from '
                  f'{pretrained_path}: {e}')

    @staticmethod
    def _split_qkv_weight(ckpt_key, ckpt_val, model_state):
        is_weight = ckpt_key.endswith('.weight')
        is_bias = ckpt_key.endswith('.bias')
        if not is_weight and not is_bias:
            return []

        base_key = ckpt_key.replace('.weight', '').replace('.bias', '')
        suffix = '.weight' if is_weight else '.bias'
        dim_out = ckpt_val.shape[0] // 3

        q_base = base_key.replace('attn.qkv', 'attn.q_proj')
        k_base = base_key.replace('attn.qkv', 'attn.k_proj')
        v_base = base_key.replace('attn.qkv', 'attn.v_proj')

        def _find_key(name_base, sfx):
            candidates = [name_base + sfx,
                          name_base + '.original_linear' + sfx]
            for c in candidates:
                if c in model_state:
                    return c
            return None

        q_key = _find_key(q_base, suffix)
        k_key = _find_key(k_base, suffix)
        v_key = _find_key(v_base, suffix)

        if q_key is None or k_key is None or v_key is None:
            return []

        model_state[q_key] = ckpt_val[:dim_out]
        model_state[k_key] = ckpt_val[dim_out:2 * dim_out]
        model_state[v_key] = ckpt_val[2 * dim_out:]

        return [q_key, k_key, v_key]

    def _init_label_feature(self):
        if self.label_feature_path is not None:
            try:
                label_feature = torch.load(self.label_feature_path)
                self.register_buffer('label_feature', label_feature)
                self._label_feature_cache = label_feature
                print(f'Loaded label features from {self.label_feature_path}')
            except Exception as e:
                print(f'Failed to load label features from '
                      f'{self.label_feature_path}: {e}')
                self.label_feature = None
        else:
            self.label_feature = None

    def _get_label_feature(self):
        if self._label_feature_cache is not None:
            return self._label_feature_cache.to(next(self.parameters()).device)

        if self.label_feature is not None:
            return self.label_feature.to(next(self.parameters()).device)

        if self.clip_checkpoint is not None and self.class_names is not None:
            return self._compute_clip_text_features()

        raise ValueError('Either label_feature_path or (clip_checkpoint + class_names) '
                         'must be provided')

    def _compute_clip_text_features(self):
        if self._label_feature_cache is not None:
            return self._label_feature_cache

        from mmseg.models.text_encoder.clip import clip

        if self._clip_model is None:
            device = next(self.parameters()).device
            model, _ = clip.load(self.clip_checkpoint, device=device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._clip_model = model

        device = next(self.parameters()).device
        templates = [
            'a photo of a {}.', 'a photograph of a {}.',
            'an image of a {}.', 'a picture of a {}.',
            'the photo of a {}.', 'the photograph of a {}.',
            'the image of a {}.', 'the picture of a {}.',
        ]

        text_embeds_list = []
        for template in templates:
            texts = [template.format(name) for name in self.class_names]
            tokens = clip.tokenize(texts).to(device)
            with torch.no_grad():
                features = self._clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            text_embeds_list.append(features)

        text_features = torch.stack(text_embeds_list).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._label_feature_cache = text_features.to(next(self.parameters()).dtype)
        return self._label_feature_cache

    def extract_feat(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        B = inputs.shape[0] // 2
        x_rgbt = self.backbone(inputs)

        x_rgb_list = [feat[:B] for feat in x_rgbt]
        x_t_list = [feat[B:] for feat in x_rgbt]

        num_stages = len(x_rgb_list)
        fused_feats = []
        for i in range(num_stages):
            if self.fusion_type == 'mlp':
                concat_feat = torch.cat([x_rgb_list[i], x_t_list[i]], dim=1)
                fused_feat = self.fusion_mlps[i](concat_feat)
            elif self.fusion_type == 'add':
                fused_feat = x_rgb_list[i] + x_t_list[i]
            else:
                raise ValueError(f'Unknown fusion_type: {self.fusion_type}')
            fused_feats.append(fused_feat)

        return fused_feats

    def _prepare_rgbt_input(self, inputs):
        input_rgb = inputs[:, 0:3, :, :]
        input_ir = inputs[:, 3:6, :, :]
        return torch.cat([input_rgb, input_ir], dim=0)

    def encode_decode(self, inputs, batch_img_metas):
        label_feature = self._get_label_feature()
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        seg_logits = self.decode_head(x, label_feature)
        return seg_logits

    def loss(self, inputs, data_samples):
        label_feature = self._get_label_feature()
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        seg_logits = self.decode_head(x, label_feature)

        losses = dict()
        seg_label = self._stack_batch_gt(data_samples)
        if seg_label.ndim == 4 and seg_label.shape[1] == 1:
            seg_label = seg_label.squeeze(1)
        seg_logits_resized = F.interpolate(
            seg_logits, size=seg_label.shape[1:],
            mode='bilinear', align_corners=self.align_corners)

        if isinstance(self.decode_head.loss_decode, nn.ModuleList):
            for loss_module in self.decode_head.loss_decode:
                if loss_module.loss_name == 'loss_ce':
                    losses[loss_module.loss_name] = loss_module(
                        seg_logits_resized, seg_label,
                        ignore_index=self.decode_head.ignore_index)
                else:
                    losses[loss_module.loss_name] = loss_module(
                        seg_logits_resized, seg_label)
        else:
            losses[self.decode_head.loss_decode.loss_name] = \
                self.decode_head.loss_decode(
                    seg_logits_resized, seg_label,
                    ignore_index=self.decode_head.ignore_index)

        from mmseg.models.losses import accuracy
        losses['acc_seg'] = accuracy(
            seg_logits_resized, seg_label,
            ignore_index=self.decode_head.ignore_index)

        return losses

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            if i < len(data_samples):
                img_meta = data_samples[i].metainfo if hasattr(data_samples[i], 'metainfo') else {}
                ori_shape = img_meta.get('ori_shape', seg_logits.shape[2:])
                seg_logit = seg_logits[i]
                seg_logit = F.interpolate(
                    seg_logit.unsqueeze(0),
                    size=ori_shape[:2] if isinstance(ori_shape, (tuple, list)) else (ori_shape, ori_shape),
                    mode='bilinear',
                    align_corners=self.align_corners).squeeze(0)
            else:
                seg_logit = seg_logits[i]

            data_samples[i].set_data({
                'pred_sem_seg': PixelData(data=seg_logit.argmax(dim=0)),
                'seg_logits': PixelData(data=seg_logit),
            })

        return data_samples

    def _forward(self, inputs, data_samples=None):
        label_feature = self._get_label_feature()
        input_rgbt = self._prepare_rgbt_input(inputs)
        x = self.extract_feat(input_rgbt)
        seg_logits = self.decode_head(x, label_feature)
        return seg_logits

    @staticmethod
    def _stack_batch_gt(batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def get_lora_info(self) -> Dict:
        lora_params = 0
        frozen_params = 0
        trainable_params = 0
        total_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if 'lora_A' in name or 'lora_B' in name:
                lora_params += num_params
                trainable_params += num_params
            elif param.requires_grad:
                trainable_params += num_params
            else:
                frozen_params += num_params

        return dict(
            lora_params=lora_params,
            frozen_params=frozen_params,
            trainable_params=trainable_params,
            total_params=total_params,
            lora_ratio=lora_params / total_params * 100,
            trainable_ratio=trainable_params / total_params * 100)

    def merge_lora(self):
        if hasattr(self.backbone, 'merge_lora'):
            self.backbone.merge_lora()
