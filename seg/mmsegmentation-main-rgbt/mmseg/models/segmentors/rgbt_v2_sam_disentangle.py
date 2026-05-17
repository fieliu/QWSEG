import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from mmengine.logging import print_log
from mmengine.model.weight_init import trunc_normal_

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from .rgbt_v2_disentangle import (compute_hsic, compute_infonce,
                                  CrossAttentionFusion)


@MODELS.register_module()
class RGBTv2SAMDisentangle(BaseSegmentor):

    def __init__(self,
                 universal_branch: ConfigType,
                 private_branch_rgb: ConfigType,
                 private_branch_t: ConfigType,
                 decode_head: ConfigType,
                 zc_seg_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 loss_seg_zc_weight=0.3,
                 loss_modal_weight=0.2,
                 loss_disentangle_weights=(0.1, 0.2, 0.3, 0.4),
                 loss_invariance_weight=0.01):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if pretrained is not None:
            universal_branch['pretrained'] = pretrained

        self.universal_branch = MODELS.build(universal_branch)
        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_zc_seg_head(zc_seg_head)
        self._init_auxiliary_head(auxiliary_head)

        num_stages = 4
        self.cross_attn_rgb = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=768)
             for _ in range(num_stages)])
        self.cross_attn_t = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=768)
             for _ in range(num_stages)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        trunc_normal_(self.cls_token, std=.02)

        self.loss_seg_zc_weight = loss_seg_zc_weight
        self.loss_modal_weight = loss_modal_weight
        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.loss_invariance_weight = loss_invariance_weight

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def init_weights(self):
        if self.init_cfg is not None and 'checkpoint' in self.init_cfg:
            checkpoint_path = self.init_cfg['checkpoint']
            print_log(
                f'Loading checkpoint from: {checkpoint_path}',
                logger='current')

            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt

                new_state_dict = {}
                loaded_keys = []
                skipped_keys = []

                for k, v in state_dict.items():
                    if k.startswith('backbone.'):
                        new_key = 'universal_branch.' + k
                    elif k.startswith('neck.'):
                        new_key = 'universal_branch.' + k
                    elif k.startswith('decode_head.'):
                        new_key = 'universal_branch.' + k
                    elif k.startswith('auxiliary_head.'):
                        continue
                    else:
                        new_key = k

                    if new_key in self.state_dict().keys():
                        if self.state_dict()[new_key].shape == v.shape:
                            new_state_dict[new_key] = v
                            loaded_keys.append(new_key)
                        else:
                            skipped_keys.append(
                                (new_key, v.shape,
                                 self.state_dict()[new_key].shape))

                print_log(
                    f'Loaded {len(loaded_keys)} keys from checkpoint',
                    logger='current')
                if skipped_keys:
                    print_log(
                        f'Skipped {len(skipped_keys)} keys due to shape '
                        f'mismatch:', logger='current')
                    for key, src_shape, dst_shape in skipped_keys[:10]:
                        print_log(
                            f'  {key}: {src_shape} -> {dst_shape}',
                            logger='current')

                self.load_state_dict(new_state_dict, strict=False)

                print_log(
                    'Successfully initialized universal branch from checkpoint',
                    logger='current')

            except Exception as e:
                print_log(
                    f'Failed to load checkpoint: {e}', logger='current',
                    level=logging.WARNING)
                super().init_weights()
        else:
            self.universal_branch.init_weights()
            super().init_weights()

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_zc_seg_head(self, zc_seg_head: ConfigType) -> None:
        self.zc_seg_head = MODELS.build(zc_seg_head)

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _get_sam_embeddings(self, x):
        backbone = self.universal_branch.backbone
        tokens = backbone.patch_embed(x)
        if backbone.pos_embed is not None:
            tokens = tokens + backbone.pos_embed
        hw_shape = (tokens.shape[1], tokens.shape[2])
        return tokens, hw_shape

    def _prepare_tokens_for_private(self, tokens):
        B, H, W, C = tokens.shape
        tokens_flat = tokens.reshape(B, H * W, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens_with_cls = torch.cat([cls_tokens, tokens_flat], dim=1)
        return tokens_with_cls

    def _run_sam_universal_stages(self, tokens, hw_shape):
        backbone = self.universal_branch.backbone
        outs = []
        for i, blk in enumerate(backbone.blocks):
            tokens = blk(tokens)
            if i in backbone.out_indices:
                out = tokens.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def extract_feat(self, inputs: torch.Tensor):
        B, C, H, W = inputs.shape
        assert C == 6, f'Input should be 6-channel (RGB+Thermal), got {C}'

        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        rgb_tokens, hw_shape = self._get_sam_embeddings(input_rgb)
        t_tokens, _ = self._get_sam_embeddings(input_ir)

        rgb_tokens_with_cls = self._prepare_tokens_for_private(rgb_tokens)
        t_tokens_with_cls = self._prepare_tokens_for_private(t_tokens)

        zp_rgb_list = self.private_branch_rgb(rgb_tokens_with_cls, hw_shape)
        zp_t_list = self.private_branch_t(t_tokens_with_cls, hw_shape)

        tokens_combined = torch.cat(
            [rgb_tokens, t_tokens], dim=0)
        zc_combined = self._run_sam_universal_stages(
            tokens_combined, hw_shape)
        zc_rgb_list = [f[:B] for f in zc_combined]
        zc_t_list = [f[B:] for f in zc_combined]

        num_stages = len(zc_rgb_list)
        assert len(zp_rgb_list) == num_stages

        zc_fused_feats = []
        for i in range(num_stages):
            zc_fused_feats.append(zc_rgb_list[i] + zc_t_list[i])

        fused_feats = []
        for i in range(num_stages):
            H_i, W_i = zc_rgb_list[i].shape[-2:]
            zc_sum_tokens = zc_fused_feats[i].flatten(2).transpose(1, 2)
            zp_rgb_tokens = zp_rgb_list[i].flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_list[i].flatten(2).transpose(1, 2)

            fused_rgb_tokens = self.cross_attn_rgb[i](
                zc_sum_tokens, zp_rgb_tokens, zp_rgb_tokens)
            fused_t_tokens = self.cross_attn_t[i](
                fused_rgb_tokens, zp_t_tokens, zp_t_tokens)

            fused_feat_i = fused_t_tokens.permute(
                0, 2, 1).reshape(B, 768, H_i, W_i)
            fused_feats.append(fused_feat_i)

        if self.with_neck:
            fused_feats = self.neck(fused_feats)
            zc_fused_feats_neck = self.neck(zc_fused_feats)
        else:
            zc_fused_feats_neck = zc_fused_feats

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_feats_neck, fused_feats)

    def encode_decode(self, inputs: torch.Tensor,
                      batch_img_metas: List[dict]) -> torch.Tensor:
        (_, _, _, _, _, fused_feats) = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples):
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _zc_seg_head_forward_train(self, zc_fused_feats, data_samples):
        losses = dict()
        loss_zc = self.zc_seg_head.loss(
            zc_fused_feats, data_samples, self.train_cfg)
        loss_val = loss_zc.get('loss_ce', list(loss_zc.values())[0])
        losses['loss_seg_zc'] = loss_val
        return losses

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(
                    inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(
                inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def loss(self, inputs: torch.Tensor,
             data_samples: SampleList) -> dict:
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_feats, fused_feats) = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            fused_feats, data_samples)
        losses.update(loss_decode)

        with torch.no_grad():
            zc_fused_feats_detached = [f.detach() for f in zc_fused_feats]
        loss_zc = self._zc_seg_head_forward_train(
            zc_fused_feats_detached, data_samples)
        losses['loss_seg_zc'] = loss_zc['loss_seg_zc'] * \
            self.loss_seg_zc_weight

        num_stages = len(zc_rgb_list)
        for i in range(num_stages):
            zc_rgb_tokens_i = zc_rgb_list[i].flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_list[i].flatten(2).transpose(1, 2).float()
            zp_rgb_tokens_i = zp_rgb_list[i].flatten(2).transpose(1, 2).float()
            zp_t_tokens_i = zp_t_list[i].flatten(2).transpose(1, 2).float()

            w = self.loss_disentangle_weights[i] \
                if i < len(self.loss_disentangle_weights) \
                else self.loss_disentangle_weights[-1]

            loss_hsic_rgb = compute_hsic(zc_rgb_tokens_i, zp_rgb_tokens_i)
            loss_hsic_t = compute_hsic(zc_t_tokens_i, zp_t_tokens_i)
            losses[f'loss_disentangle_s{i}'] = \
                (loss_hsic_rgb + loss_hsic_t) * w

            loss_modal_i = compute_infonce(zc_rgb_tokens_i, zc_t_tokens_i)
            losses[f'loss_modal_s{i}'] = loss_modal_i * \
                self.loss_modal_weight

        loss_invariance = torch.tensor(0.0, device=inputs.device)
        for i in range(num_stages):
            zc_rgb_tokens_i = zc_rgb_list[i].flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_list[i].flatten(2).transpose(1, 2).float()
            zc_rgb_norm = F.normalize(zc_rgb_tokens_i, dim=-1)
            zc_t_norm = F.normalize(zc_t_tokens_i, dim=-1)
            loss_invariance = loss_invariance + \
                F.mse_loss(zc_rgb_norm, zc_t_norm.detach()) + \
                F.mse_loss(zc_t_norm, zc_rgb_norm.detach())
        losses['loss_invariance'] = loss_invariance * \
            self.loss_invariance_weight

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                fused_feats, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: torch.Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:],
                     img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:],
                     padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs: torch.Tensor,
                 data_samples: OptSampleList = None) -> torch.Tensor:
        (_, _, _, _, _, fused_feats) = self.extract_feat(inputs)
        return self.decode_head.forward(fused_feats)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros(
            (batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros(
            (batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(
                    crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2),
                                int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

    def merge_lora(self):
        if hasattr(self.universal_branch, 'merge_lora'):
            self.universal_branch.merge_lora()

    def get_lora_info(self):
        if hasattr(self.universal_branch, 'get_lora_info'):
            return self.universal_branch.get_lora_info()
        return {}
