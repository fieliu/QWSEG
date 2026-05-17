import logging
import math
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.model.weight_init import trunc_normal_
from mmseg.registry import MODELS
from .rgbt_v1_sam_native import (
    SAMPromptEncoder, TwoWayTransformer, SAMMaskDecoder,
    DecoderAttention, apply_lora_to_decoder_qv, generate_point_grid)
from ..utils.lora import LoRALinear, apply_lora_to_model, freeze_non_lora_params


def compute_hsic(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    n = x.shape[1]
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    kx = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (2 * sigma ** 2))
    ky = torch.exp(-torch.cdist(y, y, p=2) ** 2 / (2 * sigma ** 2))
    h = torch.eye(n, device=x.device, dtype=x.dtype) - 1.0 / n
    kxh = kx @ h
    kyh = ky @ h
    hsic = torch.trace(kxh @ kyh) / ((n - 1) ** 2)
    return hsic


def compute_infonce(x: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    b, n, c = x.shape
    x_flat = x.reshape(b * n, c)
    y_flat = y.reshape(b * n, c)
    sim = torch.matmul(x_flat, y_flat.t()) / temperature
    labels = torch.arange(b * n, device=x.device)
    loss = F.cross_entropy(sim, labels)
    return loss


class CrossAttentionFusion(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        B, N, D = q.shape
        q_out = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_out = self.k_proj(k).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_out = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q_out, k_out.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v_out).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + q)
        return out


@MODELS.register_module()
class RGBTv2SAMNative(nn.Module):

    def __init__(self,
                 backbone: dict,
                 private_branch_rgb: dict,
                 private_branch_t: dict,
                 num_classes: int = 15,
                 image_size: int = 480,
                 prompt_embed_dim: int = 256,
                 decoder_depth: int = 2,
                 decoder_num_heads: int = 8,
                 decoder_mlp_dim: int = 2048,
                 num_multimask_outputs: int = 3,
                 clip_embed_dim: int = 768,
                 clip_checkpoint: str = None,
                 class_names: List[str] = None,
                 point_grid_size: int = 32,
                 use_lora_backbone: bool = True,
                 lora_rank: int = 4,
                 lora_alpha: float = 4.0,
                 lora_dropout: float = 0.1,
                 lora_target_modules: Optional[List[str]] = None,
                 use_lora_decoder: bool = True,
                 decoder_lora_rank: int = 4,
                 decoder_lora_alpha: float = 4.0,
                 decoder_lora_dropout: float = 0.0,
                 freeze_backbone_non_lora: bool = True,
                 freeze_decoder_non_lora: bool = True,
                 freeze_prompt_encoder: bool = True,
                 freeze_clip: bool = True,
                 train_patch_embed: bool = True,
                 loss_decode: dict = None,
                 loss_seg_zc_weight: float = 0.3,
                 loss_modal_weight: float = 0.2,
                 loss_disentangle_weights: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
                 loss_invariance_weight: float = 0.01,
                 init_cfg=None):
        super().__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.prompt_embed_dim = prompt_embed_dim
        self.point_grid_size = point_grid_size
        self.use_lora_backbone = use_lora_backbone
        self.use_lora_decoder = use_lora_decoder
        self.train_patch_embed = train_patch_embed
        self.loss_seg_zc_weight = loss_seg_zc_weight
        self.loss_modal_weight = loss_modal_weight
        self.loss_disentangle_weights = list(loss_disentangle_weights)
        self.loss_invariance_weight = loss_invariance_weight

        from mmseg.models.backbones.sam_vit import SAMViT
        backbone_cfg = {k: v for k, v in backbone.items() if k != 'type'}
        self.backbone = SAMViT(**backbone_cfg)

        if use_lora_backbone:
            if lora_target_modules is None:
                lora_target_modules = ['qkv', 'proj']
            apply_lora_to_model(
                self.backbone, rank=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout, target_modules=lora_target_modules)
            if freeze_backbone_non_lora:
                freeze_non_lora_params(self.backbone)

        if train_patch_embed:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = True

        self.private_branch_rgb = MODELS.build(private_branch_rgb)
        self.private_branch_t = MODELS.build(private_branch_t)

        vit_patch_size = backbone.get('patch_size', 16)
        self.image_embedding_size = image_size // vit_patch_size

        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16)

        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        transformer = TwoWayTransformer(
            depth=decoder_depth,
            embedding_dim=prompt_embed_dim,
            num_heads=decoder_num_heads,
            mlp_dim=decoder_mlp_dim)

        self.mask_decoder = SAMMaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=transformer,
            num_multimask_outputs=num_multimask_outputs,
            clip_embed_dim=clip_embed_dim)

        if use_lora_decoder:
            apply_lora_to_decoder_qv(
                self.mask_decoder,
                rank=decoder_lora_rank,
                alpha=decoder_lora_alpha,
                dropout=decoder_lora_dropout)
            if freeze_decoder_non_lora:
                for name, param in self.mask_decoder.named_parameters():
                    if 'lora_A' not in name and 'lora_B' not in name:
                        param.requires_grad = False

        self.clip_checkpoint = clip_checkpoint
        self.class_names = class_names
        self._clip_model = None
        self._clip_text_cache = None

        self._register_point_prompts()

        num_stages = 4
        self.cross_attn_rgb = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=768) for _ in range(num_stages)])
        self.cross_attn_t = nn.ModuleList(
            [CrossAttentionFusion(embed_dim=768) for _ in range(num_stages)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        trunc_normal_(self.cls_token, std=.02)

        if loss_decode is None:
            loss_decode = [
                dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce'),
                dict(type='DiceLoss', use_sigmoid=True, activate=True, naive_dice=False, loss_weight=1.0, loss_name='loss_dice')]
        self.loss_decode = loss_decode
        self._init_loss_modules()

    def _register_point_prompts(self):
        device = next(self.parameters()).device
        point_coords = generate_point_grid(self.point_grid_size, device)
        point_coords = point_coords.unsqueeze(0)
        point_labels = torch.ones(1, point_coords.shape[1], dtype=torch.int, device=device)
        self.register_buffer('point_coords', point_coords)
        self.register_buffer('point_labels', point_labels)

    def _get_clip_text_features(self):
        if self._clip_text_cache is not None:
            return self._clip_text_cache

        from ..text_encoder.clip import clip

        if self._clip_model is None:
            device = next(self.parameters()).device
            if self.clip_checkpoint is not None:
                model, _ = clip.load(self.clip_checkpoint, device=device)
            else:
                model, _ = clip.load('ViT-B/16', device=device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._clip_model = model

        device = next(self.parameters()).device
        templates = [
            'a photo of a {}.', 'a photograph of a {}.',
            'an image of a {}.', 'a picture of a {}.',
            'the photo of a {}.', 'the photograph of a {}.',
            'the image of a {}.', 'the picture of a {}.']

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
        self._clip_text_cache = text_features.to(next(self.parameters()).dtype)
        return self._clip_text_cache

    def _init_loss_modules(self):
        from mmseg.registry import MODELS as SEG_MODELS
        self.loss_modules = nn.ModuleList()
        for loss_cfg in self.loss_decode:
            self.loss_modules.append(SEG_MODELS.build(loss_cfg))

    def _resize_to_square(self, x):
        h, w = x.shape[-2:]
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return x

    def _get_sam_embeddings(self, x):
        tokens = self.backbone.patch_embed(x)
        if self.backbone.pos_embed is not None:
            tokens = tokens + self.backbone.pos_embed
        return tokens

    def _prepare_tokens_for_private(self, tokens):
        B, H, W, C = tokens.shape
        tokens_flat = tokens.reshape(B, H * W, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens_with_cls = torch.cat([cls_tokens, tokens_flat], dim=1)
        return tokens_with_cls

    def _run_sam_universal_stages(self, tokens):
        outs = []
        for i, blk in enumerate(self.backbone.blocks):
            tokens = blk(tokens)
            if i in self.backbone.out_indices:
                out = tokens.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def extract_feat(self, inputs):
        B, C, H, W = inputs.shape
        input_rgb = inputs[:, :3, :, :]
        input_ir = inputs[:, 3:, :, :]

        input_rgb = self._resize_to_square(input_rgb)
        input_ir = self._resize_to_square(input_ir)

        rgb_tokens = self._get_sam_embeddings(input_rgb)
        t_tokens = self._get_sam_embeddings(input_ir)

        rgb_tokens_with_cls = self._prepare_tokens_for_private(rgb_tokens)
        t_tokens_with_cls = self._prepare_tokens_for_private(t_tokens)

        hw_shape = (rgb_tokens.shape[1], rgb_tokens.shape[2])
        zp_rgb_list = self.private_branch_rgb(rgb_tokens_with_cls, hw_shape)
        zp_t_list = self.private_branch_t(t_tokens_with_cls, hw_shape)

        tokens_combined = torch.cat([rgb_tokens, t_tokens], dim=0)
        zc_combined = self._run_sam_universal_stages(tokens_combined)
        zc_rgb_list = [f[:B] for f in zc_combined]
        zc_t_list = [f[B:] for f in zc_combined]

        num_stages = len(zc_rgb_list)
        zc_fused_feats = []
        for i in range(num_stages):
            zc_fused_feats.append(zc_rgb_list[i] + zc_t_list[i])

        fused_feats = []
        for i in range(num_stages):
            zc_sum_tokens = zc_fused_feats[i].flatten(2).transpose(1, 2)
            zp_rgb_tokens = zp_rgb_list[i].flatten(2).transpose(1, 2)
            zp_t_tokens = zp_t_list[i].flatten(2).transpose(1, 2)

            fused_rgb_tokens = self.cross_attn_rgb[i](zc_sum_tokens, zp_rgb_tokens, zp_rgb_tokens)
            fused_t_tokens = self.cross_attn_t[i](fused_rgb_tokens, zp_t_tokens, zp_t_tokens)

            H_i, W_i = zc_rgb_list[i].shape[-2:]
            fused_feat_i = fused_t_tokens.permute(0, 2, 1).reshape(B, 768, H_i, W_i)
            fused_feats.append(fused_feat_i)

        return (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
                zc_fused_feats, fused_feats)

    def encode_image(self, inputs):
        (_, _, _, _, _, fused_feats) = self.extract_feat(inputs)
        last_feat = fused_feats[-1]

        if last_feat.shape[1] != self.prompt_embed_dim:
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = nn.Conv2d(
                    last_feat.shape[1], self.prompt_embed_dim, 1).to(last_feat.device)
            last_feat = self.channel_proj(last_feat)

        return last_feat

    def forward_decoder(self, image_embeddings, clip_text_features):
        B = image_embeddings.shape[0]
        target_dtype = image_embeddings.dtype
        point_coords = self.point_coords.expand(B, -1, -1)
        point_labels = self.point_labels.expand(B, -1)

        points = (point_coords, point_labels)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points)
        sparse_embeddings = sparse_embeddings.to(target_dtype)
        dense_embeddings = dense_embeddings.to(target_dtype)
        image_pe = self.prompt_encoder.get_dense_pe().to(target_dtype)
        clip_text_features = clip_text_features.to(target_dtype)

        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            clip_text_features=clip_text_features)

        return low_res_masks

    def loss(self, inputs, data_samples):
        (zc_rgb_list, zc_t_list, zp_rgb_list, zp_t_list,
         zc_fused_feats, fused_feats) = self.extract_feat(inputs)

        last_feat = fused_feats[-1]
        if last_feat.shape[1] != self.prompt_embed_dim:
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = nn.Conv2d(
                    last_feat.shape[1], self.prompt_embed_dim, 1).to(last_feat.device)
            last_feat = self.channel_proj(last_feat)

        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(last_feat, clip_text_features)

        seg_logits = F.interpolate(
            low_res_masks, size=inputs.shape[-2:],
            mode='bilinear', align_corners=False)

        seg_label = self._stack_seg_gt(data_samples)
        losses = dict()

        for loss_module in self.loss_modules:
            loss_name = getattr(loss_module, 'loss_name', loss_module.__class__.__name__)
            if isinstance(loss_module, (nn.CrossEntropyLoss,)):
                loss_val = loss_module(seg_logits, seg_label)
            else:
                seg_logits_sig = torch.sigmoid(seg_logits)
                one_hot = F.one_hot(seg_label, self.num_classes).permute(0, 3, 1, 2).to(seg_logits_sig.dtype)
                loss_val = loss_module(seg_logits_sig, one_hot)
            losses[loss_name] = loss_val

        num_stages = len(zc_rgb_list)
        for i in range(num_stages):
            zc_rgb_tokens_i = zc_rgb_list[i].flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_list[i].flatten(2).transpose(1, 2).float()
            zp_rgb_tokens_i = zp_rgb_list[i].flatten(2).transpose(1, 2).float()
            zp_t_tokens_i = zp_t_list[i].flatten(2).transpose(1, 2).float()

            w = self.loss_disentangle_weights[i] if i < len(self.loss_disentangle_weights) else self.loss_disentangle_weights[-1]

            loss_hsic_rgb = compute_hsic(zc_rgb_tokens_i, zp_rgb_tokens_i)
            loss_hsic_t = compute_hsic(zc_t_tokens_i, zp_t_tokens_i)
            losses[f'loss_disentangle_s{i}'] = (loss_hsic_rgb + loss_hsic_t) * w

            loss_modal_i = compute_infonce(zc_rgb_tokens_i, zc_t_tokens_i)
            losses[f'loss_modal_s{i}'] = loss_modal_i * self.loss_modal_weight

        loss_invariance = torch.tensor(0.0, device=inputs.device)
        for i in range(num_stages):
            zc_rgb_tokens_i = zc_rgb_list[i].flatten(2).transpose(1, 2).float()
            zc_t_tokens_i = zc_t_list[i].flatten(2).transpose(1, 2).float()
            zc_rgb_norm = F.normalize(zc_rgb_tokens_i, dim=-1)
            zc_t_norm = F.normalize(zc_t_tokens_i, dim=-1)
            loss_invariance = loss_invariance + \
                F.mse_loss(zc_rgb_norm, zc_t_norm.detach()) + \
                F.mse_loss(zc_t_norm, zc_rgb_norm.detach())
        losses['loss_invariance'] = loss_invariance * self.loss_invariance_weight

        return losses

    def predict(self, inputs, data_samples=None):
        image_embeddings = self.encode_image(inputs)
        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(image_embeddings, clip_text_features)

        seg_logits = F.interpolate(
            low_res_masks, size=inputs.shape[-2:],
            mode='bilinear', align_corners=False)

        if data_samples is None:
            from mmseg.structures import SegDataSample, PixelData
            data_samples = [SegDataSample() for _ in range(seg_logits.shape[0])]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(seg_logits.shape[0]):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                _, _, H, W = seg_logits.shape
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                from mmseg.models.utils import resize
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=False,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            data_samples[i].set_data({
                'pred_sem_seg': PixelData(data=i_seg_pred),
                'seg_logits': PixelData(data=i_seg_logits),
            })

        return data_samples

    def _stack_seg_gt(self, data_samples):
        gt_semantic_segs = []
        for data_sample in data_samples:
            gt_semantic_segs.append(data_sample.gt_sem_seg.data)
        gt_semantic_segs = torch.stack(gt_semantic_segs, dim=0).squeeze(1).long()
        return gt_semantic_segs

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'text_encoder'):
            self.text_encoder.eval()
        return self

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
