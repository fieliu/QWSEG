import math
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.models.segmentors import BaseSegmentor
from mmseg.utils import OptConfigType, OptMultiConfig
from ..utils.lora import LoRALinear, apply_lora_to_model, freeze_non_lora_params


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionEmbeddingRandom(nn.Module):

    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            'positional_encoding_gaussian_matrix',
            scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class SAMPromptEncoder(nn.Module):

    def __init__(self, embed_dim: int = 256,
                 image_embedding_size: Tuple[int, int] = (30, 30),
                 input_image_size: Tuple[int, int] = (480, 480),
                 mask_in_chans: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def forward(self, points: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        coords, labels = points
        bs = coords.shape[0]
        point_embeddings = self._embed_points(coords, labels, pad=True)
        sparse_embeddings = point_embeddings
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


class DecoderAttention(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class TwoWayAttentionBlock(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = DecoderAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, embedding_dim))
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                query_pe: torch.Tensor, key_pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0)))
        self.final_attn_token_to_image = DecoderAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor,
                point_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys,
                query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class SAMMaskDecoder(nn.Module):

    def __init__(self, transformer_dim: int = 256,
                 transformer: nn.Module = None,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 clip_embed_dim: int = 768):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.clip_proj = nn.Linear(clip_embed_dim, transformer_dim // 8)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                clip_text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_dtype = image_embeddings.dtype
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0).to(target_dtype)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe.expand(src.shape[0], -1, -1, -1)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        clip_proj_feat = self.clip_proj(clip_text_features)
        masks = torch.matmul(
            upscaled_embedding.permute(0, 2, 3, 1),
            clip_proj_feat.t()).permute(0, 3, 1, 2)

        return masks, iou_token_out


def apply_lora_to_decoder_qv(model, rank=4, alpha=4.0, dropout=0.0):
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parent_name_parts = name.split('.')
        child_name = parent_name_parts[-1]
        is_q = child_name == 'q_proj'
        is_v = child_name == 'v_proj'
        if not (is_q or is_v):
            continue
        parent_name = '.'.join(parent_name_parts[:-1])
        replacements.append((parent_name, child_name, module))

    for parent_name, child_name, module in replacements:
        parent = model
        for attr in parent_name.split('.'):
            if attr:
                parent = getattr(parent, attr)
        lora_layer = LoRALinear(
            module.in_features, module.out_features,
            rank=rank, alpha=alpha, dropout=dropout,
            bias=module.bias is not None)
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)
        setattr(parent, child_name, lora_layer)

    return model


def generate_point_grid(n_per_side: int, device: torch.device) -> torch.Tensor:
    offset = 0.5 / n_per_side
    points_one_side = torch.linspace(offset, 1.0 - offset, n_per_side, device=device)
    coords_x, coords_y = torch.meshgrid(points_one_side, points_one_side, indexing='xy')
    coords = torch.stack([coords_y, coords_x], dim=-1)
    return coords.reshape(-1, 2)


@MODELS.register_module()
class RGBTv1SAMNative(BaseSegmentor):

    def __init__(self,
                 backbone: dict,
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
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.num_classes = num_classes
        self.image_size = image_size
        self.prompt_embed_dim = prompt_embed_dim
        self.point_grid_size = point_grid_size
        self.use_lora_backbone = use_lora_backbone
        self.use_lora_decoder = use_lora_decoder
        self.train_patch_embed = train_patch_embed

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

        if class_names is None:
            class_names = [f'class_{i}' for i in range(num_classes)]

        self.class_names = class_names
        self.clip_checkpoint = clip_checkpoint
        self._clip_model = None
        self._clip_text_cache = None

        self.channel_proj = nn.Conv2d(768, prompt_embed_dim, 1)

        self._register_point_prompts()

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

    def _register_point_prompts(self):
        device = next(self.parameters()).device
        point_coords = generate_point_grid(self.point_grid_size, device)
        point_coords = point_coords.unsqueeze(0)
        point_labels = torch.ones(1, point_coords.shape[1], dtype=torch.int, device=device)
        self.register_buffer('point_coords', point_coords)
        self.register_buffer('point_labels', point_labels)

    def _prepare_rgbt_input(self, inputs):
        input_rgb = inputs[:, 0:3, :, :]
        input_ir = inputs[:, 3:6, :, :]
        return torch.cat([input_rgb, input_ir], dim=0)

    def _resize_to_square(self, x):
        h, w = x.shape[-2:]
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return x

    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        return x

    def encode_decode(self, inputs, batch_data_samples):
        image_embeddings = self.encode_image(inputs)
        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(image_embeddings, clip_text_features)
        seg_logits = F.interpolate(
            low_res_masks,
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return seg_logits

    def encode_image(self, inputs):
        x_rgbt = self._prepare_rgbt_input(inputs)
        x_rgbt = self._resize_to_square(x_rgbt)
        feats = self.extract_feat(x_rgbt)

        last_feat = feats[-1]
        b, c, h, w = last_feat.shape
        B = b // 2
        x_rgb_feat = last_feat[:B]
        x_ir_feat = last_feat[B:]
        fused_feat = x_rgb_feat + x_ir_feat

        if fused_feat.shape[1] != self.prompt_embed_dim:
            fused_feat = self.channel_proj(fused_feat)

        return fused_feat

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
        image_embeddings = self.encode_image(inputs)
        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(image_embeddings, clip_text_features)

        seg_logits = F.interpolate(
            low_res_masks,
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False)

        seg_label = self._stack_seg_gt(data_samples)
        losses = dict()

        loss_ce = F.cross_entropy(seg_logits, seg_label, ignore_index=255)
        losses['loss_ce'] = loss_ce

        seg_probs = F.softmax(seg_logits, dim=1)
        seg_probs = seg_probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        seg_label_clamped = seg_label.reshape(-1).clamp(0, self.num_classes - 1)
        valid_mask = (seg_label.reshape(-1) >= 0) & (seg_label.reshape(-1) < self.num_classes)
        seg_one_hot = F.one_hot(seg_label_clamped, self.num_classes).to(seg_probs.dtype)
        seg_one_hot = seg_one_hot * valid_mask.unsqueeze(-1).to(seg_probs.dtype)
        intersection = (seg_probs * seg_one_hot).sum()
        union = seg_probs.sum() + seg_one_hot.sum()
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        losses['loss_dice'] = 1.0 - dice

        return losses

    def predict(self, inputs, data_samples=None):
        image_embeddings = self.encode_image(inputs)
        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(image_embeddings, clip_text_features)

        seg_logits = F.interpolate(
            low_res_masks,
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False)

        if data_samples is None:
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

    def _forward(self, inputs, data_samples=None):
        image_embeddings = self.encode_image(inputs)
        clip_text_features = self._get_clip_text_features()
        low_res_masks = self.forward_decoder(image_embeddings, clip_text_features)
        seg_logits = F.interpolate(
            low_res_masks,
            size=inputs.shape[-2:],
            mode='bilinear',
            align_corners=False)
        return seg_logits

    def _stack_seg_gt(self, data_samples):
        gt_semantic_segs = []
        for data_sample in data_samples:
            gt_semantic_segs.append(data_sample.gt_sem_seg.data)
        gt_semantic_segs = torch.stack(gt_semantic_segs, dim=0).squeeze(1).long()
        return gt_semantic_segs

    def train(self, mode=True):
        super().train(mode)
        if self._clip_model is not None:
            self._clip_model.eval()
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
