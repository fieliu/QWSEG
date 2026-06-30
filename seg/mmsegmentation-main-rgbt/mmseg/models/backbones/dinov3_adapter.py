"""DINOv3 + ViT-Adapter backbone for dense prediction.

Official ViT-Adapter implementation from:
  "Vision Transformer Adapter for Dense Predictions" (Chen et al., ICLR 2023)
  https://github.com/czczup/ViT-Adapter

This faithfully reproduces the official architecture:
1. SpatialPriorModule: CNN stem → 3 levels of spatial features (1/8, 1/16, 1/32)
2. InteractionBlock = Injector + Extractor:
   - Injector: deformable cross-attn (spatial prior → ViT tokens), gated residual
   - Extractor: deformable cross-attn (ViT tokens → spatial prior), with ConvFFN
3. After N interactions, split & reshape spatial prior into C2, C3, C4;
   upsample C2 → C1; add ViT intermediate features; SyncBN norm.

We replace the official CUDA MSDeformAttn with mmdet's MultiScaleDeformableAttention,
which is functionally identical but works without custom CUDA compilation.
"""
import math
import logging
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmengine.model import BaseModule
from mmseg.registry import MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: mmdet MultiScaleDeformableAttention wrapper
# ---------------------------------------------------------------------------

class MSDeformAttnWrapper(nn.Module):
    """Wraps mmdet's MultiScaleDeformableAttention to match the interface
    used in the official ViT-Adapter code."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        from mmdet.models.layers.transformer.multi_scale_deform_attn import \
            MultiScaleDeformableAttention
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        # Ensure im2col_step is available
        self.attn.im2col_step = 64

    def forward(self, query, reference_points, value, spatial_shapes,
                level_start_index, attention_weights=None):
        """Match official interface: attn(query, ref_pts, value, spatial_shapes, level_start_index, None)"""
        return self.attn(
            query=query,
            key=value,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )


# ---------------------------------------------------------------------------
# Deformable input helpers (from official adapter_modules.py)
# ---------------------------------------------------------------------------

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)[:, :, None]
    return reference_points


def deform_inputs(x):
    """Compute reference points and spatial_shapes for deformable attention.

    Returns (deform_inputs1, deform_inputs2):
    - deform_inputs1: for Injector (query=ViT tokens, key=spatial prior at 3 levels)
    - deform_inputs2: for Extractor (query=spatial prior, key=ViT tokens at 1 level)
    """
    bs, c, h, w = x.shape
    # Injector: query at 1/16 scale, key at 3 scales (1/8, 1/16, 1/32)
    spatial_shapes_1 = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long, device=x.device)
    level_start_index_1 = torch.cat((
        spatial_shapes_1.new_zeros((1,)),
        spatial_shapes_1.prod(1).cumsum(0)[:-1]))
    reference_points_1 = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points_1, spatial_shapes_1, level_start_index_1]

    # Extractor: query at 3 scales (1/8, 1/16, 1/32), key at 1 scale (1/16)
    spatial_shapes_2 = torch.as_tensor(
        [(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index_2 = torch.cat((
        spatial_shapes_2.new_zeros((1,)),
        spatial_shapes_2.prod(1).cumsum(0)[:-1]))
    reference_points_2 = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points_2, spatial_shapes_2, level_start_index_2]

    return deform_inputs1, deform_inputs2


# ---------------------------------------------------------------------------
# DWConv + ConvFFN (from official adapter_modules.py)
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21  # split ratio: 16/21 at 1/8, 4/21 at 1/16, 1/21 at 1/32
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Injector & Extractor (from official adapter_modules.py)
# ---------------------------------------------------------------------------

class Injector(nn.Module):
    """Injects spatial prior into ViT tokens via deformable cross-attention.

    query = ViT tokens, key/value = spatial prior features.
    Output: ViT tokens + gamma * cross_attn_output (gated residual).
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttnWrapper(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query


class Extractor(nn.Module):
    """Extracts spatial features from ViT tokens via deformable cross-attention.

    query = spatial prior, key/value = ViT tokens.
    Updates spatial prior with ViT information + ConvFFN.
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttnWrapper(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index)
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query


# ---------------------------------------------------------------------------
# InteractionBlock (from official adapter_modules.py)
# ---------------------------------------------------------------------------

class InteractionBlock(nn.Module):
    """One interaction round: Inject spatial prior → Run ViT blocks → Extract features.

    This is the core of the ViT-Adapter. For a 12-layer ViT with 4 interactions,
    each InteractionBlock processes 3 consecutive ViT blocks.
    """
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()
        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        """Args:
            x: ViT tokens [B, N, C]
            c: spatial prior tokens (concat of 3 levels) [B, Nc, C]
            blocks: list of ViT blocks to run
            deform_inputs1: [ref_pts, spatial_shapes, level_start_index] for injector
            deform_inputs2: same for extractor
            H, W: spatial shape of ViT token grid
        Returns:
            x: updated ViT tokens
            c: updated spatial prior tokens
        """
        # 1. Inject spatial prior into ViT tokens
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        # 2. Run ViT blocks
        for blk in blocks:
            x = blk(x, H, W)

        # 3. Extract features from ViT tokens back to spatial prior
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


# ---------------------------------------------------------------------------
# SpatialPriorModule (from official adapter_modules.py)
# ---------------------------------------------------------------------------

class SpatialPriorModule(nn.Module):
    """CNN-based spatial prior module that extracts multi-scale local features.

    Architecture follows ResNet stem style:
    stem (3 conv + maxpool) → conv2 (1/8) → conv3 (1/16) → conv4 (1/32)
    Then 1x1 conv projections to embed_dim at each level.
    """
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)    # 1/4
            c2 = self.conv2(c1)  # 1/8
            c3 = self.conv3(c2)  # 1/16
            c4 = self.conv4(c3)  # 1/32
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 1/8 tokens
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 1/16 tokens
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 1/32 tokens
            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


# ---------------------------------------------------------------------------
# DINOv3 ViT Block wrapper (to match timm's Block interface)
# ---------------------------------------------------------------------------

class DINOv3BlockWrapper(nn.Module):
    """Wraps a DINOv3 ViT block to accept (x, H, W) like timm's Block."""

    def __init__(self, block, rope=None):
        super().__init__()
        self.block = block
        self.rope = rope

    def forward(self, x, H, W):
        block = self.block
        B, L, C = x.shape

        # Attention
        residual = x
        if hasattr(block, 'norm1'):
            x_norm = block.norm1(x)
        else:
            x_norm = x

        attn = block.attn if hasattr(block, 'attn') else block.attention
        if hasattr(attn, 'q_proj'):
            # DINOv3 split QKV
            q = attn.q_proj(x_norm)
            k = attn.k_proj(x_norm)
            v = attn.v_proj(x_norm)
            num_heads = attn.num_heads
            head_dim = C // num_heads
            q = q.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)

            if self.rope is not None:
                q = self.rope(q)
                k = self.rope(k)

            attn_weights = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn_weights = attn_weights.softmax(dim=-1)
            out = (attn_weights @ v).transpose(1, 2).reshape(B, L, C)
            out = attn.o_proj(out)
        else:
            out = attn(x_norm)

        if hasattr(block, 'ls1'):
            out = block.ls1(out)
        elif hasattr(block, 'layer_scale1'):
            out = block.layer_scale1(out)

        if hasattr(block, 'drop_path'):
            x = residual + block.drop_path(out)
        else:
            x = residual + out

        # MLP
        residual = x
        if hasattr(block, 'norm2'):
            x_norm = block.norm2(x)
        else:
            x_norm = x

        mlp_out = block.mlp(x_norm)
        if hasattr(block, 'ls2'):
            mlp_out = block.ls2(mlp_out)
        elif hasattr(block, 'layer_scale2'):
            mlp_out = block.layer_scale2(mlp_out)

        if hasattr(block, 'drop_path'):
            x = residual + block.drop_path(mlp_out)
        else:
            x = residual + mlp_out

        return x


# ---------------------------------------------------------------------------
# DINOv3 ViT-Adapter Backbone
# ---------------------------------------------------------------------------

@MODELS.register_module()
class DINOv3Adapter(nn.Module):
    """DINOv3 ViT with official ViT-Adapter for dense prediction.

    This faithfully follows the official ViT-Adapter (czczup/ViT-Adapter):
    1. SpatialPriorModule: ResNet-style CNN → multi-scale spatial priors
    2. N InteractionBlocks: each injects spatial prior into ViT, runs blocks,
       then extracts features back
    3. After interactions: split & reshape spatial tokens → C1..C4 feature maps
    4. Optionally add ViT intermediate features (add_vit_feature)
    5. SyncBN norm → return 4-level features for Mask2Former

    Args:
        backbone_name: HuggingFace model name for DINOv3
        backbone_ckpt: Path to local DINOv3 checkpoint (.pth)
        img_size: Input image size (h, w)
        patch_size: ViT patch size
        embed_dims: ViT embedding dimension (768 for ViT-B)
        num_heads: Number of attention heads (12 for ViT-B)
        depth: Number of ViT blocks (12 for ViT-B)
        conv_inplane: SPM intermediate channels (default 64)
        n_points: Number of sampling points in deformable attention
        deform_num_heads: Number of heads in deformable attention
        init_values: Initial value for gamma in Injector (0 = no injection at start)
        interaction_indexes: Which ViT blocks each InteractionBlock covers
            e.g. [[0,2],[3,5],[6,8],[9,11]] for 4 interactions on 12-layer ViT
        with_cffn: Whether to use ConvFFN in Extractor
        cffn_ratio: Hidden dim ratio for ConvFFN
        deform_ratio: Ratio for deformable attention
        add_vit_feature: Whether to add ViT intermediate features to output
        use_extra_extractor: Whether to add extra extractors in last interaction
        freeze_vit: Whether to freeze the ViT backbone
        local_files_only: Whether to load DINOv3 from local files only
    """

    def __init__(
        self,
        backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        backbone_ckpt: Optional[str] = None,
        img_size=(480, 640),
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        depth=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        init_values=0.,
        interaction_indexes=None,
        use_thermal=False,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        freeze_vit=False,
        local_files_only=True,
        with_cp=False,
        init_cfg=None,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.depth = depth
        self.add_vit_feature = add_vit_feature
        self.freeze_vit = freeze_vit
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.use_thermal = use_thermal

        # Default interaction_indexes: 4 interactions for 12-layer ViT
        if interaction_indexes is None:
            if depth == 12:
                interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
            elif depth == 24:
                interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
            else:
                # Generic: 4 equal splits
                chunk = depth // 4
                interaction_indexes = [
                    [i * chunk, (i + 1) * chunk - 1] for i in range(4)]
        self.interaction_indexes = interaction_indexes

        # --- Load DINOv3 backbone ---
        from .eomt_core_import import load_dinov3_backbone
        self.backbone = load_dinov3_backbone(
            backbone_name=backbone_name,
            backbone_ckpt=backbone_ckpt,
            img_size=self.img_size,
            patch_size=patch_size,
            local_files_only=local_files_only,
        )

        # Wrap ViT blocks to accept (x, H, W)
        blocks = self.backbone.blocks if hasattr(self.backbone, 'blocks') else self.backbone.layer
        rope = self.backbone.rope_embeddings if hasattr(self.backbone, 'rope_embeddings') else None
        self.wrapped_blocks = nn.ModuleList([
            DINOv3BlockWrapper(b, rope=rope) for b in blocks
        ])

        # Freeze ViT
        if freeze_vit:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- ViT-Adapter modules (official architecture) ---
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dims, with_cp=with_cp)
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dims,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=0.,  # Can be configured per block
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dims)  # Use BN instead of SyncBN for single-GPU
        self.norm2 = nn.BatchNorm2d(embed_dims)
        self.norm3 = nn.BatchNorm2d(embed_dims)
        self.norm4 = nn.BatchNorm2d(embed_dims)

        # Channel info for downstream decoder
        self.out_channels = embed_dims
        self.num_features = 4

        # Initialize weights
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        nn.init.normal_(self.level_embed)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            from timm.models.layers import trunc_normal_
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        """Interpolate position embeddings to match current feature map size."""
        pretrain_size = self.img_size
        pos_embed = pos_embed.reshape(
            1, pretrain_size[0] // self.patch_size,
            pretrain_size[1] // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic',
                                  align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass following the official ViT-Adapter.

        Args:
            x: Input image [B, 3, H, W] or [B, 6, H, W] (RGB-T)

        Returns:
            List of 4 feature maps [f1, f2, f3, f4] at strides [4, 8, 16, 32]
        """
        # Channel selection for 6-channel RGB-T input
        if x.shape[1] == 6:
            if self.use_thermal:
                x = x[:, 3:, :, :]
            else:
                x = x[:, :3, :, :]

        # Pre-compute deformable attention inputs
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # 1. SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # Concat 3 levels for interaction

        # 2. Patch embedding forward
        backbone = self.backbone
        if hasattr(backbone, 'patch_embed'):
            x_tokens, H, W = backbone.patch_embed(x)
        else:
            x_tokens = backbone.embeddings(x)
            H, W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size

        # Add position embeddings
        if hasattr(backbone, '_pos_embed'):
            x_tokens = backbone._pos_embed(x_tokens)
        elif hasattr(backbone, 'pos_embed'):
            pos_embed = self._get_pos_embed(backbone.pos_embed[:, 1:], H, W)
            x_tokens = backbone.pos_drop(x_tokens + pos_embed)

        # Remove CLS token if present (DINOv3 may not have one)
        if hasattr(backbone, 'cls_token') and backbone.cls_token is not None:
            # Keep only patch tokens
            x_tokens = x_tokens[:, 1:]

        # 3. Interaction: inject → ViT blocks → extract
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blocks_slice = self.wrapped_blocks[indexes[0]:indexes[-1] + 1]
            x_tokens, c = layer(x_tokens, c, blocks_slice,
                                deform_inputs1, deform_inputs2, H, W)
            outs.append(x_tokens.transpose(1, 2).view(
                x_tokens.shape[0], self.embed_dims, H, W).contiguous())

        # 4. Split & reshape spatial prior tokens
        bs = x_tokens.shape[0]
        dim = self.embed_dims
        c2_len = c2.shape[1]
        c3_len = c3.shape[1]

        c2_out = c[:, 0:c2_len, :]
        c3_out = c[:, c2_len:c2_len + c3_len, :]
        c4_out = c[:, c2_len + c3_len:, :]

        h8, w8 = self.img_size[0] // 8, self.img_size[1] // 8
        h16, w16 = H, W
        h32, w32 = H // 2, W // 2

        c2_out = c2_out.transpose(1, 2).view(bs, dim, h8, w8).contiguous()   # 1/8
        c3_out = c3_out.transpose(1, 2).view(bs, dim, h16, w16).contiguous()  # 1/16
        c4_out = c4_out.transpose(1, 2).view(bs, dim, h32, w32).contiguous()  # 1/32
        c1_out = self.up(c2_out) + c1  # 1/4 (c1 is still 2D from SPM)

        # 5. Add ViT intermediate features
        if self.add_vit_feature and len(outs) >= 4:
            x1, x2, x3, x4 = outs[0], outs[1], outs[2], outs[3]
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_out = c1_out + x1
            c2_out = c2_out + x2
            c3_out = c3_out + x3
            c4_out = c4_out + x4

        # 6. Final norm
        f1 = self.norm1(c1_out)
        f2 = self.norm2(c2_out)
        f3 = self.norm3(c3_out)
        f4 = self.norm4(c4_out)

        return [f1, f2, f3, f4]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vit:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# RGBT DINOv3-Adapter Backbone (follows RGBTSwinTransformer pattern)
# ---------------------------------------------------------------------------

@MODELS.register_module()
class RGBTDINOv3Adapter(BaseModule):
    """Dual-branch DINOv3 + ViT-Adapter for RGB-T segmentation.

    Two adapter branches share the same DINOv3 backbone. At each interaction
    stage, both branches run their adapter interaction, then bidirectional
    CrossAttn allows each modality to "sense" the other. The sensed features
    propagate to subsequent stages, so deeper stages receive cross-modal
    information from all previous stages.

    Architecture:
      Stage 0: RGB_adapter_interaction → T_adapter_interaction → CrossAttn
      Stage 1: RGB_adapter_interaction → T_adapter_interaction → CrossAttn
      Stage 2: RGB_adapter_interaction → T_adapter_interaction → CrossAttn
      Stage 3: RGB_adapter_interaction → T_adapter_interaction → CrossAttn
      Output:  4 multi-scale features (fused or dual)

    Args:
        fusion_type: 'ADD' or 'MAX' for fusing RGB and T features at output
        thr_in_channels: Number of channels for thermal input (1 or 3)
        return_dual: If True, return (rgb_feats, thr_feats) separately
        cross_attn_heads: Number of attention heads for cross-modal attention
        **kwargs: All DINOv3Adapter arguments (backbone_name, embed_dims, etc.)
    """

    def __init__(self,
                 fusion_type='ADD',
                 thr_in_channels=3,
                 return_dual=False,
                 cross_attn_heads=8,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.fusion_type = fusion_type
        self.thr_in_channels = thr_in_channels
        self.embed_dims = kwargs.get('embed_dims', 768)
        self.return_dual = return_dual

        # Build two adapter branches; they share the DINOv3 backbone
        self.rgb_adapter = DINOv3Adapter(**kwargs)

        # For thermal branch, we need a separate adapter but shared backbone
        # Create thermal adapter with its own SPM + Interactions
        self.thr_spm = SpatialPriorModule(
            inplanes=kwargs.get('conv_inplane', 64),
            embed_dim=self.embed_dims,
            with_cp=kwargs.get('with_cp', False))
        self.thr_level_embed = nn.Parameter(torch.zeros(3, self.embed_dims))

        interaction_indexes = kwargs.get('interaction_indexes', None)
        if interaction_indexes is None:
            depth = kwargs.get('depth', 12)
            if depth == 12:
                interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
            elif depth == 24:
                interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
            else:
                chunk = depth // 4
                interaction_indexes = [
                    [i * chunk, (i + 1) * chunk - 1] for i in range(4)]

        self.thr_interactions = nn.Sequential(*[
            InteractionBlock(
                dim=self.embed_dims,
                num_heads=kwargs.get('deform_num_heads', 6),
                n_points=kwargs.get('n_points', 4),
                init_values=kwargs.get('init_values', 0.),
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=kwargs.get('with_cffn', True),
                cffn_ratio=kwargs.get('cffn_ratio', 0.25),
                deform_ratio=kwargs.get('deform_ratio', 1.0),
                extra_extractor=((i == len(interaction_indexes) - 1)
                                 and kwargs.get('use_extra_extractor', True)),
                with_cp=kwargs.get('with_cp', False))
            for i in range(len(interaction_indexes))
        ])

        self.thr_up = nn.ConvTranspose2d(self.embed_dims, self.embed_dims, 2, 2)
        self.thr_norm1 = nn.BatchNorm2d(self.embed_dims)
        self.thr_norm2 = nn.BatchNorm2d(self.embed_dims)
        self.thr_norm3 = nn.BatchNorm2d(self.embed_dims)
        self.thr_norm4 = nn.BatchNorm2d(self.embed_dims)

        # Share the frozen DINOv3 backbone (same object reference)
        self.thr_backbone = self.rgb_adapter.backbone
        self.thr_wrapped_blocks = self.rgb_adapter.wrapped_blocks

        # Cross-modal attention: one per stage (4 stages)
        # After each stage, both modalities sense each other via bidirectional cross-attn
        from mmseg.models.segmentors.eomt_fusion_blocks import CrossAttnFusion
        self.cross_fusions = nn.ModuleList([
            CrossAttnFusion(self.embed_dims, num_heads=cross_attn_heads)
            for _ in range(len(interaction_indexes))
        ])

        # Fusion norms (one per output level)
        self.fusion_norms = nn.ModuleList([
            nn.BatchNorm2d(self.embed_dims) for _ in range(4)
        ])

        # Store config for forward
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = kwargs.get('add_vit_feature', True)
        self.img_size = tuple(kwargs.get('img_size', (480, 640)))
        self.patch_size = kwargs.get('patch_size', 16)

        # Init weights for thermal adapter
        self.thr_up.apply(self.rgb_adapter._init_weights)
        self.thr_spm.apply(self.rgb_adapter._init_weights)
        self.thr_interactions.apply(self.rgb_adapter._init_weights)
        nn.init.normal_(self.thr_level_embed)

    def _init_adapter_state(self, x_input, spm, level_embed, backbone,
                            wrapped_blocks, img_size, patch_size):
        """Initialize one adapter branch: SPM + patch embed + pos embed.

        Returns:
            x_tokens: [B, N, C] ViT tokens after patch+pos embed
            c: [B, Nc, C] spatial prior tokens (concat of 3 levels)
            c1: [B, C, H/4, W/4] stage-1 spatial prior (for final c1 output)
            c2_len, c3_len: lengths for splitting c back into levels
            H, W: token grid size
            deform_inputs1, deform_inputs2: for injector/extractor
        """
        deform_inputs1, deform_inputs2 = deform_inputs(x_input)

        c1, c2, c3, c4 = spm(x_input)
        c2 = c2 + level_embed[0]
        c3 = c3 + level_embed[1]
        c4 = c4 + level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)
        c2_len = c2.shape[1]
        c3_len = c3.shape[1]

        # Patch embedding
        if hasattr(backbone, 'patch_embed'):
            x_tokens, H, W = backbone.patch_embed(x_input)
        else:
            x_tokens = backbone.embeddings(x_input)
            H, W = img_size[0] // patch_size, img_size[1] // patch_size

        # Position embeddings
        if hasattr(backbone, '_pos_embed'):
            x_tokens = backbone._pos_embed(x_tokens)
        elif hasattr(backbone, 'pos_embed'):
            pos_embed = self.rgb_adapter._get_pos_embed(
                backbone.pos_embed[:, 1:], H, W)
            x_tokens = backbone.pos_drop(x_tokens + pos_embed)

        if hasattr(backbone, 'cls_token') and backbone.cls_token is not None:
            x_tokens = x_tokens[:, 1:]

        return (x_tokens, c, c1, c2_len, c3_len, H, W,
                deform_inputs1, deform_inputs2)

    def _run_stage(self, x_tokens, c, interaction, wrapped_blocks,
                   indexes, deform_inputs1, deform_inputs2, H, W):
        """Run one interaction stage for one branch.

        Returns:
            x_tokens: updated ViT tokens
            c: updated spatial prior tokens
            out_feat: [B, C, H, W] ViT feature at this stage
        """
        blocks_slice = wrapped_blocks[indexes[0]:indexes[-1] + 1]
        x_tokens, c = interaction(x_tokens, c, blocks_slice,
                                  deform_inputs1, deform_inputs2, H, W)
        out_feat = x_tokens.transpose(1, 2).view(
            x_tokens.shape[0], self.embed_dims, H, W).contiguous()
        return x_tokens, c, out_feat

    def _assemble_outputs(self, rgb_c, rgb_c1, thr_c, thr_c1,
                          rgb_outs, thr_outs,
                          rgb_up, thr_up,
                          rgb_norms, thr_norms,
                          c2_len, c3_len, img_size, H, W,
                          add_vit_feature, fusion_type, fusion_norms):
        """Assemble final multi-scale outputs from both branches.

        Returns:
            If return_dual: (rgb_feats, thr_feats) as two lists of 4 feature maps
            Else: list of 4 fused feature maps
        """
        bs = rgb_c.shape[0]
        dim = self.embed_dims
        h8, w8 = img_size[0] // 8, img_size[1] // 8
        h16, w16 = H, W
        h32, w32 = H // 2, W // 2

        def split_and_reshape(c, c1, up, norms, outs):
            c2_out = c[:, 0:c2_len, :].transpose(1, 2).view(bs, dim, h8, w8).contiguous()
            c3_out = c[:, c2_len:c2_len + c3_len, :].transpose(1, 2).view(bs, dim, h16, w16).contiguous()
            c4_out = c[:, c2_len + c3_len:, :].transpose(1, 2).view(bs, dim, h32, w32).contiguous()
            c1_out = up(c2_out) + c1

            if add_vit_feature and len(outs) >= 4:
                x1, x2, x3, x4 = outs[0], outs[1], outs[2], outs[3]
                x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
                x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
                c1_out = c1_out + x1
                c2_out = c2_out + x2
                c3_out = c3_out + x3
                c4_out = c4_out + x4

            return [norms[0](c1_out), norms[1](c2_out),
                    norms[2](c3_out), norms[3](c4_out)]

        rgb_feats = split_and_reshape(
            rgb_c, rgb_c1, rgb_up, rgb_norms, rgb_outs)
        thr_feats = split_and_reshape(
            thr_c, thr_c1, thr_up, thr_norms, thr_outs)

        if self.return_dual:
            return rgb_feats, thr_feats

        fused = []
        for i in range(4):
            if fusion_type == 'MAX':
                out = torch.max(rgb_feats[i], thr_feats[i])
            else:
                out = rgb_feats[i] + thr_feats[i]
            out = fusion_norms[i](out)
            fused.append(out)
        return fused

    def forward(self, x):
        """Forward pass for RGB-T input with inter-stage cross-modal attention.

        Flow:
          For each of 4 stages:
            1. RGB branch runs adapter interaction (Injector → ViT blocks → Extractor)
            2. T branch runs adapter interaction
            3. Bidirectional CrossAttn: each modality senses the other
               -> updated x_tokens propagate to the next stage
          Then assemble multi-scale outputs from both branches.

        Args:
            x: 6-channel input [B, 6, H, W] (first 3=RGB, last 3=T),
               or 4-channel [B, 4, H, W] (3 RGB + 1 T)

        Returns:
            List of 4 feature maps at strides [4, 8, 16, 32]
        """
        if x.shape[1] == 6:
            rgb_input = x[:, :3, :, :]
            thr_input = x[:, 3:6, :, :]
        elif x.shape[1] == 4:
            rgb_input = x[:, :3, :, :]
            thr_input = x[:, 3:4, :, :]
        else:
            B = x.shape[0] // 2
            rgb_input = x[:B, :3, :, :]
            thr_input = x[B:, :, :, :]
            if thr_input.shape[1] == 3:
                thr_input = thr_input[:, :1, :, :]

        # If thermal is 1-channel, expand to 3 for SPM input
        if thr_input.shape[1] == 1:
            thr_input = thr_input.expand(-1, 3, -1, -1)

        # Initialize both branches
        (rgb_x, rgb_c, rgb_c1, rgb_c2_len, rgb_c3_len, rgb_H, rgb_W,
         rgb_di1, rgb_di2) = self._init_adapter_state(
            rgb_input, self.rgb_adapter.spm, self.rgb_adapter.level_embed,
            self.rgb_adapter.backbone, self.rgb_adapter.wrapped_blocks,
            self.img_size, self.patch_size)

        (thr_x, thr_c, thr_c1, thr_c2_len, thr_c3_len, thr_H, thr_W,
         thr_di1, thr_di2) = self._init_adapter_state(
            thr_input, self.thr_spm, self.thr_level_embed,
            self.thr_backbone, self.thr_wrapped_blocks,
            self.img_size, self.patch_size)

        # Interleave: each stage runs both branches then quality repair + CrossAttn
        rgb_outs = []
        thr_outs = []
        all_quality = []  # for quality loss in segmentor

        for i in range(len(self.interaction_indexes)):
            indexes = self.interaction_indexes[i]

            # RGB branch stage
            rgb_x, rgb_c, rgb_out = self._run_stage(
                rgb_x, rgb_c, self.rgb_adapter.interactions[i],
                self.rgb_adapter.wrapped_blocks, indexes,
                rgb_di1, rgb_di2, rgb_H, rgb_W)
            rgb_outs.append(rgb_out)

            # T branch stage
            thr_x, thr_c, thr_out = self._run_stage(
                thr_x, thr_c, self.thr_interactions[i],
                self.thr_wrapped_blocks, indexes,
                thr_di1, thr_di2, thr_H, thr_W)
            thr_outs.append(thr_out)

            # ===== [INCREMENT] Quality repair (before baseline CrossAttn) =====
            if getattr(self, 'use_quality', False):
                from mmseg.models.segmentors.eomt_quality_attn import (
                    soft_keep_mask, complementary_fix_tokens)
                s_rgb = self.quality_predictors[i](rgb_x, thr_x)  # [B,N,1]
                s_thr = self.quality_predictors[i](thr_x, rgb_x)

                D_rgb = soft_keep_mask(s_rgb, self.quality_tau, self.mask_temperature)
                D_thr = soft_keep_mask(s_thr, self.quality_tau, self.mask_temperature)
                D_rgb, D_thr = complementary_fix_tokens(D_rgb, D_thr, s_rgb, s_thr)

                if getattr(self, 'use_quality_gate', False):
                    m_thr = D_thr.squeeze(-1)
                    m_rgb = D_rgb.squeeze(-1)
                    repaired_rgb = self.quality_cross_fusions[i](rgb_x, thr_x, keep_mask=m_thr)
                    repaired_thr = self.quality_cross_fusions[i](thr_x, rgb_x, keep_mask=m_rgb)
                else:
                    repaired_rgb = self.quality_cross_fusions[i](rgb_x, thr_x, keep_mask=None)
                    repaired_thr = self.quality_cross_fusions[i](thr_x, rgb_x, keep_mask=None)

                if getattr(self, 'use_quality_merge', False):
                    repaired_rgb = repaired_rgb * D_rgb + rgb_x * (1 - D_rgb)
                    repaired_thr = repaired_thr * D_thr + thr_x * (1 - D_thr)

                rgb_x = repaired_rgb
                thr_x = repaired_thr
                all_quality.append((s_rgb, s_thr))
            else:
                all_quality.append(None)

            # ===== [BASELINE] Cross-modal attention: sense each other =====
            new_rgb_x = self.cross_fusions[i](rgb_x, thr_x, keep_mask=None)
            new_thr_x = self.cross_fusions[i](thr_x, rgb_x, keep_mask=None)
            rgb_x = new_rgb_x
            thr_x = new_thr_x

        # Store quality predictions for segmentor's quality loss
        self._all_quality = all_quality

        # Assemble final outputs
        return self._assemble_outputs(
            rgb_c, rgb_c1, thr_c, thr_c1,
            rgb_outs, thr_outs,
            self.rgb_adapter.up, self.thr_up,
            [self.rgb_adapter.norm1, self.rgb_adapter.norm2,
             self.rgb_adapter.norm3, self.rgb_adapter.norm4],
            [self.thr_norm1, self.thr_norm2,
             self.thr_norm3, self.thr_norm4],
            rgb_c2_len, rgb_c3_len,
            self.img_size, rgb_H, rgb_W,
            self.add_vit_feature, self.fusion_type, self.fusion_norms)

    def train(self, mode=True):
        super().train(mode)
        # Keep ViT backbone frozen
        self.rgb_adapter.backbone.eval()
        for p in self.rgb_adapter.backbone.parameters():
            p.requires_grad = False
