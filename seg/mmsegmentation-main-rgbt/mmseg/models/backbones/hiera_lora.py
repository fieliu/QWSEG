import math
from typing import List, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class LoRALinear(nn.Module):

    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = self.original_linear(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return original_out + lora_out

    @property
    def weight(self):
        return self.original_linear.weight

    @property
    def bias(self):
        return self.original_linear.bias


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, activation: nn.Module = nn.ReLU,
                 sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, kernel_size: Tuple[int, ...] = (7, 7),
                 stride: Tuple[int, ...] = (4, 4),
                 padding: Tuple[int, ...] = (3, 3),
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size,
            stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats, temperature: int = 10000,
                 normalize: bool = True, scale: Optional[float] = None,
                 warmup_cache: bool = True, image_size: int = 1024,
                 strides: Tuple = (4, 8, 16, 32)):
        super().__init__()
        assert num_pos_feats % 2 == 0
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}
        if warmup_cache and torch.cuda.is_available():
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1).repeat(B, 1, W))
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1).repeat(B, H, 1))
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):

    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.q_pool = q_pool
        self.q_proj = nn.Linear(dim, dim_out)
        self.k_proj = nn.Linear(dim, dim_out)
        self.v_proj = nn.Linear(dim, dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

    def _init_from_qkv(self, qkv_weight, qkv_bias):
        dim_out = self.dim_out
        self.q_proj.weight.data.copy_(qkv_weight[:dim_out])
        self.k_proj.weight.data.copy_(qkv_weight[dim_out:2 * dim_out])
        self.v_proj.weight.data.copy_(qkv_weight[2 * dim_out:])
        if qkv_bias is not None:
            self.q_proj.bias.data.copy_(qkv_bias[:dim_out])
            self.k_proj.bias.data.copy_(qkv_bias[dim_out:2 * dim_out])
            self.v_proj.bias.data.copy_(qkv_bias[2 * dim_out:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        q = self.q_proj(x).reshape(B, H * W, self.num_heads, -1)
        k = self.k_proj(x).reshape(B, H * W, self.num_heads, -1)
        v = self.v_proj(x).reshape(B, H * W, self.num_heads, -1)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim: int, dim_out: int, num_heads: int,
                 mlp_ratio: float = 4.0, drop_path: float = 0.0,
                 norm_layer: Union[nn.Module, str] = "LayerNorm",
                 q_stride: Tuple[int, int] = None, act_layer: nn.Module = nn.GELU,
                 window_size: int = 0):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)

        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        xn = self.norm2(x)
        x = x + self.drop_path(self.mlp(xn))
        return x


class HieraBackbone(nn.Module):

    def __init__(self, embed_dim: int = 96, num_heads: int = 1,
                 drop_path_rate: float = 0.0, q_pool: int = 3,
                 q_stride: Tuple[int, int] = (2, 2),
                 stages: Tuple[int, ...] = (2, 3, 16, 3),
                 dim_mul: float = 2.0, head_mul: float = 2.0,
                 window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
                 window_spec: Tuple[int, ...] = (8, 4, 14, 7),
                 global_att_blocks: Tuple[int, ...] = (12, 16, 20),
                 return_interm_layers: bool = True):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size)
            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out])

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        tile_h = (h + window_embed.shape[2] - 1) // window_embed.shape[2]
        tile_w = (w + window_embed.shape[3] - 1) // window_embed.shape[3]
        window_embed = window_embed.tile([1, 1, tile_h, tile_w])
        window_embed = F.interpolate(window_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs


class FpnNeck(nn.Module):

    def __init__(self, position_encoding: nn.Module, d_model: int,
                 backbone_channel_list: List[int], kernel_size: int = 1,
                 stride: int = 1, padding: int = 0,
                 fpn_interp_model: str = "bilinear",
                 fuse_type: str = "sum",
                 fpn_top_down_levels: Optional[List[int]] = None):
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module("conv", nn.Conv2d(
                in_channels=dim, out_channels=d_model,
                kernel_size=kernel_size, stride=stride, padding=padding))
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    size=lateral_features.shape[-2:],
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)
        return out, pos


def apply_lora_to_hiera(model, rank=4, alpha=4.0, dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = ['qkv', 'proj']

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        should_apply = any(t in name for t in target_modules)
        if not should_apply:
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        replacements.append((parent_name, child_name, module))

    for parent_name, child_name, module in replacements:
        parent = model
        for attr in parent_name.split('.'):
            if attr:
                parent = getattr(parent, attr)
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)

    return model


def freeze_non_lora_params_hiera(model):
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def merge_lora_weights_hiera(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_weight = (module.lora_B.data @ module.lora_A.data) * module.scaling
            module.original_linear.weight.data += lora_weight.t()
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()


@MODELS.register_module()
class HieraLoRA(BaseModule):

    ARCH_CONFIGS = {
        't': {
            'embed_dim': 96, 'num_heads': 1,
            'stages': (1, 2, 7, 2),
            'global_att_blocks': (5, 7, 9),
            'window_pos_embed_bkg_spatial_size': (7, 7),
            'window_spec': (8, 4, 14, 7),
            'backbone_channel_list': [768, 384, 192, 96],
        },
        's': {
            'embed_dim': 96, 'num_heads': 1,
            'stages': (1, 2, 11, 2),
            'global_att_blocks': (7, 10, 13),
            'window_pos_embed_bkg_spatial_size': (7, 7),
            'window_spec': (8, 4, 14, 7),
            'backbone_channel_list': [768, 384, 192, 96],
        },
        'b': {
            'embed_dim': 112, 'num_heads': 2,
            'stages': (2, 3, 16, 3),
            'global_att_blocks': (12, 16, 20),
            'window_pos_embed_bkg_spatial_size': (14, 14),
            'window_spec': (8, 4, 14, 7),
            'backbone_channel_list': [896, 448, 224, 112],
        },
        'l': {
            'embed_dim': 144, 'num_heads': 2,
            'stages': (2, 6, 36, 4),
            'global_att_blocks': (23, 33, 43),
            'window_pos_embed_bkg_spatial_size': (7, 7),
            'window_spec': (8, 4, 16, 8),
            'backbone_channel_list': [1152, 576, 288, 144],
        },
    }

    FPN_CONFIGS = {
        't': {'num_pos_feats': 256, 'normalize': True, 'scale': None, 'temperature': 10000,
               'd_model': 256, 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest'},
        's': {'num_pos_feats': 256, 'normalize': True, 'scale': None, 'temperature': 10000,
               'd_model': 256, 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest'},
        'b': {'num_pos_feats': 256, 'normalize': True, 'scale': None, 'temperature': 10000,
               'd_model': 256, 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest'},
        'l': {'num_pos_feats': 256, 'normalize': True, 'scale': None, 'temperature': 10000,
               'd_model': 256, 'fpn_top_down_levels': [2, 3], 'fpn_interp_model': 'nearest'},
    }

    def __init__(self,
                 arch: str = 's',
                 use_lora: bool = True,
                 lora_rank: int = 4,
                 lora_alpha: float = 4.0,
                 lora_dropout: float = 0.0,
                 lora_target_modules: Optional[List[str]] = None,
                 freeze_backbone: bool = True,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert arch in self.ARCH_CONFIGS, f'Unknown arch: {arch}'
        self.arch = arch
        self.out_indices = out_indices
        self.use_lora = use_lora

        arch_cfg = self.ARCH_CONFIGS[arch]
        fpn_cfg = self.FPN_CONFIGS[arch]

        self.trunk = HieraBackbone(
            embed_dim=arch_cfg['embed_dim'],
            num_heads=arch_cfg['num_heads'],
            stages=arch_cfg['stages'],
            global_att_blocks=arch_cfg['global_att_blocks'],
            window_pos_embed_bkg_spatial_size=arch_cfg['window_pos_embed_bkg_spatial_size'],
            window_spec=arch_cfg['window_spec'],
        )

        self.neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=fpn_cfg['num_pos_feats'],
                normalize=fpn_cfg['normalize'],
                scale=fpn_cfg['scale'],
                temperature=fpn_cfg['temperature']),
            d_model=fpn_cfg['d_model'],
            backbone_channel_list=arch_cfg['backbone_channel_list'],
            fpn_top_down_levels=fpn_cfg['fpn_top_down_levels'],
            fpn_interp_model=fpn_cfg['fpn_interp_model'],
        )

        self.d_model = fpn_cfg['d_model']
        self.backbone_channel_list = arch_cfg['backbone_channel_list']

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ['qkv', 'proj']
            apply_lora_to_hiera(
                self.trunk, rank=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout, target_modules=lora_target_modules)
            if freeze_backbone:
                freeze_non_lora_params_hiera(self.trunk)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        trunk_out = self.trunk(x)
        features, pos = self.neck(trunk_out)
        out = [features[i] for i in self.out_indices]
        return out

    def merge_lora(self):
        if self.use_lora:
            merge_lora_weights_hiera(self.trunk)
