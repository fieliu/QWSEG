import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_, constant_init
from mmengine.runner.checkpoint import _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.registry import MODELS


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


class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SAMAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, \
                "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        q = self.q_proj(x).reshape(B, H * W, self.num_heads,
                                   -1).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, H * W, self.num_heads,
                                   -1).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, H * W, self.num_heads,
                                   -1).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(
            0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class SAMBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAMAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size
            if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SAMPatchEmbed(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                   max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
        q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    if q.ndim == 4:
        B, n_heads, _, dim = q.shape
        r_q = q.reshape(B, n_heads, q_h, q_w, dim)
        rel_h = torch.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
        rel_w = torch.einsum("bnhwc,wkc->bnhwk", r_q, Rw)
        attn = (attn.view(B, n_heads, q_h, q_w, k_h, k_w)
                + rel_h[:, :, :, :, :, None]
                + rel_w[:, :, :, :, None, :]).view(B, n_heads, q_h * q_w,
                                                    k_h * k_w)
    else:
        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        attn = (attn.view(B, q_h, q_w, k_h, k_w)
                + rel_h[:, :, :, :, None]
                + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


@MODELS.register_module()
class SAMViT(BaseModule):

    def __init__(
        self,
        img_size: int = 480,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
        out_indices: Tuple[int, ...] = (3, 5, 7, 11),
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        pretrained: Optional[str] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes

        if isinstance(out_indices, int):
            self.out_indices = [out_indices]
        else:
            self.out_indices = list(out_indices)

        self.patch_embed = SAMPatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_channels,
            embed_dim=embed_dims,
        )

        self.patch_shape = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_shape[0], self.patch_shape[1],
                            embed_dims))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SAMBlock(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size
                if i not in global_attn_indexes else 0,
                input_size=self.patch_shape,
            )
            self.blocks.append(block)

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LayerNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint_path = self.init_cfg['checkpoint']
            checkpoint = _load_checkpoint(
                checkpoint_path, logger=None, map_location='cpu')
            state_dict = self._load_sam_checkpoint(checkpoint)
            self.load_state_dict(state_dict, False)
        elif self.init_cfg is not None:
            super().init_weights()

    def _load_sam_checkpoint(self, checkpoint):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('image_encoder.'):
                new_key = key[len('image_encoder.'):]
                new_state_dict[new_key] = value

        model_state_dict = self.state_dict()
        loaded_keys = set()
        for key in new_state_dict:
            if '.attn.qkv.' in key:
                base_key = key.replace('.attn.qkv.', '.attn.')
                qkv_weight = new_state_dict[key]
                dim = qkv_weight.shape[0] // 3
                if 'weight' in key:
                    model_state_dict[base_key + 'q_proj.weight'] = qkv_weight[:dim]
                    model_state_dict[base_key + 'k_proj.weight'] = qkv_weight[dim:2*dim]
                    model_state_dict[base_key + 'v_proj.weight'] = qkv_weight[2*dim:]
                elif 'bias' in key:
                    model_state_dict[base_key + 'q_proj.bias'] = qkv_weight[:dim]
                    model_state_dict[base_key + 'k_proj.bias'] = qkv_weight[dim:2*dim]
                    model_state_dict[base_key + 'v_proj.bias'] = qkv_weight[2*dim:]
                loaded_keys.add(base_key + 'q_proj.weight')
                loaded_keys.add(base_key + 'k_proj.weight')
                loaded_keys.add(base_key + 'v_proj.weight')
                if 'bias' in key:
                    loaded_keys.add(base_key + 'q_proj.bias')
                    loaded_keys.add(base_key + 'k_proj.bias')
                    loaded_keys.add(base_key + 'v_proj.bias')
            elif key in model_state_dict:
                if key == 'pos_embed':
                    loaded_pos_embed = new_state_dict[key]
                    model_pos_shape = model_state_dict[key].shape
                    if loaded_pos_embed.shape != model_pos_shape:
                        loaded_pos_embed = self._interpolate_pos_embed(
                            loaded_pos_embed, model_pos_shape)
                    model_state_dict[key] = loaded_pos_embed
                    loaded_keys.add(key)
                elif 'rel_pos_h' in key or 'rel_pos_w' in key:
                    loaded_rel_pos = new_state_dict[key]
                    model_rel_pos = model_state_dict[key]
                    if loaded_rel_pos.shape != model_rel_pos.shape:
                        loaded_rel_pos = self._interpolate_rel_pos(
                            loaded_rel_pos, model_rel_pos.shape[0])
                    model_state_dict[key] = loaded_rel_pos
                    loaded_keys.add(key)
                else:
                    model_state_dict[key] = new_state_dict[key]
                    loaded_keys.add(key)

        return model_state_dict

    def _interpolate_pos_embed(self, pos_embed, target_shape):
        ori_h, ori_w = pos_embed.shape[1], pos_embed.shape[2]
        new_h, new_w = target_shape[1], target_shape[2]
        if ori_h == new_h and ori_w == new_w:
            return pos_embed

        pos_embed = pos_embed.float().permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).to(pos_embed.dtype)
        return pos_embed

    def _interpolate_rel_pos(self, rel_pos, target_size):
        if rel_pos.shape[0] == target_size:
            return rel_pos
        rel_pos = rel_pos.float().unsqueeze(0).permute(0, 2, 1)
        rel_pos = F.interpolate(
            rel_pos, size=target_size, mode='linear', align_corners=False)
        rel_pos = rel_pos.permute(0, 2, 1).squeeze(0).to(rel_pos.dtype)
        return rel_pos

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)
