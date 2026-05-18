import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_norm_layer
from mmengine.model import BaseModule, ModuleList

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down1 = nn.Linear(dim, dim // ratio)
        self.down2 = nn.Linear(dim, dim // ratio)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        x = (self.gap_pool(x) + self.gmp_pool(x)) * 0.5
        y = self.up(self.act(self.down1(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2).view(b, c)
        y = (y / y.norm(dim=1, keepdim=True)).contiguous().view(b, c)
        return y


class EAEF_clip(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb_r = Feature_Pool(dim, ratio=4)
        self.emb_x = Feature_Pool(dim, ratio=4)
        self.logit_scale = nn.Parameter(torch.ones([]) * dim)

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        x1, x2 = self.emb_r(x1), self.emb_x(x2)
        logits_per = self.logit_scale * torch.mul(x1, x2)
        return logits_per


class local_Feature_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, logits_per, RGB, X):
        b, c, h, w = RGB.size()
        add_gate = self.sigmoid(-1 * logits_per).contiguous().view(b, c, 1, 1)
        x1 = X * add_gate + RGB
        x2 = RGB * add_gate + X
        x = (x1 + x2) / 2
        return x


class MixFFN(BaseModule):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = nn.GELU()

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        from mmengine.model import Sequential
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        from mmcv.cnn.bricks.drop import build_dropout
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        self.batch_first = batch_first
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        if dropout_layer is not None:
            from mmcv.cnn.bricks.drop import build_dropout
            self.dropout_layer = build_dropout(dropout_layer)
        else:
            self.dropout_layer = torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        B, N, C = x_q.shape
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return identity + self.dropout_layer(x)


class TransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)
        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as cp
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class BIMixVisionTransformer(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]

        cur = 0
        self.layers_rgb = ModuleList()
        self.layers_x = ModuleList()
        self.fuse_module = ModuleList()

        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)

            x_patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)

            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])

            x_layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])

            EAEF1s = EAEF_clip(dim=embed_dims * num_heads[i])
            LFFM1s = local_Feature_Fusion(dim=embed_dims * num_heads[i])

            in_channels = embed_dims_i

            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            x_norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers_rgb.append(ModuleList([patch_embed, layer, norm]))
            self.layers_x.append(ModuleList([x_patch_embed, x_layer, x_norm]))
            self.fuse_module.append(ModuleList([EAEF1s, LFFM1s]))
            cur += num_layer

    def init_weights(self):
        from mmengine.runner.checkpoint import CheckpointLoader
        if self.init_cfg is not None and hasattr(self.init_cfg, 'type') and \
                self.init_cfg.type == 'Pretrained' and self.init_cfg.checkpoint:
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            # strip 'backbone.' prefix
            clean_sd = {}
            for k, v in state_dict.items():
                nk = k.replace('backbone.', '', 1) if k.startswith('backbone.') else k
                clean_sd[nk] = v

            model_sd = self.state_dict()

            def _map_branch(branch_prefix):
                loaded = {}
                for model_key, model_val in model_sd.items():
                    if not model_key.startswith(branch_prefix):
                        continue
                    suffix = model_key[len(branch_prefix):]
                    parts = suffix.strip('.').split('.')
                    if len(parts) < 2:
                        continue
                    try:
                        stage = int(parts[0])
                    except ValueError:
                        continue

                    pretrained_key = 'layers.' + str(stage) + '.' + '.'.join(parts[1:])
                    if pretrained_key in clean_sd and clean_sd[pretrained_key].shape == model_val.shape:
                        loaded[model_key] = clean_sd[pretrained_key]
                return loaded

            rgb_load = _map_branch('layers_rgb.')
            x_load = _map_branch('layers_x.')
            model_sd.update(rgb_load)
            model_sd.update(x_load)
            self.load_state_dict(model_sd, strict=False)

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f'BIMixVisionTransformer: loaded {len(rgb_load)} keys to rgb branch, '
                        f'{len(x_load)} keys to x branch')
        elif self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    from mmengine.model.weight_init import trunc_normal_init
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    from mmengine.model.weight_init import constant_init
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    from mmengine.model.weight_init import normal_init
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        x, x_modal = x[:, :3, :, :], x[:, 3:, :, :]
        if x_modal.shape[1] == 1:
            x_modal = x_modal.repeat(1, 3, 1, 1)
        teacher = []
        rgb = []
        thermal = []
        for i, layer in enumerate(zip(self.layers_rgb, self.layers_x, self.fuse_module)):
            x, hw_shape = layer[0][0](x)
            x_modal, hw_shape = layer[1][0](x_modal)
            for block in layer[0][1]:
                x = block(x, hw_shape)
            for x_block in layer[1][1]:
                x_modal = x_block(x_modal, hw_shape)
            x = layer[0][2](x)
            x_modal = layer[1][2](x_modal)
            x = nlc_to_nchw(x, hw_shape)
            x_modal = nlc_to_nchw(x_modal, hw_shape)
            N_logits_per = layer[2][0](x, x_modal)
            x_fuse = layer[2][1](N_logits_per, x, x_modal)
            rgb.append(x)
            thermal.append(x_modal)
            teacher.append(x_fuse)
        return rgb, thermal, teacher
