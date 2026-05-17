import math

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_, kaiming_init

from mmseg.registry import MODELS
from .mae import MAETransformerEncoderLayer


@MODELS.register_module()
class LightweightMAEBranch(nn.Module):

    def __init__(self,
                 embed_dims=192,
                 num_layers=12,
                 num_heads=3,
                 mlp_ratio=4,
                 out_indices=(3, 5, 7, 11),
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 init_values=1.0,
                 universal_embed_dims=768,
                 final_norm=True,
                 proj_hidden_dim=768,
                 img_size=(480, 480),
                 patch_size=16,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_indices = list(out_indices)
        self.universal_embed_dims = universal_embed_dims
        self.final_norm = final_norm
        self.init_cfg = init_cfg

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        window_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.input_proj = nn.Linear(
            universal_embed_dims, embed_dims, bias=False)

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                MAETransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=0.,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    bias=True,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    window_size=window_size,
                    init_values=init_values))

        if final_norm:
            self.norm = nn.LayerNorm(embed_dims)

        self.proj_layers = ModuleList()
        for _ in range(len(self.out_indices)):
            self.proj_layers.append(
                nn.Sequential(
                    nn.Linear(embed_dims, proj_hidden_dim),
                    nn.GELU(),
                    nn.Linear(proj_hidden_dim, universal_embed_dims)))

    def forward(self, x, hw_shape):
        B = x.shape[0]

        x = self.input_proj(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm(x)
            if i in self.out_indices:
                out = x[:, 1:]
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  self.embed_dims).permute(
                    0, 3, 1, 2).contiguous()

                stage_idx = self.out_indices.index(i)
                B_o, _, H_o, W_o = out.shape
                out_flat = out.permute(0, 2, 3, 1).reshape(
                    B_o * H_o * W_o, self.embed_dims)
                out_proj = self.proj_layers[stage_idx](out_flat)
                out_proj = out_proj.reshape(
                    B_o, H_o, W_o, self.universal_embed_dims)
                out_proj = out_proj.permute(
                    0, 3, 1, 2).contiguous()
                outs.append(out_proj)

        return tuple(outs)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)

        self.apply(_init_weights)

        for layer_id, layer in enumerate(self.layers):
            def rescale(param, lid):
                param.div_(math.sqrt(2.0 * lid))
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)
