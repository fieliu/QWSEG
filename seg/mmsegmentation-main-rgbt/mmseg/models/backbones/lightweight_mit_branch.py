import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_, constant_init

from mmseg.registry import MODELS
from .mit import TransformerEncoderLayer
from ..utils import nlc_to_nchw


@MODELS.register_module()
class LightweightMITBranch(nn.Module):

    def __init__(self,
                 embed_dims=32,
                 num_stages=4,
                 num_layers=[2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 universal_embed_dims_list=None,
                 with_cp=False,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.init_cfg = init_cfg

        if universal_embed_dims_list is None:
            universal_embed_dims_list = [64, 128, 320, 512]

        self.universal_embed_dims_list = universal_embed_dims_list

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.channel_proj = ModuleList()
        self.layers = ModuleList()
        self.norms = ModuleList()

        for i in range(num_stages):
            embed_dims_i = embed_dims * num_heads[i]
            universal_dims_i = universal_embed_dims_list[i]

            self.channel_proj.append(
                nn.Linear(universal_dims_i, embed_dims_i, bias=False))

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
                    sr_ratio=sr_ratios[i]) for idx in range(num_layers[i])
            ])
            self.layers.append(layer)

            from mmcv.cnn import build_norm_layer
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.norms.append(norm)

            cur += num_layers[i]

    def forward(self, inputs_list, hw_shapes):
        outs = []
        for i in range(self.num_stages):
            x = inputs_list[i]
            hw_shape = hw_shapes[i]

            x = self.channel_proj[i](x)

            for block in self.layers[i]:
                x = block(x, hw_shape)
            x = self.norms[i](x)
            x = nlc_to_nchw(x, hw_shape)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.weight, 1.0)
                constant_init(m.bias, 0.)
