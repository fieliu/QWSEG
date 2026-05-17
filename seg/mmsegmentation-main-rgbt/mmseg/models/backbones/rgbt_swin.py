import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS
from .swin import SwinTransformer


@MODELS.register_module()
class RGBTSwinTransformer(BaseModule):

    def __init__(self,
                 embed_dims=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 with_cp=False,
                 frozen_stages=-1,
                 share_start_idx=4,
                 fusion_type='MAX',
                 thr_in_channels=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.share_start_idx = share_start_idx
        self.fusion_type = fusion_type
        self.out_indices = out_indices

        rgb_kwargs = dict(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            out_indices=out_indices,
            with_cp=with_cp,
            frozen_stages=frozen_stages,
            init_cfg=init_cfg,
        )

        self.rgb_branch = SwinTransformer(in_channels=3, **rgb_kwargs)

        thr_kwargs = rgb_kwargs.copy()
        thr_kwargs['init_cfg'] = init_cfg
        self.thr_branch = SwinTransformer(in_channels=thr_in_channels, **thr_kwargs)

        if share_start_idx < len(depths):
            for i in range(share_start_idx, len(depths)):
                self.thr_branch.stages[i] = self.rgb_branch.stages[i]
                if i in out_indices:
                    norm_name = f'norm{i}'
                    setattr(self.thr_branch, norm_name,
                            getattr(self.rgb_branch, norm_name))

        in_channels_list = [embed_dims * (2 ** i) for i in range(len(depths))]
        self.fusion_norms = nn.ModuleList()
        for idx in out_indices:
            ch = in_channels_list[idx]
            self.fusion_norms.append(nn.BatchNorm2d(ch))

    def init_weights(self):
        if self.init_cfg is not None and self.init_cfg.type == 'Pretrained':
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            rgb_state = {}
            thr_state = {}
            for k, v in state_dict.items():
                key = k.replace('backbone.', '', 1) if k.startswith('backbone.') else k
                rgb_state[f'rgb_branch.{key}'] = v
                thr_state[f'thr_branch.{key}'] = v

            model_dict = self.state_dict()
            rgb_loaded = {k: v for k, v in rgb_state.items() if k in model_dict and v.shape == model_dict[k].shape}
            thr_loaded = {k: v for k, v in thr_state.items() if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(rgb_loaded)
            model_dict.update(thr_loaded)
            self.load_state_dict(model_dict, strict=False)

            rgb_keys = set(rgb_state.keys()) - set(rgb_loaded.keys())
            thr_keys = set(thr_state.keys()) - set(thr_loaded.keys())
            if rgb_keys:
                print(f'RGB branch skipped keys: {sorted(rgb_keys)}')
            if thr_keys:
                print(f'THR branch skipped keys: {sorted(thr_keys)}')
        else:
            self.rgb_branch.init_weights()
            self.thr_branch.init_weights()

    def _fuse_feats(self, rgb_feats, thr_feats):
        fused = []
        for i, idx in enumerate(self.out_indices):
            rgb_f = rgb_feats[idx]
            thr_f = thr_feats[idx]
            if self.fusion_type == 'ADD':
                out = (rgb_f + thr_f) / 2.0
            elif self.fusion_type == 'MAX':
                out = torch.max(rgb_f, thr_f)
            else:
                out = (rgb_f + thr_f) / 2.0
            out = self.fusion_norms[i](out)
            fused.append(out)
        return fused

    def forward(self, x):
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

        rgb_feats = self.rgb_branch(rgb_input)
        thr_feats = self.thr_branch(thr_input)

        rgb_feat_dict = {}
        thr_feat_dict = {}
        for i, idx in enumerate(self.out_indices):
            rgb_feat_dict[idx] = rgb_feats[i]
            thr_feat_dict[idx] = thr_feats[i]

        fused_feats = self._fuse_feats(rgb_feat_dict, thr_feat_dict)

        self._rgb_feats = [rgb_feat_dict[idx] for idx in self.out_indices]
        self._thr_feats = [thr_feat_dict[idx] for idx in self.out_indices]

        return fused_feats

    def forward_with_mask(self, x, rgb_mask=None, thr_mask=None):
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

        rgb_feats = self.rgb_branch(rgb_input)
        thr_feats = self.thr_branch(thr_input)

        rgb_feat_dict = {}
        thr_feat_dict = {}
        for i, idx in enumerate(self.out_indices):
            rgb_feat_dict[idx] = rgb_feats[i]
            thr_feat_dict[idx] = thr_feats[i]

        fused_feats = self._fuse_feats(rgb_feat_dict, thr_feat_dict)

        self._rgb_feats = [rgb_feat_dict[idx] for idx in self.out_indices]
        self._thr_feats = [thr_feat_dict[idx] for idx in self.out_indices]

        return fused_feats
