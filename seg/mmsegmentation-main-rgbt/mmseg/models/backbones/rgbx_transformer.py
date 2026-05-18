import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList

from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        from timm.models.layers import to_2tuple
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SemanticConsistencyInference(nn.Module):
    def __init__(self, input_dim, lambda_init=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fin, Fvi, t=0.4):
        cos_sim = F.cosine_similarity(Fin, Fvi, dim=-1)
        Sm = 0.1 * cos_sim.mean(dim=-1)

        if torch.any(Sm < t):
            concatenated = torch.cat((Fin, Fvi), dim=-1)
            Mmid = self.mlp(concatenated).squeeze(-1)
            Msc = self.sigmoid(Mmid)

            Pin = Fin * (1 - Msc.unsqueeze(-1)) + Fvi * Msc.unsqueeze(-1)
            Pvi = Fvi * (1 - Msc.unsqueeze(-1)) + Fin * Msc.unsqueeze(-1)
            Kvi = Fvi - Pvi
            Kin = Fin - Pin

            aFvi = Fvi - self.lambda_param * Kvi
            aFin = Fin - self.lambda_param * Kin
        else:
            aFvi, aFin = Fvi, Fin

        return aFvi, aFin


class From_Channel(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg_v = self.avg_pool(x).view(B, self.dim * 2)
        max_v = self.max_pool(x).view(B, self.dim * 2)

        avg_se = self.mlp(avg_v).view(B, self.dim * 2, 1)
        max_se = self.mlp(max_v).view(B, self.dim * 2, 1)

        Stat_out = self.sigmoid(avg_se + max_se).view(B, self.dim * 2, 1)
        channel_weights = Stat_out.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class From_Spatial(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 2, kernel_size),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x1_mean_out = torch.mean(x1, dim=1, keepdim=True)
        x1_max_out, _ = torch.max(x1, dim=1, keepdim=True)
        x2_mean_out = torch.mean(x2, dim=1, keepdim=True)
        x2_max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x_cat = torch.cat((x1_mean_out, x1_max_out, x2_mean_out, x2_max_out), dim=1)
        spatial_weights = self.mlp(x_cat).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class Fuse_s_c(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cha = From_Channel(self.dim)
        self.sap = From_Spatial(reduction=4)

    def forward(self, x1, x2):
        f_cha = self.cha(x1, x2)
        f_sap = self.sap(x1, x2)
        mixatt_out = f_cha.mul(f_sap)
        return mixatt_out


class Local_Eliminating(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_sc = Fuse_s_c(dim)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        mid_feature = self.gate(torch.cat((x1_flat, x2_flat), dim=2))
        mid_feature = mid_feature.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous()
        fusion = self.fuse_sc(x1, x2)
        channel_feature = mid_feature * fusion[0]
        spatial_feature = mid_feature * fusion[1]
        out_x1 = x1 + channel_feature * x2
        out_x2 = x2 + spatial_feature * x1
        out_1 = self.sigmoid(out_x1 * channel_feature - out_x1) * out_x1 + out_x1 * channel_feature
        out_2 = self.sigmoid(out_x2 * spatial_feature - out_x2) * out_x2 + out_x2 * spatial_feature
        return out_1, out_2


class Salient_Enhancement(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_atten = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_atten + x)
        x_out = self.proj_drop(x_out)
        return x_out


class Cross_Modal_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2, H, W):
        B1, N1, C1 = x1.shape
        B2, N2, C2 = x2.shape
        q1 = self.q1(x1).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x2_ = x2.permute(0, 2, 1).reshape(B2, C2, H, W)
            x2_ = self.sr(x2_).reshape(B2, C2, -1).permute(0, 2, 1)
            x2_ = self.norm(x2_)
            kv2 = self.kv2(x2_).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv2 = self.kv2(x2).reshape(B2, -1, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_atten = (attn @ v2).transpose(1, 2).reshape(B2, N2, C2)
        x_out = self.proj(x_atten + x1)
        x_out = self.proj_drop(x_out)
        return x_out


class Global_Eliminating(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.SE_x1 = Salient_Enhancement(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None,
                                          attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.SE_x2 = Salient_Enhancement(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None,
                                          attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CM_x1toX2 = Cross_Modal_Attention(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None,
                                                attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.CM_x2toX1 = Cross_Modal_Attention(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None,
                                                attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)

        x1_self_enhance = self.SE_x1(x1_flat, H1, W1)
        x2_self_enhance = self.SE_x2(x2_flat, H1, W1)
        x1_cross_enhance = self.CM_x1toX2(x1_self_enhance, x2_self_enhance, H1, W1)
        x2_cross_enhance = self.CM_x2toX1(x2_self_enhance, x1_cross_enhance, H1, W1)
        Fuse = self.proj(x2_cross_enhance)
        Fuse_out = Fuse.permute(0, 2, 1).reshape(B1, C1, H1, W1).contiguous()

        return Fuse_out


@MODELS.register_module()
class RGBXTransformer(BaseModule):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        self.sci_1 = SemanticConsistencyInference(input_dim=embed_dims[0])
        self.sci_2 = SemanticConsistencyInference(input_dim=embed_dims[1])
        self.sci_3 = SemanticConsistencyInference(input_dim=embed_dims[2])
        self.sci_4 = SemanticConsistencyInference(input_dim=embed_dims[3])

        self.LEs = nn.ModuleList([Local_Eliminating(dim=embed_dims[0], reduction=4),
                                  Local_Eliminating(dim=embed_dims[1], reduction=4),
                                  Local_Eliminating(dim=embed_dims[2], reduction=4),
                                  Local_Eliminating(dim=embed_dims[3], reduction=4)])

        self.GEs = nn.ModuleList([Global_Eliminating(dim=embed_dims[0], num_heads=num_heads[0], qkv_bias=qkv_bias,
                                                     qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                                     proj_drop=drop_rate, sr_ratio=sr_ratios[0]),
                                  Global_Eliminating(dim=embed_dims[1], num_heads=num_heads[1], qkv_bias=qkv_bias,
                                                     qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                                     proj_drop=drop_rate, sr_ratio=sr_ratios[1]),
                                  Global_Eliminating(dim=embed_dims[2], num_heads=num_heads[2], qkv_bias=qkv_bias,
                                                     qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                                     proj_drop=drop_rate, sr_ratio=sr_ratios[2]),
                                  Global_Eliminating(dim=embed_dims[3], num_heads=num_heads[3], qkv_bias=qkv_bias,
                                                     qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                                     proj_drop=drop_rate, sr_ratio=sr_ratios[3])])

    def init_weights(self):
        if self.init_cfg is None:
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
            from mmengine.runner.checkpoint import CheckpointLoader
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            clean_sd = {}
            for k, v in state_dict.items():
                nk = k.replace('backbone.', '', 1) if k.startswith('backbone.') else k
                clean_sd[nk] = v

            model_sd = self.state_dict()
            key_map = {}
            in_proj_pending = {}

            for pretrained_key, pretrained_val in clean_sd.items():
                if not pretrained_key.startswith('layers.'):
                    continue
                parts = pretrained_key.split('.')
                stage_idx = int(parts[1])
                sub_idx = parts[2]
                model_stage = stage_idx + 1

                if sub_idx == '0':
                    rest = '.'.join(parts[3:])
                    rest = rest.replace('projection.', 'proj.')
                    new_key = f'patch_embed{model_stage}.{rest}'
                    extra_key = f'extra_patch_embed{model_stage}.{rest}'
                elif sub_idx == '1':
                    block_idx = parts[3]
                    rest = '.'.join(parts[4:])
                    mapped, splits = self._map_block_subkey(rest, pretrained_val)
                    if mapped is None:
                        continue
                    if splits is not None:
                        for split_key, split_val in splits:
                            nk = f'block{model_stage}.{block_idx}.{split_key}'
                            ek = f'extra_block{model_stage}.{block_idx}.{split_key}'
                            if nk in model_sd and model_sd[nk].shape == split_val.shape:
                                key_map[nk] = split_val
                            if ek in model_sd and model_sd[ek].shape == split_val.shape:
                                key_map[ek] = split_val
                        continue
                    new_key = f'block{model_stage}.{block_idx}.{mapped}'
                    extra_key = f'extra_block{model_stage}.{block_idx}.{mapped}'
                elif sub_idx == '2':
                    rest = '.'.join(parts[3:])
                    new_key = f'norm{model_stage}.{rest}'
                    extra_key = f'extra_norm{model_stage}.{rest}'
                else:
                    continue

                if new_key in model_sd and model_sd[new_key].shape == pretrained_val.shape:
                    key_map[new_key] = pretrained_val
                if extra_key in model_sd and model_sd[extra_key].shape == pretrained_val.shape:
                    key_map[extra_key] = pretrained_val

            model_sd.update(key_map)
            self.load_state_dict(model_sd, strict=False)

            import logging
            logger = logging.getLogger(__name__)
            loaded_rgb = sum(1 for k in key_map if not k.startswith('extra_'))
            loaded_extra = sum(1 for k in key_map if k.startswith('extra_'))
            logger.info(f'RGBXTransformer: loaded {loaded_rgb} keys to rgb branch, '
                        f'{loaded_extra} keys to extra branch')

    @staticmethod
    def _map_block_subkey(rest, pretrained_val):
        if rest == 'attn.attn.in_proj_weight':
            dim = pretrained_val.shape[0] // 3
            q_val = pretrained_val[:dim]
            kv_val = pretrained_val[dim:]
            return None, [
                ('attn.q.weight', q_val),
                ('attn.kv.weight', kv_val),
            ]
        if rest == 'attn.attn.in_proj_bias':
            dim = pretrained_val.shape[0] // 3
            q_val = pretrained_val[:dim]
            kv_val = pretrained_val[dim:]
            return None, [
                ('attn.q.bias', q_val),
                ('attn.kv.bias', kv_val),
            ]
        if rest.startswith('attn.attn.out_proj.'):
            param_type = rest.split('.')[-1]
            return f'attn.proj.{param_type}', None
        if rest.startswith('ffn.layers.0.'):
            param_type = rest.split('.')[-1]
            return f'mlp.fc1.{param_type}', None
        if rest.startswith('ffn.layers.1.'):
            param_type = rest.split('.')[-1]
            return f'mlp.dwconv.dwconv.{param_type}', None
        if rest.startswith('ffn.layers.4.'):
            param_type = rest.split('.')[-1]
            return f'mlp.fc2.{param_type}', None
        return rest, None

    def forward_features(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs_semantic = []
        outs_vision = []

        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_e, _, _ = self.extra_patch_embed1(x_e)
        x_rgb, x_e = self.sci_1(x_rgb, x_e)
        for blk in self.block1:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.LEs[0](x_rgb, x_e)
        x_fused = self.GEs[0](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        x_rgb, x_e = self.sci_2(x_rgb, x_e)
        for blk in self.block2:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.LEs[1](x_rgb, x_e)
        x_fused = self.GEs[1](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        x_rgb, x_e = self.sci_3(x_rgb, x_e)
        for blk in self.block3:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.LEs[2](x_rgb, x_e)
        x_fused = self.GEs[2](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        x_rgb, x_e = self.sci_4(x_rgb, x_e)
        for blk in self.block4:
            x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4:
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.LEs[3](x_rgb, x_e)
        x_fused = self.GEs[3](x_rgb, x_e)

        outs_vision.append(x_rgb)
        outs_vision.append(x_e)
        outs_semantic.append(x_fused)

        return outs_vision, outs_semantic

    def forward(self, x):
        x_rgb = x[:, :3, :, :]
        x_e = x[:, 3:, :, :]
        if x_e.shape[1] == 1:
            x_e = x_e.repeat(1, 3, 1, 1)
        out_vision, out_semantic = self.forward_features(x_rgb, x_e)
        return out_vision, out_semantic
