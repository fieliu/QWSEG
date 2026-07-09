"""Single-modal EoMT segmentor wrapped for MMSegmentation.

Deliverable 1: a faithful EoMT (Encoder-only Mask Transformer, CVPR 2025)
wrapped as an mmseg BaseSegmentor. Uses a DINOv3 ViT backbone via the original
EoMT ViT wrapper, and the original HF Mask2Former set-prediction loss.

Only the RGB channels (first 3 of the 6-channel RGB-T input) are used here.
This is the smoke-test baseline to confirm EoMT runs inside mmseg.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

from .eomt_core import ViT, EoMT
from .eomt_core.mask_classification_loss import MaskClassificationLoss
from .eomt_utils import build_targets, mask_class_to_seg_logits, resize_seg_logits


@MODELS.register_module()
class EoMTSegmentor(BaseSegmentor):
    def __init__(
        self,
        img_size,
        num_classes,
        backbone_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        backbone_ckpt: Optional[str] = None,
        patch_size=16,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
        local_files_only=True,
        # loss coefficients (EoMT defaults)
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        no_object_coefficient=0.1,
        ignore_index=255,
        data_preprocessor=None,
        init_cfg=None,
        # mmseg standard kwargs (accepted for compatibility, stored)
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.img_size = tuple(img_size)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.align_corners = False
        self.out_channels = num_classes

        encoder = ViT(
            img_size=self.img_size,
            patch_size=patch_size,
            backbone_name=backbone_name,
            ckpt_path=backbone_ckpt,
            local_files_only=local_files_only,
        )
        self.network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

    # ---- input handling -------------------------------------------------
    def _get_rgb(self, inputs):
        # 6-channel RGB-T input -> take RGB (first 3). Already normalized by
        # the mmseg data_preprocessor, so EoMT internal normalization is OFF
        # (we feed network.encoder.backbone directly, bypassing forward's norm).
        return inputs[:, :3]

    def _network_forward(self, x):
        """Replicates EoMT.forward but WITHOUT the internal pixel_mean/std
        normalization (mmseg data_preprocessor already normalized the input)."""
        net = self.network
        backbone = net.encoder.backbone

        rope = None
        if hasattr(backbone, "rope_embeddings"):
            rope = backbone.rope_embeddings(x)

        x = backbone.patch_embed(x)
        if hasattr(backbone, "_pos_embed"):
            x = backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        num_layers = len(backbone.blocks)

        for i, block in enumerate(backbone.blocks):
            if i == num_layers - net.num_blocks:
                x = torch.cat(
                    (net.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )
            if net.masked_attn_enabled and i >= num_layers - net.num_blocks:
                ml, cl = net._predict(backbone.norm(x))
                mask_logits_per_layer.append(ml)
                class_logits_per_layer.append(cl)
                attn_mask = net._attn_mask(x, ml, i)

            attn = block.attn if hasattr(block, "attn") else block.attention
            attn_out = net._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                attn_out = block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                attn_out = block.layer_scale1(attn_out)
            x = x + block.drop_path(attn_out)
            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                mlp_out = block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                mlp_out = block.layer_scale2(mlp_out)
            x = x + block.drop_path(mlp_out)

        ml, cl = net._predict(backbone.norm(x))
        mask_logits_per_layer.append(ml)
        class_logits_per_layer.append(cl)
        return mask_logits_per_layer, class_logits_per_layer

    # ---- training -------------------------------------------------------
    def loss(self, inputs, data_samples):
        x = self._get_rgb(inputs)
        mask_logits_per_layer, class_logits_per_layer = self._network_forward(x)
        targets = build_targets(data_samples, self.num_classes, self.ignore_index)

        losses = {}
        n_layers = len(mask_logits_per_layer)
        for li, (ml, cl) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
            ml_up = F.interpolate(ml, size=self.img_size, mode="bilinear", align_corners=False)
            layer_loss = self.criterion(ml_up, targets, cl)
            for k, v in layer_loss.items():
                losses[f"l{li}.{k}"] = v
        return losses

    # ---- inference ------------------------------------------------------
    def encode_decode(self, inputs, batch_img_metas):
        x = self._get_rgb(inputs)
        mask_logits_per_layer, class_logits_per_layer = self._network_forward(x)
        ml, cl = mask_logits_per_layer[-1], class_logits_per_layer[-1]
        seg_logits = mask_class_to_seg_logits(ml, cl)
        seg_logits = resize_seg_logits(seg_logits, inputs.shape[2:])
        return seg_logits

    def predict(self, inputs, data_samples=None):
        if data_samples:
            bm = [ds.metainfo for ds in data_samples]
        else:
            bm = [dict(ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                       pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])] * inputs.shape[0]
        seg_logits = self.encode_decode(inputs, bm)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples=None):
        x = self._get_rgb(inputs)
        ml, cl = self._network_forward(x)
        return mask_class_to_seg_logits(ml[-1], cl[-1])

    def extract_feat(self, inputs):
        return self._get_rgb(inputs)

    def postprocess_result(self, seg_logits, data_samples):
        B, C, H, W = seg_logits.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(B)]
        for i in range(B):
            img_meta = data_samples[i].metainfo
            ps = img_meta.get('padding_size', [0, 0, 0, 0])
            pl, pr, pt, pb = ps
            i_sl = seg_logits[i:i + 1, :, pt:H - pb, pl:W - pr]
            flip = img_meta.get('flip', None)
            if flip:
                fd = img_meta.get('flip_direction', None)
                i_sl = i_sl.flip(dims=(3,) if fd == 'horizontal' else (2,))
            from mmseg.models.utils import resize
            i_sl = resize(i_sl, size=img_meta['ori_shape'], mode='bilinear',
                          align_corners=self.align_corners, warning=False).squeeze(0)
            pred = i_sl.argmax(dim=0, keepdim=True)
            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_sl),
                'pred_sem_seg': PixelData(data=pred),
            })
        return data_samples
