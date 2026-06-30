"""DINOv3 + ViT-Adapter + Mask2Former segmentor for RGB-T segmentation.

This is a standard comparison baseline that uses:
- DINOv3 ViT-B/16 as the frozen backbone
- ViT-Adapter (Chen et al., ICLR 2023) to inject spatial priors and extract
  multi-scale features
- Mask2Former (Cheng et al., CVPR 2022) as the decoder head

For RGB-T: two modalities share the same DINOv3 backbone (Siamese), with
separate adapters for each modality. Features are fused via addition before
the Mask2Former decoder.

This baseline provides a fair comparison to the EoMT-based approach:
- Same backbone (DINOv3-B), same pretraining
- Standard decoder (Mask2Former) vs EoMT's encoder-only decoder
- ViT-Adapter spatial injection vs EoMT's direct query-decode
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList, add_prefix


@MODELS.register_module()
class DINOv3AdapterM2F(BaseSegmentor):
    """DINOv3 + ViT-Adapter + Mask2Former for RGB-T segmentation.

    Two-branch architecture:
    - RGB branch: DINOv3Adapter (shared backbone, separate adapter) -> multi-scale feats
    - T branch: DINOv3Adapter (shared backbone, separate adapter) -> multi-scale feats
    - Fusion: element-wise addition of multi-scale features
    - Decode: Mask2Former head

    Args:
        backbone: DINOv3Adapter backbone config
        decode_head: Mask2Former decode head config
        fusion_type: How to fuse RGB and T features ('add', 'cat', 'avg')
        freeze_backbone: Whether to freeze the shared DINOv3 backbone
    """

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
        fusion_type: str = 'add',
        freeze_backbone: bool = False,
    ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build two adapter branches (they share the DINOv3 backbone weights
        # but have separate adapter parameters)
        self.backbone_rgb = MODELS.build(backbone)
        self.backbone_t = MODELS.build(backbone)

        # Share the frozen DINOv3 backbone between branches
        # The adapter parameters (spatial_prior, injectors, feature_extractor)
        # are SEPARATE for each modality
        self.backbone_t.backbone = self.backbone_rgb.backbone

        if freeze_backbone:
            for p in self.backbone_rgb.backbone.parameters():
                p.requires_grad = False

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fusion_type = fusion_type

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def _split(self, inputs: torch.Tensor):
        """Split 6-channel RGB-T input into RGB and T."""
        return inputs[:, :3], inputs[:, 3:]

    def extract_feat(self, inputs: torch.Tensor):
        """Extract fused multi-scale features from RGB-T input.

        Returns 4-level features at strides [4, 8, 16, 32].
        """
        rgb, t = self._split(inputs)

        feats_rgb = self.backbone_rgb(rgb)
        feats_t = self.backbone_t(t)

        # Fuse at each scale
        if self.fusion_type == 'add':
            fused_feats = [fr + ft for fr, ft in zip(feats_rgb, feats_t)]
        elif self.fusion_type == 'avg':
            fused_feats = [0.5 * (fr + ft) for fr, ft in zip(feats_rgb, feats_t)]
        elif self.fusion_type == 'cat':
            fused_feats = [torch.cat([fr, ft], dim=1) for fr, ft in zip(feats_rgb, feats_t)]
        else:
            raise ValueError(f'Unknown fusion_type: {self.fusion_type}')

        if self.with_neck:
            fused_feats = self.neck(fused_feats)

        return fused_feats

    def encode_decode(self, inputs, batch_img_metas):
        fused_feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(
            fused_feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        fused_feats = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self.decode_head.loss(fused_feats, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(fused_feats, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:],
                     img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:],
                     padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples=None):
        fused_feats = self.extract_feat(inputs)
        return self.decode_head.forward(fused_feats)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2),
                                int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

    def _auxiliary_head_forward_train(self, inputs, data_samples):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def postprocess_result(self, seg_logits, data_samples):
        from mmseg.models.utils import resize
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            img_meta = data_samples[i].metainfo
            if 'img_padding_size' in img_meta:
                padding_size = img_meta['img_padding_size']
            else:
                padding_size = img_meta.get('padding_size', [0] * 4)
            padding_left, padding_right, padding_top, padding_bottom = padding_size
            i_seg_logits = seg_logits[i:i + 1, :,
                                      padding_top:H - padding_bottom,
                                      padding_left:W - padding_right]
            flip = img_meta.get('flip', None)
            if flip:
                flip_direction = img_meta.get('flip_direction', None)
                if flip_direction == 'horizontal':
                    i_seg_logits = i_seg_logits.flip(dims=(3,))
                else:
                    i_seg_logits = i_seg_logits.flip(dims=(2,))
            i_seg_logits = resize(
                i_seg_logits,
                size=img_meta['ori_shape'],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False).squeeze(0)

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_seg_logits),
                'pred_sem_seg': PixelData(data=i_seg_pred)
            })
        return data_samples


@MODELS.register_module()
class DINOv3AdapterM2FSingle(BaseSegmentor):
    """DINOv3 + ViT-Adapter + Mask2Former for single-modal segmentation.

    Used for RGB-only and T-only baselines.
    """

    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
        use_thermal: bool = False,
    ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.use_thermal = use_thermal

        if neck is not None:
            self.neck = MODELS.build(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: OptConfigType) -> None:
        if auxiliary_head is not None:
            self.auxiliary_head = MODELS.build(auxiliary_head)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    def extract_feat(self, inputs: torch.Tensor):
        # Select RGB or T from 6-channel input
        if self.use_thermal:
            x = inputs[:, 3:]
        else:
            x = inputs[:, :3]

        feats = self.backbone(x)

        if self.with_neck:
            feats = self.neck(feats)

        return feats

    def encode_decode(self, inputs, batch_img_metas):
        feats = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(feats, batch_img_metas, self.test_cfg)
        return seg_logits

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self.decode_head.loss(feats, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        if self.with_auxiliary_head:
            loss_aux = self.auxiliary_head.loss(feats, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [
                dict(ori_shape=inputs.shape[2:],
                     img_shape=inputs.shape[2:],
                     pad_shape=inputs.shape[2:],
                     padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self, inputs, data_samples=None):
        feats = self.extract_feat(inputs)
        return self.decode_head.forward(feats)

    def whole_inference(self, inputs, batch_img_metas):
        return self.encode_decode(inputs, batch_img_metas)

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2),
                                int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def inference(self, inputs, batch_img_metas):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            return self.slide_inference(inputs, batch_img_metas)
        return self.whole_inference(inputs, batch_img_metas)

    def postprocess_result(self, seg_logits, data_samples):
        from mmseg.models.utils import resize
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]

        for i in range(batch_size):
            img_meta = data_samples[i].metainfo
            padding_size = img_meta.get('img_padding_size', img_meta.get('padding_size', [0] * 4))
            padding_left, padding_right, padding_top, padding_bottom = padding_size
            i_seg_logits = seg_logits[i:i + 1, :,
                                      padding_top:H - padding_bottom,
                                      padding_left:W - padding_right]
            flip = img_meta.get('flip', None)
            if flip:
                flip_direction = img_meta.get('flip_direction', None)
                if flip_direction == 'horizontal':
                    i_seg_logits = i_seg_logits.flip(dims=(3,))
                else:
                    i_seg_logits = i_seg_logits.flip(dims=(2,))
            i_seg_logits = resize(
                i_seg_logits,
                size=img_meta['ori_shape'],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False).squeeze(0)

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits > 0.5).to(i_seg_logits)

            data_samples[i].set_data({
                'seg_logits': PixelData(data=i_seg_logits),
                'pred_sem_seg': PixelData(data=i_seg_pred)
            })
        return data_samples
