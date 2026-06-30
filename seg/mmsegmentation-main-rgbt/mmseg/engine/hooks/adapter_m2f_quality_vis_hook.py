"""Visualization hook for DINOv3AdapterM2FQuality (Adapter+M2F architecture).

Every `interval` epochs, dumps PNGs:
  - epoch{N}_feature.png : input | RGB feat | T feat | fused feat | pred | GT
  - epoch{N}_quality.png : RGB quality heatmap | T quality heatmap

Quality maps use the red-blue convention (red = high quality, blue = low).
"""

import os
import os.path as osp

import cv2
import numpy as np
import torch

from mmengine.hooks import Hook
from mmseg.registry import HOOKS
from mmseg.utils.vis_utils import (
    _to_uint8, _feat_top3_rgb, _quality_to_red_blue, _apply_palette)


@HOOKS.register_module()
class AdapterM2FQualityVisHook(Hook):
    """Feature + quality visualization for DINOv3AdapterM2FQuality."""
    priority = 'LOW'

    def __init__(self, interval=10, num_samples=1, short_side=240):
        super().__init__()
        self.interval = interval
        self.num_samples = num_samples
        self.short_side = short_side
        self._vis_dir = None
        self._cached_batch = None

    def _get_model(self, runner):
        model = runner.model
        while hasattr(model, 'module'):
            model = model.module
        return model

    def before_train_epoch(self, runner):
        model = self._get_model(runner)
        if hasattr(model, 'current_epoch'):
            model.current_epoch = runner.epoch

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if batch_idx == 0 and runner.epoch % self.interval == 0:
            self._cached_batch = data_batch

    def after_train_epoch(self, runner, metrics=None):
        if runner.epoch % self.interval != 0 or self._cached_batch is None:
            return
        if self._vis_dir is None:
            self._vis_dir = osp.join(runner.work_dir, 'vis_data')
            os.makedirs(self._vis_dir, exist_ok=True)
        model = self._get_model(runner)
        try:
            self._visualize(runner, model, runner.epoch)
        except Exception as e:
            import traceback
            runner.logger.warning(f'AdapterM2FQualityVisHook error: {e}')
            traceback.print_exc()
        finally:
            self._cached_batch = None

    def _get_palette(self, runner):
        try:
            mi = runner.train_dataloader.dataset.metainfo
            if isinstance(mi, dict):
                return mi.get('palette')
        except Exception:
            pass
        return None

    def _visualize(self, runner, model, epoch):
        batch = self._cached_batch
        raw_inputs = batch.get('inputs')
        data_samples = batch.get('data_samples')
        if raw_inputs is None:
            return
        palette = self._get_palette(runner)
        use_quality = getattr(model, 'use_quality', False)

        was_training = model.training
        model.eval()
        try:
            n = min(self.num_samples, len(raw_inputs))
            feat_rows = []
            qual_rows = []

            with torch.no_grad():
                for bi in range(n):
                    proc = model.data_preprocessor(
                        {'inputs': [raw_inputs[bi]],
                         'data_samples': [data_samples[bi]] if data_samples else None},
                        True)
                    xi = proc['inputs']  # [1, 6, H, W]

                    # Forward
                    seg_logits = model.whole_inference(
                        xi, [{'ori_shape': raw_inputs[bi].shape[1:],
                              'img_shape': xi.shape[2:],
                              'pad_shape': xi.shape[2:]}])
                    pred = seg_logits.argmax(dim=1)[0].cpu().numpy()

                    # Get visualization features stored during extract_feat
                    vis_rgb = getattr(model, '_vis_rgb_feats', None)
                    vis_thr = getattr(model, '_vis_thr_feats', None)
                    vis_fused = getattr(model, '_vis_fused_feats', None)
                    vis_q_rgb = getattr(model, '_vis_quality_rgb', None)
                    vis_q_thr = getattr(model, '_vis_quality_thr', None)

                    # Input image
                    inp = raw_inputs[bi].cpu().float().numpy()  # [6, H, W]
                    rgb_img = _to_uint8(inp[:3].transpose(1, 2, 0))
                    thr_img = _to_uint8(inp[3:].transpose(1, 2, 0))

                    # GT
                    gt = None
                    if data_samples is not None and bi < len(data_samples):
                        gt_seg = data_samples[bi].gt_sem_seg.data.cpu().numpy()
                        gt = gt_seg.squeeze()

                    # Feature maps (use deepest scale = index 3)
                    rgb_feat_vis = _feat_top3_rgb(vis_rgb[3]) if vis_rgb else np.zeros_like(rgb_img)
                    thr_feat_vis = _feat_top3_rgb(vis_thr[3]) if vis_thr else np.zeros_like(thr_img)
                    fused_feat_vis = _feat_top3_rgb(vis_fused[3]) if vis_fused else np.zeros_like(rgb_img)

                    # Pred + GT with palette
                    pred_vis = _apply_palette(pred, palette)
                    gt_vis = _apply_palette(gt, palette) if gt is not None else np.zeros_like(pred_vis)

                    # Resize feature maps to input size
                    h, w = rgb_img.shape[:2]
                    rgb_feat_vis = cv2.resize(rgb_feat_vis, (w, h))
                    thr_feat_vis = cv2.resize(thr_feat_vis, (w, h))
                    fused_feat_vis = cv2.resize(fused_feat_vis, (w, h))

                    # Compose feature row
                    row = np.concatenate([
                        rgb_img, thr_img, rgb_feat_vis, thr_feat_vis,
                        fused_feat_vis, pred_vis, gt_vis], axis=1)
                    feat_rows.append(row)

                    # Quality maps
                    if use_quality and vis_q_rgb is not None:
                        s_rgb = vis_q_rgb[0].squeeze(-1)  # [N]
                        s_thr = vis_q_thr[0].squeeze(-1)  # [N]
                        N = s_rgb.shape[0]
                        side = int(np.sqrt(N))
                        if side * side == N:
                            q_rgb_map = s_rgb.reshape(side, side).cpu().numpy()
                            q_thr_map = s_thr.reshape(side, side).cpu().numpy()
                            q_rgb_vis = _quality_to_red_blue(q_rgb_map)
                            q_thr_vis = _quality_to_red_blue(q_thr_map)
                            q_rgb_vis = cv2.resize(q_rgb_vis, (w, h))
                            q_thr_vis = cv2.resize(q_thr_vis, (w, h))
                            qual_row = np.concatenate([
                                rgb_img, q_rgb_vis, thr_img, q_thr_vis], axis=1)
                            qual_rows.append(qual_row)

            # Save
            if feat_rows:
                canvas = np.concatenate(feat_rows, axis=0)
                path = osp.join(self._vis_dir, f'epoch{epoch}_feature.png')
                cv2.imwrite(path, canvas)
                runner.logger.info(f'Feature vis saved: {path}')
            if qual_rows:
                canvas = np.concatenate(qual_rows, axis=0)
                path = osp.join(self._vis_dir, f'epoch{epoch}_quality.png')
                cv2.imwrite(path, canvas)
                runner.logger.info(f'Quality vis saved: {path}')

        finally:
            model.train(was_training)
