"""Lightweight training-time visualization for EoMTRGBTQuality (DINOv3 RGB-T).

Dumps, every `interval` epochs, a composite PNG per sample:
  row 0: degraded RGB | degraded T | GT seg | pred seg
  row k: per-SEGMENT quality heatmap (RGB) | quality (T) | merged-feature top3 | (blank)
Quality maps use the same red-blue convention as the project's other vis
(red = high quality, blue = low). Self-contained: reads model._all_quality and
model._last_merged_feat set during the forward pass.
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


def _seq_to_2d_feat(z, grid):
    """[B,N,C] token seq (maybe with prefix tokens) -> [B,C,gh,gw] for vis."""
    gh, gw = grid
    B, N, C = z.shape
    if N > gh * gw:               # drop leading prefix tokens (cls/register)
        z = z[:, N - gh * gw:, :]
    return z.transpose(1, 2).reshape(B, C, gh, gw)


def _quality_seq_to_2d(s, grid):
    """[B,N,1] or [B,N] quality -> [B,gh,gw] numpy (sample 0)."""
    gh, gw = grid
    if s.dim() == 3:
        s = s.squeeze(-1)
    N = s.shape[1]
    if N > gh * gw:
        s = s[:, N - gh * gw:]
    return s[0].reshape(gh, gw).detach().cpu().float().numpy()


@HOOKS.register_module()
class EoMTRGBTVisHook(Hook):
    """Per-segment quality + merged-feature visualization for EoMTRGBTQuality."""
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
            runner.logger.warning(f'EoMTRGBTVisHook error: {e}')
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

        was_training = model.training
        model.eval()
        try:
            # MFNet train aug yields per-sample sizes -> the data_preprocessor
            # rejects a mixed-size batch. Visualize the FIRST sample only, fed
            # as a length-1 batch so the size-consistency assert always holds.
            n = min(self.num_samples, len(raw_inputs))
            grid = model.network.encoder.backbone.patch_embed.grid_size

            grids = []
            with torch.no_grad():
                for bi in range(n):
                    proc = model.data_preprocessor(
                        {'inputs': [raw_inputs[bi]],
                         'data_samples': [data_samples[bi]] if data_samples else None},
                        False)
                    xi = proc['inputs']
                    rgb, t = model._split(xi)
                    # degrade exactly like training (single forward), then read
                    # the quality/merged-feature side-channels the model stores.
                    drgb, dir_, m_rgb, m_t = model.degrader(
                        rgb, t, epoch=epoch)
                    ml, cl = model._dual_stream_forward(drgb, dir_)
                    all_q = model._all_quality
                    merged = model._last_merged_feat
                    stage_feats = (
                        getattr(model, '_vis_rgb_feats', None),
                        getattr(model, '_vis_t_feats', None),
                        getattr(model, '_vis_fused_feats', None))
                    pred = self._predict(model, ml, cl, xi)
                    grids.append(self._compose(
                        drgb, dir_, all_q, merged, grid,
                        data_samples[bi] if data_samples else None,
                        pred, palette, stage_feats))
            self._save(grids, epoch)
            runner.logger.info(
                f'EoMTRGBTVisHook: epoch {epoch}, {n} samples -> {self._vis_dir}')
        finally:
            if was_training:
                model.train()

    def _predict(self, model, ml_layers, cl_layers, xi):
        from mmseg.models.segmentors.eomt_utils import (
            mask_class_to_seg_logits, resize_seg_logits)
        seg = mask_class_to_seg_logits(ml_layers[-1], cl_layers[-1])
        seg = resize_seg_logits(seg, xi.shape[2:])
        return seg.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def _input_to_img(self, x_norm):
        """Normalized [1,3,H,W] -> uint8 RGB HxWx3 (best-effort de-norm)."""
        img = x_norm[0].detach().cpu().float().numpy().transpose(1, 2, 0)
        return _to_uint8(img)

    def _cell(self, img, h, w):
        if img is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    def _seg_vis(self, seg, palette, h, w):
        if palette is not None:
            v = _apply_palette(seg, palette)
        else:
            v = cv2.cvtColor(cv2.applyColorMap(seg, cv2.COLORMAP_VIRIDIS),
                             cv2.COLOR_BGR2RGB)
        return self._cell(v, h, w)

    def _compose(self, drgb, dir_, all_q, merged, grid, sample, pred, palette,
                 stage_feats=None):
        s = self.short_side
        cell = lambda im: self._cell(im, s, s)  # noqa: E731

        # row 0: degraded inputs + GT + pred
        gt = None
        if sample is not None and hasattr(sample, 'gt_sem_seg') \
                and sample.gt_sem_seg is not None:
            gt = sample.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        row0 = np.concatenate([
            cell(self._input_to_img(drgb)),
            cell(self._input_to_img(dir_)),
            self._seg_vis(gt, palette, s, s) if gt is not None
            else np.zeros((s, s, 3), np.uint8),
            self._seg_vis(pred, palette, s, s)], axis=1)

        # per-segment quality rows: RGB-q | T-q | merged-feat (last row only) | blank
        rows = [row0]
        merged_2d = _seq_to_2d_feat(merged, grid) if merged is not None else None
        merged_rgb = _feat_top3_rgb(merged_2d) if merged_2d is not None else None
        for si, (s_rgb, s_t) in enumerate(all_q):
            q_rgb = _quality_seq_to_2d(s_rgb, grid)
            q_t = _quality_seq_to_2d(s_t, grid)
            mcell = cell(merged_rgb) if si == len(all_q) - 1 and merged_rgb is not None \
                else np.zeros((s, s, 3), np.uint8)
            rows.append(np.concatenate([
                cell(_quality_to_red_blue(q_rgb, s, s)),
                cell(_quality_to_red_blue(q_t, s, s)),
                mcell,
                np.zeros((s, s, 3), np.uint8)], axis=1))

        # Swin-style per-stage feature block: pre-merge RGB | pre-merge T |
        # fused. One row per fusion stage. 4th column blank to match width.
        rgb_f, t_f, fused_f = stage_feats if stage_feats else (None, None, None)
        if rgb_f:
            blank = np.zeros((s, s, 3), np.uint8)
            for si in range(len(rgb_f)):
                def _ft(seq):
                    if seq is None or si >= len(seq) or seq[si] is None:
                        return blank
                    return cell(_feat_top3_rgb(_seq_to_2d_feat(seq[si], grid)))
                rows.append(np.concatenate([
                    _ft(rgb_f), _ft(t_f), _ft(fused_f), blank], axis=1))
        return np.concatenate(rows, axis=0)

    def _save(self, grids, epoch):
        if not grids:
            return
        max_w = max(g.shape[1] for g in grids)
        padded = [np.pad(g, ((0, 0), (0, max_w - g.shape[1]), (0, 0)))
                  for g in grids]
        out = np.concatenate(padded, axis=0)
        path = osp.join(self._vis_dir, f'eomt_rgbt_quality_epoch{epoch}.png')
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))



