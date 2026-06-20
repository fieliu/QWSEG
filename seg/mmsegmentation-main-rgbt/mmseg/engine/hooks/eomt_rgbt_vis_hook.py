"""Training-time visualization for EoMTRGBTQuality (DINOv3 RGB-T).

Every `interval` epochs, dumps TWO separate PNGs per visualized sample, on the
SAME degraded inputs the model is actually trained on:
  - epoch{N}_feature.png : input | RGB feat | T feat | fused feat | pred | GT
  - epoch{N}_quality.png : per-segment RGB-quality | T-quality heatmaps
The quality PNG is only written when the model's quality mechanism is on
(use_quality=True); E0b/E1 (mechanism off) get only the feature PNG.

For E2 (paired multi-level degradation) the hook reproduces the training-time
`make_paired` and visualizes BOTH the light and heavy versions that the model
actually sees, so the picture matches what is fed to the network.
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


def _seq_to_2d_feat(z, grid):
    """[B,N,C] token seq (maybe with prefix tokens) -> [B,C,gh,gw] for vis."""
    gh, gw = grid
    B, N, C = z.shape
    if N > gh * gw:               # drop leading prefix tokens (cls/register)
        z = z[:, N - gh * gw:, :]
    return z.transpose(1, 2).reshape(B, C, gh, gw)


def _quality_seq_to_2d(s, grid):
    """[B,N,1] or [B,N] quality -> [gh,gw] numpy (sample 0)."""
    gh, gw = grid
    if s.dim() == 3:
        s = s.squeeze(-1)
    N = s.shape[1]
    if N > gh * gw:
        s = s[:, N - gh * gw:]
    return s[0].reshape(gh, gw).detach().cpu().float().numpy()


@HOOKS.register_module()
class EoMTRGBTVisHook(Hook):
    """Feature + quality visualization for EoMTRGBTQuality, on real inputs."""
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

    # ---- build the actual degraded inputs the model is trained on ----
    def _make_versions(self, model, rgb, t, epoch):
        """Return a list of (tag, rgb, t) versions matching training.

        E2 (use_quality + paired make_paired): light & heavy versions.
        E0b/E1 (single-level degrader): one degraded version.
        """
        use_quality = getattr(model, 'use_quality', False)
        has_paired = hasattr(model.degrader, 'make_paired')
        if use_quality and has_paired:
            mean = model.data_preprocessor.mean.flatten()
            std = model.data_preprocessor.std.flatten()
            (l_rgb, l_ir, h_rgb, h_ir, _rr, _ri) = model.degrader.make_paired(
                rgb, t, mean, std, epoch=epoch)
            return [('light', l_rgb, l_ir), ('heavy', h_rgb, h_ir)]
        drgb, dir_, _mr, _mt = model.degrader(rgb, t, epoch=epoch)
        return [('deg', drgb, dir_)]

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
            grid = model.network.encoder.backbone.patch_embed.grid_size
            feat_grids, qual_grids = [], []
            with torch.no_grad():
                for bi in range(n):
                    # training=True so the preprocessor pads/crops to crop_size
                    # (e.g. 480x640) -> token count matches the backbone's FIXED
                    # patch_embed.grid_size. With training=False it keeps the
                    # raw augmented size, the token count != grid_size, and
                    # EoMT._predict's reshape to grid_size crashes.
                    proc = model.data_preprocessor(
                        {'inputs': [raw_inputs[bi]],
                         'data_samples': [data_samples[bi]] if data_samples else None},
                        True)
                    xi = proc['inputs']
                    rgb, t = model._split(xi)
                    sample = data_samples[bi] if data_samples else None
                    # reproduce the training-time degraded version(s)
                    for tag, drgb, dir_ in self._make_versions(model, rgb, t, epoch):
                        ml, cl = model._dual_stream_forward(drgb, dir_)
                        all_q = model._all_quality
                        merged = model._last_merged_feat
                        stage_feats = (
                            getattr(model, '_vis_rgb_feats', None),
                            getattr(model, '_vis_t_feats', None),
                            getattr(model, '_vis_fused_feats', None))
                        pred = self._predict(model, ml, cl, xi)
                        feat_grids.append(self._compose_feature(
                            tag, drgb, dir_, merged, grid, sample, pred,
                            palette, stage_feats))
                        if use_quality:
                            qual_grids.append(
                                self._compose_quality(tag, all_q, grid))
            self._save(feat_grids, epoch, 'feature')
            if use_quality:
                self._save(qual_grids, epoch, 'quality')
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

    # ---- FEATURE png: input | RGB feat | T feat | fused feat | pred | GT ----
    def _compose_feature(self, tag, drgb, dir_, merged, grid, sample, pred,
                         palette, stage_feats=None):
        s = self.short_side
        cell = lambda im: self._cell(im, s, s)  # noqa: E731

        gt = None
        if sample is not None and hasattr(sample, 'gt_sem_seg') \
                and sample.gt_sem_seg is not None:
            gt = sample.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

        # row 0: degraded RGB | degraded T | pred | GT
        row0 = np.concatenate([
            cell(self._input_to_img(drgb)),
            cell(self._input_to_img(dir_)),
            self._seg_vis(pred, palette, s, s),
            self._seg_vis(gt, palette, s, s) if gt is not None
            else np.zeros((s, s, 3), np.uint8)], axis=1)
        rows = [row0]

        # per-stage feature rows: pre-merge RGB | pre-merge T | fused | merged
        rgb_f, t_f, fused_f = stage_feats if stage_feats else (None, None, None)
        merged_2d = _seq_to_2d_feat(merged, grid) if merged is not None else None
        merged_rgb = _feat_top3_rgb(merged_2d) if merged_2d is not None else None
        if rgb_f:
            blank = np.zeros((s, s, 3), np.uint8)
            for si in range(len(rgb_f)):
                def _ft(seq):
                    if seq is None or si >= len(seq) or seq[si] is None:
                        return blank
                    return cell(_feat_top3_rgb(_seq_to_2d_feat(seq[si], grid)))
                mcell = cell(merged_rgb) if (si == len(rgb_f) - 1
                                             and merged_rgb is not None) else blank
                rows.append(np.concatenate([
                    _ft(rgb_f), _ft(t_f), _ft(fused_f), mcell], axis=1))
        return np.concatenate(rows, axis=0)

    # ---- QUALITY png: per-segment RGB-quality | T-quality heatmaps ----
    def _compose_quality(self, tag, all_q, grid):
        s = self.short_side
        cell = lambda im: self._cell(im, s, s)  # noqa: E731
        rows = []
        for (s_rgb, s_t) in all_q:
            q_rgb = _quality_seq_to_2d(s_rgb, grid)
            q_t = _quality_seq_to_2d(s_t, grid)
            rows.append(np.concatenate([
                cell(_quality_to_red_blue(q_rgb, s, s)),
                cell(_quality_to_red_blue(q_t, s, s))], axis=1))
        if not rows:
            return np.zeros((s, s * 2, 3), np.uint8)
        return np.concatenate(rows, axis=0)

    def _save(self, grids, epoch, kind):
        if not grids:
            return
        max_w = max(g.shape[1] for g in grids)
        padded = [np.pad(g, ((0, 0), (0, max_w - g.shape[1]), (0, 0)))
                  for g in grids]
        out = np.concatenate(padded, axis=0)
        path = osp.join(self._vis_dir, f'epoch{epoch}_{kind}.png')
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))



