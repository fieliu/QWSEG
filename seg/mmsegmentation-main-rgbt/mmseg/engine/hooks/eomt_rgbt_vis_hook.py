"""Training-time visualization for EoMTRGBTQuality (DINOv3 RGB-T).

Every `interval` epochs, dumps PNGs per visualized sample, on the SAME
degraded inputs the model is actually trained on:
  - epoch{N}_feature.png : input | RGB feat | T feat | fused feat | pred | GT
  - epoch{N}_quality.png : per-segment RGB-quality | T-quality heatmaps
  - epoch{N}_level.png   : input | level mask (L1-L5) for rgb/t
  - epoch{N}_rank.png    : per-segment s_light vs s_heavy scatter+hist

All PNGs share the SAME row layout: one row per (sample, version), so
different samples / light-heavy / stages can be compared in a single image.

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
        """Return a list of (tag, rgb, t, lvl_rgb, lvl_t) versions matching
        training. lvl_* are [B,1,H,W] long level masks (L1-L5).

        E2 (use_quality + paired make_paired): light & heavy versions with
        level masks for supervision visualization.
        E0b/E1 (single-level degrader): one degraded version with binary mask
        converted to level mask (0->L1 clean, 1->L5 degraded).
        """
        use_quality = getattr(model, 'use_quality', False)
        has_paired = hasattr(model.degrader, 'make_paired')
        if use_quality and has_paired:
            mean = model.data_preprocessor.mean.flatten()
            std = model.data_preprocessor.std.flatten()
            (l_rgb, l_ir, h_rgb, h_ir,
             llr, lli, hlr, hli, _rm, _lg) = \
                model.degrader.make_paired(
                    rgb, t, mean, std, epoch=epoch)
            return [
                ('light', l_rgb, l_ir, llr, lli),
                ('heavy', h_rgb, h_ir, hlr, hli),
            ]
        drgb, dir_, mr, mt = model.degrader(rgb, t, epoch=epoch)
        # Convert binary mask (0/1) to level mask (L1/L5) for uniform vis
        llr = torch.ones_like(mr).long()                      # L1 clean
        llr[mr > 0.5] = 5                                      # L5 degraded
        llt = torch.ones_like(mt).long()
        llt[mt > 0.5] = 5
        return [('deg', drgb, dir_, llr, llt)]

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
            feat_grids, qual_grids, level_grids = [], [], []
            # rank data: list of (sample_idx, tag, all_q, lvl_tok)
            # collected per sample-version, paired light/heavy for rank PNG
            rank_rows = []  # each entry = one row of the rank PNG
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
                    # collect per-sample light/heavy quality for rank PNG
                    sample_q = {}  # tag -> all_q
                    sample_lvl = {}  # tag -> (lvl_rgb_tok, lvl_t_tok)
                    # reproduce the training-time degraded version(s)
                    for tag, drgb, dir_, lvl_rgb, lvl_t in \
                            self._make_versions(model, rgb, t, epoch):
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
                                self._compose_quality(
                                    tag, drgb, dir_, all_q, grid))
                            level_grids.append(self._compose_level(
                                tag, drgb, dir_, lvl_rgb, lvl_t))
                            sample_q[tag] = all_q
                            # downsample level mask to token grid for rank
                            lr_tok = self._mask_to_token(lvl_rgb, grid)
                            lt_tok = self._mask_to_token(lvl_t, grid)
                            sample_lvl[tag] = (lr_tok, lt_tok)
                    # build rank row if both light & heavy available
                    if use_quality and 'light' in sample_q and 'heavy' in sample_q:
                        rank_rows.append(self._compose_rank(
                            bi, sample_q['light'], sample_q['heavy'],
                            sample_lvl['light'], sample_lvl['heavy'], grid))
            self._save(feat_grids, epoch, 'feature')
            if use_quality:
                self._save(qual_grids, epoch, 'quality')
                self._save(level_grids, epoch, 'level')
                if rank_rows:
                    self._save(rank_rows, epoch, 'rank')
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

    # ---- QUALITY png: input | per-segment RGB-quality | T-quality heatmaps ----
    def _compose_quality(self, tag, drgb, dir_, all_q, grid):
        """Quality heatmaps with the INPUT images alongside for direct visual
        alignment. Layout per segment row:
            [deg RGB input | deg T input | RGB quality (red=high) | T quality]
        The input images are repeated in every row so each quality map can be
        compared to the actual degraded input without eye-tracking across rows.
        """
        s = self.short_side
        cell = lambda im: self._cell(im, s, s)  # noqa: E731
        rgb_img = self._input_to_img(drgb)
        t_img = self._input_to_img(dir_)
        rows = []
        for (s_rgb, s_t) in all_q:
            q_rgb = _quality_seq_to_2d(s_rgb, grid)
            q_t = _quality_seq_to_2d(s_t, grid)
            rows.append(np.concatenate([
                cell(rgb_img),
                cell(t_img),
                cell(_quality_to_red_blue(q_rgb, s, s)),
                cell(_quality_to_red_blue(q_t, s, s))], axis=1))
        if not rows:
            return np.zeros((s, s * 4, 3), np.uint8)
        return np.concatenate(rows, axis=0)

    # ---- LEVEL png: input | level mask (L1-L5) for rgb/t ----
    def _compose_level(self, tag, drgb, dir_, lvl_rgb, lvl_t):
        """Level mask visualization. Layout:
            [deg RGB | deg T | RGB level mask | T level mask]
        Level mask color coding (L1-L5):
            L1=white (clean), L2=light red, L3=red, L4=dark red, L5=black
        This shows EXACTLY which pixels the supervisor marks as degraded and
        at what level, so one can verify the quality heatmaps align with the
        actual degradation region and severity.
        """
        s = self.short_side
        cell = lambda im: self._cell(im, s, s)  # noqa: E731
        rgb_img = self._input_to_img(drgb)
        t_img = self._input_to_img(dir_)
        lvl_rgb_vis = self._level_to_color(lvl_rgb, s, s)
        lvl_t_vis = self._level_to_color(lvl_t, s, s)
        return np.concatenate([
            cell(rgb_img), cell(t_img), cell(lvl_rgb_vis), cell(lvl_t_vis)],
            axis=1)

    def _level_to_color(self, lvl_mask, h, w):
        """[B,1,H,W] long (L1-L5) -> [h,w,3] uint8 color map.
        L1=white(255), L2=200, L3=140, L4=80, L5=0 (black). Single-channel
        gradient so the severity is immediately readable."""
        m = lvl_mask[0, 0].detach().cpu().numpy().astype(np.float32)
        # L1->255 (white), L5->0 (black), linear in between
        val = np.clip((6.0 - m) / 5.0 * 255.0, 0, 255).astype(np.uint8)
        val = cv2.resize(val, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.stack([val, val, val], axis=-1)

    def _mask_to_token(self, mask, grid):
        """Downsample [B,1,H,W] level mask to token grid [B,N] (max-pool,
        conservative: take highest level in the patch). Mirrors the model's
        _mask_to_token so the rank PNG uses the SAME token labels as loss."""
        gh, gw = grid
        m = torch.nn.functional.adaptive_max_pool2d(
            mask.float(), (gh, gw))            # [B,1,gh,gw]
        return m.flatten(2).squeeze(1).long()  # [B, N]

    # ---- RANK png: per-segment s_light vs s_heavy scatter + gap histogram ----
    def _compose_rank(self, sample_idx, all_q_light, all_q_heavy,
                      lvl_light, lvl_heavy, grid):
        """One row per sample. Columns = segments (RGB quality + T quality).
        Each cell is a scatter plot: x=s_light, y=s_heavy.
            red dot  : degraded-region token (lvl_heavy > 1)
            gray dot : clean token (lvl == 1)
            diagonal : y=x (ideal: red dots BELOW diagonal => s_light > s_heavy)
        Also draws a small histogram of (s_light - s_heavy) in the last column.
        This directly answers: "does the rank signal work at every stage?"
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        s = self.short_side
        n_seg = len(all_q_light)
        # one figure per modality per segment, plus a histogram column
        n_cols = n_seg + 1
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        lvl_h_rgb, lvl_h_t = lvl_heavy     # [N] each (token-level)
        deg_rgb = (lvl_h_rgb[0] > 1).cpu().numpy()    # bool [N]
        deg_t = (lvl_h_t[0] > 1).cpu().numpy()

        for seg_i in range(n_seg):
            for row, (q_l, q_h, deg, modality) in enumerate([
                    (all_q_light[seg_i][0], all_q_heavy[seg_i][0], deg_rgb, 'RGB'),
                    (all_q_light[seg_i][1], all_q_heavy[seg_i][1], deg_t, 'T')]):
                ax = axes[row, seg_i]
                sl = q_l.squeeze(-1).squeeze(0)   # [N]
                sh = q_h.squeeze(-1).squeeze(0)
                if sl.shape[0] > deg.shape[0]:
                    sl = sl[sl.shape[0] - deg.shape[0]:]
                    sh = sh[sh.shape[0] - deg.shape[0]:]
                sl = sl.cpu().numpy()
                sh = sh.cpu().numpy()
                clean = ~deg
                ax.scatter(sl[clean], sh[clean], c='gray', s=3, alpha=0.3,
                           label='clean')
                ax.scatter(sl[deg], sh[deg], c='red', s=5, alpha=0.6,
                           label='degraded')
                ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.set_title(f's{seg_i} {modality}', fontsize=8)
                if seg_i == 0:
                    ax.set_ylabel(f'sample{sample_idx}\n{modality}', fontsize=7)
                ax.tick_params(labelsize=6)

        # last column: gap histogram (s_light - s_heavy) for degraded tokens
        for row, (seg_q_l, seg_q_h, deg, modality) in enumerate([
                (all_q_light[-1][0], all_q_heavy[-1][0], deg_rgb, 'RGB'),
                (all_q_light[-1][1], all_q_heavy[-1][1], deg_t, 'T')]):
            ax = axes[row, -1]
            sl = seg_q_l.squeeze(-1).squeeze(0)
            sh = seg_q_h.squeeze(-1).squeeze(0)
            if sl.shape[0] > deg.shape[0]:
                sl = sl[sl.shape[0] - deg.shape[0]:]
                sh = sh[sh.shape[0] - deg.shape[0]:]
            gap = (sl - sh).cpu().numpy()
            gap_deg = gap[deg]
            gap_clean = gap[~deg]
            bins = np.linspace(-1, 1, 40)
            ax.hist(gap_clean, bins=bins, color='gray', alpha=0.4,
                    label='clean')
            ax.hist(gap_deg, bins=bins, color='red', alpha=0.6,
                    label='degraded')
            ax.axvline(0.20, color='k', ls='--', lw=0.5, label='margin')
            ax.set_title(f'gap (s_l - s_h)\n{modality} last seg', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)

        fig.tight_layout()
        # render to numpy
        fig.canvas.draw()
        # matplotlib >=3.8 移除了 tostring_rgb(), 用 buffer_rgba() 替代
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(
                fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return buf

    def _save(self, grids, epoch, kind):
        if not grids:
            return
        max_w = max(g.shape[1] for g in grids)
        padded = [np.pad(g, ((0, 0), (0, max_w - g.shape[1]), (0, 0)))
                  for g in grids]
        out = np.concatenate(padded, axis=0)
        path = osp.join(self._vis_dir, f'epoch{epoch}_{kind}.png')
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))



