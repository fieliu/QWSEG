import os
import os.path as osp
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

from mmseg.registry import HOOKS
from mmseg.utils.vis_utils import (
    _apply_palette,
    _build_row,
    _feat_top3_rgb,
    _make_cell,
    _quality_to_red_blue,
    _threshold_to_bw,
    _to_uint8,
    detect_model_type,
)


def _quality_to_heatmap(q_scores, h, w):
    B, N = q_scores.shape
    H = W = int(N ** 0.5)
    q_map = q_scores.view(B, 1, H, W)
    q_map = torch.nn.functional.interpolate(
        q_map, size=(h, w), mode='bilinear', align_corners=False)
    return q_map.squeeze(1).cpu().numpy()


def _add_text_overlay(img, text, font_scale=0.6, thickness=2):
    result = img.copy()
    h, w = result.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.rectangle(result, (x - 4, y - th - 4), (x + tw + 4, y + 4),
                  (0, 0, 0), -1)
    cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 255), thickness)
    return result


def _quality_to_red_blue(q_np, target_h, target_w):
    if q_np.ndim == 1:
        H = W = int(q_np.shape[0] ** 0.5)
        q_np = q_np.reshape(H, W)
    q_np = np.clip(q_np.astype(np.float32), 0.0, 1.0)
    rgb = np.zeros((q_np.shape[0], q_np.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = (q_np * 255).astype(np.uint8)
    rgb[:, :, 2] = ((1 - q_np) * 255).astype(np.uint8)
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _threshold_to_bw(mask, target_h, target_w):
    mask_f = mask.astype(np.float32)
    if mask_f.ndim == 1:
        H = W = int(mask_f.shape[0] ** 0.5)
        mask_f = mask_f.reshape(H, W)
    mask_resized = cv2.resize(
        mask_f, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    val = (mask_resized * 255).astype(np.uint8)
    vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    vis[:, :, 0] = val
    vis[:, :, 1] = val
    vis[:, :, 2] = val
    return vis


def _add_missing_overlay(img, alpha=0.5):
    result = img.copy()
    h, w = result.shape[:2]
    overlay = np.zeros_like(result)
    overlay[:, :] = [0, 0, 200]
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    text = 'MISSING'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return result


def _quality_to_rgb_heatmap(q_scores, target_h, target_w, vmin=0.0, vmax=1.0,
                            cmap='rdBu_r'):
    if q_scores.ndim == 1 or (q_scores.ndim == 2 and q_scores.shape[0] == 1):
        if q_scores.ndim == 2:
            q_scores = q_scores.squeeze(0)
        N = q_scores.shape[0]
        H = W = int(N ** 0.5)
        q_2d = q_scores.reshape(H, W)
    else:
        q_2d = q_scores
    q_np = q_2d.astype(np.float32)
    q_np = np.clip(q_np, vmin, vmax)
    q_norm = (q_np - vmin) / (vmax - vmin + 1e-8)
    if cmap == 'rdBu_r':
        try:
            import matplotlib.pyplot as plt
            cmap_obj = plt.get_cmap('RdBu_r')
            rgba = cmap_obj(q_norm)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            q_uint8 = (q_norm * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(q_uint8, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        q_uint8 = (q_norm * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(q_uint8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _feat_top3_rgb_with_mask(feature_map, mask=None, sample_idx=0):
    result = _feat_top3_rgb(feature_map, sample_idx)
    if mask is None:
        return result
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    if mask_np.ndim == 1:
        H_m = W_m = int(mask_np.shape[0] ** 0.5)
        mask_2d = mask_np.reshape(H_m, W_m)
    else:
        mask_2d = mask_np
    mask_2d = cv2.resize(
        mask_2d.astype(np.float32), (result.shape[1], result.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    result[mask_2d < 0.5] = 0
    return result


def _build_feat_rows(feat_sources, sample_idx=0, masks=None):
    num_stages = len(feat_sources[0])
    rows = []
    for stage in range(num_stages):
        cells = []
        for col_idx, feat_list in enumerate(feat_sources):
            stage_feat = feat_list[stage]
            if stage_feat is None:
                cells.append(None)
            elif masks is not None and col_idx < len(masks) and masks[col_idx] is not None:
                cells.append(_feat_top3_rgb_with_mask(
                    stage_feat, masks[col_idx], sample_idx))
            else:
                cells.append(_feat_top3_rgb(stage_feat, sample_idx))
        rows.append(cells)
    return rows


def _add_col_headers(image, headers, cell_w, short_side,
                     font_scale=0.45, thickness=1):
    (tw, th), _ = cv2.getTextSize('Ay', cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, thickness)
    pad = 4
    bar_h = th + 2 * pad
    bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
    for i, h in enumerate(headers):
        x = i * cell_w + (cell_w - len(h) * tw // 2) // 2
        cv2.putText(bar, h, (max(pad, x), th + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (200, 200, 200), thickness)
    return np.concatenate([bar, image], axis=0)


def _add_row_label(image, label, font_scale=0.45, thickness=1):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, thickness)
    pad = 4
    bar_h = th + 2 * pad
    bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (180, 220, 180), thickness)
    return np.concatenate([bar, image], axis=0)


def _compute_cell_size(short_side, aspect_ratio=1.0):
    cell_h = short_side
    cell_w = max(short_side, int(short_side * aspect_ratio + 0.5))
    return cell_h, cell_w


def _compose_vis(rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                 short_side=250, title=None,
                 deg_rgb_vis=None, deg_t_vis=None,
                 deg_pred_vis=None,
                 col_headers=None, row_labels=None,
                 decoder_preds=None, aspect_ratio=1.0,
                 deg_decoder_preds=None):
    num_cols = max(2, len(feat_rows[0]) if feat_rows else 2)
    cell_h, cell_w = _compute_cell_size(short_side, aspect_ratio)

    rows = [_build_row([rgb_vis, t_vis],
                       cell_h, cell_w, short_side, num_cols)]
    if deg_rgb_vis is not None and deg_t_vis is not None:
        deg_cells = [deg_rgb_vis, deg_t_vis]
        if deg_pred_vis is not None:
            deg_cells.append(label_vis)
            deg_cells.append(deg_pred_vis)
        rows.append(_build_row(deg_cells, cell_h, cell_w,
                               short_side, num_cols))
    for idx, stage_cells in enumerate(feat_rows):
        row_img = _build_row(stage_cells, cell_h, cell_w, short_side, num_cols)
        if row_labels is not None and idx < len(row_labels):
            row_img = _add_row_label(row_img, row_labels[idx])
        rows.append(row_img)
    if decoder_preds is not None and len(decoder_preds) > 0:
        clean_label_row = [None] * num_cols
        clean_label_row[0] = _add_text_overlay(
            label_vis.copy(), 'GT', font_scale=0.5, thickness=1)
        rows.append(_build_row(clean_label_row, cell_h, cell_w,
                               short_side, num_cols))
        clean_pred_cells = [None] * num_cols
        for col_idx, pred_img in decoder_preds.items():
            if col_idx < num_cols:
                clean_pred_cells[col_idx] = pred_img
        clean_pred_cells[0] = _add_text_overlay(
            clean_pred_cells[0] if clean_pred_cells[0] is not None
            else np.zeros_like(label_vis),
            'Clean Seg', font_scale=0.4, thickness=1)
        rows.append(_build_row(clean_pred_cells, cell_h, cell_w,
                               short_side, num_cols))
    else:
        rows.append(_build_row([label_vis, pred_vis],
                               cell_h, cell_w, short_side, num_cols))
    if deg_decoder_preds is not None and len(deg_decoder_preds) > 0:
        deg_pred_cells = [None] * num_cols
        for col_idx, pred_img in deg_decoder_preds.items():
            if col_idx < num_cols:
                deg_pred_cells[col_idx] = pred_img
        deg_pred_cells[0] = _add_text_overlay(
            deg_pred_cells[0] if deg_pred_cells[0] is not None
            else np.zeros_like(label_vis),
            'Deg Seg', font_scale=0.4, thickness=1)
        rows.append(_build_row(deg_pred_cells, cell_h, cell_w,
                               short_side, num_cols))
    composite = np.concatenate(rows, axis=0)

    if col_headers is not None:
        composite = _add_col_headers(composite, col_headers, cell_w, short_side)
    if title is not None:
        composite = _add_title_bar(composite, title)
    return composite


def _add_title_bar(image, text, font_scale=0.6, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 6
    bar = np.zeros((th + 2 * pad, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)
    return np.concatenate([bar, image], axis=0)


def _compose_ablation_vis(rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                          short_side=250, title=None,
                          col_headers=None, row_labels=None,
                          decoder_preds=None, aspect_ratio=1.0,
                          deg_rgb_vis=None, deg_t_vis=None,
                          deg_feat_rows=None,
                          deg_decoder_preds=None,
                          deg_pred_vis=None):
    num_cols = max(2, len(feat_rows[0]) if feat_rows else 2)
    cell_h, cell_w = _compute_cell_size(short_side, aspect_ratio)
    empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    rows = []

    input_cells = [rgb_vis, t_vis] + [empty.copy() for _ in range(num_cols - 2)]
    rows.append(_build_row(input_cells, cell_h, cell_w, short_side, num_cols))

    for idx, stage_cells in enumerate(feat_rows):
        row_img = _build_row(stage_cells, cell_h, cell_w, short_side, num_cols)
        if row_labels is not None and idx < len(row_labels):
            row_img = _add_row_label(row_img, row_labels[idx])
        rows.append(row_img)

    if deg_rgb_vis is not None and deg_t_vis is not None and deg_feat_rows is not None:
        deg_input_cells = [deg_rgb_vis, deg_t_vis] + [empty.copy() for _ in range(num_cols - 2)]
        rows.append(_build_row(deg_input_cells, cell_h, cell_w, short_side, num_cols))

        for idx, stage_cells in enumerate(deg_feat_rows):
            row_img = _build_row(stage_cells, cell_h, cell_w, short_side, num_cols)
            if row_labels is not None:
                deg_idx = idx + len(feat_rows)
                if deg_idx < len(row_labels):
                    row_img = _add_row_label(row_img, row_labels[deg_idx])
            rows.append(row_img)

    seg_row = [None] * num_cols
    seg_row[0] = _add_text_overlay(label_vis.copy(), 'GT', font_scale=0.5, thickness=1)
    if decoder_preds is not None:
        for col_idx, pred_img in decoder_preds.items():
            if col_idx < num_cols:
                seg_row[col_idx] = pred_img
    if seg_row[1] is None and pred_vis is not None:
        seg_row[1] = pred_vis
    rows.append(_build_row(seg_row, cell_h, cell_w, short_side, num_cols))

    if deg_decoder_preds is not None and len(deg_decoder_preds) > 0:
        deg_seg_row = [None] * num_cols
        for col_idx, pred_img in deg_decoder_preds.items():
            if col_idx < num_cols:
                deg_seg_row[col_idx] = pred_img
        if deg_pred_vis is not None and deg_seg_row[1] is None:
            deg_seg_row[1] = deg_pred_vis
        deg_seg_row[0] = _add_text_overlay(
            deg_seg_row[0] if deg_seg_row[0] is not None else empty.copy(),
            'Deg', font_scale=0.5, thickness=1)
        rows.append(_build_row(deg_seg_row, cell_h, cell_w, short_side, num_cols))

    composite = np.concatenate(rows, axis=0)

    if col_headers is not None:
        composite = _add_col_headers(composite, col_headers, cell_w, short_side)
    if title is not None:
        composite = _add_title_bar(composite, title)
    return composite


def _compose_v9_ablation_vis(rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                              short_side=250, title=None,
                              col_headers=None, row_labels=None,
                              decoder_preds=None, aspect_ratio=1.0,
                              deg_rgb_vis=None, deg_t_vis=None,
                              deg_feat_rows=None,
                              deg_decoder_preds=None,
                              deg_pred_vis=None):
    num_cols = 8
    cell_h, cell_w = _compute_cell_size(short_side, aspect_ratio)
    empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    inner_gap = 5
    block_gap = 10
    h_gap_strip = np.zeros((cell_h, inner_gap, 3), dtype=np.uint8)
    total_w = num_cols * cell_w + (num_cols - 1) * inner_gap

    def _gapped_row(cells):
        processed = []
        for c in cells:
            if c is None:
                processed.append(empty.copy())
            else:
                processed.append(_make_cell(c, cell_h, cell_w, short_side))
        while len(processed) < num_cols:
            processed.append(empty.copy())
        result = processed[0]
        for cell in processed[1:]:
            result = np.concatenate([result, h_gap_strip, cell], axis=1)
        return result

    def _feat_block(rgb, t, gt, f_rows, d_preds):
        rows = []
        rows.append(_gapped_row([rgb, t, gt] + [None] * 5))
        for stage_cells in f_rows:
            rows.append(_gapped_row(stage_cells))
        pred_cells = [None] * num_cols
        if d_preds is not None:
            for col_idx, pred_img in d_preds.items():
                if col_idx < num_cols:
                    pred_cells[col_idx] = pred_img
        rows.append(_gapped_row(pred_cells))
        return rows

    all_rows = []
    clean_rows = _feat_block(rgb_vis, t_vis, label_vis, feat_rows, decoder_preds)
    v_gap = np.zeros((inner_gap, total_w, 3), dtype=np.uint8)
    for i, row in enumerate(clean_rows):
        if i > 0:
            all_rows.append(v_gap)
        all_rows.append(row)

    if deg_rgb_vis is not None and deg_t_vis is not None and deg_feat_rows is not None:
        b_gap = np.zeros((block_gap, total_w, 3), dtype=np.uint8)
        all_rows.append(b_gap)
        deg_rows = _feat_block(deg_rgb_vis, deg_t_vis, empty.copy(),
                               deg_feat_rows, deg_decoder_preds)
        for i, row in enumerate(deg_rows):
            if i > 0:
                all_rows.append(v_gap)
            all_rows.append(row)

    return np.concatenate(all_rows, axis=0)


def _hstack_samples(grids, gap=0):
    if not grids:
        return None
    max_h = max(g.shape[0] for g in grids)
    padded = []
    for g in grids:
        if g.shape[0] < max_h:
            pad = np.zeros((max_h - g.shape[0], g.shape[1], 3), dtype=np.uint8)
            g = np.concatenate([g, pad], axis=0)
        padded.append(g)
    if gap > 0 and len(padded) > 1:
        gap_strip = np.zeros((max_h, gap, 3), dtype=np.uint8)
        result = padded[0]
        for g in padded[1:]:
            result = np.concatenate([result, gap_strip, g], axis=1)
        return result
    return np.concatenate(padded, axis=1)


def _get_palette_from_runner(runner):
    try:
        dataloader = runner.train_dataloader
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'metainfo'):
            metainfo = dataloader.dataset.metainfo
            if isinstance(metainfo, dict) and 'palette' in metainfo:
                return metainfo['palette']
    except Exception:
        pass
    return None


@HOOKS.register_module()
class TrainVisHook(Hook):
    priority = 'LOW'

    def __init__(self, interval=20, num_samples=2, short_side=250,
                 mask_threshold=0.5):
        super().__init__()
        self.interval = interval
        self.num_samples = num_samples
        self.short_side = short_side
        self.mask_threshold = mask_threshold
        self._writer = None
        self._vis_dir = None
        self._cached_batch = None

    def _ensure_writer(self, runner, model_type):
        if self._writer is not None:
            return
        self._vis_dir = osp.join(runner.work_dir, 'vis_data')
        os.makedirs(self._vis_dir, exist_ok=True)
        if HAS_TENSORBOARD:
            self._writer = SummaryWriter(log_dir=runner._log_dir)
        else:
            self._writer = None

    def _get_model(self, runner):
        model = runner.model
        while hasattr(model, 'module'):
            model = model.module
        return model

    def before_train_epoch(self, runner):
        model = self._get_model(runner)
        if hasattr(model, 'update_prune_temperature'):
            model.update_prune_temperature(
                runner.epoch, runner.max_epochs)
        if hasattr(model, 'epoch'):
            model.epoch = runner.epoch
        if hasattr(model, '_update_quality_freeze_status'):
            model._update_quality_freeze_status(runner.epoch)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if batch_idx != 0:
            return
        if runner.epoch % self.interval != 0:
            return
        self._cached_batch = data_batch

    def after_train_epoch(self, runner, metrics=None):
        epoch = runner.epoch
        if epoch % self.interval != 0:
            return
        if self._cached_batch is None:
            return

        runner.logger.info(f'TrainVisHook: visualizing epoch {epoch}')
        model = self._get_model(runner)
        model_type = detect_model_type(model)
        self._ensure_writer(runner, model_type)

        try:
            self._visualize(runner, model, model_type, epoch)
        except Exception as e:
            import traceback
            runner.logger.warning(f'TrainVisHook error: {e}')
            traceback.print_exc()
        finally:
            self._cached_batch = None

    def _get_loss_summary(self, runner):
        try:
            message_hub = runner.message_hub
            log_scalars = message_hub.log_scalars
            parts = []
            for key in sorted(log_scalars.keys()):
                if key.startswith('loss') and '/' not in key:
                    val = log_scalars[key].current()
                    if isinstance(val, (int, float)):
                        parts.append(f'{key}={val:.4f}')
            return ', '.join(parts) if parts else ''
        except Exception:
            return ''

    def _visualize(self, runner, model, model_type, epoch):
        data_batch = self._cached_batch
        raw_inputs = data_batch.get('inputs', None)
        data_samples = data_batch.get('data_samples', None)
        if raw_inputs is None:
            runner.logger.warning('No inputs in cached data_batch')
            return

        num_total = (raw_inputs.shape[0] if isinstance(raw_inputs, torch.Tensor)
                     else len(raw_inputs))
        num_vis = min(self.num_samples, num_total)
        palette = _get_palette_from_runner(runner)
        loss_str = self._get_loss_summary(runner)

        was_training = model.training
        model.eval()

        try:
            sample_grids = []
            quality_grids = []

            for b_idx in range(num_vis):
                grid, q_grid = self._visualize_sample(
                    runner, model, model_type,
                    raw_inputs, data_samples, b_idx, epoch, palette, loss_str)
                if grid is not None:
                    sample_grids.append(grid)
                if q_grid is not None:
                    quality_grids.append(q_grid)

            self._save_results(runner, sample_grids, quality_grids, epoch,
                               sample_gap=10, quality_gap=10)
        finally:
            if was_training:
                model.train()

        self._log_lora_params(runner, model, epoch)
        runner.logger.info(
            f'TrainVisHook: epoch {epoch}, {num_vis} samples -> {self._vis_dir}')

    def _preprocess_input(self, model, single_input, single_sample, runner, b_idx):
        if not (hasattr(model, 'data_preprocessor')
                and model.data_preprocessor is not None):
            return single_input.to(torch.float32) / 255.0, None

        try:
            saved_meta = []
            if single_sample:
                for ds in single_sample:
                    meta = ds.metainfo
                    saved_meta.append({
                        k: meta[k] for k in
                        ['ori_shape', 'img_shape', 'pad_shape', 'padding_size']
                        if k in meta})

            batch = {'inputs': single_input}
            if single_sample:
                batch['data_samples'] = single_sample
            proc_data = model.data_preprocessor(batch, False)

            if single_sample and saved_meta:
                for ds, sm in zip(proc_data.get('data_samples', single_sample),
                                  saved_meta):
                    ds.set_metainfo(sm)

            return proc_data.get('inputs', None), proc_data
        except Exception as e:
            runner.logger.warning(f'data_preprocessor failed for sample {b_idx}: {e}')
            return single_input.to(torch.float32) / 255.0, None

    def _split_rgb_thermal(self, proc_inputs, proc_data, raw_input=None):
        if raw_input is not None:
            img_np = raw_input[0].detach().cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in (3, 6):
                img_np = img_np.transpose(1, 2, 0)
            if img_np.dtype != np.uint8:
                img_np = _to_uint8(img_np)
        else:
            img_for_vis = (proc_data['inputs']
                           if proc_data and 'inputs' in proc_data
                           else proc_inputs)
            img_np = img_for_vis[0].detach().cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] in (3, 6):
                img_np = img_np.transpose(1, 2, 0)
            img_np = _to_uint8(img_np)

        if img_np.shape[-1] >= 6:
            rgb_raw, t_raw = img_np[:, :, :3], img_np[:, :, 3:6]
        else:
            rgb_raw, t_raw = img_np[:, :, :3], img_np[:, :, :3]

        rgb_vis = rgb_raw if rgb_raw.dtype == np.uint8 else _to_uint8(rgb_raw)
        t_gray = t_raw if t_raw.dtype == np.uint8 else _to_uint8(t_raw)
        if t_gray.ndim == 2:
            t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)
        else:
            t_gray = cv2.cvtColor(t_gray, cv2.COLOR_RGB2GRAY)
            t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)
        return rgb_vis, t_vis

    def _extract_features(self, model, model_type, proc_inputs, runner):
        with torch.no_grad():
            if model_type in ('v2', 'v3'):
                feats = model.extract_feat(proc_inputs)
                return dict(zc_rgb=feats[0], zc_t=feats[1],
                            zp_rgb=feats[2], zp_t=feats[3],
                            fused=feats[5])
            elif model_type in ('v4', 'v5'):
                feats = model.extract_feat(proc_inputs)
                return dict(zc_rgb=feats[0], zc_t=feats[1],
                            zp_rgb=feats[2], zp_t=feats[3],
                            fused=feats[5], q_rgb=feats[6], q_t=feats[7])
            elif model_type == 'original':
                b, c, h, w = proc_inputs.shape
                input_rgbt = proc_inputs.view(b * 2, c // 2, h, w)
                return dict(raw=model.extract_feat(input_rgbt))
            elif model_type == 'v1':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                if hasattr(model.backbone, 'img_size'):
                    img_size = model.backbone.img_size
                    if isinstance(img_size, (tuple, list)):
                        target_h, target_w = img_size
                    else:
                        target_h = target_w = img_size
                    if input_rgbt.shape[-2] != target_h or input_rgbt.shape[-1] != target_w:
                        input_rgbt = F.interpolate(
                            input_rgbt, size=(target_h, target_w),
                            mode='bilinear', align_corners=False)
                fused_feats = model.extract_feat(input_rgbt)
                with torch.no_grad():
                    x_rgbt = model.backbone(input_rgbt)
                B = input_rgbt.shape[0] // 2
                x_rgb_list = [feat[:B] for feat in x_rgbt]
                x_t_list = [feat[B:] for feat in x_rgbt]
                return dict(x_rgb=x_rgb_list, x_t=x_t_list,
                            fused=fused_feats)
            elif model_type == 'v6_baseline':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                fused_feats = model.extract_feat(input_rgbt)
                x_rgbt = model.backbone(input_rgbt)
                B = input_rgbt.shape[0] // 2
                x_rgb_list = [feat[:B] for feat in x_rgbt]
                x_t_list = [feat[B:] for feat in x_rgbt]
                return dict(zc_rgb=x_rgb_list, zc_t=x_t_list,
                            fused=fused_feats)
            elif model_type == 'v6_disentangle':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                zc_rgb, zc_t, zp_rgb, zp_t, zc_enhanced, fused = \
                    model.extract_feat(input_rgbt)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused)
            elif model_type == 'v7_degradation':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                zc_rgb, zc_t, zp_rgb, zp_t, zc_enhanced, fused = \
                    model.extract_feat(input_rgbt)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused)
            elif model_type == 'v7_degradation_full':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                zc_rgb, zc_t, zp_rgb, zp_t, zc_enhanced, fused = \
                    model.extract_feat(input_rgbt)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused)
            elif model_type == 'v8_quality_pyramid':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                (zc_rgb, zc_t, zp_rgb, zp_t,
                 zc_enhanced, fused, q_rgb_maps, q_t_maps,
                 rgb_weighted, t_weighted,
                 universal_enhanced) = \
                    model.extract_feat_with_quality(input_rgbt)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused,
                            q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                            rgb_weighted=rgb_weighted, t_weighted=t_weighted,
                            universal_enhanced=universal_enhanced)
            elif model_type == 'v9_quality_gated':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                (fused_feats, zc_rgb, zc_t,
                 zp_rgb, zp_t, q_rgb_maps, q_t_maps) = \
                    model.extract_feat(input_rgbt)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            fused=fused_feats,
                            q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps)
            elif model_type == 'v7_quality_adaptive':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                (zc_rgb, zc_t, zp_rgb, zp_t,
                 zc_enhanced, fused, q_rgb_maps, q_t_maps) = \
                    model.extract_feat(input_rgb, input_ir)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused,
                            q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps)
            elif model_type == 'v10_quality_embed':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                (zc_rgb, zc_t, zp_rgb, zp_t,
                 zc_enhanced, fused, q_rgb_maps, q_t_maps) = \
                    model.extract_feat(input_rgb, input_ir)
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_enhanced=zc_enhanced, fused=fused,
                            q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps)
            elif model_type == 'v6_add_fusion':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                fused_feats = model.extract_feat(input_rgbt)
                x_rgbt = model.backbone(input_rgbt)
                B = input_rgbt.shape[0] // 2
                x_rgb_list = [feat[:B] for feat in x_rgbt]
                x_t_list = [feat[B:] for feat in x_rgbt]
                return dict(zc_rgb=x_rgb_list, zc_t=x_t_list,
                            fused=fused_feats)
            elif model_type == 'v11_mask_mae':
                fused_feats = model.extract_feat(proc_inputs)
                x_rgb_list = model._last_rgb_feats
                x_t_list = model._last_t_feats
                masks = model._last_masks
                return dict(zc_rgb=x_rgb_list, zc_t=x_t_list,
                            fused=fused_feats, masks=masks)
            elif model_type == 'v6_mask2former':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                fused_feats = model.extract_feat(input_rgbt)
                x_rgb_list = model._last_rgb_feats
                x_t_list = model._last_t_feats
                return dict(zc_rgb=x_rgb_list, zc_t=x_t_list,
                            fused=fused_feats)
            elif model_type == 'v12d_quality_disentangle':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_pf, t_pf,
                     final_fused, q_rgb_maps, q_t_maps) = clean_results

                    deg_inputs = model._generate_degraded_inputs(input_rgb, input_ir)
                    deg_rgb, deg_t, deg_type_rgb, deg_type_t = deg_inputs
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_pf_deg, t_pf_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_pf, t_pf=t_pf,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_pf_deg, t_pf_deg=t_pf_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                )
            elif model_type == 'v12_nodeg_quality_disentangle':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_pf, t_pf,
                     final_fused, q_rgb_maps, q_t_maps) = clean_results

                    deg_rgb, deg_t, deg_type_rgb, deg_type_t = \
                        model._generate_degraded_vis_inputs(input_rgb, input_ir)
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_pf_deg, t_pf_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_pf, t_pf=t_pf,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_pf_deg, t_pf_deg=t_pf_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                )
            elif model_type == 'v12_disentangle_only':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_pf, t_pf,
                     final_fused) = results
                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_pf, t_pf=t_pf,
                    final_fused=final_fused,
                )
            elif model_type == 'swin_baseline':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
                fused_feats = model.extract_feat(input_rgbt)
                x_rgbt = model.backbone(input_rgbt)
                B = input_rgbt.shape[0] // 2
                x_rgb_list = [feat[:B] for feat in x_rgbt]
                x_t_list = [feat[B:] for feat in x_rgbt]
                return dict(zc_rgb=x_rgb_list, zc_t=x_t_list,
                            fused=fused_feats)
            elif model_type == 'mask2former_rgbt_add':
                fused_feats = model.backbone(proc_inputs)
                rgb_feats = model.backbone._rgb_feats
                thr_feats = model.backbone._thr_feats
                if model.with_neck:
                    fused_feats = model.neck(fused_feats)
                return dict(zc_rgb=rgb_feats, zc_t=thr_feats,
                            fused=fused_feats)
            elif model_type == 'ab_baseline':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    results = model._extract_feat(input_rgb, input_ir)
                (x_rgb_list, x_t_list, fused_list,
                 has_rgb, has_t, both_present) = results
                return dict(x_rgb=x_rgb_list, x_t=x_t_list,
                            fused=fused_list)
            elif model_type == 'ab_v1':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    results = model._extract_feat(input_rgb, input_ir)
                (zc_rgb, zc_t, zp_rgb, zp_t,
                 zc_fused, rgb_enh, t_enh,
                 final_fused, has_rgb, has_t, both_present) = results
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_fused=zc_fused,
                            rgb_pf=rgb_enh, t_pf=t_enh,
                            final_fused=final_fused)
            elif model_type == 'ab_v2':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    results = model._extract_feat(input_rgb, input_ir)
                (zc_rgb, zc_t, zp_rgb, zp_t,
                 zc_fused, rgb_enh, t_enh,
                 final_fused, q_rgb_maps, q_t_maps,
                 has_rgb, has_t, both_present) = results
                return dict(zc_rgb=zc_rgb, zc_t=zc_t,
                            zp_rgb=zp_rgb, zp_t=zp_t,
                            zc_fused=zc_fused,
                            rgb_pf=rgb_enh, t_pf=t_enh,
                            final_fused=final_fused,
                            q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps)
            elif model_type == 'ab_v3':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_enh, t_enh,
                     final_fused, q_rgb_maps, q_t_maps,
                     has_rgb, has_t, both_present) = clean_results

                    deg_inputs = model._generate_degraded_inputs(input_rgb, input_ir)
                    if deg_inputs is None:
                        deg_rgb, deg_t = input_rgb, input_ir
                        deg_type_rgb, deg_type_t = 'none', 'none'
                    else:
                        deg_rgb, deg_t, deg_type_rgb, deg_type_t = deg_inputs
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_enh_deg, t_enh_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg,
                     has_rgb_deg, has_t_deg, both_present_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_enh, t_pf=t_enh,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_enh_deg, t_pf_deg=t_enh_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                )
            elif model_type == 'ab_v4':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_enh, t_enh,
                     final_fused, q_rgb_maps, q_t_maps,
                     has_rgb, has_t, both_present) = clean_results

                    deg_inputs = model._generate_degraded_inputs(input_rgb, input_ir)
                    if deg_inputs is None:
                        deg_rgb, deg_t = input_rgb, input_ir
                        deg_type_rgb, deg_type_t = 'none', 'none'
                    else:
                        deg_rgb, deg_t, deg_type_rgb, deg_type_t = deg_inputs
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_enh_deg, t_enh_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg,
                     has_rgb_deg, has_t_deg, both_present_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_enh, t_pf=t_enh,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_enh_deg, t_pf_deg=t_enh_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                )
            elif model_type == 'ab_v9':
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_enh, t_enh,
                     final_fused, q_rgb_maps, q_t_maps,
                     all_D_for_loss,
                     D_rgb_list, D_t_list,
                     D_rgb_priv_list, D_t_priv_list,
                     q_rgb_priv_list, q_t_priv_list,
                     cum_D_rgb_list, cum_D_t_list,
                     cum_D_rgb_priv_list, cum_D_t_priv_list) = clean_results

                    deg_inputs = model._generate_degraded_inputs(input_rgb, input_ir)
                    deg_rgb, deg_t, deg_type_rgb, deg_type_t = deg_inputs
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_enh_deg, t_enh_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg,
                     _, D_rgb_deg, D_t_deg,
                     D_rgb_priv_deg, D_t_priv_deg,
                     q_rgb_priv_deg, q_t_priv_deg,
                     cum_D_rgb_deg, cum_D_t_deg,
                     cum_D_rgb_priv_deg, cum_D_t_priv_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_enh, t_pf=t_enh,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    D_rgb=D_rgb_list, D_t=D_t_list,
                    D_rgb_priv=D_rgb_priv_list, D_t_priv=D_t_priv_list,
                    q_rgb_priv=q_rgb_priv_list, q_t_priv=q_t_priv_list,
                    cum_D_rgb=cum_D_rgb_list, cum_D_t=cum_D_t_list,
                    cum_D_rgb_priv=cum_D_rgb_priv_list, cum_D_t_priv=cum_D_t_priv_list,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_enh_deg, t_pf_deg=t_enh_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                    D_rgb_deg=D_rgb_deg, D_t_deg=D_t_deg,
                    D_rgb_priv_deg=D_rgb_priv_deg, D_t_priv_deg=D_t_priv_deg,
                    q_rgb_priv_deg=q_rgb_priv_deg, q_t_priv_deg=q_t_priv_deg,
                    cum_D_rgb_deg=cum_D_rgb_deg, cum_D_t_deg=cum_D_t_deg,
                    cum_D_rgb_priv_deg=cum_D_rgb_priv_deg, cum_D_t_priv_deg=cum_D_t_priv_deg,
                )
            elif model_type in ('ab_v5', 'ab_v6', 'ab_v7', 'ab_v8'):
                input_rgb = proc_inputs[:, :3, :, :]
                input_ir = proc_inputs[:, 3:, :, :]
                with torch.no_grad():
                    clean_results = model._extract_feat_single(input_rgb, input_ir)
                    (zc_rgb, zc_t, zp_rgb, zp_t,
                     zc_fused, rgb_enh, t_enh,
                     final_fused, q_rgb_maps, q_t_maps,
                     has_rgb, has_t, both_present) = clean_results

                    deg_inputs = model._generate_degraded_inputs(input_rgb, input_ir)
                    if deg_inputs is None:
                        deg_rgb, deg_t = input_rgb, input_ir
                        deg_type_rgb, deg_type_t = 'none', 'none'
                    else:
                        deg_rgb, deg_t, deg_type_rgb, deg_type_t = deg_inputs
                    deg_results = model._extract_feat_single(deg_rgb, deg_t)
                    (zc_rgb_deg, zc_t_deg, zp_rgb_deg, zp_t_deg,
                     zc_fused_deg, rgb_enh_deg, t_enh_deg,
                     final_fused_deg, q_rgb_deg, q_t_deg,
                     has_rgb_deg, has_t_deg, both_present_deg) = deg_results

                return dict(
                    zc_rgb=zc_rgb, zc_t=zc_t,
                    zp_rgb=zp_rgb, zp_t=zp_t,
                    zc_fused=zc_fused,
                    rgb_pf=rgb_enh, t_pf=t_enh,
                    final_fused=final_fused,
                    q_rgb_maps=q_rgb_maps, q_t_maps=q_t_maps,
                    deg_rgb_img=deg_rgb, deg_t_img=deg_t,
                    deg_type_rgb=deg_type_rgb[0] if isinstance(deg_type_rgb, list) else deg_type_rgb,
                    deg_type_t=deg_type_t[0] if isinstance(deg_type_t, list) else deg_type_t,
                    zc_rgb_deg=zc_rgb_deg, zc_t_deg=zc_t_deg,
                    zp_rgb_deg=zp_rgb_deg, zp_t_deg=zp_t_deg,
                    zc_fused_deg=zc_fused_deg,
                    rgb_pf_deg=rgb_enh_deg, t_pf_deg=t_enh_deg,
                    final_fused_deg=final_fused_deg,
                    q_rgb_deg=q_rgb_deg, q_t_deg=q_t_deg,
                )
            else:
                runner.logger.warning(f'Unknown model type: {model_type}')
                return None

    def _get_label(self, single_sample, h, w):
        label_seg = np.zeros((h, w), dtype=np.uint8)
        if single_sample and len(single_sample) > 0:
            ds = single_sample[0]
            if hasattr(ds, 'gt_sem_seg') and ds.gt_sem_seg is not None:
                gt = ds.gt_sem_seg.data
                if gt is not None:
                    label_seg = gt.squeeze().cpu().numpy().astype(np.uint8)
        return label_seg

    def _get_prediction(self, model, proc_inputs, single_sample, runner):
        h, w = proc_inputs.shape[-2], proc_inputs.shape[-1]
        pred_seg = np.zeros((h, w), dtype=np.uint8)
        try:
            with torch.no_grad():
                batch_img_metas = [dict(
                    ori_shape=proc_inputs.shape[2:],
                    img_shape=proc_inputs.shape[2:],
                    pad_shape=proc_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])]
                seg_logits = model.encode_decode(proc_inputs, batch_img_metas)
                if isinstance(seg_logits, torch.Tensor):
                    pred_seg = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
        except Exception as e:
            runner.logger.warning(f'Prediction failed: {e}')
        return pred_seg

    def _render_seg(self, label_seg, pred_seg, palette):
        if palette is not None:
            return _apply_palette(label_seg, palette), _apply_palette(pred_seg, palette)
        label_vis = cv2.cvtColor(
            cv2.applyColorMap(label_seg, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        pred_vis = cv2.cvtColor(
            cv2.applyColorMap(pred_seg, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        return label_vis, pred_vis

    def _get_decoder_predictions(self, model, feats, palette, runner,
                                 prefix=''):
        decoder_preds = {}
        if prefix:
            zc_fused_key = f'zc_fused_{prefix}'
            rgb_pf_key = f'rgb_pf_{prefix}'
            t_pf_key = f't_pf_{prefix}'
            final_fused_key = f'final_fused_{prefix}'
        else:
            zc_fused_key = 'zc_fused'
            rgb_pf_key = 'rgb_pf'
            t_pf_key = 't_pf'
            final_fused_key = 'final_fused'
        try:
            with torch.no_grad():
                if hasattr(model, 'common_decode_head') and \
                        model.common_decode_head is not None and \
                        zc_fused_key in feats:
                    common_feats = feats[zc_fused_key]
                    if model.with_neck:
                        common_feats = model.neck(common_feats)
                    logits = model._decode_head_predict_logits(
                        common_feats, model.common_decode_head)
                    if isinstance(logits, torch.Tensor):
                        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                        pred_vis = _apply_palette(pred, palette) if palette else \
                            cv2.cvtColor(cv2.applyColorMap(pred, cv2.COLORMAP_VIRIDIS),
                                         cv2.COLOR_BGR2RGB)
                        decoder_preds[4] = pred_vis

                if hasattr(model, 'rgb_private_decode_head') and \
                        model.rgb_private_decode_head is not None and \
                        rgb_pf_key in feats:
                    rgb_pf = feats[rgb_pf_key]
                    if model.with_neck:
                        rgb_pf = model.neck(rgb_pf)
                    logits = model._decode_head_predict_logits(
                        rgb_pf, model.rgb_private_decode_head)
                    if isinstance(logits, torch.Tensor):
                        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                        pred_vis = _apply_palette(pred, palette) if palette else \
                            cv2.cvtColor(cv2.applyColorMap(pred, cv2.COLORMAP_VIRIDIS),
                                         cv2.COLOR_BGR2RGB)
                        decoder_preds[5] = pred_vis

                if hasattr(model, 't_private_decode_head') and \
                        model.t_private_decode_head is not None and \
                        t_pf_key in feats:
                    t_pf = feats[t_pf_key]
                    if model.with_neck:
                        t_pf = model.neck(t_pf)
                    logits = model._decode_head_predict_logits(
                        t_pf, model.t_private_decode_head)
                    if isinstance(logits, torch.Tensor):
                        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                        pred_vis = _apply_palette(pred, palette) if palette else \
                            cv2.cvtColor(cv2.applyColorMap(pred, cv2.COLORMAP_VIRIDIS),
                                         cv2.COLOR_BGR2RGB)
                        decoder_preds[6] = pred_vis

                if hasattr(model, 'decode_head') and final_fused_key in feats:
                    final_feats = feats[final_fused_key]
                    if model.with_neck:
                        final_feats = model.neck(final_feats)
                    logits = model._decode_head_predict_logits(final_feats)
                    if isinstance(logits, torch.Tensor):
                        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
                        pred_vis = _apply_palette(pred, palette) if palette else \
                            cv2.cvtColor(cv2.applyColorMap(pred, cv2.COLORMAP_VIRIDIS),
                                         cv2.COLOR_BGR2RGB)
                        decoder_preds[7] = pred_vis
        except Exception as e:
            runner.logger.warning(f'Decoder prediction failed: {e}')
        return decoder_preds

    def _build_feat_vis(self, model_type, feats, runner,
                        rgb_vis=None, t_vis=None,
                        img_h=None, img_w=None):
        q_grid = None
        if model_type == 'original':
            rgb_feats, t_feats = [], []
            for feat in feats['raw']:
                fc = feat.shape[1]
                half_c = fc // 2
                rgb_feats.append(feat[:, :half_c, :, :])
                t_feats.append(feat[:, half_c:, :, :])
            feat_rows = _build_feat_rows([rgb_feats, t_feats])
        elif model_type == 'v1':
            feat_rows = _build_feat_rows(
                [feats['x_rgb'], feats['x_t'], feats['fused']])
        elif model_type in ('v2', 'v3'):
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'], feats['fused']])
        elif model_type in ('v4', 'v5'):
            q_info = self._create_quality_vis(feats)
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'], feats['fused']],
                masks=[q_info['rgb_mask'], q_info['t_mask'],
                       q_info['rgb_mask'], q_info['t_mask'], None])
            q_grid = self._compose_quality_vis(q_info)
        elif model_type == 'v6_baseline':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']])
        elif model_type == 'v6_disentangle':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_enhanced'], feats['fused']])
        elif model_type == 'v7_degradation':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_enhanced'], feats['fused']])
        elif model_type == 'v7_degradation_full':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_enhanced'], feats['fused']])
        elif model_type == 'v8_quality_pyramid':
            q_info = self._create_pyramid_quality_vis(feats, img_h=img_h, img_w=img_w)
            feat_rows = _build_feat_rows(
                [feats['rgb_weighted'], feats['t_weighted'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['universal_enhanced'], feats['fused']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_pyramid_quality_vis(
                q_info, rgb_vis=rgb_vis, t_vis=t_vis, aspect_ratio=aspect_ratio)
        elif model_type == 'v9_quality_gated':
            q_info = self._create_v9_quality_vis(feats, img_h=img_h, img_w=img_w)
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['fused']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v9_quality_vis(
                q_info, rgb_vis=rgb_vis, t_vis=t_vis, aspect_ratio=aspect_ratio)
        elif model_type == 'v7_quality_adaptive':
            q_info = self._create_v9_quality_vis(feats, img_h=img_h, img_w=img_w)
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_enhanced'], feats['fused']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v9_quality_vis(
                q_info, rgb_vis=rgb_vis, t_vis=t_vis, aspect_ratio=aspect_ratio)
        elif model_type == 'v10_quality_embed':
            q_info = self._create_v9_quality_vis(feats, img_h=img_h, img_w=img_w)
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_enhanced'], feats['fused']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v9_quality_vis(
                q_info, rgb_vis=rgb_vis, t_vis=t_vis, aspect_ratio=aspect_ratio)
        elif model_type == 'v6_add_fusion':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']])
        elif model_type == 'v11_mask_mae':
            masks = feats.get('masks', None)
            rgb_mask_vis = None
            t_mask_vis = None
            if masks is not None:
                rgb_mask_1d, t_mask_1d, strategy = masks
                rgb_mask_vis = ~rgb_mask_1d[0]
                t_mask_vis = ~t_mask_1d[0]
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']],
                masks=[rgb_mask_vis, t_mask_vis, None])
        elif model_type == 'v6_mask2former':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']])
        elif model_type == 'v12d_quality_disentangle':
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v12d_deg_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_deg'])
            deg_q_t_mask = self._get_v12d_quality_mask(feats['q_t_deg'])
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_fused'], feats['rgb_pf'],
                 feats['t_pf'], feats['final_fused']])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['zc_fused_deg'], feats['rgb_pf_deg'],
                 feats['t_pf_deg'], feats['final_fused_deg']],
                masks=[deg_q_rgb_mask, deg_q_t_mask,
                       deg_q_rgb_mask, deg_q_t_mask,
                       None, deg_q_rgb_mask,
                       deg_q_t_mask, None])
            feat_rows.extend(deg_feat_rows)
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
        elif model_type == 'v12_nodeg_quality_disentangle':
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v12d_deg_quality_vis(feats, img_h=img_h, img_w=img_w)
            q_rgb_deg_mask = self._get_v12d_quality_mask(feats['q_rgb_deg'])
            q_t_deg_mask = self._get_v12d_quality_mask(feats['q_t_deg'])
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_fused'], feats['rgb_pf'],
                 feats['t_pf'], feats['final_fused']])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['zc_fused_deg'], feats['rgb_pf_deg'],
                 feats['t_pf_deg'], feats['final_fused_deg']],
                masks=[q_rgb_deg_mask, q_t_deg_mask,
                       q_rgb_deg_mask, q_t_deg_mask,
                       None, q_rgb_deg_mask,
                       q_t_deg_mask, None])
            feat_rows.extend(deg_feat_rows)
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
        elif model_type == 'v12_disentangle_only':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['zc_fused'], feats['rgb_pf'],
                 feats['t_pf'], feats['final_fused']])
            q_grid = None
        elif model_type == 'swin_baseline':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']])
        elif model_type == 'mask2former_rgbt_add':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['fused']])
        elif model_type == 'ab_baseline':
            feat_rows = _build_feat_rows(
                [feats['x_rgb'], feats['x_t'], feats['fused']])
        elif model_type == 'ab_v1':
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']])
        elif model_type == 'ab_v2':
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_maps'])
            q_t_mask = self._get_v12d_quality_mask(feats['q_t_maps'])
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']],
                masks=[q_rgb_mask, q_t_mask, None,
                       q_rgb_mask, q_t_mask,
                       q_rgb_mask, q_t_mask, None])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
        elif model_type == 'ab_v3':
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v12d_deg_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_deg'])
            deg_q_t_mask = self._get_v12d_quality_mask(feats['q_t_deg'])
            q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_maps'])
            q_t_mask = self._get_v12d_quality_mask(feats['q_t_maps'])
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']],
                masks=[q_rgb_mask, q_t_mask, None,
                       q_rgb_mask, q_t_mask,
                       q_rgb_mask, q_t_mask, None])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'], feats['zc_fused_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['rgb_pf_deg'], feats['t_pf_deg'], feats['final_fused_deg']],
                masks=[deg_q_rgb_mask, deg_q_t_mask, None,
                       deg_q_rgb_mask, deg_q_t_mask,
                       deg_q_rgb_mask, deg_q_t_mask, None])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
            return feat_rows, q_grid, deg_feat_rows
        elif model_type == 'ab_v4':
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v12d_deg_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_deg'])
            deg_q_t_mask = self._get_v12d_quality_mask(feats['q_t_deg'])
            q_rgb_mask = self._get_v12d_quality_mask(feats['q_rgb_maps'])
            q_t_mask = self._get_v12d_quality_mask(feats['q_t_maps'])
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']],
                masks=[q_rgb_mask, q_t_mask, None,
                       q_rgb_mask, q_t_mask,
                       q_rgb_mask, q_t_mask, None])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'], feats['zc_fused_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['rgb_pf_deg'], feats['t_pf_deg'], feats['final_fused_deg']],
                masks=[deg_q_rgb_mask, deg_q_t_mask, None,
                       deg_q_rgb_mask, deg_q_t_mask,
                       deg_q_rgb_mask, deg_q_t_mask, None])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
            return feat_rows, q_grid, deg_feat_rows
        elif model_type == 'ab_v9':
            q_info = self._create_v9_ablation_quality_vis(
                feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v9_ablation_deg_quality_vis(
                feats, img_h=img_h, img_w=img_w)
            self._v9_q_info = q_info
            self._v9_deg_q_info = deg_q_info
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'], feats['zc_fused_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['rgb_pf_deg'], feats['t_pf_deg'], feats['final_fused_deg']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v9_ablation_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                aspect_ratio=aspect_ratio)
            return feat_rows, q_grid, deg_feat_rows
        elif model_type in ('ab_v5', 'ab_v6', 'ab_v7', 'ab_v8'):
            q_info = self._create_v12d_quality_vis(feats, img_h=img_h, img_w=img_w)
            deg_q_info = self._create_v12d_deg_quality_vis(feats, img_h=img_h, img_w=img_w)
            feat_rows = _build_feat_rows(
                [feats['zc_rgb'], feats['zc_t'], feats['zc_fused'],
                 feats['zp_rgb'], feats['zp_t'],
                 feats['rgb_pf'], feats['t_pf'], feats['final_fused']])
            deg_feat_rows = _build_feat_rows(
                [feats['zc_rgb_deg'], feats['zc_t_deg'], feats['zc_fused_deg'],
                 feats['zp_rgb_deg'], feats['zp_t_deg'],
                 feats['rgb_pf_deg'], feats['t_pf_deg'], feats['final_fused_deg']])
            aspect_ratio = (img_w / max(img_h, 1)) if (img_h and img_w) else 1.0
            q_grid = self._compose_v12d_quality_vis(
                q_info, deg_q_info=deg_q_info,
                rgb_vis=rgb_vis, t_vis=t_vis,
                feats=feats, aspect_ratio=aspect_ratio)
            return feat_rows, q_grid, deg_feat_rows
        else:
            feat_rows = []
        return feat_rows, q_grid, None

    def _create_quality_vis(self, feats):
        zc_rgb = feats['zc_rgb']
        q_rgb, q_t = feats['q_rgb'], feats['q_t']
        h, w = zc_rgb[0].shape[-2:]

        q_rgb_map = _quality_to_heatmap(q_rgb, h, w)
        q_t_map = _quality_to_heatmap(q_t, h, w)

        q_rgb_colored = cv2.cvtColor(
            cv2.applyColorMap(_to_uint8(q_rgb_map[0]), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)
        q_t_colored = cv2.cvtColor(
            cv2.applyColorMap(_to_uint8(q_t_map[0]), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB)

        rgb_mask = (q_rgb[0] >= self.mask_threshold).cpu().numpy()
        t_mask = (q_t[0] >= self.mask_threshold).cpu().numpy()
        return dict(q_rgb=q_rgb_colored, q_t=q_t_colored,
                    rgb_mask=rgb_mask, t_mask=t_mask)

    def _create_pyramid_quality_vis(self, feats, img_h=None, img_w=None):
        q_rgb_maps = feats['q_rgb_maps']
        q_t_maps = feats['q_t_maps']
        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        rgb_heatmaps = []
        t_heatmaps = []
        rgb_masks = []
        t_masks = []
        rgb_mean_scores = []
        t_mean_scores = []

        for i in range(len(q_rgb_maps)):
            q_rgb_i = q_rgb_maps[i]
            q_t_i = q_t_maps[i]

            q_rgb_np = q_rgb_i[0, 0].detach().cpu().numpy()
            q_t_np = q_t_i[0, 0].detach().cpu().numpy()

            rgb_mean_scores.append(float(q_rgb_np.mean()))
            t_mean_scores.append(float(q_t_np.mean()))

            rgb_heatmaps.append(
                _quality_to_rgb_heatmap(q_rgb_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))
            t_heatmaps.append(
                _quality_to_rgb_heatmap(q_t_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))

            q_rgb_stage = q_rgb_i[0, 0].detach().cpu().numpy()
            q_t_stage = q_t_i[0, 0].detach().cpu().numpy()
            rgb_masks.append(q_rgb_stage >= self.mask_threshold)
            t_masks.append(q_t_stage >= self.mask_threshold)

        return dict(rgb_heatmaps=rgb_heatmaps, t_heatmaps=t_heatmaps,
                    rgb_masks=rgb_masks, t_masks=t_masks,
                    rgb_mean_scores=rgb_mean_scores, t_mean_scores=t_mean_scores)

    def _compose_quality_vis(self, q_info):
        cell_h = cell_w = self.short_side
        num_cols = 2
        row1 = _build_row(
            [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * 2,
            cell_h, cell_w, self.short_side, num_cols)
        row2 = _build_row([q_info['q_rgb'], q_info['q_t']],
                          cell_h, cell_w, self.short_side, num_cols)
        return np.concatenate([row1, row2], axis=0)

    def _compose_pyramid_quality_vis(self, q_info, rgb_vis=None, t_vis=None,
                                     aspect_ratio=1.0):
        cell_h, cell_w = _compute_cell_size(self.short_side, aspect_ratio)
        num_cols = 2
        rows = []
        if rgb_vis is not None and t_vis is not None:
            header = _build_row(
                [rgb_vis, t_vis],
                cell_h, cell_w, self.short_side, num_cols)
        else:
            header = _build_row(
                [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * 2,
                cell_h, cell_w, self.short_side, num_cols)
        rows.append(header)
        for i in range(len(q_info['rgb_heatmaps'])):
            rgb_hm = q_info['rgb_heatmaps'][i]
            t_hm = q_info['t_heatmaps'][i]
            rgb_score = q_info.get('rgb_mean_scores', [None])[i]
            t_score = q_info.get('t_mean_scores', [None])[i]
            if rgb_score is not None:
                rgb_hm = _add_text_overlay(rgb_hm, f'S{i} q={rgb_score:.2f}',
                                           font_scale=0.4, thickness=1)
            if t_score is not None:
                t_hm = _add_text_overlay(t_hm, f'S{i} q={t_score:.2f}',
                                         font_scale=0.4, thickness=1)
            row = _build_row(
                [rgb_hm, t_hm],
                cell_h, cell_w, self.short_side, num_cols)
            rows.append(row)
        return np.concatenate(rows, axis=0)

    def _create_v9_quality_vis(self, feats, img_h=None, img_w=None):
        q_rgb_maps = feats['q_rgb_maps']
        q_t_maps = feats['q_t_maps']
        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        rgb_heatmaps = []
        t_heatmaps = []
        rgb_mean_scores = []
        t_mean_scores = []

        for i in range(len(q_rgb_maps)):
            q_rgb_i = q_rgb_maps[i]
            q_t_i = q_t_maps[i]

            q_rgb_np = q_rgb_i[0, 0].detach().cpu().numpy()
            q_t_np = q_t_i[0, 0].detach().cpu().numpy()

            rgb_mean_scores.append(float(q_rgb_np.mean()))
            t_mean_scores.append(float(q_t_np.mean()))

            rgb_heatmaps.append(
                _quality_to_rgb_heatmap(q_rgb_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))
            t_heatmaps.append(
                _quality_to_rgb_heatmap(q_t_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))

        return dict(rgb_heatmaps=rgb_heatmaps, t_heatmaps=t_heatmaps,
                    rgb_mean_scores=rgb_mean_scores, t_mean_scores=t_mean_scores)

    def _compose_v9_quality_vis(self, q_info, rgb_vis=None, t_vis=None,
                                aspect_ratio=1.0):
        cell_h, cell_w = _compute_cell_size(self.short_side, aspect_ratio)
        num_cols = 2
        rows = []
        if rgb_vis is not None and t_vis is not None:
            header = _build_row(
                [rgb_vis, t_vis],
                cell_h, cell_w, self.short_side, num_cols)
        else:
            header = _build_row(
                [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * 2,
                cell_h, cell_w, self.short_side, num_cols)
        rows.append(header)
        for i in range(len(q_info['rgb_heatmaps'])):
            rgb_hm = q_info['rgb_heatmaps'][i]
            t_hm = q_info['t_heatmaps'][i]
            rgb_score = q_info.get('rgb_mean_scores', [None])[i]
            t_score = q_info.get('t_mean_scores', [None])[i]
            if rgb_score is not None:
                rgb_hm = _add_text_overlay(rgb_hm, f'S{i} q={rgb_score:.2f}',
                                           font_scale=0.4, thickness=1)
            if t_score is not None:
                t_hm = _add_text_overlay(t_hm, f'S{i} q={t_score:.2f}',
                                         font_scale=0.4, thickness=1)
            row = _build_row(
                [rgb_hm, t_hm],
                cell_h, cell_w, self.short_side, num_cols)
            rows.append(row)
        return np.concatenate(rows, axis=0)

    def _create_v12d_quality_vis(self, feats, img_h=None, img_w=None):
        q_rgb_maps = feats['q_rgb_maps']
        q_t_maps = feats['q_t_maps']
        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        rgb_heatmaps = []
        t_heatmaps = []
        rgb_mean_scores = []
        t_mean_scores = []

        for i in range(len(q_rgb_maps)):
            q_rgb_i = q_rgb_maps[i]
            q_t_i = q_t_maps[i]

            q_rgb_np = q_rgb_i[0, 0].detach().cpu().numpy()
            q_t_np = q_t_i[0, 0].detach().cpu().numpy()

            rgb_mean_scores.append(float(q_rgb_np.mean()))
            t_mean_scores.append(float(q_t_np.mean()))

            rgb_heatmaps.append(
                _quality_to_rgb_heatmap(q_rgb_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))
            t_heatmaps.append(
                _quality_to_rgb_heatmap(q_t_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))

        return dict(rgb_heatmaps=rgb_heatmaps, t_heatmaps=t_heatmaps,
                    rgb_mean_scores=rgb_mean_scores, t_mean_scores=t_mean_scores)

    def _create_v12d_deg_quality_vis(self, feats, img_h=None, img_w=None):
        q_rgb_deg = feats['q_rgb_deg']
        q_t_deg = feats['q_t_deg']
        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        rgb_heatmaps = []
        t_heatmaps = []
        rgb_mean_scores = []
        t_mean_scores = []

        for i in range(len(q_rgb_deg)):
            q_rgb_i = q_rgb_deg[i]
            q_t_i = q_t_deg[i]

            q_rgb_np = q_rgb_i[0, 0].detach().cpu().numpy()
            q_t_np = q_t_i[0, 0].detach().cpu().numpy()

            rgb_mean_scores.append(float(q_rgb_np.mean()))
            t_mean_scores.append(float(q_t_np.mean()))

            rgb_heatmaps.append(
                _quality_to_rgb_heatmap(q_rgb_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))
            t_heatmaps.append(
                _quality_to_rgb_heatmap(q_t_np, h, w, vmin=0.0, vmax=1.0,
                                        cmap='rdBu_r'))

        return dict(rgb_heatmaps=rgb_heatmaps, t_heatmaps=t_heatmaps,
                    rgb_mean_scores=rgb_mean_scores, t_mean_scores=t_mean_scores)

    def _get_v12d_quality_mask(self, q_maps, stage_idx=0):
        if q_maps is None or len(q_maps) == 0:
            return None
        q_stage = q_maps[stage_idx]
        q_np = q_stage[0, 0].detach().cpu().numpy()
        H_m = W_m = int(q_np.shape[0] ** 0.5) if q_np.ndim == 1 else q_np.shape[0]
        if q_np.ndim == 1:
            q_2d = q_np.reshape(H_m, -1)
        else:
            q_2d = q_np
        mask = (q_2d >= self.mask_threshold).astype(np.float32)
        return mask

    def _render_degraded_images(self, feats, model, raw_rgb=None, raw_t=None):
        deg_rgb = feats['deg_rgb_img']
        deg_t = feats['deg_t_img']
        deg_type_rgb = feats.get('deg_type_rgb', 'none')
        deg_type_t = feats.get('deg_type_t', 'none')

        try:
            rgb_mean = model.data_preprocessor.mean[:3].flatten().cpu()
            rgb_std = model.data_preprocessor.std[:3].flatten().cpu()
            ir_mean = model.data_preprocessor.mean[3:].flatten().cpu()
            ir_std = model.data_preprocessor.std[3:].flatten().cpu()

            if deg_type_rgb == 'missing':
                if raw_rgb is not None:
                    rgb_vis = _add_missing_overlay(raw_rgb)
                else:
                    rgb_vis = np.zeros((self.short_side, self.short_side, 3), dtype=np.uint8)
                    rgb_vis = _add_text_overlay(rgb_vis, 'MISSING')
            else:
                rgb_raw = deg_rgb[0].cpu() * rgb_std.view(3, 1, 1) + rgb_mean.view(3, 1, 1)
                rgb_raw = rgb_raw.clamp(0, 255) / 255.0
                rgb_np = rgb_raw.permute(1, 2, 0).numpy()
                rgb_vis = _to_uint8(rgb_np)
                if deg_type_rgb != 'none':
                    rgb_vis = _add_text_overlay(rgb_vis, deg_type_rgb)

            if deg_type_t == 'missing':
                if raw_t is not None:
                    t_vis = _add_missing_overlay(raw_t)
                else:
                    t_vis = np.zeros((self.short_side, self.short_side, 3), dtype=np.uint8)
                    t_vis = _add_text_overlay(t_vis, 'MISSING')
            else:
                t_raw = deg_t[0].cpu() * ir_std.view(3, 1, 1) + ir_mean.view(3, 1, 1)
                t_raw = t_raw.clamp(0, 255) / 255.0
                t_np = t_raw.permute(1, 2, 0).numpy()
                t_gray = cv2.cvtColor(_to_uint8(t_np), cv2.COLOR_RGB2GRAY)
                t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)
                if deg_type_t != 'none':
                    t_vis = _add_text_overlay(t_vis, deg_type_t)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rgb_vis = np.zeros((self.short_side, self.short_side, 3), dtype=np.uint8)
            t_vis = np.zeros((self.short_side, self.short_side, 3), dtype=np.uint8)

        return rgb_vis, t_vis

    def _render_degraded_images_raw(self, feats, model):
        deg_rgb = feats['deg_rgb_img']
        deg_t = feats['deg_t_img']

        try:
            rgb_mean = model.data_preprocessor.mean[:3].flatten().cpu()
            rgb_std = model.data_preprocessor.std[:3].flatten().cpu()
            ir_mean = model.data_preprocessor.mean[3:].flatten().cpu()
            ir_std = model.data_preprocessor.std[3:].flatten().cpu()

            rgb_raw = deg_rgb[0].cpu() * rgb_std.view(3, 1, 1) + rgb_mean.view(3, 1, 1)
            rgb_raw = rgb_raw.clamp(0, 255) / 255.0
            rgb_np = rgb_raw.permute(1, 2, 0).numpy()
            rgb_vis = _to_uint8(rgb_np)

            t_raw = deg_t[0].cpu() * ir_std.view(3, 1, 1) + ir_mean.view(3, 1, 1)
            t_raw = t_raw.clamp(0, 255) / 255.0
            t_np = t_raw.permute(1, 2, 0).numpy()
            t_uint8 = _to_uint8(t_np)
            t_gray = cv2.cvtColor(t_uint8, cv2.COLOR_RGB2GRAY)
            t_vis = np.stack([t_gray, t_gray, t_gray], axis=-1)
        except Exception:
            import traceback
            traceback.print_exc()
            h, w = deg_rgb.shape[-2], deg_rgb.shape[-1]
            rgb_vis = np.zeros((h, w, 3), dtype=np.uint8)
            t_vis = np.zeros((h, w, 3), dtype=np.uint8)

        return rgb_vis, t_vis

    def _compose_v12d_quality_vis(self, q_info, deg_q_info=None,
                                  rgb_vis=None, t_vis=None, feats=None,
                                  aspect_ratio=1.0):
        cell_h, cell_w = _compute_cell_size(self.short_side, aspect_ratio)
        num_cols = 2
        rows = []
        if rgb_vis is not None and t_vis is not None:
            header = _build_row(
                [rgb_vis, t_vis],
                cell_h, cell_w, self.short_side, num_cols)
        else:
            header = _build_row(
                [np.zeros((cell_h, cell_w, 3), dtype=np.uint8)] * 2,
                cell_h, cell_w, self.short_side, num_cols)
        rows.append(header)
        for i in range(len(q_info['rgb_heatmaps'])):
            rgb_hm = q_info['rgb_heatmaps'][i]
            t_hm = q_info['t_heatmaps'][i]
            rgb_score = q_info.get('rgb_mean_scores', [None])[i]
            t_score = q_info.get('t_mean_scores', [None])[i]
            if rgb_score is not None:
                rgb_hm = _add_text_overlay(rgb_hm, f'S{i} q={rgb_score:.2f}',
                                           font_scale=0.4, thickness=1)
            if t_score is not None:
                t_hm = _add_text_overlay(t_hm, f'S{i} q={t_score:.2f}',
                                         font_scale=0.4, thickness=1)
            row = _build_row(
                [rgb_hm, t_hm],
                cell_h, cell_w, self.short_side, num_cols)
            rows.append(row)
        if deg_q_info is not None:
            deg_label_row = _add_title_bar(
                np.zeros((cell_h, cell_w * 2, 3), dtype=np.uint8),
                'Degraded Quality Maps', font_scale=0.5, thickness=1)
            rows.append(deg_label_row)
            for i in range(len(deg_q_info['rgb_heatmaps'])):
                rgb_hm = deg_q_info['rgb_heatmaps'][i]
                t_hm = deg_q_info['t_heatmaps'][i]
                rgb_score = deg_q_info.get('rgb_mean_scores', [None])[i]
                t_score = deg_q_info.get('t_mean_scores', [None])[i]
                if rgb_score is not None:
                    rgb_hm = _add_text_overlay(rgb_hm, f'S{i} q={rgb_score:.2f}',
                                               font_scale=0.4, thickness=1)
                if t_score is not None:
                    t_hm = _add_text_overlay(t_hm, f'S{i} q={t_score:.2f}',
                                             font_scale=0.4, thickness=1)
                row = _build_row(
                    [rgb_hm, t_hm],
                    cell_h, cell_w, self.short_side, num_cols)
                rows.append(row)
        return np.concatenate(rows, axis=0)

    def _visualize_sample(self, runner, model, model_type,
                          raw_inputs, data_samples, b_idx, epoch, palette,
                          loss_str):
        single_input = (raw_inputs[b_idx].unsqueeze(0)
                        if isinstance(raw_inputs, torch.Tensor)
                        else raw_inputs[b_idx].unsqueeze(0))
        single_sample = ([data_samples[b_idx]]
                         if data_samples and b_idx < len(data_samples)
                         else None)

        proc_inputs, proc_data = self._preprocess_input(
            model, single_input, single_sample, runner, b_idx)
        if proc_inputs is None:
            return None, None

        rgb_vis, t_vis = self._split_rgb_thermal(proc_inputs, proc_data,
                                                  raw_input=single_input)

        feats = self._extract_features(model, model_type, proc_inputs, runner)
        if feats is None:
            return None, None

        label_seg = self._get_label(single_sample, proc_inputs.shape[-2],
                                    proc_inputs.shape[-1])
        pred_seg = self._get_prediction(model, proc_inputs, single_sample, runner)

        pad_left = pad_right = pad_top = pad_bottom = 0
        if single_sample and len(single_sample) > 0:
            ps = single_sample[0].metainfo.get('padding_size', [0, 0, 0, 0])
            pad_left, pad_right, pad_top, pad_bottom = ps

        if pad_bottom > 0 or pad_right > 0:
            H, W = rgb_vis.shape[:2]
            rgb_vis = rgb_vis[pad_top:H - pad_bottom, pad_left:W - pad_right]
            t_vis = t_vis[pad_top:H - pad_bottom, pad_left:W - pad_right]
            if label_seg.shape[0] == H and label_seg.shape[1] == W:
                label_seg = label_seg[pad_top:H - pad_bottom,
                                      pad_left:W - pad_right]
            if pred_seg.shape[0] == H and pred_seg.shape[1] == W:
                pred_seg = pred_seg[pad_top:H - pad_bottom,
                                    pad_left:W - pad_right]

        label_vis, pred_vis = self._render_seg(label_seg, pred_seg, palette)

        feat_rows, q_grid, deg_feat_rows = self._build_feat_vis(
            model_type, feats, runner, rgb_vis=rgb_vis, t_vis=t_vis,
            img_h=proc_inputs.shape[-2], img_w=proc_inputs.shape[-1])

        col_headers = None
        row_labels = None
        decoder_preds = None
        aspect_ratio = 1.0

        if model_type in ('v12d_quality_disentangle',
                          'v12_nodeg_quality_disentangle',
                          'v12_disentangle_only'):
            col_headers = ['zc_rgb', 'zc_t', 'zp_rgb', 'zp_t',
                           'zc_fused', 'rgb_enh', 't_enh', 'final']
            num_stages = len(feats.get('zc_rgb', feats.get('final_fused', [])))
            row_labels = []
            for s in range(num_stages):
                row_labels.append(f'Stage {s} (clean)')
            if model_type in ('v12d_quality_disentangle',
                              'v12_nodeg_quality_disentangle'):
                for s in range(num_stages):
                    row_labels.append(f'Stage {s} (degraded)')

            decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner)

            img_h, img_w = proc_inputs.shape[-2], proc_inputs.shape[-1]
            aspect_ratio = img_w / max(img_h, 1)

        if model_type == 'ab_baseline':
            col_headers = ['rgb_feat', 't_feat', 'fused']
            num_stages = len(feats.get('x_rgb', feats.get('fused', [])))
            row_labels = [f'Stage {s}' for s in range(num_stages)]
            aspect_ratio = proc_inputs.shape[-1] / max(proc_inputs.shape[-2], 1)

        if model_type == 'mask2former_rgbt_add':
            col_headers = ['rgb_feat', 't_feat', 'fused']
            num_stages = len(feats.get('zc_rgb', feats.get('fused', [])))
            row_labels = [f'Stage {s}' for s in range(num_stages)]
            aspect_ratio = proc_inputs.shape[-1] / max(proc_inputs.shape[-2], 1)

        if model_type == 'ab_v9':
            col_headers = ['zc_rgb', 'zc_t', 'zc_fused',
                           'zp_rgb', 'zp_t',
                           'rgb_pf', 't_pf', 'final']
            num_stages = len(feats.get('zc_rgb', feats.get('final_fused', [])))
            row_labels = [f'Stage {s} (clean)' for s in range(num_stages)]
            row_labels += [f'Stage {s} (degraded)' for s in range(num_stages)]

            decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner)

            img_h, img_w = proc_inputs.shape[-2], proc_inputs.shape[-1]
            aspect_ratio = img_w / max(img_h, 1)

        elif model_type in ('ab_v1', 'ab_v2', 'ab_v3', 'ab_v4', 'ab_v5', 'ab_v6', 'ab_v7', 'ab_v8'):
            col_headers = ['zc_rgb', 'zc_t', 'zc_fused',
                           'zp_rgb', 'zp_t',
                           'rgb_pf', 't_pf', 'final']
            num_stages = len(feats.get('zc_rgb', feats.get('final_fused', [])))
            row_labels = [f'Stage {s}' for s in range(num_stages)]
            if model_type in ('ab_v3', 'ab_v4', 'ab_v5', 'ab_v6', 'ab_v7', 'ab_v8'):
                row_labels += [f'Stage {s}(deg)' for s in range(num_stages)]

            decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner)

            img_h, img_w = proc_inputs.shape[-2], proc_inputs.shape[-1]
            aspect_ratio = img_w / max(img_h, 1)

        deg_decoder_preds = None
        if model_type in ('v12d_quality_disentangle',
                          'v12_nodeg_quality_disentangle',
                          'ab_v9') and 'deg_rgb_img' in feats:
            deg_rgb_vis, deg_t_vis = self._render_degraded_images(
                feats, model, raw_rgb=rgb_vis, raw_t=t_vis)

            deg_rgb_t = feats['deg_rgb_img']
            deg_t_t = feats['deg_t_img']
            deg_input = torch.cat([deg_rgb_t, deg_t_t], dim=1)
            deg_pred_seg = self._get_prediction(
                model, deg_input, single_sample, runner)
            if pad_bottom > 0 or pad_right > 0:
                dH, dW = deg_rgb_vis.shape[:2]
                if dH == H and dW == W:
                    deg_rgb_vis = deg_rgb_vis[pad_top:dH - pad_bottom,
                                              pad_left:dW - pad_right]
                    deg_t_vis = deg_t_vis[pad_top:dH - pad_bottom,
                                          pad_left:dW - pad_right]
                dpH, dpW = deg_pred_seg.shape[:2]
                if dpH == H and dpW == W:
                    deg_pred_seg = deg_pred_seg[pad_top:dpH - pad_bottom,
                                                pad_left:dpW - pad_right]
            _, deg_pred_vis = self._render_seg(label_seg, deg_pred_seg, palette)

            deg_decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner, prefix='deg')

            deg_type_rgb = feats.get('deg_type_rgb', 'none')
            deg_type_t = feats.get('deg_type_t', 'none')
            label_parts = []
            if deg_type_rgb != 'none':
                label_parts.append(f'RGB:{deg_type_rgb}')
            if deg_type_t != 'none':
                label_parts.append(f'T:{deg_type_t}')
            deg_label = ' | '.join(label_parts) if label_parts else 'clean'

            title = f'Epoch {epoch} | Sample {b_idx} | Deg: {deg_label}'
            if loss_str:
                title += f' | {loss_str}'
            if model_type == 'ab_v9':
                grid = _compose_v9_ablation_vis(
                    rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                    self.short_side, title=title,
                    col_headers=col_headers, row_labels=row_labels,
                    decoder_preds=decoder_preds, aspect_ratio=aspect_ratio,
                    deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                    deg_feat_rows=deg_feat_rows,
                    deg_decoder_preds=deg_decoder_preds,
                    deg_pred_vis=deg_pred_vis)
            else:
                grid = _compose_vis(
                    rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                    self.short_side, title=title,
                    deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                    deg_pred_vis=deg_pred_vis,
                    col_headers=col_headers, row_labels=row_labels,
                    decoder_preds=decoder_preds, aspect_ratio=aspect_ratio,
                    deg_decoder_preds=deg_decoder_preds)
        elif model_type == 'ab_baseline':
            title = f'Epoch {epoch} | Sample {b_idx}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=decoder_preds, aspect_ratio=aspect_ratio)
        elif model_type in ('ab_v1', 'ab_v2'):
            ab_decoder_preds = {}
            if decoder_preds is not None:
                col_remap = {4: 2, 5: 5, 6: 6, 7: 7}
                for k, v in decoder_preds.items():
                    ab_decoder_preds[col_remap.get(k, k)] = v
            title = f'Epoch {epoch} | Sample {b_idx}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=ab_decoder_preds, aspect_ratio=aspect_ratio)
        elif model_type == 'ab_v3':
            deg_rgb_vis, deg_t_vis = self._render_degraded_images(
                feats, model, raw_rgb=rgb_vis, raw_t=t_vis)

            deg_rgb_t = feats['deg_rgb_img']
            deg_t_t = feats['deg_t_img']
            deg_input = torch.cat([deg_rgb_t, deg_t_t], dim=1)
            deg_pred_seg = self._get_prediction(
                model, deg_input, single_sample, runner)
            if pad_bottom > 0 or pad_right > 0:
                dH, dW = deg_rgb_vis.shape[:2]
                if dH == H and dW == W:
                    deg_rgb_vis = deg_rgb_vis[pad_top:dH - pad_bottom,
                                              pad_left:dW - pad_right]
                    deg_t_vis = deg_t_vis[pad_top:dH - pad_bottom,
                                          pad_left:dW - pad_right]
                dpH, dpW = deg_pred_seg.shape[:2]
                if dpH == H and dpW == W:
                    deg_pred_seg = deg_pred_seg[pad_top:dpH - pad_bottom,
                                                pad_left:dpW - pad_right]
            _, deg_pred_vis = self._render_seg(label_seg, deg_pred_seg, palette)

            deg_decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner, prefix='deg')

            ab_decoder_preds = {}
            if decoder_preds is not None:
                col_remap = {4: 2, 5: 5, 6: 6, 7: 7}
                for k, v in decoder_preds.items():
                    ab_decoder_preds[col_remap.get(k, k)] = v
            ab_deg_decoder_preds = {}
            if deg_decoder_preds is not None:
                for k, v in deg_decoder_preds.items():
                    ab_deg_decoder_preds[col_remap.get(k, k)] = v

            deg_type_rgb = feats.get('deg_type_rgb', 'none')
            deg_type_t = feats.get('deg_type_t', 'none')
            label_parts = []
            if deg_type_rgb != 'none':
                label_parts.append(f'RGB:{deg_type_rgb}')
            if deg_type_t != 'none':
                label_parts.append(f'T:{deg_type_t}')
            deg_label = ' | '.join(label_parts) if label_parts else 'clean'

            title = f'Epoch {epoch} | Sample {b_idx} | Deg: {deg_label}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=ab_decoder_preds, aspect_ratio=aspect_ratio,
                deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                deg_feat_rows=deg_feat_rows,
                deg_decoder_preds=ab_deg_decoder_preds,
                deg_pred_vis=deg_pred_vis)
        elif model_type == 'ab_v4':
            deg_rgb_vis, deg_t_vis = self._render_degraded_images(
                feats, model, raw_rgb=rgb_vis, raw_t=t_vis)

            deg_rgb_t = feats['deg_rgb_img']
            deg_t_t = feats['deg_t_img']
            deg_input = torch.cat([deg_rgb_t, deg_t_t], dim=1)
            deg_pred_seg = self._get_prediction(
                model, deg_input, single_sample, runner)
            if pad_bottom > 0 or pad_right > 0:
                dH, dW = deg_rgb_vis.shape[:2]
                if dH == H and dW == W:
                    deg_rgb_vis = deg_rgb_vis[pad_top:dH - pad_bottom,
                                              pad_left:dW - pad_right]
                    deg_t_vis = deg_t_vis[pad_top:dH - pad_bottom,
                                          pad_left:dW - pad_right]
                dpH, dpW = deg_pred_seg.shape[:2]
                if dpH == H and dpW == W:
                    deg_pred_seg = deg_pred_seg[pad_top:dpH - pad_bottom,
                                                pad_left:dpW - pad_right]
            _, deg_pred_vis = self._render_seg(label_seg, deg_pred_seg, palette)

            deg_decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner, prefix='deg')

            ab_decoder_preds = {}
            if decoder_preds is not None:
                col_remap = {4: 2, 5: 5, 6: 6, 7: 7}
                for k, v in decoder_preds.items():
                    ab_decoder_preds[col_remap.get(k, k)] = v
            ab_deg_decoder_preds = {}
            if deg_decoder_preds is not None:
                for k, v in deg_decoder_preds.items():
                    ab_deg_decoder_preds[col_remap.get(k, k)] = v

            deg_type_rgb = feats.get('deg_type_rgb', 'none')
            deg_type_t = feats.get('deg_type_t', 'none')
            label_parts = []
            if deg_type_rgb != 'none':
                label_parts.append(f'RGB:{deg_type_rgb}')
            if deg_type_t != 'none':
                label_parts.append(f'T:{deg_type_t}')
            deg_label = ' | '.join(label_parts) if label_parts else 'clean'

            title = f'Epoch {epoch} | Sample {b_idx} | Deg: {deg_label}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=ab_decoder_preds, aspect_ratio=aspect_ratio,
                deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                deg_feat_rows=deg_feat_rows,
                deg_decoder_preds=ab_deg_decoder_preds,
                deg_pred_vis=deg_pred_vis)
        elif model_type == 'ab_v9':
            deg_rgb_vis, deg_t_vis = self._render_degraded_images_raw(
                feats, model)

            deg_rgb_t = feats['deg_rgb_img']
            deg_t_t = feats['deg_t_img']
            deg_input = torch.cat([deg_rgb_t, deg_t_t], dim=1)
            deg_pred_seg = self._get_prediction(
                model, deg_input, single_sample, runner)
            if pad_bottom > 0 or pad_right > 0:
                dH, dW = deg_rgb_vis.shape[:2]
                if dH == H and dW == W:
                    deg_rgb_vis = deg_rgb_vis[pad_top:dH - pad_bottom,
                                              pad_left:dW - pad_right]
                    deg_t_vis = deg_t_vis[pad_top:dH - pad_bottom,
                                          pad_left:dW - pad_right]
                dpH, dpW = deg_pred_seg.shape[:2]
                if dpH == H and dpW == W:
                    deg_pred_seg = deg_pred_seg[pad_top:dpH - pad_bottom,
                                                pad_left:dpW - pad_right]
            _, deg_pred_vis = self._render_seg(label_seg, deg_pred_seg, palette)

            deg_decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner, prefix='deg')

            ab_decoder_preds = {}
            if decoder_preds is not None:
                col_remap = {4: 2, 5: 5, 6: 6, 7: 7}
                for k, v in decoder_preds.items():
                    ab_decoder_preds[col_remap.get(k, k)] = v
            ab_deg_decoder_preds = {}
            if deg_decoder_preds is not None:
                for k, v in deg_decoder_preds.items():
                    ab_deg_decoder_preds[col_remap.get(k, k)] = v

            title = f'Epoch {epoch} | Sample {b_idx}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_v9_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=ab_decoder_preds, aspect_ratio=aspect_ratio,
                deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                deg_feat_rows=deg_feat_rows,
                deg_decoder_preds=ab_deg_decoder_preds,
                deg_pred_vis=deg_pred_vis)

            q_grid = self._compose_v9_ablation_quality_vis(
                 self._v9_q_info, deg_q_info=self._v9_deg_q_info,
                 rgb_vis=rgb_vis, t_vis=t_vis,
                 deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                 aspect_ratio=aspect_ratio)
        elif model_type in ('ab_v5', 'ab_v6', 'ab_v7', 'ab_v8'):
            deg_rgb_vis, deg_t_vis = self._render_degraded_images(
                feats, model, raw_rgb=rgb_vis, raw_t=t_vis)

            deg_rgb_t = feats['deg_rgb_img']
            deg_t_t = feats['deg_t_img']
            deg_input = torch.cat([deg_rgb_t, deg_t_t], dim=1)
            deg_pred_seg = self._get_prediction(
                model, deg_input, single_sample, runner)
            if pad_bottom > 0 or pad_right > 0:
                dH, dW = deg_rgb_vis.shape[:2]
                if dH == H and dW == W:
                    deg_rgb_vis = deg_rgb_vis[pad_top:dH - pad_bottom,
                                              pad_left:dW - pad_right]
                    deg_t_vis = deg_t_vis[pad_top:dH - pad_bottom,
                                          pad_left:dW - pad_right]
                dpH, dpW = deg_pred_seg.shape[:2]
                if dpH == H and dpW == W:
                    deg_pred_seg = deg_pred_seg[pad_top:dpH - pad_bottom,
                                                pad_left:dpW - pad_right]
            _, deg_pred_vis = self._render_seg(label_seg, deg_pred_seg, palette)

            deg_decoder_preds = self._get_decoder_predictions(
                model, feats, palette, runner, prefix='deg')

            ab_decoder_preds = {}
            if decoder_preds is not None:
                col_remap = {4: 2, 5: 5, 6: 6, 7: 7}
                for k, v in decoder_preds.items():
                    ab_decoder_preds[col_remap.get(k, k)] = v
            ab_deg_decoder_preds = {}
            if deg_decoder_preds is not None:
                for k, v in deg_decoder_preds.items():
                    ab_deg_decoder_preds[col_remap.get(k, k)] = v

            deg_type_rgb = feats.get('deg_type_rgb', 'none')
            deg_type_t = feats.get('deg_type_t', 'none')
            label_parts = []
            if deg_type_rgb != 'none':
                label_parts.append(f'RGB:{deg_type_rgb}')
            if deg_type_t != 'none':
                label_parts.append(f'T:{deg_type_t}')
            deg_label = ' | '.join(label_parts) if label_parts else 'clean'

            title = f'Epoch {epoch} | Sample {b_idx} | Deg: {deg_label}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_ablation_vis(
                rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                self.short_side, title=title,
                col_headers=col_headers, row_labels=row_labels,
                decoder_preds=ab_decoder_preds, aspect_ratio=aspect_ratio,
                deg_rgb_vis=deg_rgb_vis, deg_t_vis=deg_t_vis,
                deg_feat_rows=deg_feat_rows,
                deg_decoder_preds=ab_deg_decoder_preds,
                deg_pred_vis=deg_pred_vis)
        else:
            title = f'Epoch {epoch} | Sample {b_idx}'
            if loss_str:
                title += f' | {loss_str}'
            grid = _compose_vis(rgb_vis, t_vis, feat_rows, label_vis, pred_vis,
                                self.short_side, title=title,
                                col_headers=col_headers, row_labels=row_labels,
                                decoder_preds=decoder_preds,
                                aspect_ratio=aspect_ratio)
        return grid, q_grid

    def _save_results(self, runner, sample_grids, quality_grids, epoch,
                      sample_gap=0, quality_gap=0):
        if sample_grids:
            composite = _hstack_samples(sample_grids, gap=sample_gap)
            img_tensor = composite.transpose(2, 0, 1).astype(np.float32) / 255.0
            png_path = osp.join(self._vis_dir, f'epoch{epoch}_all_samples.png')
            cv2.imwrite(png_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            runner.logger.info(f'Saved: {png_path}')

            if self._writer is not None:
                self._writer.add_image(
                    f'epoch{epoch}/all_samples', img_tensor, epoch)

            if hasattr(runner, 'visualizer') and runner.visualizer is not None:
                try:
                    runner.visualizer.add_image(
                        'train_vis/all_samples',
                        img_tensor,
                        step=epoch)
                except Exception as e:
                    runner.logger.warning(
                        f'Failed to add image to runner visualizer: {e}')

        if quality_grids:
            q_composite = _hstack_samples(quality_grids, gap=quality_gap)
            q_img_tensor = q_composite.transpose(2, 0, 1).astype(np.float32) / 255.0
            png_q = osp.join(self._vis_dir, f'epoch{epoch}_all_quality.png')
            cv2.imwrite(png_q, cv2.cvtColor(q_composite, cv2.COLOR_RGB2BGR))

            if self._writer is not None:
                self._writer.add_image(
                    f'epoch{epoch}/all_quality', q_img_tensor, epoch)

            if hasattr(runner, 'visualizer') and runner.visualizer is not None:
                try:
                    runner.visualizer.add_image(
                        'train_vis/all_quality',
                        q_img_tensor,
                        step=epoch)
                except Exception as e:
                    runner.logger.warning(
                        f'Failed to add quality image to runner visualizer: {e}')

        if self._writer is not None:
            self._writer.flush()

    def _create_v9_ablation_quality_vis(self, feats, img_h=None, img_w=None):
        q_rgb_maps = feats['q_rgb_maps']
        q_t_maps = feats['q_t_maps']
        cum_D_rgb_list = feats.get('cum_D_rgb', feats['D_rgb'])
        cum_D_t_list = feats.get('cum_D_t', feats['D_t'])
        q_rgb_priv_list = feats['q_rgb_priv']
        q_t_priv_list = feats['q_t_priv']
        cum_D_rgb_priv_list = feats.get('cum_D_rgb_priv', feats['D_rgb_priv'])
        cum_D_t_priv_list = feats.get('cum_D_t_priv', feats['D_t_priv'])

        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        def _safe_q_np(q_tensor, h, w):
            if q_tensor is None:
                return np.ones((h, w), dtype=np.float32)
            return q_tensor[0, 0].detach().cpu().numpy()

        def _safe_D_np(D_tensor, h, w):
            if D_tensor is None:
                return np.ones((h, w), dtype=np.float32)
            return D_tensor[0, 0].detach().cpu().numpy()

        stages = []
        for i in range(len(q_rgb_maps)):
            q_rgb_np = _safe_q_np(q_rgb_maps[i], h, w)
            q_t_np = _safe_q_np(q_t_maps[i], h, w)
            q_rgb_priv_np = _safe_q_np(q_rgb_priv_list[i], h, w)
            q_t_priv_np = _safe_q_np(q_t_priv_list[i], h, w)

            rgb_hm = _quality_to_red_blue(q_rgb_np, h, w)
            t_hm = _quality_to_red_blue(q_t_np, h, w)
            rgb_priv_hm = _quality_to_red_blue(q_rgb_priv_np, h, w)
            t_priv_hm = _quality_to_red_blue(q_t_priv_np, h, w)

            D_rgb_np = _safe_D_np(cum_D_rgb_list[i], h, w)
            D_t_np = _safe_D_np(cum_D_t_list[i], h, w)
            D_rgb_priv_np = _safe_D_np(cum_D_rgb_priv_list[i], h, w)
            D_t_priv_np = _safe_D_np(cum_D_t_priv_list[i], h, w)

            D_rgb_bw = _threshold_to_bw(
                (D_rgb_np >= self.mask_threshold).astype(np.float32), h, w)
            D_t_bw = _threshold_to_bw(
                (D_t_np >= self.mask_threshold).astype(np.float32), h, w)
            D_rgb_priv_bw = _threshold_to_bw(
                (D_rgb_priv_np >= self.mask_threshold).astype(np.float32), h, w)
            D_t_priv_bw = _threshold_to_bw(
                (D_t_priv_np >= self.mask_threshold).astype(np.float32), h, w)

            stages.append(dict(
                rgb_hm=rgb_hm, t_hm=t_hm,
                rgb_priv_hm=rgb_priv_hm, t_priv_hm=t_priv_hm,
                D_rgb_bw=D_rgb_bw, D_t_bw=D_t_bw,
                D_rgb_priv_bw=D_rgb_priv_bw, D_t_priv_bw=D_t_priv_bw,
            ))

        return dict(stages=stages)

    def _create_v9_ablation_deg_quality_vis(self, feats, img_h=None,
                                             img_w=None):
        q_rgb_deg = feats['q_rgb_deg']
        q_t_deg = feats['q_t_deg']
        cum_D_rgb_deg = feats.get('cum_D_rgb_deg', feats['D_rgb_deg'])
        cum_D_t_deg = feats.get('cum_D_t_deg', feats['D_t_deg'])
        q_rgb_priv_deg = feats['q_rgb_priv_deg']
        q_t_priv_deg = feats['q_t_priv_deg']
        cum_D_rgb_priv_deg = feats.get('cum_D_rgb_priv_deg', feats['D_rgb_priv_deg'])
        cum_D_t_priv_deg = feats.get('cum_D_t_priv_deg', feats['D_t_priv_deg'])

        if img_h is not None and img_w is not None:
            h, w = img_h, img_w
        else:
            h, w = feats['zc_rgb'][0].shape[-2:]

        def _safe_q_np(q_tensor, h, w):
            if q_tensor is None:
                return np.ones((h, w), dtype=np.float32)
            return q_tensor[0, 0].detach().cpu().numpy()

        def _safe_D_np(D_tensor, h, w):
            if D_tensor is None:
                return np.ones((h, w), dtype=np.float32)
            return D_tensor[0, 0].detach().cpu().numpy()

        stages = []
        for i in range(len(q_rgb_deg)):
            q_rgb_np = _safe_q_np(q_rgb_deg[i], h, w)
            q_t_np = _safe_q_np(q_t_deg[i], h, w)
            q_rgb_priv_np = _safe_q_np(q_rgb_priv_deg[i], h, w)
            q_t_priv_np = _safe_q_np(q_t_priv_deg[i], h, w)

            rgb_hm = _quality_to_red_blue(q_rgb_np, h, w)
            t_hm = _quality_to_red_blue(q_t_np, h, w)
            rgb_priv_hm = _quality_to_red_blue(q_rgb_priv_np, h, w)
            t_priv_hm = _quality_to_red_blue(q_t_priv_np, h, w)

            D_rgb_np = _safe_D_np(cum_D_rgb_deg[i], h, w)
            D_t_np = _safe_D_np(cum_D_t_deg[i], h, w)
            D_rgb_priv_np = _safe_D_np(cum_D_rgb_priv_deg[i], h, w)
            D_t_priv_np = _safe_D_np(cum_D_t_priv_deg[i], h, w)

            D_rgb_bw = _threshold_to_bw(
                (D_rgb_np >= self.mask_threshold).astype(np.float32), h, w)
            D_t_bw = _threshold_to_bw(
                (D_t_np >= self.mask_threshold).astype(np.float32), h, w)
            D_rgb_priv_bw = _threshold_to_bw(
                (D_rgb_priv_np >= self.mask_threshold).astype(np.float32), h, w)
            D_t_priv_bw = _threshold_to_bw(
                (D_t_priv_np >= self.mask_threshold).astype(np.float32), h, w)

            stages.append(dict(
                rgb_hm=rgb_hm, t_hm=t_hm,
                rgb_priv_hm=rgb_priv_hm, t_priv_hm=t_priv_hm,
                D_rgb_bw=D_rgb_bw, D_t_bw=D_t_bw,
                D_rgb_priv_bw=D_rgb_priv_bw, D_t_priv_bw=D_t_priv_bw,
            ))

        return dict(stages=stages)

    def _compose_v9_ablation_quality_vis(self, q_info, deg_q_info=None,
                                          rgb_vis=None, t_vis=None,
                                          deg_rgb_vis=None, deg_t_vis=None,
                                          aspect_ratio=1.0):
        cell_h, cell_w = _compute_cell_size(self.short_side, aspect_ratio)
        num_cols = 4
        inner_gap = 5
        block_gap = 10
        empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        h_gap_strip = np.zeros((cell_h, inner_gap, 3), dtype=np.uint8)
        total_w = num_cols * cell_w + (num_cols - 1) * inner_gap

        def _gapped_row(cells):
            processed = []
            for c in cells:
                if c is None:
                    processed.append(empty.copy())
                else:
                    processed.append(_make_cell(c, cell_h, cell_w, self.short_side))
            while len(processed) < num_cols:
                processed.append(empty.copy())
            result = processed[0]
            for cell in processed[1:]:
                result = np.concatenate([result, h_gap_strip, cell], axis=1)
            return result

        def _build_quality_block(stages_data, rgb_base=None, t_base=None):
            block_rows = []
            block_rows.append(_gapped_row([rgb_base, t_base, None, None]))
            for stage in stages_data:
                q_row = _gapped_row([
                    stage['rgb_hm'], stage['t_hm'],
                    stage['rgb_priv_hm'], stage['t_priv_hm']])
                block_rows.append(q_row)
                d_row = _gapped_row([
                    stage['D_rgb_bw'], stage['D_t_bw'],
                    stage['D_rgb_priv_bw'], stage['D_t_priv_bw']])
                block_rows.append(d_row)
            return block_rows

        all_rows = []
        v_gap = np.zeros((inner_gap, total_w, 3), dtype=np.uint8)

        clean_rows = _build_quality_block(
            q_info['stages'], rgb_base=rgb_vis, t_base=t_vis)
        for i, row in enumerate(clean_rows):
            if i > 0:
                all_rows.append(v_gap)
            all_rows.append(row)

        if deg_q_info is not None:
            b_gap = np.zeros((block_gap, total_w, 3), dtype=np.uint8)
            all_rows.append(b_gap)
            deg_rows = _build_quality_block(
                deg_q_info['stages'], rgb_base=deg_rgb_vis, t_base=deg_t_vis)
            for i, row in enumerate(deg_rows):
                if i > 0:
                    all_rows.append(v_gap)
                all_rows.append(row)

        if not all_rows:
            return np.zeros((cell_h, total_w, 3), dtype=np.uint8)

        return np.concatenate(all_rows, axis=0)

    def _log_lora_params(self, runner, model, epoch):
        if not hasattr(model, 'backbone') or epoch != 0:
            return
        lora_params = [(n, p.data.abs().mean().item())
                       for n, p in model.backbone.named_parameters()
                       if 'lora' in n.lower()]
        if lora_params:
            runner.logger.info(f'LoRA parameters (epoch {epoch}):')
            for name, val in lora_params:
                runner.logger.info(f'  {name}: {val:.6f}')

    def after_run(self, runner):
        if self._writer is not None:
            self._writer.close()
