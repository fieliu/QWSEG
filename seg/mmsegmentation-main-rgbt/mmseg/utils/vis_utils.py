import cv2
import numpy as np
import torch


def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def _apply_cmap(img, colormap=cv2.COLORMAP_JET):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = _to_uint8(img)
    bgr = cv2.applyColorMap(img, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_short_side(img, short_side=250):
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _pad_to_size(img, target_h, target_w, pad_value=0):
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    if img.ndim == 2:
        pad = np.full((target_h, target_w), pad_value, dtype=img.dtype)
    else:
        pad = np.full((target_h, target_w, img.shape[2]), pad_value,
                       dtype=img.dtype)
    pad[:min(h, target_h), :min(w, target_w)] = img[:target_h, :target_w]
    return pad


def _quality_to_red_blue(q_np, h, w):
    q_resized = cv2.resize(q_np, (w, h), interpolation=cv2.INTER_LINEAR)
    q_clipped = np.clip(q_resized, 0.0, 1.0)
    r = (q_clipped * 255).astype(np.uint8)
    b = ((1.0 - q_clipped) * 255).astype(np.uint8)
    g = np.zeros_like(r)
    return np.stack([r, g, b], axis=-1)


def _threshold_to_bw(mask_np, h, w):
    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_clipped = np.clip(mask_resized, 0.0, 1.0)
    v = (mask_clipped * 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


def _make_cell(img, cell_h, cell_w, short_side=250):
    h, w = img.shape[:2]
    if h < w:
        scale = short_side / h
    else:
        scale = short_side / w
    new_h = int(h * scale + 0.5)
    new_w = int(w * scale + 0.5)
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    img = _pad_to_size(img, cell_h, cell_w)
    return img


def _feat_top3_rgb(feature_map, sample_idx=0):
    feat = feature_map[sample_idx].cpu().float()
    mean_activation = feat.abs().mean(dim=(1, 2))
    k = min(3, feat.shape[0])
    topk_indices = mean_activation.topk(k).indices
    channels = []
    for c in topk_indices:
        ch = feat[c].detach().numpy()
        vmin, vmax = ch.min(), ch.max()
        if vmax - vmin < 1e-8:
            ch = np.zeros_like(ch)
        else:
            ch = (ch - vmin) / (vmax - vmin)
        channels.append(ch)
    while len(channels) < 3:
        channels.append(np.zeros_like(channels[0]))
    rgb = np.stack(channels[:3], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _build_row(cells, cell_h, cell_w, short_side, num_cols):
    empty = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    row_cells = []
    for c in cells:
        if c is None:
            row_cells.append(empty.copy())
        else:
            row_cells.append(_make_cell(c, cell_h, cell_w, short_side))
    while len(row_cells) < num_cols:
        row_cells.append(empty.copy())
    return np.concatenate(row_cells, axis=1)


def _get_backbone_input_size(model):
    backbone = model.backbone if hasattr(model, 'backbone') else model
    if hasattr(backbone, 'img_size'):
        img_size = backbone.img_size
        if isinstance(img_size, (list, tuple)):
            return img_size[0], img_size[1]
        return img_size, img_size
    if hasattr(backbone, 'patch_embed'):
        pe = backbone.patch_embed
        if hasattr(pe, 'image_size'):
            s = pe.image_size
            if isinstance(s, (list, tuple)):
                return s[0], s[1]
            return s, s
    return 768, 768


def _apply_palette(label, palette):
    if label.ndim != 2:
        label = label.squeeze()
    h, w = label.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(palette):
        mask = label == cls_id
        color_map[mask] = color
    return color_map


def _unwrap_model(model):
    while hasattr(model, 'module'):
        model = model.module
    return model


def detect_model_type(model):
    type_name = type(model).__name__
    if type_name == 'EncoderDecoder':
        return 'original'
    if type_name == 'RGBTv1Baseline':
        return 'v1'
    if type_name == 'RGBTv2Disentangle':
        return 'v2'
    if type_name == 'RGBTv3Degradation':
        return 'v3'
    if type_name == 'RGBTv4QualityPruning':
        return 'v4'
    if type_name == 'RGBTv5QualityJoint':
        return 'v5'
    if type_name == 'MiTMulV6Baseline':
        return 'v6_baseline'
    if type_name == 'MiTMulV6Disentangle':
        return 'v6_disentangle'
    if type_name == 'MiTMulV7Degradation':
        return 'v7_degradation'
    if type_name == 'MiTMulV7DegradationFull':
        return 'v7_degradation_full'
    if type_name == 'MiTMulV8QualityPyramid':
        return 'v8_quality_pyramid'
    if type_name == 'MiTMulV9QualityGated':
        return 'v9_quality_gated'
    if type_name == 'MiTMulV6AddFusion':
        return 'v6_add_fusion'
    if type_name == 'MiTMulV7QualityAdaptive':
        return 'v7_quality_adaptive'
    if type_name == 'MiTMulV10QualityEmbed':
        return 'v10_quality_embed'
    if type_name == 'SwinMulV11MaskMAE':
        return 'v11_mask_mae'
    if type_name == 'SwinMulV6Mask2Former':
        return 'v6_mask2former'
    if type_name == 'SwinMulV12DQualityDisentangle':
        return 'v12d_quality_disentangle'
    if type_name == 'SwinMulV12QualityDisentangleNoDeg':
        return 'v12_nodeg_quality_disentangle'
    if type_name == 'SwinMulV12DisentangleOnly':
        return 'v12_disentangle_only'
    if type_name == 'MiTMulV12DQualityDisentangle':
        return 'v12d_quality_disentangle'
    if type_name == 'MiTMulV12DisentangleOnly':
        return 'v12_disentangle_only'
    if type_name == 'MiTMulV12QualityDisentangleNoDeg':
        return 'v12_nodeg_quality_disentangle'
    if type_name == 'SwinBaseline':
        return 'swin_baseline'
    if type_name == 'Mask2FormerRGBTAdd':
        return 'mask2former_rgbt_add'
    if type_name == 'MiTMulABBaseline':
        return 'ab_baseline'
    if type_name == 'MiTMulABV1':
        return 'ab_v1'
    if type_name == 'MiTMulABV2':
        return 'ab_v2'
    if type_name == 'MiTMulABV3':
        return 'ab_v3'
    if type_name == 'MiTMulABV3Replace':
        return 'ab_v3'
    if type_name == 'MiTMulABV4':
        return 'ab_v4'
    if type_name == 'MiTMulABV5':
        return 'ab_v5'
    if type_name == 'MiTMulABV6':
        return 'ab_v6'
    if type_name == 'MiTMulABV7':
        return 'ab_v7'
    if type_name == 'MiTMulABV8':
        return 'ab_v8'
    if type_name == 'MiTMulABV9':
        return 'ab_v9'
    if type_name == 'QualityGatedMiTMamba':
        return 'mit_quality_mamba'
    if type_name == 'QualityGatedSwinMask2Former':
        return 'swin_quality_mask2former'
    return 'unknown'


def extract_features_original(model, inputs):
    input_rgb = inputs[:, :3, :, :]
    input_ir = inputs[:, 3:, :, :]
    input_rgbt = torch.cat([input_rgb, input_ir], dim=1)
    b, c, h, w = input_rgbt.shape
    input_rgbt = input_rgbt.view(b * 2, c // 2, h, w)
    target_h, target_w = _get_backbone_input_size(model)
    if h != target_h or w != target_w:
        input_rgbt = torch.nn.functional.interpolate(
            input_rgbt, size=(target_h, target_w), mode='bilinear',
            align_corners=False)
    with torch.no_grad():
        x_rgbt = model.backbone(input_rgbt)
    rgb_feats = []
    t_feats = []
    for feat in x_rgbt:
        fb, fc, fh, fw = feat.shape
        B = fb // 2
        rgb_feats.append(feat[:B])
        t_feats.append(feat[B:])
    return [rgb_feats, t_feats]


def extract_features_v1(model, inputs):
    input_rgb = inputs[:, :3, :, :]
    input_ir = inputs[:, 3:, :, :]
    input_rgbt = torch.cat([input_rgb, input_ir], dim=1)
    b, c, h, w = input_rgbt.shape
    input_rgbt = input_rgbt.view(b * 2, c // 2, h, w)
    target_h, target_w = _get_backbone_input_size(model)
    if h != target_h or w != target_w:
        input_rgbt = torch.nn.functional.interpolate(
            input_rgbt, size=(target_h, target_w), mode='bilinear',
            align_corners=False)
    with torch.no_grad():
        x_rgbt = model.backbone(input_rgbt)
    rgb_feats = []
    t_feats = []
    for feat in x_rgbt:
        fb, fc, fh, fw = feat.shape
        B = fb // 2
        rgb_feats.append(feat[:B])
        t_feats.append(feat[B:])
    return [rgb_feats, t_feats]


def extract_features_v2(model, inputs):
    with torch.no_grad():
        feats = model.extract_feat(inputs)
    zc_rgb = feats[0]
    zc_t = feats[1]
    zp_rgb = feats[2]
    zp_t = feats[3]
    return [zc_rgb, zc_t, zp_rgb, zp_t]


def extract_features_v3(model, inputs):
    with torch.no_grad():
        feats = model.extract_feat(inputs)
    return [feats[0], feats[1], feats[2], feats[3]]


def extract_features_v4(model, inputs):
    with torch.no_grad():
        feats = model.extract_feat(inputs)
    return [feats[0], feats[1], feats[2], feats[3]]


def extract_features_v5(model, inputs):
    with torch.no_grad():
        feats = model.extract_feat(inputs)
    return [feats[0], feats[1], feats[2], feats[3]]


def extract_features_v6_baseline(model, inputs):
    input_rgb = inputs[:, :3, :, :]
    input_ir = inputs[:, 3:, :, :]
    input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
    with torch.no_grad():
        fused_feats = model.extract_feat(input_rgbt)
    x_rgbt = model.backbone(input_rgbt)
    B = input_rgbt.shape[0] // 2
    x_rgb_list = [feat[:B] for feat in x_rgbt]
    x_t_list = [feat[B:] for feat in x_rgbt]
    return [x_rgb_list, x_t_list, fused_feats]


def extract_features_v6_disentangle(model, inputs):
    input_rgb = inputs[:, :3, :, :]
    input_ir = inputs[:, 3:, :, :]
    input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
    with torch.no_grad():
        zc_rgb, zc_t, zp_rgb, zp_t, zc_enhanced, fused = model.extract_feat(
            input_rgbt)
    return [zc_rgb, zc_t, zp_rgb, zp_t, zc_enhanced, fused]


def extract_features_v11_mask_mae(model, inputs):
    with torch.no_grad():
        fused_feats = model.extract_feat(inputs)
    x_rgb_list = model._last_rgb_feats
    x_t_list = model._last_t_feats
    return [x_rgb_list, x_t_list, fused_feats]


def extract_features_v6_mask2former(model, inputs):
    input_rgb = inputs[:, :3, :, :]
    input_ir = inputs[:, 3:, :, :]
    input_rgbt = torch.cat([input_rgb, input_ir], dim=0)
    with torch.no_grad():
        fused_feats = model.extract_feat(input_rgbt)
    x_rgb_list = model._last_rgb_feats
    x_t_list = model._last_t_feats
    return [x_rgb_list, x_t_list, fused_feats]


_EXTRACT_FN = {
    'original': extract_features_original,
    'v1': extract_features_v1,
    'v2': extract_features_v2,
    'v3': extract_features_v3,
    'v4': extract_features_v4,
    'v5': extract_features_v5,
    'v6_baseline': extract_features_v6_baseline,
    'v6_disentangle': extract_features_v6_disentangle,
    'v7_degradation': extract_features_v6_disentangle,
    'v11_mask_mae': extract_features_v11_mask_mae,
    'v6_mask2former': extract_features_v6_mask2former,
}


def extract_features(model, inputs):
    model_type = detect_model_type(model)
    fn = _EXTRACT_FN.get(model_type)
    if fn is None:
        return None
    return fn(model, inputs)


def create_sample_vis(rgb_img, t_img, feat_lists, label, pred,
                      short_side=250, palette=None):
    num_feat_cols = len(feat_lists) if feat_lists is not None else 0
    num_stages = len(feat_lists[0]) if feat_lists is not None else 0
    num_cols = max(2, num_feat_cols)

    cell_h = cell_w = short_side

    row1 = _build_row([rgb_img, t_img], cell_h, cell_w, short_side,
                      num_cols)
    rows = [row1]

    if feat_lists is not None:
        for stage in range(num_stages):
            stage_cells = []
            for f_idx in range(num_feat_cols):
                feat_vis = _feat_top3_rgb(feat_lists[f_idx][stage])
                stage_cells.append(feat_vis)
            rows.append(_build_row(stage_cells, cell_h, cell_w, short_side,
                                   num_cols))

    if palette is not None:
        label_vis = _apply_palette(label.astype(np.uint8), palette)
        pred_vis = _apply_palette(pred.astype(np.uint8), palette)
    else:
        label_vis = _apply_cmap(label.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        pred_vis = _apply_cmap(pred.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    rows.append(_build_row([label_vis, pred_vis], cell_h, cell_w, short_side,
                           num_cols))

    return np.concatenate(rows, axis=0)


def create_composite_vis(vis_data_list, short_side=250, palette=None):
    grids = []
    for d in vis_data_list:
        g = create_sample_vis(
            d['rgb'], d['thermal'], d['features'],
            d['label'], d['pred'], short_side, palette=palette)
        grids.append(g)
    max_h = max(g.shape[0] for g in grids)
    padded = []
    for g in grids:
        if g.shape[0] < max_h:
            pad = np.zeros((max_h - g.shape[0], g.shape[1], 3),
                           dtype=np.uint8)
            g = np.concatenate([g, pad], axis=0)
        padded.append(g)
    return np.concatenate(padded, axis=1)
