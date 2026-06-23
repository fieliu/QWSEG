"""像素级跨模态对比损失 (CLIP 风格 + 采样).

用于 teacher (EoMTRGBTFusion) 训练, 让 RGB 和 T 双流特征在语义空间对齐:
  - 正样本: 同一 token 位置的 RGB 和 T 特征
  - 负样本: batch 内不同类 token 的另一模态特征
  - 双向 InfoNCE: RGB→T (行) + T→RGB (列), 共用一个 [K,K] 相似度矩阵
  - 每类采样固定数量 anchor, 保证类别平衡, 避免背景类主导

设计参考:
  - CLIP (Radford et al. 2021): 双塔对比 + 双向 InfoNCE
  - ContrastiveSeg (Zhou et al. ICCV2021): 像素级 + 每类采样 + memory bank
  - ReCo (Liu et al. ICLR2022): 主动稀疏采样 (<5% 像素)
"""
import torch
import torch.nn.functional as F


def _downsample_labels_to_token(labels_hw, token_grid):
    """将 [B, H, W] 像素级标签下采样到 [B, N] token 级标签.

    用最近邻下采样, 保证每个 token 对应一个语义类别.
    token_grid = (h_tok, w_tok), N = h_tok * w_tok.
    """
    B = labels_hw.shape[0]
    h_tok, w_tok = token_grid
    # [B, 1, H, W] -> [B, 1, h_tok, w_tok] -> [B, h_tok, w_tok] -> [B, N]
    lab = F.interpolate(
        labels_hw.unsqueeze(1).float(),
        size=(h_tok, w_tok),
        mode="nearest",
    ).squeeze(1).long()
    return lab.reshape(B, -1)  # [B, N]


def _sample_anchors(z_rgb, z_t, labels, samples_per_class, ignore_index=255):
    """每类采样 anchor, 返回 anchor 的 RGB/T 特征和类别.

    z_rgb, z_t: [B, N, C]
    labels:     [B, N]  token 级语义标签
    返回:
      a_rgb:    [K, C]
      a_t:      [K, C]
      a_labels: [K]
    """
    B, N, C = z_rgb.shape
    classes = labels.unique()
    # 过滤 ignore_index
    classes = classes[(classes != ignore_index) & (classes >= 0)]

    a_rgb_list, a_t_list, a_lab_list = [], [], []
    for b in range(B):
        for c in classes:
            mask = (labels[b] == c)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)  # [num_c]
            if idx.numel() == 0:
                continue
            n_sample = min(samples_per_class, idx.numel())
            perm = torch.randperm(idx.numel(), device=idx.device)[:n_sample]
            sel = idx[perm]
            a_rgb_list.append(z_rgb[b, sel])
            a_t_list.append(z_t[b, sel])
            a_lab_list.append(labels[b, sel])

    if not a_rgb_list:
        return None, None, None
    return torch.cat(a_rgb_list, 0), torch.cat(a_t_list, 0), torch.cat(a_lab_list, 0)


def cross_modal_contrast_loss(z_rgb, z_t, labels,
                               samples_per_class=15, tau=0.1,
                               ignore_index=255):
    """双向像素级跨模态对比损失 (CLIP 风格).

    z_rgb, z_t: [B, N, C]  双流 token 特征 (融合前)
    labels:     [B, N]     token 级语义标签
    返回: 标量损失
    """
    a_rgb, a_t, a_labels = _sample_anchors(
        z_rgb, z_t, labels, samples_per_class, ignore_index)
    if a_rgb is None:
        return z_rgb.new_zeros(())

    K = a_rgb.shape[0]
    a_rgb = F.normalize(a_rgb, dim=-1)
    a_t = F.normalize(a_t, dim=-1)

    # 一个相似度矩阵 [K, K]
    sim = a_rgb @ a_t.t() / tau

    # mask (行列共用)
    pos_mask = torch.eye(K, device=sim.device, dtype=torch.bool)  # 对角线 (同位置)
    label_eq = (a_labels.unsqueeze(0) == a_labels.unsqueeze(1))   # [K,K] 同类
    neg_mask = (~pos_mask) & (~label_eq)  # 不同位置 且 不同类

    # 方向 1 (RGB→T): 按行算 InfoNCE
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    pos_sum = (exp_sim * pos_mask).sum(1)
    den_sum = (exp_sim * (pos_mask | neg_mask)).sum(1)
    loss_rgb2t = -(torch.log(pos_sum / (den_sum + 1e-8))).mean()

    # 方向 2 (T→RGB): 按列算 (= 转置后按行)
    sim_t = sim.t()
    exp_sim_t = torch.exp(sim_t - sim_t.max(dim=1, keepdim=True).values)
    pos_sum_t = (exp_sim_t * pos_mask).sum(1)
    den_sum_t = (exp_sim_t * (pos_mask | neg_mask)).sum(1)
    loss_t2rgb = -(torch.log(pos_sum_t / (den_sum_t + 1e-8))).mean()

    return (loss_rgb2t + loss_t2rgb) / 2
