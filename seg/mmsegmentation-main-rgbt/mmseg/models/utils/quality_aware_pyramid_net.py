import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class QualityStage(nn.Module):

    def __init__(self, in_channels, mid_channels, kernel_size, stride,
                 prev_channels=0):
        super().__init__()
        self.stride = stride
        score_dim = max(mid_channels // 4, 8)

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride,
                      padding=0, bias=False),
            nn.GroupNorm(max(mid_channels // 8, 1), mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(max(mid_channels // 8, 1), mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.has_prev = prev_channels > 0
        if self.has_prev:
            self.prev_downsample = nn.Conv2d(
                prev_channels, prev_channels, 2, stride=2, bias=False)
            self.prev_fuse = nn.Sequential(
                nn.Conv2d(prev_channels + mid_channels, mid_channels, 1,
                          bias=False),
                nn.GroupNorm(max(mid_channels // 8, 1), mid_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.score_mlp = nn.Sequential(
            nn.Conv2d(mid_channels, score_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(score_dim, 1, 1),
        )

        self.log_tau = nn.Parameter(torch.tensor(-2.0))

    def _init_refine_center_bias(self):
        with torch.no_grad():
            w = self.refine[0].weight.data
            C_out, C_in, kH, kW = w.shape
            center_h, center_w = kH // 2, kW // 2
            neighbor_mean = w[:, :, 0, 0].mean()
            w[:, :, center_h, center_w] = neighbor_mean * 4.0
            w[:, :, 0, 0] = neighbor_mean * 0.5
            w[:, :, 0, kW - 1] = neighbor_mean * 0.5
            w[:, :, kH - 1, 0] = neighbor_mean * 0.5
            w[:, :, kH - 1, kW - 1] = neighbor_mean * 0.5
            w[:, :, center_h, 0] = neighbor_mean * 0.75
            w[:, :, center_h, kW - 1] = neighbor_mean * 0.75
            w[:, :, 0, center_w] = neighbor_mean * 0.75
            w[:, :, kH - 1, center_w] = neighbor_mean * 0.75

    def forward(self, x, prev_feat=None):
        B, C, H, W = x.shape
        pad_h = (self.stride - H % self.stride) % self.stride
        pad_w = (self.stride - W % self.stride) % self.stride
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0)
        feat = self.down(x)
        feat = feat + self.refine(feat)

        if self.has_prev and prev_feat is not None:
            if prev_feat.shape[2] >= 2 and prev_feat.shape[3] >= 2:
                prev_down = self.prev_downsample(prev_feat)
            else:
                prev_down = F.adaptive_avg_pool2d(
                    prev_feat, (max(prev_feat.shape[2] // 2, 1),
                                max(prev_feat.shape[3] // 2, 1)))
            if prev_down.shape[2:] != feat.shape[2:]:
                prev_down = F.adaptive_avg_pool2d(
                    prev_down, feat.shape[2:])
            feat = self.prev_fuse(torch.cat([feat, prev_down], dim=1))

        score = self.score_mlp(feat)
        quality = torch.sigmoid(score)
        return feat, quality


@MODELS.register_module()
class QualityAwarePyramidNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_stages=4,
                 in_channels_rgb=None,
                 in_channels_t=None,
                 embed_dims=None,
                 out_indices=None):
        super().__init__()
        self.num_stages = num_stages

        if in_channels_rgb is not None:
            in_channels = in_channels_rgb
        if embed_dims is not None:
            mid_channels = embed_dims

        self.rgb_stages = nn.ModuleList()
        self.t_stages = nn.ModuleList()

        for i in range(num_stages):
            if i == 0:
                k, s, in_ch, prev_ch = 4, 4, in_channels, 0
            else:
                k, s, in_ch, prev_ch = 2, 2, mid_channels, mid_channels
            self.rgb_stages.append(
                QualityStage(in_ch, mid_channels, k, s, prev_channels=prev_ch))

        t_in = in_channels_t if in_channels_t is not None else in_channels
        for i in range(num_stages):
            if i == 0:
                k, s, in_ch, prev_ch = 4, 4, t_in, 0
            else:
                k, s, in_ch, prev_ch = 2, 2, mid_channels, mid_channels
            self.t_stages.append(
                QualityStage(in_ch, mid_channels, k, s, prev_channels=prev_ch))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        for stage in list(self.rgb_stages) + list(self.t_stages):
            last = stage.score_mlp[-1]
            nn.init.normal_(last.weight, mean=0.0, std=0.01)
            nn.init.constant_(last.bias, 0.0)
            stage._init_refine_center_bias()

    @staticmethod
    def _compute_local_var_map(img, patch_sizes):
        B, C, H, W = img.shape
        var_maps = []
        for ps in patch_sizes:
            H_p = H // ps
            W_p = W // ps
            if H_p == 0 or W_p == 0:
                var_maps.append(None)
                continue
            cropped = img[:, :, :H_p * ps, :W_p * ps]
            mean_local = F.avg_pool2d(cropped, ps, stride=ps)
            mean_up = F.interpolate(mean_local, size=(H_p * ps, W_p * ps),
                                    mode='nearest')
            diff_sq = (cropped - mean_up) ** 2
            var_map = F.avg_pool2d(diff_sq, ps, stride=ps).mean(dim=1, keepdim=True)
            var_maps.append(var_map)
        return var_maps

    @staticmethod
    def _var_to_gate(var_map, log_tau):
        tau = log_tau.exp().clamp(min=1e-6, max=10.0)
        normalized_var = var_map / (tau + 1e-8)
        gate = torch.sigmoid(50.0 * (normalized_var - 0.05))
        return gate.clamp(min=0.0, max=1.0)

    def _forward_branch(self, img, stages):
        B, C, H, W = img.shape

        patch_sizes = [4 * (2 ** i) for i in range(self.num_stages)]
        var_maps = self._compute_local_var_map(img, patch_sizes)

        quality_maps = []
        feat = img
        prev_feat = None
        for i, stage in enumerate(stages):
            feat, q_map = stage(feat, prev_feat=prev_feat)
            prev_feat = feat

            if i < len(var_maps) and var_maps[i] is not None:
                vm = var_maps[i]
                if vm.shape[2:] != q_map.shape[2:]:
                    vm = F.interpolate(
                        vm, size=q_map.shape[2:], mode='nearest')
                var_gate = self._var_to_gate(vm, stage.log_tau)
                q_map = q_map * var_gate

            quality_maps.append(q_map)

        return quality_maps

    def forward_rgb(self, img):
        return self._forward_branch(img, self.rgb_stages)

    def forward_thermal(self, img):
        return self._forward_branch(img, self.t_stages)

    def forward(self, img_rgb, img_t):
        q_rgb = self.forward_rgb(img_rgb)
        q_t = self.forward_thermal(img_t)
        return q_rgb, q_t


class PyramidRankingLoss(nn.Module):

    def __init__(self, margin=0.1,
                 max_deg5_quality=0.1, consistency_margin=0.05,
                 contrast_margin=0.3, contrast_topk=0.2,
                 min_clean_quality=0.7,
                 cross_scale_margin=0.05,
                 deg5_ratio_threshold=0.8,
                 w_rank=1.0, w_deg5=0.1, w_consist=0.5,
                 w_contrast=0.3, w_clean=0.3,
                 w_cross_scale=1.0):
        super().__init__()
        self.margin = margin
        self.max_deg5_quality = max_deg5_quality
        self.consistency_margin = consistency_margin
        self.contrast_margin = contrast_margin
        self.contrast_topk = contrast_topk
        self.min_clean_quality = min_clean_quality
        self.cross_scale_margin = cross_scale_margin
        self.deg5_ratio_threshold = deg5_ratio_threshold
        self.w_rank = w_rank
        self.w_deg5 = w_deg5
        self.w_consist = w_consist
        self.w_contrast = w_contrast
        self.w_clean = w_clean
        self.w_cross_scale = w_cross_scale

    def forward(self, quality_maps_list, token_masks_list):
        device = quality_maps_list[0][0].device
        num_levels = len(quality_maps_list)
        num_stages = len(quality_maps_list[0])

        loss_rank = torch.tensor(0.0, device=device)
        loss_deg5 = torch.tensor(0.0, device=device)
        loss_consist = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)
        loss_clean = torch.tensor(0.0, device=device)
        loss_cross_scale = torch.tensor(0.0, device=device)

        for s in range(num_stages):
            target_size = quality_maps_list[0][s].shape[2:]

            q_per_level = []
            ratio_per_level = []
            for lvl in range(num_levels):
                q = quality_maps_list[lvl][s]
                ratio = F.adaptive_avg_pool2d(
                    token_masks_list[lvl], target_size)
                q_per_level.append(q)
                ratio_per_level.append(ratio)

            orig_q = q_per_level[0]

            deg5_q = q_per_level[-1]
            deg5_ratio = ratio_per_level[-1]
            strict_mask = (deg5_ratio >= self.deg5_ratio_threshold).float()
            n_strict = strict_mask.sum()
            if n_strict > 0:
                loss_deg5 = loss_deg5 + (
                    F.relu(deg5_q - self.max_deg5_quality) *
                    strict_mask).sum() / (n_strict + 1e-8)

            for i in range(num_levels - 1):
                q_higher = q_per_level[i]
                q_lower = q_per_level[i + 1]
                ratio_i = ratio_per_level[i]
                ratio_i1 = ratio_per_level[i + 1]
                joint_deg = ((ratio_i + ratio_i1) > 0.5).float()
                n_joint = joint_deg.sum()
                if n_joint > 0:
                    violation = self.margin + q_lower - q_higher
                    loss_rank = loss_rank + (
                        F.relu(violation) * joint_deg).sum() / (
                        n_joint + 1e-8)

            for lvl in range(1, num_levels):
                deg_non_deg = (ratio_per_level[lvl] < 0.5).float()
                n_non_deg = deg_non_deg.sum()
                if n_non_deg > 0:
                    diff = (q_per_level[lvl] - orig_q).abs()
                    violation = diff - self.consistency_margin
                    loss_consist = loss_consist + (
                        F.relu(violation) * deg_non_deg).sum() / (
                        n_non_deg + 1e-8)

            clean_q = q_per_level[0]
            q_flat = clean_q.flatten(1)
            N_q = q_flat.shape[1]
            k = max(1, int(N_q * self.contrast_topk))

            q_top_vals = q_flat.topk(k, dim=1).values
            q_bottom_vals = q_flat.topk(k, dim=1, largest=False).values

            loss_gap = F.relu(
                self.contrast_margin - (q_top_vals.mean(dim=1) -
                                        q_bottom_vals.mean(dim=1))).mean()

            loss_contrast = loss_contrast + loss_gap

            violation = self.min_clean_quality - orig_q
            loss_clean = loss_clean + F.relu(violation).mean()

        for lvl in range(num_levels):
            q_maps = quality_maps_list[lvl]
            for i in range(num_stages - 1):
                q_shallow = q_maps[i]
                q_deep = q_maps[i + 1]
                q_shallow_down = F.adaptive_avg_pool2d(
                    q_shallow, q_deep.shape[2:])
                violation = q_deep - q_shallow_down - self.cross_scale_margin
                loss_cross_scale = loss_cross_scale + F.relu(violation).mean()

        loss_rank = loss_rank / (num_stages * max(num_levels - 1, 1))
        loss_deg5 = loss_deg5 / num_stages
        loss_consist = loss_consist / (num_stages * max(num_levels - 1, 1))
        loss_contrast = loss_contrast / num_stages
        loss_clean = loss_clean / num_stages
        loss_cross_scale = loss_cross_scale / (
            num_levels * max(num_stages - 1, 1))

        total_loss = (
            self.w_rank * loss_rank
            + self.w_deg5 * loss_deg5
            + self.w_consist * loss_consist
            + self.w_contrast * loss_contrast
            + self.w_clean * loss_clean
            + self.w_cross_scale * loss_cross_scale
        )

        loss_dict = {
            'total_loss': total_loss,
            'loss_rank': loss_rank,
            'loss_deg5': loss_deg5,
            'loss_consist': loss_consist,
            'loss_contrast': loss_contrast,
            'loss_clean': loss_clean,
            'loss_cross_scale': loss_cross_scale,
        }

        return total_loss, loss_dict
