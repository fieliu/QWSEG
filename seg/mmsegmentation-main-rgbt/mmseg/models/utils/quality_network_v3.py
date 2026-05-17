import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


class GlobalNorm(nn.Module):

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 4
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim))
        self.beta_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim))

    def forward(self, tokens):
        B, N, D = tokens.shape
        global_desc = self.global_token.expand(B, -1, -1).squeeze(1)
        gamma = self.gamma_mlp(global_desc).unsqueeze(1)
        beta = self.beta_mlp(global_desc).unsqueeze(1)
        out = tokens * (1.0 + gamma) + beta
        return out


class LocalChannelAttention(nn.Module):

    def __init__(self, embed_dim, mid_dim=None):
        super().__init__()
        if mid_dim is None:
            mid_dim = embed_dim // 4
        self.token_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 1),
            nn.GELU())
        self.context_proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 3, padding=1, groups=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 1))
        self.attn_gate = nn.Sequential(
            nn.Conv2d(mid_dim * 2, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, embed_dim, 1),
            nn.Sigmoid())
        self.score_head = nn.Sequential(
            nn.Conv2d(embed_dim, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 1, 1))

    def forward(self, x_2d):
        token_feat = self.token_proj(x_2d)
        ctx_feat = self.context_proj(x_2d)
        gate_input = torch.cat([token_feat, ctx_feat], dim=1)
        gate = self.attn_gate(gate_input)
        modulated = x_2d * gate
        score_map = self.score_head(modulated)
        return score_map


@MODELS.register_module()
class QualityNetworkV3(nn.Module):

    def __init__(self,
                 embed_dim=768,
                 proj_dim=128,
                 global_norm_hidden=None,
                 local_attn_mid=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.rgb_input_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU())
        self.rgb_global_norm = GlobalNorm(proj_dim, global_norm_hidden)
        self.rgb_local_attn = LocalChannelAttention(proj_dim, local_attn_mid)

        self.t_input_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU())
        self.t_global_norm = GlobalNorm(proj_dim, global_norm_hidden)
        self.t_local_attn = LocalChannelAttention(proj_dim, local_attn_mid)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for name, attn in [('rgb', self.rgb_local_attn),
                           ('t', self.t_local_attn)]:
            last_conv = attn.score_head[-1]
            nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(last_conv.bias, 0.0)

    def _forward_branch(self, tokens, input_proj, global_norm, local_attn):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f'Token count {N} is not a perfect square'

        projected = input_proj(tokens)
        normed = global_norm(projected)
        x_2d = normed.transpose(1, 2).reshape(B, self.proj_dim, H, W)
        score_map = local_attn(x_2d)
        scores = score_map.reshape(B, 1, -1).squeeze(1)
        quality = torch.sigmoid(scores)
        return quality

    def forward_rgb(self, tokens):
        return self._forward_branch(
            tokens, self.rgb_input_proj,
            self.rgb_global_norm, self.rgb_local_attn)

    def forward_thermal(self, tokens):
        return self._forward_branch(
            tokens, self.t_input_proj,
            self.t_global_norm, self.t_local_attn)

    def forward(self, tokens, modality='rgb'):
        if modality == 'rgb':
            return self.forward_rgb(tokens)
        elif modality in ('t', 'thermal'):
            return self.forward_thermal(tokens)
        else:
            raise ValueError(f'Unsupported modality: {modality}')


class ImagePseudoLabelGenerator(nn.Module):

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        self.laplacian_kernel = nn.Conv2d(
            1, 1, 3, padding=1, bias=False)
        self.laplacian_kernel.weight.data = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32).reshape(1, 1, 3, 3)
        self.laplacian_kernel.weight.requires_grad_(False)

        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad_(False)

        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y.weight.requires_grad_(False)

        self.median_kernel = nn.Conv2d(
            1, 1, 3, padding=1, bias=False)
        self.median_kernel.weight.data = torch.ones(
            1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.median_kernel.weight.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img):
        B, C, H, W = img.shape
        ps = self.patch_size
        h_tok = H // ps
        w_tok = W // ps

        gray = img.mean(dim=1, keepdim=True)

        lap = self.laplacian_kernel(gray)
        lap_var = self._patch_stat(lap.pow(2).mean(dim=1, keepdim=True), h_tok, w_tok)

        contrast = self._patch_stat(gray, h_tok, w_tok, stat='std')

        mu = self._patch_stat(gray, h_tok, w_tok, stat='mean')
        bright_score = torch.exp(-((mu - 128.0) / 64.0).pow(2))

        smooth = self.median_kernel(gray)
        residual = (gray - smooth).abs()
        noise_std = self._patch_stat(residual, h_tok, w_tok, stat='mean')

        gx = self.sobel_x(gray)
        gy = self.sobel_y(gray)
        grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()
        grad_entropy = self._patch_entropy(grad_mag, h_tok, w_tok, num_bins=16)

        lap_var_norm = self._minmax_norm(lap_var)
        contrast_norm = self._minmax_norm(contrast)
        noise_std_norm = self._minmax_norm(noise_std)
        grad_entropy_norm = self._minmax_norm(grad_entropy)

        raw = (0.4 * lap_var_norm
               + 0.2 * contrast_norm
               + 0.15 * bright_score
               + 0.15 * (1.0 - noise_std_norm)
               + 0.1 * grad_entropy_norm)

        return raw.reshape(B, -1)

    def _patch_stat(self, x, h_tok, w_tok, stat='mean'):
        ps = self.patch_size
        B = x.shape[0]
        x_patched = x.reshape(B, 1, h_tok, ps, w_tok, ps)
        x_patched = x_patched.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, h_tok, w_tok, ps * ps)
        x_mean = x_patched.mean(dim=-1, keepdim=False)
        if stat == 'std':
            x_sq = x_patched.pow(2)
            x_sq_mean = x_sq.mean(dim=-1, keepdim=False)
            var = (x_sq_mean - x_mean.pow(2)).clamp(min=0)
            result = var.sqrt()
        elif stat == 'mean':
            result = x_mean
        else:
            result = x_mean
        return result.squeeze(1)

    def _patch_entropy(self, grad_mag, h_tok, w_tok, num_bins=16):
        B = grad_mag.shape[0]
        ps = self.patch_size
        H = h_tok * ps
        W = w_tok * ps
        grad_patched = grad_mag[:, :, :H, :W].reshape(B, 1, h_tok, ps, w_tok, ps)
        grad_per_patch = grad_patched.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, h_tok, w_tok, ps * ps)

        g_min = grad_per_patch.min(dim=-1, keepdim=True)[0]
        g_max = grad_per_patch.max(dim=-1, keepdim=True)[0]
        g_range = g_max - g_min + 1e-8
        bin_idx = ((grad_per_patch - g_min) / g_range * num_bins).long().clamp(0, num_bins - 1)

        one_hot = torch.zeros(
            B, 1, h_tok, w_tok, num_bins,
            dtype=grad_per_patch.dtype, device=grad_per_patch.device)
        one_hot.scatter_(4, bin_idx, 1.0)
        hist = one_hot / (one_hot.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(hist * (hist + 1e-8).log()).sum(dim=-1)
        return entropy.squeeze(1)

    @staticmethod
    def _minmax_norm(x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        x_min = x_flat.min(dim=-1, keepdim=True)[0]
        x_max = x_flat.max(dim=-1, keepdim=True)[0]
        normalized = (x_flat - x_min) / (x_max - x_min + 1e-8)
        return normalized.reshape(x.shape)


class ThermalPseudoLabelGenerator(nn.Module):

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_x.weight.requires_grad_(False)

        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y.weight.requires_grad_(False)

        self.median_kernel = nn.Conv2d(
            1, 1, 3, padding=1, bias=False)
        self.median_kernel.weight.data = torch.ones(
            1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.median_kernel.weight.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img):
        B, C, H, W = img.shape
        ps = self.patch_size
        h_tok = H // ps
        w_tok = W // ps

        gray = img.mean(dim=1, keepdim=True)

        gx = self.sobel_x(gray)
        gy = self.sobel_y(gray)
        grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()
        hf_energy = self._patch_stat(grad_mag, h_tok, w_tok, stat='mean')

        contrast = self._patch_stat(gray, h_tok, w_tok, stat='std')

        dynamic_range = self._patch_dynamic_range(gray, h_tok, w_tok)

        smooth = self.median_kernel(gray)
        residual = (gray - smooth).abs()
        noise_std = self._patch_stat(residual, h_tok, w_tok, stat='mean')

        texture_activity = self._patch_texture_activity(gray, h_tok, w_tok)

        hf_energy_norm = self._minmax_norm(hf_energy)
        contrast_norm = self._minmax_norm(contrast)
        dynamic_range_norm = self._minmax_norm(dynamic_range)
        noise_score = 1.0 - self._minmax_norm(noise_std)
        texture_norm = self._minmax_norm(texture_activity)

        raw = (0.30 * hf_energy_norm
               + 0.25 * contrast_norm
               + 0.20 * dynamic_range_norm
               + 0.15 * noise_score
               + 0.10 * texture_norm)

        return raw.reshape(B, -1)

    def _patch_stat(self, x, h_tok, w_tok, stat='mean'):
        ps = self.patch_size
        B = x.shape[0]
        x_patched = x.reshape(B, 1, h_tok, ps, w_tok, ps)
        x_patched = x_patched.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, h_tok, w_tok, ps * ps)
        x_mean = x_patched.mean(dim=-1, keepdim=False)
        if stat == 'std':
            x_sq = x_patched.pow(2)
            x_sq_mean = x_sq.mean(dim=-1, keepdim=False)
            var = (x_sq_mean - x_mean.pow(2)).clamp(min=0)
            result = var.sqrt()
        elif stat == 'mean':
            result = x_mean
        else:
            result = x_mean
        return result.squeeze(1)

    def _patch_dynamic_range(self, gray, h_tok, w_tok):
        ps = self.patch_size
        B = gray.shape[0]
        valid_low = 5.0
        valid_high = 250.0
        in_range = ((gray >= valid_low) & (gray <= valid_high)).float()
        in_range_patched = in_range.reshape(B, 1, h_tok, ps, w_tok, ps)
        in_range_patched = in_range_patched.permute(0, 1, 2, 4, 3, 5).reshape(
            B, 1, h_tok, w_tok, ps * ps)
        ratio = in_range_patched.mean(dim=-1, keepdim=False)
        return ratio.squeeze(1)

    def _patch_texture_activity(self, gray, h_tok, w_tok):
        ps = self.patch_size
        B = gray.shape[0]
        gray_patched = gray.reshape(B, 1, h_tok, ps, w_tok, ps)
        gray_patched = gray_patched.permute(0, 1, 2, 4, 3, 5).reshape(
            B, 1, h_tok, w_tok, ps * ps)
        p_max = gray_patched.max(dim=-1, keepdim=False)[0]
        p_min = gray_patched.min(dim=-1, keepdim=False)[0]
        local_range = (p_max - p_min).squeeze(1)
        global_std = gray.std()
        texture = local_range / (global_std + 1e-8)
        return texture

    @staticmethod
    def _minmax_norm(x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        x_min = x_flat.min(dim=-1, keepdim=True)[0]
        x_max = x_flat.max(dim=-1, keepdim=True)[0]
        normalized = (x_flat - x_min) / (x_max - x_min + 1e-8)
        return normalized.reshape(x.shape)


class IntraImageRankingLoss(nn.Module):

    def __init__(self, margin=0.0, top_k_ratio=0.3):
        super().__init__()
        self.margin = margin
        self.top_k_ratio = top_k_ratio

    def forward(self, q_pred, pseudo_labels):
        B, N = q_pred.shape
        loss = torch.tensor(0.0, device=q_pred.device)
        count = 0

        for b in range(B):
            q = q_pred[b]
            p = pseudo_labels[b]

            idx = torch.randperm(N, device=q.device)
            half = N // 2
            i_idx = idx[:half]
            j_idx = idx[half:2 * half]

            q_i = q[i_idx]
            q_j = q[j_idx]
            p_i = p[i_idx]
            p_j = p[j_idx]

            p_diff = (p_i - p_j).abs()
            if p_diff.max() < 1e-8:
                continue

            k = max(1, int(half * self.top_k_ratio))
            _, topk_idx = p_diff.topk(k)
            p_diff_topk = p_diff[topk_idx]
            confidence = p_diff_topk / (p_diff_topk.max() + 1e-8)

            p_i_sel = p_i[topk_idx]
            p_j_sel = p_j[topk_idx]
            q_i_sel = q_i[topk_idx]
            q_j_sel = q_j[topk_idx]

            target = torch.sign(p_i_sel - p_j_sel)
            diff = q_i_sel - q_j_sel
            violation = -target * diff + self.margin
            per_pair_loss = F.relu(violation).pow(2) * confidence

            loss = loss + per_pair_loss.mean()
            count += 1

        if count > 0:
            loss = loss / count
        return loss


class RankingMarginLoss(nn.Module):

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, q_high, q_low, mask=None):
        diff = q_high - q_low
        violation = self.margin - diff
        if mask is not None:
            loss = (F.relu(violation).pow(2) * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = F.relu(violation).pow(2).mean()
        return loss


class LevelConsistencyLoss(nn.Module):

    def __init__(self, num_levels=5, level_margin=0.1, min_low_quality=0.1,
                 min_clean_quality=0.5):
        super().__init__()
        self.num_levels = num_levels
        self.level_margin = level_margin
        self.min_low_quality = min_low_quality
        self.min_clean_quality = min_clean_quality

    def forward(self, q_list, mask=None):
        loss = torch.tensor(0.0, device=q_list[0].device)

        if mask is not None:
            if mask.shape[0] == 1 and q_list[0].shape[0] > 1:
                mask = mask.expand(q_list[0].shape[0], -1)

            deg_tokens = (mask > 0.5).float()
            clean_tokens = 1.0 - deg_tokens
            n_deg = deg_tokens.sum()
            n_clean = clean_tokens.sum()

            for i in range(len(q_list) - 1):
                diff = q_list[i] - q_list[i + 1]
                violation = self.level_margin - diff
                if n_deg > 0:
                    loss = loss + (F.relu(violation).pow(2) * deg_tokens).sum() / (n_deg + 1e-8)

            for i in range(len(q_list) - 1):
                q_i = q_list[i]
                floor_violation = self.min_low_quality - q_i
                if n_deg > 0:
                    loss = loss + (F.relu(floor_violation).pow(2) * deg_tokens).sum() / (n_deg + 1e-8)
                if n_clean > 0:
                    loss = loss + (F.relu(floor_violation).pow(2) * clean_tokens).sum() / (n_clean + 1e-8)

            if n_deg > 0:
                max_deg_q = q_list[-1]
                ceiling_violation = max_deg_q - self.min_low_quality
                loss = loss + (F.relu(ceiling_violation).pow(2) * deg_tokens).sum() / (n_deg + 1e-8)

            if len(q_list) > 0:
                q_clean = q_list[0]
                clean_floor = self.min_clean_quality - q_clean
                loss = loss + F.relu(clean_floor).pow(2).mean()
        else:
            for i in range(len(q_list) - 1):
                diff = q_list[i] - q_list[i + 1]
                violation = self.level_margin - diff
                loss = loss + F.relu(violation).pow(2).mean()

            for i in range(len(q_list) - 1):
                floor_violation = self.min_low_quality - q_list[i]
                loss = loss + F.relu(floor_violation).pow(2).mean()

            if len(q_list) > 0:
                ceiling_violation = q_list[-1] - self.min_low_quality
                loss = loss + F.relu(ceiling_violation).pow(2).mean()

            if len(q_list) > 0:
                clean_floor = self.min_clean_quality - q_list[0]
                loss = loss + F.relu(clean_floor).pow(2).mean()

        return loss
