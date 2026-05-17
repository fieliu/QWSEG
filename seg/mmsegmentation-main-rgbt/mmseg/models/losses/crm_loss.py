import numpy as np
import random
import torch

from mmseg.registry import MODELS


class MaskGenerator:
    def __init__(self,
                 input_size=(256, 320),
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.5,
                 mask_type='patch',
                 strategy='rand_comp'):
        self.input_size = np.array(input_size)
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size[0] % self.mask_patch_size == 0
        assert self.input_size[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if mask_type == 'patch':
            self.gen_mask = self._gen_patch_mask
        elif mask_type == 'square':
            self.gen_mask = self._gen_square_mask
        else:
            raise AssertionError(f'Not valid mask type: {mask_type}')

        if strategy == 'comp':
            self.strategy = self._gen_comp_masks
        elif strategy == 'rand_comp':
            self.strategy = self._gen_rand_comp_masks
        elif strategy == 'indiv':
            self.strategy = self._gen_indiv_masks
        else:
            raise AssertionError(f'Not valid strategy: {strategy}')

    def _gen_patch_mask(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = np.expand_dims(
            mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1), axis=0)
        return mask

    def _gen_square_mask(self):
        mask = np.zeros((self.input_size[0], self.input_size[1]), dtype=int)
        h1 = np.random.randint(0, self.input_size[0] * self.mask_ratio)
        w1 = np.random.randint(0, self.input_size[1] * self.mask_ratio)
        h2 = int(h1 + self.input_size[0] * self.mask_ratio)
        w2 = int(w1 + self.input_size[1] * self.mask_ratio)
        mask[w1:w2, h1:h2] = 1
        return np.expand_dims(mask, axis=0)

    def _gen_comp_masks(self):
        mask = self.gen_mask()
        return mask, 1 - mask

    def _gen_rand_comp_masks(self):
        mask = self.gen_mask()
        nomask = np.zeros_like(mask)
        idx = random.randrange(3)
        if idx == 0:
            return nomask, 1 - mask
        elif idx == 1:
            return mask, nomask
        else:
            return mask, 1 - mask

    def _gen_indiv_masks(self):
        mask1 = self.gen_mask()
        mask2 = self.gen_mask()
        return mask1, mask2

    def __call__(self):
        return self.strategy()


@MODELS.register_module()
class CRMLoss(torch.nn.Module):
    def __init__(self,
                 mws_weight=1.0,
                 sdc_weight=1.0,
                 sdn_weight=1.0,
                 mask_enabled=True,
                 mask_size=(256, 320),
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.5,
                 mask_type='patch',
                 mask_strategy='rand_comp'):
        super().__init__()
        self.mws_weight = mws_weight
        self.sdc_weight = sdc_weight
        self.sdn_weight = sdn_weight
        self.mask_enabled = mask_enabled

        if mask_enabled:
            self.mask_generator = MaskGenerator(
                input_size=mask_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=model_patch_size,
                mask_ratio=mask_ratio,
                mask_type=mask_type,
                strategy=mask_strategy)

    def generate_masks(self, batch_size):
        masks = []
        for _ in range(batch_size):
            mask1, mask2 = self.mask_generator()
            masks.append(torch.as_tensor(np.stack([mask1, mask2], axis=0)))
        return torch.stack(masks, dim=0)

    def forward(self, losses_rgbt, losses_rgb, losses_thr,
                losses_masked=None, losses_self_comp=None,
                losses_self_nlocal=None):
        loss = sum(losses_rgbt.values())
        loss += self.mws_weight * sum(losses_rgb.values())
        loss += self.mws_weight * sum(losses_thr.values())

        if losses_masked is not None and self.mask_enabled:
            loss += sum(losses_masked.values())
            if losses_self_comp is not None:
                loss += self.sdc_weight * losses_self_comp
            if losses_self_nlocal is not None:
                loss += self.sdn_weight * losses_self_nlocal

        return loss
