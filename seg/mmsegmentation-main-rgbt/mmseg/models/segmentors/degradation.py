"""In-training degradation generator for RGB-T MISSING-modality robustness.

Missing-only variant (Paradigm One). Degrades ONE modality at a time (the other
stays fully usable, honoring "at least one modality usable per position") via:
  - 'missing'       : whole modality zeroed
  - 'local_missing' : a random region zeroed (local dropout)
Returns degraded tensors + per-modality masks (1 = missing/zeroed at that pixel).

Missing is binary by nature (zeroed or not), so there is no severity grading and
quality supervision is a clean 0/1 target. Region size for local_missing covers
a range of scales for coverage.
"""
import random
import torch

# region area fractions for local_missing (covers small..large holes)
_LOCAL_AREA = [0.10, 0.25, 0.40, 0.60, 0.80]


class DegradationGenerator:
    def __init__(
        self,
        kinds=("missing", "local_missing"),
        kind_probs=(0.5, 0.5),
        degrade_prob=0.8,          # prob that a sample is degraded at all
        curriculum=False,
        total_epochs=200,
        **kwargs,                  # tolerate legacy keys (severity_range, etc.)
    ):
        self.kinds = list(kinds)
        self.kind_probs = list(kind_probs)
        self.degrade_prob = degrade_prob
        self.curriculum = curriculum
        self.total_epochs = total_epochs

    @torch.no_grad()
    def __call__(self, rgb, ir, epoch=0):
        """Returns degraded_rgb, degraded_ir, mask_rgb, mask_ir.

        mask_* : [B, 1, H, W] float, 1 where that modality is MISSING (zeroed).
        """
        B, _, H, W = rgb.shape
        dev = rgb.device
        drgb = rgb.clone()
        dir_ = ir.clone()
        mask_rgb = torch.zeros(B, 1, H, W, device=dev, dtype=rgb.dtype)
        mask_ir = torch.zeros(B, 1, H, W, device=dev, dtype=ir.dtype)

        for b in range(B):
            if random.random() > self.degrade_prob:
                continue  # this sample stays clean
            mod = random.choice(["rgb", "ir"])  # degrade one modality only
            kind = random.choices(self.kinds, weights=self.kind_probs, k=1)[0]

            img = drgb[b:b + 1] if mod == "rgb" else dir_[b:b + 1]
            m = mask_rgb[b:b + 1] if mod == "rgb" else mask_ir[b:b + 1]

            if kind == "missing":
                img.zero_()
                m.fill_(1.0)
            else:  # local_missing: zero a random region
                area = random.choice(_LOCAL_AREA)
                rh = max(1, int(H * (area ** 0.5)))
                rw = max(1, int(W * (area ** 0.5)))
                y0 = random.randint(0, max(0, H - rh))
                x0 = random.randint(0, max(0, W - rw))
                img[:, :, y0:y0 + rh, x0:x0 + rw] = 0.0
                m[:, :, y0:y0 + rh, x0:x0 + rw] = 1.0

            if mod == "rgb":
                drgb[b:b + 1] = img
                mask_rgb[b:b + 1] = m
            else:
                dir_[b:b + 1] = img
                mask_ir[b:b + 1] = m

        return drgb, dir_, mask_rgb, mask_ir
