# Utilities to bridge EoMT (mask-classification) with mmseg's per-pixel
# semantic segmentation interface.
import torch
import torch.nn.functional as F


def gt_to_mask_targets(gt_sem_seg, num_classes, ignore_index=255):
    """Convert a per-pixel label map [H, W] into mask-classification targets.

    Returns a dict {"masks": [K, H, W] bool, "labels": [K] long} where K is the
    number of distinct valid classes present in this image. If the image has no
    valid class, returns one empty entry so the Hungarian matcher still works.
    """
    gt = gt_sem_seg.long()
    classes = torch.unique(gt)
    classes = classes[(classes != ignore_index) & (classes >= 0) & (classes < num_classes)]

    if classes.numel() == 0:
        masks = torch.zeros((0, *gt.shape[-2:]), dtype=torch.bool, device=gt.device)
        labels = torch.zeros((0,), dtype=torch.long, device=gt.device)
        return {"masks": masks, "labels": labels}

    masks = torch.stack([(gt == c) for c in classes], dim=0)  # [K, H, W] bool
    labels = classes.long()
    return {"masks": masks, "labels": labels}


def build_targets(data_samples, num_classes, ignore_index=255):
    """Build a list of per-image mask-classification targets from mmseg
    data_samples (each carries gt_sem_seg.data of shape [1, H, W])."""
    targets = []
    for ds in data_samples:
        gt = ds.gt_sem_seg.data.squeeze(0)  # [H, W]
        targets.append(gt_to_mask_targets(gt, num_classes, ignore_index))
    return targets


def mask_class_to_seg_logits(mask_logits, class_logits):
    """EoMT query outputs -> per-pixel class logits.

    mask_logits:  [B, Q, h, w]
    class_logits: [B, Q, num_classes + 1]
    returns:      [B, num_classes, h, w]
    """
    return torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits.softmax(dim=-1)[..., :-1],
    )


def resize_seg_logits(seg_logits, size):
    """Bilinearly resize per-pixel logits to a target (H, W)."""
    return F.interpolate(seg_logits, size=size, mode="bilinear", align_corners=False)
