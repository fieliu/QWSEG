"""Teacher-prediction -> pseudo-label conversion for Mask2Former/EoMT
distillation, ported from facebookresearch/GuidedDistillation
(prepare_ssl_outputs). The pseudo-labels are then fed to the model's OWN
standard Mask2Former criterion (Hungarian matching + class CE + mask BCE +
dice), so distillation reuses the exact standard loss — no hand-rolled query
matching or full-image BCE (the latter was background-dominated and collapsed
foreground mIoU).

Standard pipeline (GuidedDistillation):
  1. drop no-object queries (argmax == no_obj)
  2. keep only high-confidence queries (softmax max > thresh_class)
  3. drop near-empty masks (sigmoid(mask).sum > mask_size)
  4. pseudo-GT = {labels: argmax, masks: sigmoid(mask) > 0.5}  (BINARY)
Then: criterion(student_mask_logits, [pseudo_gt...], student_class_logits).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def prepare_ssl_outputs(pred_logits, pred_masks, thresh_class=0.7,
                        mask_size=5):
    """teacher per-query predictions -> list of pseudo-GT dicts (one per image).

    pred_logits: [B, Q, C+1] teacher class logits (raw)
    pred_masks:  [B, Q, h, w] teacher mask logits (raw)
    Returns: list of {'labels': [K] long, 'masks': [K, h, w] bool} per image.
    Mirrors GuidedDistillation.prepare_ssl_outputs exactly."""
    out = []
    bs = pred_logits.shape[0]
    no_obj = pred_logits.shape[-1] - 1
    for b in range(bs):
        mask_cls = pred_logits[b]                       # [Q, C+1]
        mask_pred = pred_masks[b]                       # [Q, h, w]

        objects = mask_cls.argmax(dim=1) != no_obj      # drop no-object
        mask_cls = mask_cls[objects]
        mask_pred = mask_pred[objects]

        high_conf = F.softmax(mask_cls, dim=1).max(dim=1).values > thresh_class
        mask_cls = mask_cls[high_conf]
        mask_pred = mask_pred[high_conf]

        not_empty = torch.sigmoid(mask_pred).sum(dim=(1, 2)) > mask_size
        tar_cls = mask_cls[not_empty].argmax(dim=1)     # [K] labels
        tar_mask = torch.sigmoid(mask_pred[not_empty]) > 0.5  # [K,h,w] binary

        out.append({'labels': tar_cls.clone(), 'masks': tar_mask.clone()})
    return out
