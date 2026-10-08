# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence, Union

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RemapLabels(BaseTransform):
    """Apply an explicit label-value mapping to ``gt_seg_map``.

    Used to remove classes that carry no usable supervision without leaving a
    hole in the label space.  Dropping a value is not enough on its own: if
    class ``c`` is removed and classes ``c+1..n`` keep their indices, the model
    (which has ``n`` outputs, indexed ``0..n-1``) would receive labels it cannot
    express.  The mapping must therefore also shift the classes after the gap.

    FMB is the motivating case.  After ``reduce_zero_label`` the 14 foreground
    classes are 0..13, with Bicycle at 12 and Pole at 13.  Bicycle occupies
    9,115 pixels across 12 of the 1,220 training images and exactly zero pixels
    in the 280-image validation split, so its IoU is undefined there and a
    spurious prediction would give a non-empty union with an empty intersection
    and pull the mean down.  The correct mapping is therefore::

        RemapLabels(mapping={12: 255, 13: 12})

    which removes Bicycle and closes the gap, leaving 13 contiguous classes.

    The mapping is compiled into a lookup table once, so the order of the
    entries cannot matter and a value can be both a source and a destination
    without the two rules interfering.

    Args:
        mapping (dict): ``{source_value: destination_value}``.  Values must be
            in ``[0, 255]``.
        ignore_index (int): Value used for removed classes.  Only used to
            validate; the mapping itself carries the destinations.
    """

    def __init__(self,
                 mapping: Dict[int, int],
                 ignore_index: int = 255) -> None:
        if not mapping:
            raise ValueError('mapping must not be empty')
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.ignore_index = int(ignore_index)
        for src, dst in self.mapping.items():
            if not 0 <= src <= 255 or not 0 <= dst <= 255:
                raise ValueError(
                    f'label values must be in [0, 255], got {src} -> {dst}')
        if self.ignore_index not in self.mapping.values():
            raise ValueError(
                f'ignore_index {self.ignore_index} is not among the mapping '
                'destinations; a removed class would keep a live label.')
        # Built once: a single gather makes the transform order-independent.
        self._lut = np.arange(256, dtype=np.uint8)
        for src, dst in self.mapping.items():
            self._lut[src] = dst

    def transform(self, results: dict) -> dict:
        """Apply the mapping to ``gt_seg_map``.

        Args:
            results (dict): Result dict from :obj:`LoadAnnotations`.

        Returns:
            dict: The updated results.
        """
        seg = results.get('gt_seg_map')
        if seg is None:
            return results
        if not isinstance(seg, np.ndarray):
            raise TypeError('RemapLabels expects a numpy gt_seg_map, got '
                            f'{type(seg).__name__}')
        if seg.dtype != np.uint8:
            seg = seg.astype(np.uint8)
        results['gt_seg_map'] = self._lut[seg]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(mapping={self.mapping}, '
                f'ignore_index={self.ignore_index})')
