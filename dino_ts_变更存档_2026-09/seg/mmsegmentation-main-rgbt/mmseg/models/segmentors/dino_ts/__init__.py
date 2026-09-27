"""DINO multi-stage Dense-Teacher / Sparse-Student segmentors.

Loaded via config ``custom_imports`` (these segmentors depend on mmdet's
Mask2Former head, so they are NOT added to mmseg.models.segmentors.__init__ —
that keeps `import mmseg` working when mmdet is absent, matching the EoMT
family's convention).
"""
from .base_dino_ts import DinoTSBase
from .stage1_adapt import DinoTSStage1Adapt
from .stage2a_dense import DinoTSDense
from .stage2b_ema import DinoTSDenseEMA
from .stage3_sparse import DinoTSSparse

__all__ = [
    'DinoTSBase', 'DinoTSStage1Adapt', 'DinoTSDense', 'DinoTSDenseEMA',
    'DinoTSSparse',
]
