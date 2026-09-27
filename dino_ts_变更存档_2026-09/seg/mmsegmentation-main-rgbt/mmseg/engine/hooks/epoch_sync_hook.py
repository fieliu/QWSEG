"""Epoch-sync hook for the DINO-TS stage segmentors.

Several DINO-TS stages read ``model.current_epoch`` every iteration:
  - Stage 2B: EMA momentum ramp 0.996 -> 0.9999 (doc 8.1),
  - Stage 3:  progressive Extra budget K and soft->hard gate switch (doc 9.7),
  - all stages: degradation curriculum epoch.

The existing vis / eval hooks only set ``current_epoch`` at their interval (and
some not at all), so without this hook it would stay 0 and the ramps/schedules
would never advance. This hook sets it at the start of every training epoch.
"""
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class EpochSyncHook(Hook):
    """Propagate ``runner.epoch`` to ``model.current_epoch`` each epoch."""

    priority = 'VERY_HIGH'  # run before other hooks that read current_epoch

    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'current_epoch'):
            model.current_epoch = runner.epoch
