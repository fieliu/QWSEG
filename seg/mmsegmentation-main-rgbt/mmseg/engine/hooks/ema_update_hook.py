"""EMA lifecycle hook for the DINO-TS dense teacher."""
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class EMAUpdateHook(Hook):
    """Initialize after checkpoint loading and update after optimizer steps."""

    priority = 'ABOVE_NORMAL'

    @staticmethod
    def _model(runner):
        model = runner.model
        return model.module if hasattr(model, 'module') else model

    def before_train(self, runner):
        model = self._model(runner)
        if not hasattr(model, 'initialize_ema'):
            raise TypeError('EMAUpdateHook requires a model with initialize_ema()')
        model.initialize_ema()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        accumulation = getattr(runner.optim_wrapper, '_accumulative_counts', 1)
        is_update = ((runner.iter + 1) % accumulation == 0
                     or runner.iter + 1 == runner.max_iters)
        if not is_update:
            return
        model = self._model(runner)
        model.update_ema()
