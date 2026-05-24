from mmengine.hooks import Hook


class ValDegradationHook(Hook):
    priority = 'LOW'

    def after_val_epoch(self, runner, metrics=None):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'compute_val_deg_metrics'):
            model.compute_val_deg_metrics()
