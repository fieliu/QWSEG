"""Two-step EMA runtime check; warm-start via --cfg-options load_from=... ."""
_base_ = ['autodl_stage2a_robust_smoke.py']
model = dict(type='DinoTSDenseEMA', lambda_anchor=1.0, lambda_global=0.0)
custom_hooks = [dict(type='EMAUpdateHook')]
