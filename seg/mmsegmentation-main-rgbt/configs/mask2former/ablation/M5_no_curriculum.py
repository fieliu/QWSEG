# Ablation M5: remove the three-phase training curriculum.
# skip_phases=True => phase 3 from epoch 0: quality predictors and
# degradation training are active from the start (no warmup). Architecture
# unchanged. => necessity of the staged curriculum.
_base_ = ['../swinmul_quality_mask2former_swin-t_1xb2-200E_mfnet-480x640.py']

model = dict(skip_phases=True)
