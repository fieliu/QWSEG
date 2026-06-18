# Teacher: clean dual-branch shared-Swin Mask2Former, concat+reduce fusion.
# Train this FIRST on clean RGB-T; its checkpoint feeds the student.
_base_ = ['../mask2former/swinmul_v6_mask2former_swin-t_1xb2-80K_mfnet-480x640.py']

custom_imports = dict(
    imports=['mmseg.models.segmentors.quality_distill_teacher',
             'mmseg.models.segmentors.swinmul_v6_mask2former',
             'mmseg.engine',
             'mmseg.datasets.mfnet',
             'mmseg.datasets.transforms.loading',
             'mmdet.models'],
    allow_failed_imports=False)

# only swap the segmentor type + add concat-reduce fusion dims; everything
# else (backbone / decode_head / schedule / dataset) inherited from V6.
model = dict(
    type='QualityDistillTeacher',
    fusion_dims=(96, 192, 384, 768),
)
