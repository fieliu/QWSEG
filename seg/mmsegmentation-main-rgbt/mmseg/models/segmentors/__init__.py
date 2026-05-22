# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .depth_estimator import DepthEstimator
from .amdanet import AMDANet
from .encoder_decoder import EncoderDecoder
from .encoder_decoder_mult import EncoderDecoderMult
from .multimodal_encoder_decoder import MultimodalEncoderDecoder
from .rgbt_v1_baseline import RGBTv1Baseline
from .rgbt_v1_sam_baseline import RGBTv1SAMBaseline
from .rgbt_v2_disentangle import RGBTv2Disentangle
from .rgbt_v2_sam_disentangle import RGBTv2SAMDisentangle
from .rgbt_v3_degradation import RGBTv3Degradation
from .rgbt_v3_sam_degradation import RGBTv3SAMDegradation
from .rgbt_v4_quality_pruning import RGBTv4QualityPruning
from .rgbt_v4_sam_quality_pruning import RGBTv4SAMQualityPruning
from .rgbt_v5_quality_joint import RGBTv5QualityJoint
from .rgbt_v5_sam_quality_joint import RGBTv5SAMQualityJoint
from .mitmul_v1_baseline import MiTMulV1Baseline
from .mitmul_v2_disentangle import MiTMulV2Disentangle
from .mitmul_v3_degradation import MiTMulV3Degradation
# from .mitmul_v4_degradation import MiTMulV4Degradation  # 暂时注释，导入有问题
from .mitmul_v4_quality_pruning import MiTMulV4QualityPruning
from .mitmul_v5_quality_joint import MiTMulV5QualityJoint
from .mitmul_v6_disentangle import MiTMulV6Baseline, MiTMulV6Disentangle, MiTMulV7Degradation, MiTMulV7DegradationFull
from .mitmul_v8_quality_pyramid import MiTMulV8QualityPyramid
from .mitmul_v9_quality_gated import MiTMulV9QualityGated
from .mitmul_v7_quality_adaptive import MiTMulV7QualityAdaptive
from .mitmul_v10_quality_embed import MiTMulV10QualityEmbed
from .rgbt_v1_sam_native import RGBTv1SAMNative
from .rgbt_v2_sam_native import RGBTv2SAMNative
from .rgbt_v1_sam2 import RGBTv1SAM2
from .swinmul_v11_mask_mae import SwinMulV11MaskMAE
from .swinmul_v6_mask2former import SwinMulV6Mask2Former
from .swinmul_v12_quality_disentangle import SwinMulV12QualityDisentangle
from .swinmul_v12d_quality_disentangle import SwinMulV12DQualityDisentangle
from .swinmul_v12_nodeg_quality_disentangle import SwinMulV12QualityDisentangleNoDeg
from .swinmul_v12_disentangle_only import SwinMulV12DisentangleOnly
from .mitmul_v12d_quality_disentangle import MiTMulV12DQualityDisentangle
from .mitmul_v12_disentangle_only import MiTMulV12DisentangleOnly
from .mitmul_v12_nodeg_quality_disentangle import MiTMulV12QualityDisentangleNoDeg
from .swin_baseline import SwinBaseline
from .crm_mask2former import CRMMask2Former
from .mask2former_rgbt_add import Mask2FormerRGBTAdd
from .seg_tta import SegTTAModel
from .swin_dual_add import SwinDualAdd
from .mitmul_ablation import MiTMulABBaseline, MiTMulABV1, MiTMulABV2, MiTMulABV3, MiTMulABV5, MiTMulABV6, MiTMulABV7, MiTMulABV8
from .swin_multibranch_noquality import SwinMultiBranchNoQuality
from .swin_baseline_mask2former import SwinBaselineMask2Former

__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'SegTTAModel',
    'MultimodalEncoderDecoder', 'DepthEstimator', 'RGBTv1Baseline',
    'RGBTv1SAMBaseline', 'RGBTv1SAMNative', 'RGBTv2Disentangle',
    'RGBTv2SAMDisentangle', 'RGBTv2SAMNative', 'RGBTv3Degradation',
    'RGBTv3SAMDegradation', 'RGBTv4QualityPruning',
    'RGBTv4SAMQualityPruning', 'RGBTv5QualityJoint', 'RGBTv5SAMQualityJoint',
    'MiTMulV1Baseline', 'MiTMulV2Disentangle', 'MiTMulV3Degradation',
    # 'MiTMulV4Degradation',  # 暂时注释，导入有问题
    'MiTMulV4QualityPruning', 'MiTMulV5QualityJoint',
    'MiTMulV6Baseline', 'MiTMulV6Disentangle', 'MiTMulV7Degradation',
    'MiTMulV7DegradationFull',
    'MiTMulV8QualityPyramid',
    'MiTMulV9QualityGated',
    'MiTMulV7QualityAdaptive',
    'MiTMulV10QualityEmbed',
    'RGBTv1SAM2',
    'SwinMulV11MaskMAE',
    'SwinMulV6Mask2Former',
    'SwinMulV12QualityDisentangle',
    'SwinMulV12DQualityDisentangle',
    'SwinMulV12QualityDisentangleNoDeg',
    'SwinMulV12DisentangleOnly',
    'MiTMulV12DQualityDisentangle',
    'MiTMulV12DisentangleOnly',
    'MiTMulV12QualityDisentangleNoDeg',
    'SwinBaseline',
    'MiTMulABBaseline', 'MiTMulABV1', 'MiTMulABV2', 'MiTMulABV3', 'MiTMulABV5', 'MiTMulABV6', 'MiTMulABV7', 'MiTMulABV8',
    'SwinDualAdd',
    'AMDANet', 'EncoderDecoderMult',
    'CRMMask2Former', 'Mask2FormerRGBTAdd',
    'SwinMultiBranchNoQuality',
    'SwinBaselineMask2Former',
]
