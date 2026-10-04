#!/usr/bin/env bash
# Run on Ubuntu with Miniconda installed.
# CUDA compiler and g++ must be installed. Builds MMCV for V100 (sm_70).
set -euo pipefail
QWSEG_ENV=${QWSEG_ENV:-/root/autodl-tmp/envs/qwseg}
QWSEG_ROOT=${QWSEG_ROOT:-/root/autodl-tmp/code/QWSEG}
if [ ! -x "$QWSEG_ENV/bin/python" ]; then
  "${CONDA_EXE:-/root/miniconda3/bin/conda}" create -y -p "$QWSEG_ENV" \
    --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.10 pip
fi
export PATH="$QWSEG_ENV/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export MAX_JOBS=${MAX_JOBS:-4}
export MMCV_WITH_OPS=1
QWSEG_INDEX=${QWSEG_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
test -x "$QWSEG_ENV/bin/python"
python -m pip install --timeout 30 --retries 2 -i "$QWSEG_INDEX" \
  numpy==1.26.4 torch==2.2.2 torchvision==0.17.2 setuptools==75.3.2 wheel ninja
python -m pip install --timeout 30 --retries 2 -i "$QWSEG_INDEX" \
  mmengine==0.10.7 mmdet==3.3.0 transformers==4.56.2 timm==1.0.19 \
  opencv-python==4.11.0.86 scipy matplotlib prettytable einops ftfy regex \
  tensorboard addict yapf==0.40.2
python -m pip install --timeout 30 --retries 2 -i "$QWSEG_INDEX" \
  --no-build-isolation --no-binary=mmcv mmcv==2.1.0
# The vendored MMSeg source omits this runtime asset; restore it from its wheel.
python "$QWSEG_ROOT/tools/server/restore_mmseg_assets.py" --index-url "$QWSEG_INDEX"
python - <<'PY'
import torch, mmcv, mmengine, mmdet
from mmcv.ops import MultiScaleDeformableAttention
from transformers import DINOv3ViTModel
print(torch.__version__, torch.cuda.get_arch_list())
print((torch.ones(4, device='cuda') + 1).cpu())
print('ENV_GPU_IMPORT_OK')
PY
