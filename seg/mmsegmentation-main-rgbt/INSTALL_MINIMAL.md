# QWSEG 最小环境配置
# 手动安装步骤：

# 1. 检查当前环境（请在终端执行）
# python --version
# conda env list (如果使用 conda)

# 2. 创建并激活环境（可选）
# conda create -n qwseg python=3.9 -y
# conda activate qwseg

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# 你的系统：CUDA 12.7 (NVIDIA Driver 566.14)
# 推荐方案：使用 PyTorch 2.x + CUDA 12.1 或更高版本
#
# 访问 https://pytorch.org/get-started/locally/
# 示例：
# CUDA 12.1 (兼容 12.7):
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# 或者使用 conda:
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. 安装 openmim（用于自动安装 mmcv）
pip install openmim

# 5. 使用 mim 安装 mmcv（自动匹配 PyTorch 和 CUDA）
mim install "mmcv>=2.0.0rc4,<2.2.0"

# 6. 安装 mmengine
pip install "mmengine>=0.5.0,<1.0.0"

# 7. 安装 NumPy + OpenCV（两个包必须一起装，版本互相约束）
# - NumPy 必须 < 2.0，否则 PyTorch 初始化报错 _ARRAY_API not found
# - OpenCV 必须 < 4.11 且用 headless 版本：
#   opencv-python 依赖 Qt (服务器缺 libQt5Core)
#   opencv-python-headless >= 4.11 要求 numpy>=2，与 PyTorch 冲突
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null
pip install "opencv-python-headless<4.11" "numpy<2.0"

# 8. 安装其他依赖
pip install matplotlib packaging prettytable scipy ftfy regex timm

# 9. 安装项目本身（开发模式）
cd /path/to/mmsegmentation-main-rgbt
pip install -e .

# 10. 安装 TensorBoard
pip install tensorboard tensorboardX

# 11. 验证安装
python -c "import torch; import mmcv; import mmseg; import cv2; print('All imports OK')"
