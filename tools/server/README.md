# AutoDL 环境与短程测试记录（2026-10-04）

当前服务器仅用于跑通代码，没有启动正式训练。测试配置采用真实图片、真实 DINOv3 权重、480×640 输入、batch size 1。每个任务只训练 2–4 步；所有测试 mIoU 都不代表模型效果。

## 环境和目录

| 内容 | 服务器路径或版本 |
|---|---|
| 代码 | `/root/autodl-tmp/code/QWSEG` |
| Conda 环境 | `/root/autodl-tmp/envs/qwseg`，Python 3.10 |
| PyTorch / torchvision | `2.2.2+cu121` / `0.17.2` |
| MMCV / MMEngine / MMDetection | `2.1.0` / `0.10.7` / `3.3.0` |
| Transformers / NumPy | `4.56.2` / `1.26.4` |
| DINOv3 | `/root/autodl-tmp/pretrain/dinov3-vitb16`，完整 HF 格式目录 |
| 数据 | `/root/autodl-tmp/data` |
| 安装、测试与审计日志 | `/root/autodl-tmp/code/QWSEG/work_dirs/server_setup` |

激活已安装的环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/qwseg
cd /root/autodl-tmp/code/QWSEG
```

不要使用原来的 base 环境进行 V100 训练：该环境的 PyTorch 实际 CUDA 运算曾报架构不支持。独立环境已经通过 GPU 运算、真实 DINOv3 前向和 MMCV 算子参与的训练。

`setup_v100_env.sh` 可创建 Python 3.10 环境并通过清华镜像安装依赖。MMCV 2.1.0 在本机源码编译，使用 CUDA 编译器和 g++。`restore_mmseg_assets.py` 从同版本 MMSeg wheel 提取缺失的词表，不覆盖本项目的 MMSeg 源码。

## 数据检查

- LLVIP：训练 12,025 对、测试 3,463 对，RGB/T 文件名配对完整。其框标注不是语义分割标签；当前用于 Stage 1 无标签适配。
- MFNet：train/val/test = 784/392/393。全部 1,569 个样本已检查图片可解码、尺寸、标签值与配对；集合之间没有重复。训练用 train.txt，验证用 val.txt，最终测试用 test.txt。
- FMB：RGB、T 与标签目录已发现，训练 1,220、验证 280；尚未对全量标签做与 MFNet 同等级的审计。
- PST900：已找到数据目录；尚未进行完整训练接入与审计。

## 短程验证

对应配置在 `seg/mmsegmentation-main-rgbt/configs/dino_ts/autodl_*_smoke.py`。

| 检查 | 步数 | 用途 |
|---|---:|---|
| Stage 1，LLVIP | 4 | 热红外适配、特征对齐反向传播 |
| Stage 2A，MFNet clean | 4 | 分割、验证、保存 checkpoint |
| Stage 2A，clean + degradation + missing | 2 | 退化库、模态缺失和三视图训练 |
| Stage 2B，EMA | 2 | Stage 2A 权重加载、EMA 创建、一致性损失 |
| Stage 3，soft | 2 | 教师加载、软门控训练、硬选择验证 |
| Stage 3，hard | 2 | 硬选择训练及验证 |

测试使用 AMP，分割部分动态 loss scaling 的初始值设为 128；默认 65,536 在前几步出现非有限梯度，不能把那次进程正常退出当作通过。保留的通过记录检查了实际 optimizer step 和参数有限性。

Stage 3 测试教师是短程 Stage 2A checkpoint，仅用于接口验证，没有使用训练完成的 EMA 教师。`check_smoke_checkpoint.py` 检查优化器实际更新步数，可额外逐张量核对冻结教师与源 checkpoint 完全一致，并检查软门控 Router 有参数更新。

2026-10-04 的六项检查全部通过：优化器实际步数分别为 4/4/2/2/2/2，模型参数均有限；两个 Stage 3 checkpoint 的冻结教师均逐张量一致，soft Router 的 6 个参数张量发生更新，hard Router 无更新（符合不可导 Top-K 的当前实现）。检查原始结果见服务器 `work_dirs/server_setup/verify_checkpoints.log`。训练进程已退出。

重新跑全套短程检查（会生成新的测试权重，占用磁盘）：

```bash
bash tools/server/run_smoke_checks.sh
```

这次修正了共享配置重复字段、LLVIP 无标签数据前缀、适配器零梯度初始化、退化过程对 padding 的处理，以及 Stage 3 教师加载早于框架初始化而被覆盖的问题。上述源码和辅助脚本同时保留在本地工作区和服务器，尚未提交到 Git。

## 正式四阶段训练

统一入口为 `tools/server/train_dino_ts.sh`。它使用固定工作目录，自动把上一阶段选中的权重交给下一阶段，并支持单卡或 `torchrun` 多卡训练：

```bash
cd /root/autodl-tmp/code/QWSEG
GPUS=4 bash tools/server/train_dino_ts.sh stage1
GPUS=4 bash tools/server/train_dino_ts.sh stage2a
GPUS=4 bash tools/server/train_dino_ts.sh stage2b
GPUS=4 bash tools/server/train_dino_ts.sh stage3
```

默认顺序与权重交接如下：

1. Stage 1 在 LLVIP 的 12,025 个 train 对上做无标签模态适配，每 5 epoch 在 3,463 个官方 test 对上计算不带标签的对齐损失。
2. Stage 2A 从 Stage 1 验证对齐损失最低的权重开始，在 MFNet 上进行稠密鲁棒分割训练。
3. Stage 2B 从 Stage 2A 的最佳 mIoU 权重开始，训练并验证 EMA 教师。
4. Stage 3 从 Stage 2B 的最佳 EMA 权重构造冻结教师与稀疏学生。

可用 `INIT_CKPT=/abs/path.pth` 覆盖 Stage 2A/2B 的输入权重，用 `TEACHER_CKPT=/abs/path.pth` 覆盖 Stage 3 教师。数据和预训练目录可分别通过 `QWSEG_DATA_ROOT`、`QWSEG_PRETRAIN_ROOT` 覆盖。

中断后使用相同阶段和工作目录恢复：

```bash
GPUS=4 RESUME=1 bash tools/server/train_dino_ts.sh stage2a
```

`--resume` 会从工作目录中的 `last_checkpoint` 恢复模型、优化器、AMP scaler、学习率调度器、epoch/iteration 和消息状态；Stage 2B checkpoint 也包含 EMA 教师及初始化状态。每 5 个 epoch 保存一次，因此突然中断最多需要重跑最近 5 个 epoch。更换卡数后通常也能恢复，但有效全局 batch size 会变化，不建议在同一次实验中途更换。

权重放在 `work_dirs/dino_ts/<stage>/weight/`。Stage 1 最多保留最近 2 个普通 checkpoint 和 1 个最佳对齐损失 checkpoint；Stage 2A、2B、3 各保留最近 2 个普通 checkpoint 和 1 个当前最佳 mIoU checkpoint，旧文件由 `CheckpointHook` 自动删除。EMA 和冻结教师会让 Stage 2B/3 的单个 checkpoint 较大。启动脚本默认要求工作盘至少剩余 10 GiB；空间已另行确认时可用 `MIN_FREE_GB=0` 关闭这项启动保护。

训练日志、loss、学习率、mIoU 和退化子集指标会写入普通日志及 TensorBoard；Stage 2/3 的验证预测图也会低频写入 `vis_data/`。启动查看：

```bash
tensorboard --logdir /root/autodl-tmp/code/QWSEG/work_dirs/dino_ts \
  --host 0.0.0.0 --port 6006
```

Stage 1 是无标签特征对齐，没有语义分割预测图，主要看 train loss 与 `stage1/align_loss`。这里沿用 LLVIP 官方基线把 test split 当验证集的协议；因此不能再把该 split 上的结果当作未参与模型选择的 LLVIP 测试结果。当前不建议给 DINO-TS 添加 `--visualize`：这个可选参数调用的是旧模型的重型训练特征图 Hook，尚未针对 DINO-TS 的 anchor/extra 输出适配；标量曲线和低频验证预测已经足够监控首轮正式训练。

## 换 GPU 与正式训练前

1. 保存代码、权重、数据和依赖清单。当前服务器登录提示明确写明 `/root/autodl-tmp` 不随系统镜像保存；新 Conda 环境也在此目录，不能只保存系统镜像就假定已完整备份。
2. 在目标 GPU 上核对 PyTorch/CUDA/MMCV 的架构支持。当前 MMCV 按 V100 的 `sm_70` 编译；换其他架构应重新构建或安装匹配版本。本安装脚本默认面向 V100，不保证适用于所有新 GPU。
3. 正式串联 Stage 1 → Stage 2A → Stage 2B → Stage 3 的启动、阶段交接和恢复逻辑已经补入统一脚本；Stage 2B 验证走 EMA 教师。仍需在重新开启的 GPU 服务器上完成一次“保存 → 终止 → 恢复”的端到端验收。
4. 检查正式稀疏预算课程：本次 soft 测试直接使用 K=600，验证 Router 可学习。正式配置从 K=N 起步，且 hard Top-K 索引不可导，需明确软训练到硬选择后的 Router 策略。
5. 为特征蒸馏和 logit 蒸馏补齐有效区域掩码；当前退化生成已保护 padding，但蒸馏损失的有效区域策略仍需完善。
6. 完成正式数据协议、基线、消融、独立退化 benchmark 与多随机种子实验，之后才能评价方法效果。

短程通过只说明这些配置能完成有限步训练、验证与保存，不表示长程收敛、EMA 全流程交接或论文指标已经验证。
