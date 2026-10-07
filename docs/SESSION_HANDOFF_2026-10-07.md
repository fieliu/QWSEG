# QWSEG 会话迁移与实验交接（2026-10-07）

> 把本文完整粘贴到新会话，并要求新会话先核查状态，再继续实验。本文记录的是截至 2026-10-06 21:24（北京时间）已经验证的状态。服务器可能被关机、重启或更换实例，因此所有进程、GPU、磁盘和 SSH 状态都要重新确认。

## 1. 希望新会话完成的工作

请接管 QWSEG 的 DINOv3 RGB-T 语义分割实验。不要重新采用已经证伪的旧训练方式，也不要直接用 Stage1 完整 checkpoint 初始化 Stage2。当前首要任务是从已经建立的 RGB 语义基线和 Stage1 热红外适配参数出发，完成干净双模态 Dense Teacher 的正式训练与消融；只有 Dense Teacher 达标后，才能继续退化鲁棒训练和稀疏学生蒸馏。

接管后先执行以下只读检查：

```bash
nvidia-smi
ps -ef | grep -E 'tools/(train|test)\.py' | grep -v grep
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/code/QWSEG/work_dirs/dino_ts
```

不要默认服务器仍有训练进程。最后一次检查时测试已经结束，没有继续运行训练。

## 2. 项目与环境

- 本地仓库：`C:\Users\lh\Desktop\学习资料\paper\QWSEG`
- 本地分支：`ablation-baseline`
- GitHub：`fieliu/QWSEG`
- 服务器最后使用的连接：`ssh -p 21348 root@region-9.autodl.pro`
- 不要把服务器密码、GitHub Token 或私钥写入仓库或本文档；连接时由用户单独提供。
- 服务器仓库：`/root/autodl-tmp/code/QWSEG`
- 服务器 Conda 环境：`/root/autodl-tmp/envs/qwseg`
- MMSeg 工程：`/root/autodl-tmp/code/QWSEG/seg/mmsegmentation-main-rgbt`
- MFNet：`/root/autodl-tmp/data/MFNet`
- 最后使用的 GPU：单卡 Tesla V100S 32 GB。

服务器仓库曾停留在旧提交 `573f133`，之后通过文件同步加入了本地的新修复，因此服务器工作树可能有未提交修改。本地远程分支才是源代码基准。**不要在服务器直接执行 `git reset --hard` 或盲目 `git pull`**；先查看：

```bash
cd /root/autodl-tmp/code/QWSEG
git status --short
git rev-parse --short HEAD
git diff --stat
```

然后备份服务器独有配置，或者逐文件与远程 `ablation-baseline` 比较后再同步。训练输出位于 `work_dirs/`，不应进入 Git。

## 3. 研究目标与当前结构

目标是在 MFNet RGB-T 语义分割上实现三个阶段：

1. **Stage1：热红外适配与跨模态关系对齐**
   - DINOv3 ViT-B/16 RGB 分支作为语义参照。
   - 前 3 个 Block 中，RGB/T 分别使用共享 DINO Block，并加入各自 Adapter。
   - 正式 Stage1 损失以关系对齐为主，弱区域对齐为辅：`lambda_patch=0`、`lambda_region=0.25`、`lambda_relation=1.0`。
   - 对齐只发生在 256 维公共投影空间，不用逐 Patch 强迫 Thermal 复制 RGB；原始 768 维 Thermal token 仍保留模态专有信息。

2. **Stage2A：干净 Dense RGB-T Teacher**
   - 输入 `480x640`，Patch Size 16，得到 `30x40=1200` 个位置。
   - 前 3 层后生成 `1200 Anchor + 1200 Extra`。
   - 2400 个 token 一起进入后 9 个共享 DINO Block；Decoder 读取最终 Anchor，Extra 通过注意力影响 Anchor。
   - 这一阶段只训练干净有标签分割，不混入退化、模态缺失、EMA 或剪枝。

3. **Stage2B / Stage3：鲁棒 Teacher 与稀疏 Student**
   - Stage2B 用 clean/weak EMA Teacher 监督退化或模态缺失的 Online Model。
   - Stage3 冻结 Dense Teacher，Student 保留全部 Anchor，只对 Extra 做真实 Top-K gather。
   - 这两个阶段的代码框架已经存在，但还没有得到可信的正式实验验证。

详细设计见 `docs/DINO_MULTISTAGE_TEACHER_STUDENT.md`。

## 4. 此前低精度的核心问题

旧实验第 5 轮只有约 **18.74 mIoU**，不是正常收敛，主要由以下实现错误造成。

### 4.1 随机投影破坏 DINOv3 预训练特征

旧 `AnchorExtraFusion` 中的 `v_rgb`、`v_t`、融合打分器和 Extra MLP 是随机初始化。RGB token 经过随机线性投影后，不再保留 DINOv3 的预训练表示，导致分割头几乎要重新学习特征空间。

已经修改为：

- `v_rgb`、`v_t` 单位映射初始化；
- Anchor 初始为约 `0.9 RGB + 0.1 Thermal`；
- Extra 以 Thermal 残差作为起点；
- Extra 输出末层零初始化；
- modality、anchor/extra type/position embedding 零初始化，避免训练开始时扰动预训练特征。

### 4.2 后 9 个 DINO Block 遗漏 RoPE

旧实现把 Anchor/Extra 拼接后送入后 9 层时没有继续传递旋转位置编码。空间关系因此被破坏，对密集分割影响很大。

现在已按 Anchor/Extra 的真实索引为 Dense 和 Sparse 路径重复或 gather RoPE。

### 4.3 Stage1 完整 checkpoint 污染 Stage2

Stage1 只真正训练了 Thermal PatchEmbed、Thermal modality embedding 和前 3 层 Thermal Adapter，但完整 checkpoint 还包含未训练的随机 Fusion、Decoder 等参数。旧流程把完整 Stage1 checkpoint 加载到 Stage2，覆盖了 Stage2 的正确初始化。

现在使用 `tools/extract_dino_ts_stage1.py`，只迁移 24 个张量、893,379 个已训练参数：

```text
backbone.patch_embed.thermal.*
backbone.modality_embed.thermal
backbone.adapters.thermal.*
```

### 4.4 其他已经修复的问题

- 恢复 Mask2Former 全部 Decoder 中间层的辅助分类、Mask BCE 和 Dice 损失。
- Hungarian matching cost 改为 FP32，避免 AMP/FP16 下出现 `inf`。
- Stage2A 改为纯净输入训练，不再从第一步同时学习干净、退化和缺失模态目标。
- Warmup 从 5 epoch 缩短为 1 epoch；原方案第一个验证点仍处于过低学习率阶段。
- 验证图片默认关闭绘制。此前 `SegVisualizationHook(draw=True)` 使测试约 1.54 秒/图、GPU 几乎空闲，并产生大量图片；该问题不影响 mIoU，但严重拖慢验证并占磁盘。

对应关键提交：

```text
830b569 fix: keep mask matching costs in fp32
ab51b2d fix: restore clean dense teacher training
284358c fix: preserve DINOv3 features through fusion
e7c6029 perf: disable validation image rendering by default
```

## 5. 已经完成的诊断实验

以下结果都基于 MFNet `train+val=1176` 张训练、官方 `test=393` 张评估。由于 test 被用于逐轮观察和选权重，这些是工程诊断结果，不能称为完全未参与调参的最终测试结果。

| 实验 | Epoch | mIoU | 说明 |
|---|---:|---:|---|
| 旧双模态实现 | 5 | 18.74 | 存在上述核心 bug |
| 修复后 RGB-only | 1 | 33.55 | 第 1 轮仍包含 warmup |
| 修复后 RGB-only | 2 | 46.14 | 证明数据、DINO、Decoder 和评估链路可工作 |
| 修复后 RGB-only | 3 | **52.52** | 当前 RGB 最佳诊断权重 |
| 修复后 RGB-only | 5 | 52.35 | 已接近平稳，epoch 3 仍最佳 |
| 修复后双模态从头训练 | 1 | 27.96 | 双倍 token 导致更大的预训练分布变化 |
| 修复后双模态从头训练 | 2 | 39.21 | 已远高于旧实现，但落后同轮 RGB-only |
| 双模态 epoch 2 权重强制 RGB-only 前向 | 2 | 36.36 | 同一权重的 Dense 前向为 39.21，Extra 已贡献约 +2.85 |
| RGB epoch 3 + Stage1 Thermal，直接切换 Dense、零微调 | 0 | **48.90** | 证明修复后的融合不再摧毁 DINO 特征 |

48.90 的逐类 IoU：

```text
unlabeled 97.79, car 87.95, person 58.59, bike 59.70,
curve 37.51, car_stop 40.00, guardrail 0.00,
color_cone 50.12, bump 8.39
```

最后一项从 RGB 的 52.52 下降到 48.90，但没有进行任何 Dense 双模态微调。这个结果表明当前主要矛盾已经从“错误前向”转为“2400-token 的分布变化和训练课程”。

## 6. 服务器上的重要权重和日志

```text
# Stage1 最佳完整权重（不要直接加载到 Stage2）
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage1/weight/stage1/best_stage1_align_loss_epoch_15.pth

# Stage1 可安全迁移到 Stage2 的小权重，约 3.5 MB
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage1/weight/stage1/best_stage1_align_loss_epoch_15_stage2_transfer.pth

# RGB-only 最佳权重，epoch 3，mIoU 52.52
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/diagnostic_rgb_only_trainval_e1/weight/diagnostic_rgb_only_trainval_e1/best_mIoU_epoch_3.pth

# RGB epoch 3 + Stage1 Thermal 合并权重，Dense 零微调 mIoU 48.90
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/rgb_e3_plus_stage1_thermal.pth

# 修复后双模态从头训练 epoch 2，mIoU 39.21
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_fixed_trainval_e2/weight/stage2a_fixed_trainval_e2/best_mIoU_epoch_2.pth

# 48.90 测试日志
/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/eval_rgb_warmstart_dense.log
```

权重通常约 807–845 MB。最后检查时 `dino_ts` 目录约 15 GB，数据盘剩余约 8.3 GB。应只保留关键 best、必要的 latest/resume 和日志；不要默认开启预测图片保存。删除任何文件前必须核对绝对路径，不要删上述关键权重。

## 7. 当前仍然存在的问题

### 7.1 双倍 token 的突变

Dense 路径把深层输入从 1200 个 RGB/Anchor token 突然变为 2400 个 Anchor+Extra token。即使新增 token 初始值合理，Self-Attention 的 softmax 分母和特征分布仍会发生变化。因此“双模态从头训练第 2 轮 39.21”低于“RGB-only 第 2 轮 46.14”并不等于 Thermal 无效。

当前更合理的顺序是：

```text
RGB/Anchor 语义预热
    -> 合入 Stage1 Thermal 参数
    -> 开启 Dense Extra 微调
    -> 干净 Dense Teacher 验收
    -> 鲁棒训练
    -> 稀疏蒸馏
```

### 7.2 Dense Teacher 还没有完成正式训练

48.90 是合并权重后零微调的诊断结果，不是最终 Dense Teacher。需要继续训练并确认它能稳定超过 RGB-only，而不是只看某一轮。

### 7.3 小类别仍然很弱

`guardrail=0`、`bump=8.39`，说明类别不平衡、小目标或 crop 采样仍需处理。应先确认标签统计和混淆矩阵，再考虑类别权重、稀有类采样或针对性 crop；不要在尚未确认数据统计前随意加大 loss 权重。

### 7.4 数据划分存在论文报告风险

当前 `trainval.txt` 有 1176 张，官方 `test.txt` 有 393 张，而且 test 已用于模型选择。正式论文最好同时保存两套协议：

- 开发/消融：官方 train 训练、val 选参；
- 最终模型：冻结方案后用 train+val 重训，官方 test 只评估一次。

如果继续使用当前 test-set protocol，论文必须明确 test 参与模型选择，不能与严格 held-out test 的论文结果直接等价比较。

### 7.5 尚未验证的研究模块

- Stage1 的最佳对齐损失是否对应最佳下游 mIoU，尚未做系统消融。
- Anchor-only、Thermal-only、简单 early/late fusion 等必要基线尚未完整完成。
- Stage2B EMA 鲁棒训练尚未得到可信结果。
- Stage3 Top-K 稀疏蒸馏尚未得到精度、延迟和显存结果。
- 退化库已有设计和代码基础，但尚未完成逐算子可视检查、参数标定、真实退化对照和冻结版 benchmark manifest。
- 结果尚缺多个随机种子，当前数字只能用于诊断趋势。

## 8. 建议的下一步执行顺序

### Step 1：安全同步并核查配置

确认以下本地修复在服务器存在：

```text
单位初始化的 v_rgb/v_t
零初始化 modality/type/position embedding
Dense/Sparse 深层 RoPE
Stage1 selective extractor
Mask2Former auxiliary losses
FP32 Hungarian cost
Stage2A clean-only objective
SegVisualizationHook(draw=False)
```

运行形状/前向 smoke test，并确认输出有限值。不要再跑已经完成的 5 轮 RGB 诊断。

### Step 2：从合并权重进行短周期 Dense 微调

先跑 5 个 epoch，每轮验证，用来确认 warm-start 曲线。建议命令模板：

```bash
cd /root/autodl-tmp/code/QWSEG/seg/mmsegmentation-main-rgbt
export MFNET_ROOT=/root/autodl-tmp/data/MFNet
export CUDA_VISIBLE_DEVICES=0

/root/autodl-tmp/envs/qwseg/bin/python tools/train.py \
  configs/dino_ts/stage2a_dense_mfnet.py \
  --work-dir /root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_warmstart_dense \
  --cfg-options \
    load_from=/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/rgb_e3_plus_stage1_thermal.pth \
    train_cfg.max_epochs=5 \
    train_cfg.val_interval=1 \
    default_hooks.checkpoint.interval=1 \
    default_hooks.checkpoint.max_keep_ckpts=1 \
    default_hooks.visualization.draw=False
```

不要使用 `--resume` 启动这个新实验；它是从合并权重初始化的新 run。若训练中断，需要在同一个 work-dir 使用 `--resume` 继续。

建议验收：

- 第 1–2 轮不应重新跌回 20 mIoU 左右；
- 数轮内应回到或超过 48.90；
- 目标是 Dense RGB-T 稳定超过 52.52 的 RGB-only 基线；
- 如果 5 轮仍明显落后，先做 Anchor-only 与 Extra gate/渐进加入实验，不要立即进入 Stage2B。

### Step 3：补齐最小必要消融

至少比较同一数据、训练预算和 seed 下的：

```text
RGB-only
Thermal-only
Anchor-only（仍为 1200 token）
Dense Anchor+Extra（2400 token）
无 Stage1
Stage1 关系+弱区域对齐
强逐 Patch 对齐（作为反例/消融）
```

若 Dense 始终不优于 Anchor-only，优先研究 Extra 的注意力门控或分阶段开放，而不是继续增加复杂损失。

### Step 4：Dense 达标后再进行 Stage2B

Stage2B 才加入退化、缺失模态和 EMA 一致性。先做 clean、单算子单级、整模态缺失三组验证，保证鲁棒性提升没有明显牺牲 clean mIoU。

### Step 5：最后进行 Stage3 稀疏学生

冻结通过验收的 Dense Teacher，训练 Utility Router；报告真实 gather 后的 token 数、mIoU、端到端延迟和峰值显存。不能只把未选择 token 乘零后宣称获得计算加速。

### Step 6：冻结退化 benchmark

每个算子和等级需要：固定参数范围、随机种子、作用模态、样本清单和输出 manifest；先人工检查代表图像，再批量生成。最终分别报告 clean、逐算子逐级、宏平均、RGB 缺失、Thermal 缺失以及真实夜间/恶劣条件，不把合成退化结果直接等同于真实部署能力。

## 9. 性能与资源信息

- V100S 单卡足以进行当前 Stage1/Stage2 实验，不需要为正确性改成 4 卡。
- RGB-only 诊断约 `0.34 s/iter`，框架统计显存约 3.3 GB。
- Dense 双模态约 `0.42 s/iter`，框架统计显存约 3.6 GB；`nvidia-smi` 曾观察到约 5.3 GB。
- 验证时曾看到 GPU 利用率低，原因是默认图片绘制和磁盘写入，不是模型计算本身；该默认行为已关闭。
- 当前代码可使用 PyTorch SDPA；在 V100 上不要假设能够使用现代 FlashAttention-2 内核。先完成可信基线，再做吞吐优化。

## 10. 新会话需要持续遵守的原则

1. 每次启动训练前记录 commit、完整 config、数据划分、初始化权重和随机种子。
2. 区分“工程诊断结果”“用于选参的验证结果”和“最终未参与调参的测试结果”。
3. 任何 Stage2 初始化都不能加载 Stage1 完整 checkpoint。
4. 在 Dense Teacher 未超过合理基线前，不继续堆叠 EMA、退化和剪枝。
5. 先检查 checkpoint 是否真正加载、可训练参数数量和初始验证结果，再进行长时间训练。
6. 保存空间有限：默认只保留 1 个普通 checkpoint 和 1 个 best，关闭批量图片绘制。
7. 修改代码后先在本地 `ablation-baseline` 提交并推送；服务器应尽量与远程提交对齐，避免长期保留无法复现的手工修改。

