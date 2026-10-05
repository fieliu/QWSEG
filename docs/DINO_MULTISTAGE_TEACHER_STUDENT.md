# DINO 多阶段 Dense-Teacher / Sparse-Student 设计

> 状态：目标架构设计，尚未对应当前 Swin/质量网络实现。
>
> 第一阶段研究范围：RGB-T 闭集鲁棒语义分割。
>
> 明确约束：不使用无标签图像的语义伪标签；无标签数据仅参与自监督特征学习与 Teacher–Student 特征一致性训练。

## 1. 设计目标

本方案希望同时解决三个问题：

1. 将 RGB 预训练的 DINO ViT 扩展到 Thermal，后续再扩展到 Depth、Event-frame 等图像网格模态；
2. 在单模态退化、整模态缺失和局部退化下保持语义分割性能；
3. 始终保留完整空间语义网格，仅压缩跨模态冗余 Token，从而获得可验证的计算收益。

本方案不尝试：

- 开放词汇分割；
- 从无标签数据生成语义 mask 作为监督；
- 在第一版中统一处理原始点云、异步事件流和 Radar 原始信号；
- 将合成退化等同于真实传感器鲁棒性。

## 2. 核心设计结论

### 2.1 每个空间位置始终保留一个 Anchor Token

设每个模态产生 `N` 个 patch token。RGB-T 输入最初包含 `2N` 个 Token。融合后将每个位置的两个模态 Token 重参数化为：

- `N` 个 Anchor Token：每个空间位置一个，永不剪枝；
- `N` 个 Extra Token：表示 RGB 与 Thermal 的互补差异，可按固定预算 Top-K 压缩。

因此序列长度满足：

```text
Dense Teacher: N anchors + N extras = 2N
Sparse Student: N anchors + K extras, 0 <= K <= N
```

分割解码器只读取 `N` 个 Anchor Token，并将其恢复为二维特征图。

### 2.2 共享 ViT 不等于从输入开始混合所有模态

推荐采用“浅层独立、深层混合”：

```text
RGB PatchEmbed ─────┐
                    ├─ Shared DINO Blocks 1...R（参数共享、注意力独立）
T PatchEmbed ───────┘
                              ↓
                     Anchor / Extra Fusion
                              ↓
                 Shared DINO Blocks R+1...L
                       （跨模态全局注意力）
```

前 `R` 个 Block 中可将模态维合并到 batch 维，形状从 `[B, M, N, D]` 转换为 `[B*M, N, D]`。不同模态使用相同的 Attention 和 FFN 参数，但不会互相注意。

融合后，Anchor 和 Extra 组成同一序列，后续 DINO Block 执行跨模态全局自注意力。

### 2.3 Adapter 是共享 FFN 的补充，不是替代

这里的 Adapter **不是 DINOv3 Block，也不是从 DINOv3 FFN 复制出来的一层**。RGB 与 Thermal 在融合前调用同一组预训练 DINOv3 Block；二者仅使用不同的 PatchEmbed、Modality Embedding 和低秩 Adapter。换言之，共享的是完整的 Attention/FFN 主路径，Adapter 只学习模态相关的残差修正。

融合前的第 `l` 个 Block 对模态 `m` 执行：

```text
u_m = h_m + SharedAttention_l(LN(h_m))
h'_m = u_m + SharedFFN_l(LN(u_m)) + gamma_l,m * Adapter_l,m(LN(u_m))
```

其中：

- `SharedAttention_l`：所有模态共用；
- `SharedFFN_l`：所有模态共用；
- `Adapter_l,m`：每个模态独立的低秩残差模块；
- `gamma_l,m`：可学习缩放系数；
- Adapter 上投影零初始化，使初始路径尽量保持 DINO 行为。

推荐 Adapter 结构：

```text
LayerNorm(D) -> Linear(D, d_adapter) -> GELU -> Linear(d_adapter, D)
```

对于 ViT-B，可从 `d_adapter=64` 开始。Adapter 只放在融合前的前 `R=3` 或 `R=4` 个 Block。融合后 Token 已不再属于单一模态，不再使用模态专属 Adapter。

当前 ViT-B、`R=3` 实现的初始化如下：

| 部件 | 初始化方式 | Stage 1 状态 |
|---|---|---|
| 12 个共享 DINO Block | DINOv3 ViT-B 预训练权重；RGB/T 调用同一参数对象 | 冻结 |
| RGB PatchEmbed | 复制 DINOv3 PatchEmbed 卷积权重 | 冻结 |
| Thermal PatchEmbed（三通道） | 直接复制 RGB/DINOv3 PatchEmbed 权重 | 训练 |
| Thermal PatchEmbed（单通道） | 对 RGB 三通道卷积核按输入通道求均值 | 训练 |
| RGB Adapter | 新建；`up` 权重和偏置置零，初始残差严格为零 | 冻结 |
| Thermal Adapter | 新建；`up` 权重和偏置置零，初始残差严格为零 | 训练 |
| RGB/T Modality Embedding | 截断正态小值初始化 | 仅 Thermal 训练 |
| Teacher/Student Projector | 先用相同参数初始化，随后固定 Teacher | Student 训练 |

Adapter 的瓶颈形状为 `768 -> 64 -> 768`，与 DINOv3 FFN 的形状和职责不同，不能直接用 DINOv3 前几层 FFN 权重初始化。虽然 RGB/T Adapter 结构相同、技术上可以互相复制，但初始 RGB Adapter 本身没有预训练知识，而且两者初始输出都为零，复制不会带来有效先验。当前方案保留零输出初始化，使训练起点等价于原始 DINOv3 主路径，再由 Thermal Adapter 在 Stage 1 学习红外修正量。

### 2.4 Projector 只用于训练损失

必须区分：

- `PatchEmbed_m`：模态专属输入投影，训练和推理都保留；
- `AlignmentProjector`：将 Token 投影到公共对齐空间，只在训练损失中使用，推理时删除。

推荐形式：

```text
ModalitySpecificLayerNorm -> SharedTwoLayerProjector -> L2 Normalize
```

不推荐为每个模态配置完全独立的 Projector。完全独立的 Projector 可能独自吸收对齐任务，使 Backbone Token 实际上没有对齐。

Teacher Projector 与 Teacher Backbone 一起冻结或通过 EMA 更新；Student Projector 接收梯度。固定 Teacher 目标时，不需要额外的特征方差损失来防止坍塌。

## 3. Anchor–Extra 融合

### 3.1 RGB-T 的可解释形式

融合点处，第 `i` 个空间位置具有 RGB Token `r_i` 和 Thermal Token `t_i`。先预测融合权重：

```text
c_i = concat(r_i, t_i, abs(r_i - t_i), r_i * t_i)
[alpha_rgb_i, alpha_t_i] = softmax(FusionScorer(c_i))
```

生成 Anchor：

```text
a_i = alpha_rgb_i * V_rgb(r_i) + alpha_t_i * V_t(t_i)
```

生成一个 Extra 候选：

```text
e_i = ExtraMLP(concat(r_i - t_i, r_i, t_i, a_i))
```

概念上，最简单的线性形式为：

```text
a_i = alpha_i * r_i + (1 - alpha_i) * t_i
e_i = r_i - t_i
```

在保留 `alpha_i` 的情况下，线性形式可恢复原 Token：

```text
r_i = a_i + (1 - alpha_i) * e_i
t_i = a_i - alpha_i * e_i
```

因此 Dense 阶段的 `Anchor + Extra` 可以理解为从“RGB/T 模态坐标”到“共同语义/模态差异坐标”的重参数化，而不是立即丢弃信息。实际实现使用 MLP 后不宣称严格数学可逆，只保留该设计动机。

### 3.2 Dense 与 Sparse 路径

Dense 路径保留所有 Extra：

```text
z_dense = concat([a_1, ..., a_N], [e_1, ..., e_N])
```

Sparse 路径为每个 Extra 预测任务效用：

```text
u_i = UtilityRouter(concat(a_i, e_i, abs(a_i - e_i), a_i * e_i))
indices = TopK(u, K)
z_sparse = concat(all_anchors, gather(extras, indices))
```

`u_i` 表示在固定预算下该 Extra 对恢复 Dense Teacher 语义表征的边际价值，不解释为绝对图像质量，也不能使用 Extra 范数代替。

每个 Extra 使用：

```text
对应空间位置编码 + Extra 类型编码 + 来源模态编码
```

最终 Decoder 只读取序列前 `N` 个 Anchor。Extra 通过后续全局注意力影响 Anchor，最后可直接丢弃。

### 3.3 从 RGB-T 扩展到 RGB-X

输入统一组织为：

```text
tokens: [B, M, N, D]
availability_mask: [B, M]
```

每个模态拥有独立的 `PatchEmbed`、输入归一化和浅层 Adapter；共享 ViT 与融合接口不变。

对 `M` 个模态，可生成：

- 一个加权 Anchor；
- 最多 `M-1` 组独立 Extra 基；
- 从所有 Extra 候选中选择固定预算 Top-K。

第一版仅实现 RGB-T。只有在至少增加 RGB-D 或另一种图像网格模态并完成实验后，论文才使用 RGB-X 表述。

## 4. Teacher 的三个不同角色

为避免概念混淆，训练中存在三种不同 Teacher：

### 4.1 固定 RGB DINO Teacher

仅用于阶段一，将 DINO 的 RGB 语义先验迁移到 Thermal Adapter。它不被称为多模态 Teacher，也不负责 Token 剪枝。

### 4.2 Dense EMA Target

阶段二中，在 Dense 模型已经具备基本分割能力后，由 Dense Online Model 复制并通过 EMA 更新。它负责无标签数据上的稳定特征目标。

### 4.3 Frozen Dense Robust Teacher

阶段二结束后，从验证集上选择最佳 Dense EMA 权重并冻结。阶段三中，它保留全部 Extra，指导 Sparse Student 学习固定 Token 预算。

EMA 不创造新知识。多模态知识来自：

- DINO 的 RGB 视觉先验；
- 配对 RGB-T 数据的跨模态对应；
- 有标注数据的语义分割监督；
- 模态非对称退化和整模态缺失训练。

## 5. 总体训练流程

论文中写为三个阶段；工程上阶段二包含 `2A/2B` 两个子阶段。

```text
Stage 1  Modality Adaptation
    LLVIP 配对 RGB-T，无语义标签
    ↓
Stage 2A Clean Dense Teacher
    MFNet clean，使用真实分割标签建立强干净基线
    ↓
Stage 2B Dense EMA Self-Distillation
    可选的 clean / degraded / missing 鲁棒化 + EMA 特征一致性
    ↓
Stage 3  Dense-to-Sparse Distillation
    MFNet 分割监督 + Dense Teacher 蒸馏
```

全流程不生成或使用无标签语义伪标签。

## 6. Stage 1：模态适配

### 6.1 目标

让 Thermal PatchEmbed 和浅层 Adapter 产生能够被共享 DINO Attention/FFN 处理的 Token，同时只对齐公共语义，不要求 Thermal 完全复制 RGB 表征。

### 6.2 数据

- 配准较好的无标签 RGB-T 对；
- 可使用 LLVIP、KAIST 等配对数据；
- 必须按场景或视频去重，避免与后续验证/测试数据重叠；
- 当前正式配置关闭逐 Patch 对齐，因此不会直接把严重低照 RGB 的每个位置强加给 Thermal。区域/关系损失仍要求配对数据基本可靠；首轮训练前应剔除明显错配样本，后续可将 RGB 区域可靠性加权作为消融扩展。

### 6.3 冻结与训练参数

冻结：

```text
RGB DINO PatchEmbed
共享 DINO 主干
Teacher AlignmentProjector
分割 Decoder
Fusion / Utility Router
```

训练：

```text
Thermal PatchEmbed
Thermal Modality Embedding
前 R 个 Block 的 Thermal Adapter
Student AlignmentProjector
```

RGB Adapter 在 Stage 1 冻结。其上投影为零，因此虽然模块存在，初始残差严格为零，RGB 路径等价于原始 DINOv3 主路径；进入 Stage 2 后再与 Thermal Adapter 一起训练。

### 6.4 对齐损失

对于配准可靠的 patch：

```text
z_t_i = normalize(g_student(h_t_i))
z_rgb_i = stopgrad(normalize(g_teacher(h_rgb_i)))

L_cross_patch = mean_i(1 - cosine(z_t_i, z_rgb_i))
```

对于存在轻微错位的数据，先对局部窗口或区域做池化：

```text
L_cross_region = mean_j(1 - cosine(z_t_region_j, z_rgb_region_j))
```

关系对齐先将 `30x40` Patch Token 以 `2x2` 区域池化为 `15x20=300` 个区域，并计算每个模态内部的区域余弦关系：

```text
S_t[j,k]   = cosine(q_t_j, q_t_k)
S_rgb[j,k] = cosine(q_rgb_j, q_rgb_k)

p_t[j]   = softmax(mask_diag(S_t[j]) / tau)
p_rgb[j] = stopgrad(softmax(mask_diag(S_rgb[j]) / tau))

L_cross_relation = mean_j KL(p_rgb[j] || p_t[j])
```

对角线是恒为 1 的自相似度，不提供跨模态信息，因此从 softmax 中排除。当前温度 `tau=0.2`。关系对齐只要求红外保持与 RGB 相近的区域结构，不要求红外向量逐点等于 RGB；但纯关系损失对特征空间的整体旋转不敏感，因此保留一个较弱的区域余弦约束来建立共同坐标系。

阶段一总损失：

```text
L_stage1 = lambda_patch * L_cross_patch
         + lambda_region * L_cross_region
         + lambda_relation * L_cross_relation
```

正式配置采用 `lambda_patch=0`、`lambda_region=0.25`、`lambda_relation=1.0`。
`L_cross_relation` 在 2x2 池化区域上匹配模态内部的余弦相似度分布，忽略恒为 1 的对角线；它保留空间语义关系而不要求红外特征逐点复制 RGB。弱 `L_cross_region` 用于消除纯关系对齐的坐标旋转歧义。由于 Teacher Backbone 和 Teacher Projector 固定，默认不加入 `L_var`。

三个对齐项都只作用于 256 维公共投影。原始 768 维 RGB/T Token 不被替换，并继续进入 Stage 2 的 Anchor/Extra Fusion，因此 Thermal 的热辐射等模态专有信息仍有独立通路。

### 6.5 退出条件

- Thermal 特征不出现数值坍塌；
- RGB/T 对应区域的特征相似度显著高于随机区域；
- 使用冻结共享 ViT 的轻量线性探针时，Thermal 表现优于仅复制 RGB PatchEmbed 的初始化；
- 不要求阶段一直接获得最终语义分割性能。

## 7. Stage 2A：Clean Dense Teacher

### 7.1 目标

不依赖 Teacher，先只用干净 RGB-T 和真实标签训练保留全部 Token 的 Dense 多模态分割模型。该阶段先确认语义、融合、数据和评估闭环正确，并产出后续蒸馏所需的强 Dense Teacher；退化、模态缺失、EMA 和 Token 剪枝都不得混入这一基线。

### 7.2 Dense 注意力路径

```text
Blocks 1...R:
    RGB/T 分别执行共享 Attention/FFN，并叠加各自 Adapter

Fusion:
    每个位置生成一个 Anchor 和一个 Extra

Blocks R+1...L:
    concat(N anchors, N extras)
    执行跨模态全局 Self-Attention

Decoder:
    仅读取 N anchors
```

当前输入为 `480x640`、Patch Size 为 16，因此网格为 `30x40`，`N=1200`。Stage 2A/2B 的 Dense 序列为：

```text
[1200 Anchor ; 1200 Extra] -> [B, 2400, 768]
```

拼接后的 2400 个 Token 统一进入后 9 个共享 DINO Block。Anchor 与 Extra 可以通过同一个 Self-Attention 相互读取信息。深层处理结束后只截取序列前 1200 个 Anchor，恢复为 `[B,768,30,40]`，再通过 Feature2Pyramid 和 Mask2Former 输出分割结果；Extra 不直接输入 Decoder，而是通过深层注意力影响 Anchor。Stage 3 的最终硬稀疏路径保留全部 1200 个 Anchor，只将 Extra 从 1200 个压缩到 `K=600`。

### 7.3 可训练参数

```text
RGB/T PatchEmbed 与 Modality Embedding
RGB/T 浅层 Adapter
Anchor/Extra Fusion
共享 DINO 后半部分或全部 Blocks（较小学习率）
Segmentation Decoder
```

建议差异学习率：

```text
PatchEmbed / Adapter / Fusion / Decoder: 1.0 x base_lr
DINO 后半部分: 0.1 x base_lr
DINO 前半部分: 0.01 x base_lr 或冻结
```

### 7.4 干净语义监督

当前正式实现直接使用 MFNet 这类带像素级语义标签的 RGB-T 分割数据。数据加载器读取配对 RGB、Thermal 和分割标注；训练、验证、测试分别使用互不重叠的 `train.txt`、`val.txt`、`test.txt`。Stage 2A 从 Stage 1 选出的最佳权重初始化，Stage 2B 从 Stage 2A 验证集 mIoU 最佳权重初始化。

分割损失记为：

```text
L_seg = L_CE + lambda_dice * L_Dice
```

如果使用 Mask2Former，则沿用其分类、Mask BCE 和 Dice 组合损失。

Stage 2A 只计算干净输入的分割损失：

```text
L_stage2A = L_seg(f(x), y)
```

实现中设置 `lambda_deg=0`、`lambda_missing=0`。Mask2Former 对最后一层及全部中间 decoder 层计算分类、Mask BCE 和 Dice 辅助监督，保持与标准训练方式一致。

### 7.5 Dense 模型验收条件

只有满足以下条件，才能进入 EMA 自蒸馏：

- RGB-T Dense 模型优于 RGB-only 和 Thermal-only 基线；
- 干净输入性能达到可接受基线；
- Anchor 输出能够支持稳定分割。

退化和缺失性能在此阶段只作为监控项，不参与优化。通过干净基线验收后，再进入 Stage 2B 做可选鲁棒化，或者直接以该权重作为 Stage 3 的冻结 Dense Teacher，取决于消融方案。

## 8. Stage 2B：Dense EMA 鲁棒自蒸馏

### 8.1 Teacher 初始化与更新

从通过验收的 Dense Online Model 复制：

```text
theta_ema <- theta_online
```

EMA Teacher 不接收梯度：

```text
theta_ema <- momentum * theta_ema
           + (1 - momentum) * theta_online
```

`momentum` 可从 `0.996` 平滑增加到 `0.9999`。Teacher 与 Online Model 结构相同，均保留全部 Extra；两者不是同一个参数对象。

### 8.2 输入视图

```text
Dense EMA Target: 原始或弱增强 x_weak
Dense Online Model: 强退化、局部退化或模态缺失 x_strong
```

Teacher 不接收退化类型或严重程度编码。它的权重会通过接受退化监督的 Online Model 的 EMA 更新而逐渐获得鲁棒性。

### 8.3 Anchor 特征一致性

```text
L_anchor = mean_i(
    1 - cosine(
        normalize(a_online_i(x_strong)),
        stopgrad(normalize(a_ema_i(x_weak)))
    )
)
```

### 8.4 可选全局 DINO 风格损失

```text
L_global_dino = CrossEntropy(
    sharpen(center(p_ema(x_weak))),
    p_online(x_strong)
)
```

该项借鉴 DINO 的 EMA Teacher、stop-gradient、centering 和 sharpening，仅作为辅助。密集分割的主要约束仍是逐位置 Anchor 一致性和真实分割标签。

### 8.5 有标签与无标签 Batch

有标签 Batch：

```text
L_labeled = L_stage2A
          + lambda_anchor * L_anchor
          + lambda_global * L_global_dino
```

无标签 Batch：

```text
L_unlabeled = lambda_anchor * L_anchor
            + lambda_global * L_global_dino
```

无标签 Batch 不计算语义交叉熵、不生成 hard mask、不使用 Teacher 分割 logits 作为 soft pseudo-label。

### 8.6 阶段二输出

在验证集上选择最优 EMA 权重，保存为：

```text
Frozen Dense Robust Teacher
```

它应具备完整多模态理解和鲁棒性，但没有计算预算约束。

## 9. Stage 3：Dense-to-Sparse 蒸馏

### 9.1 初始化

```text
Sparse Student <- Frozen Dense Robust Teacher weights
```

新增 Utility Router，并保持 Teacher 完全冻结。

### 9.2 Token 路径

```text
Frozen Teacher: N anchors + N extras
Sparse Student: N anchors + TopK(extras, K)
```

固定 `K` 直接约束预算，不额外使用保留率损失。训练初期可使用 soft gate 或 Straight-Through Gumbel-TopK；正式推理必须通过 `gather` 缩短序列，不能只将未选择 Token 乘零。

### 9.3 同退化压缩蒸馏

Teacher 和 Student 接收相同退化输入：

```text
L_compression = mean_i(
    1 - cosine(
        a_sparse_i(C(x)),
        stopgrad(a_dense_i(C(x)))
    )
)
```

该项隔离“仅由 Token 压缩产生的误差”，直接训练 Router 选择对当前退化场景最有用的 Extra。

### 9.4 干净到退化鲁棒蒸馏

```text
L_robust = mean_i(
    1 - cosine(
        a_sparse_i(C(x)),
        stopgrad(a_dense_i(x))
    )
)
```

该项要求 Sparse Student 在退化输入下保持接近 Dense Teacher 干净输入的语义表示。对于会造成真实信息不可恢复的极端区域，可根据已知退化 mask 或 Teacher 特征稳定性降低权重，但不根据 Teacher 预测生成语义标签。

### 9.5 有标签 Logit 蒸馏

仅在已有真实标注的数据上，可加入：

```text
L_logit = T^2 * KL(
    softmax(logits_dense(C(x)) / T),
    softmax(logits_sparse(C(x)) / T)
)
```

该项属于有标签样本上的知识蒸馏，不用于给无标签数据生成伪标签。

### 9.6 Stage 3 总损失

有标签 Batch：

```text
L_stage3_labeled = L_seg
                 + lambda_comp * L_compression
                 + lambda_rob * L_robust
                 + lambda_logit * L_logit
```

无标签 Batch：

```text
L_stage3_unlabeled = lambda_comp * L_compression
                   + lambda_rob * L_robust
```

无标签 Batch 仍不使用语义伪标签。

### 9.7 渐进预算

训练早期可逐步降低预算：

```text
K = N -> 0.75N -> 0.5N -> target_K
```

最终模型和所有速度评估必须使用固定 `target_K`。如果采用多次 Router，可设置：

```text
after block R: K1 = 0.5N
after block R+3: K2 = 0.25N
after block R+6: K3 = 0
```

第一版优先实现单次 Router，再将多阶段预算作为后续增强。

## 10. 损失与 DINO 的关系

| 组件 | 来源 | 是否核心 |
|---|---|---:|
| DINO 预训练 ViT 权重 | DINO | 是 |
| EMA Teacher / stop-gradient | DINO/BYOL 风格 | 是 |
| 全局概率交叉熵、centering、sharpening | 原始 DINO 风格 | 可选 |
| Patch/Anchor 特征一致性 | 密集任务自蒸馏 | 是 |
| Masked patch teacher target | 更接近 iBOT/DINOv2 范式 | 后续可选 |
| 跨模态对齐 `L_cross` | 本任务适配目标 | 是 |
| Dense-to-Sparse Anchor 蒸馏 | 本架构目标 | 是 |
| CE/Dice/Mask2Former Loss | 监督语义分割 | 是 |
| 固定 Top-K Utility Router | 本架构目标 | 是 |

论文表述应为：

> 使用 DINO 预训练的共享 ViT，并借鉴 DINO 风格的 EMA Teacher–Student 自蒸馏；核心训练目标是跨模态 Anchor 一致性与 Dense-to-Sparse Token 蒸馏，而不是简单复用原始 DINO 损失。

## 11. 退化数据使用规范

### 11.1 训练

- 有标签样本：在线生成标签保持的随机退化，并复用真实标签；
- 无标签样本：只构造弱增强/强退化视图，计算特征自蒸馏；
- 训练参数从连续范围采样，不要求提供离散质量等级；
- 随机种子仅控制样本多样性，不代表未见退化泛化；
- 应平衡 RGB 退化、Thermal 退化、双模态退化和整模态缺失，防止模型固定依赖某一模态。

### 11.2 评估

- 固定随机种子和参数；
- 使用可解释的五级强度或连续强度曲线；
- 单独保留未见退化类别，而不仅是未见随机种子；
- 同时报告 clean、单模态退化、双模态退化、模态缺失和真实恶劣场景；
- 合成退化结果不得替代真实场景验证。

## 12. RGB-T 数据策略（无伪标签）

### 12.1 无标签数据

当前正式流程使用 LLVIP 配对数据进行 Stage 1 模态适配。KAIST 等其他配对数据可作为扩展，但必须先完成去重和协议检查。无标签数据用于：

- Stage 1 跨模态特征适配；
- 公共关系空间学习，不参与语义分割标签监督。

不生成语义伪标签，不将 Teacher 低置信度预测写回训练集。

### 12.2 有标签数据

当前代码以 MFNet 为 Stage 2A、Stage 2B 和 Stage 3 的主要有标签训练集：

```text
Stage 2A: MFNet 真实标签 + clean-only Dense Teacher
Stage 2B: MFNet clean/degraded/missing + Dense EMA 一致性（可选鲁棒化）
Stage 3:  MFNet 真实标签 + Dense-to-Sparse 蒸馏
```

FMB、PST900 或其他带标注 RGB-T 数据用于第二数据集复现、外部泛化验证或独立微调。不同数据集类别体系不同时，采用：

```text
共享 PatchEmbed / Adapter / ViT / Fusion
+ dataset-specific segmentation heads
```

联合训练时先均匀采样数据集，再在数据集内采样，避免大数据集完全压制小数据集。

任意预训练配对数据与下游分割数据都必须按原始视频、场景 ID 或感知哈希去重，避免训练/验证/测试泄漏。

## 13. RGB-T 到 RGB-X 的扩展流程

完成 RGB-T 后，增加新模态 `X`：

1. 新建 `PatchEmbed_X`、`LayerNorm_X`、`Adapter_X` 和 Modality Embedding；
2. 冻结 RGB-T 主干，使用配对 RGB-X 数据执行 Stage 1；
3. 使用 RGB-T 与 RGB-X 混合 batch，低学习率联合训练 Dense 模型；
4. 保留旧 RGB-T batch 进行 replay，避免灾难性遗忘；
5. 为标签体系不同的数据集保留独立 Decoder；
6. 重新生成 Dense Robust Teacher；
7. 重新训练固定预算 Sparse Student。

不能仅添加新 Adapter 后直接复用原 RGB-T Sparse Student，因为 Fusion、Router 和 Teacher 均未学习过新模态。

## 14. 实现接口建议

### 14.1 输入协议

```python
inputs = {
    'rgb': rgb_tensor,
    'thermal': thermal_tensor,
}
availability = {
    'rgb': rgb_available,
    'thermal': thermal_available,
}
```

### 14.2 模块划分

```text
ModalityPatchEmbeds: ModuleDict[str, PatchEmbed]
ModalityNorms: ModuleDict[str, LayerNorm]
EarlyAdapters: ModuleDict[str, ModuleList[Adapter]]
SharedViTBlocks: ModuleList[Block]
AnchorExtraFusion
UtilityRouter
SegmentationDecoder
AlignmentProjector（仅训练）
```

### 14.3 前向模式

```text
mode='adapt'         Stage 1，仅输出模态对齐特征
mode='dense'         Stage 2，保留全部 Extra
mode='sparse_soft'   Stage 3 初期，可微 soft gate
mode='sparse_hard'   Stage 3 后期与推理，固定 Top-K gather
```

Teacher 和 Student 必须共享架构定义，但拥有独立参数实例。Teacher 前向在 `torch.no_grad()` 下执行。

## 15. 必要消融

### 15.1 架构消融

```text
RGB-only
Thermal-only
RGB-T concat + Dense DINO
每位置直接 2N -> N Anchor
N Anchor + 全部 N Extra
N Anchor + Top-K Extra
无 Adapter
独立 Backbone（不共享 ViT）
从输入开始混合全部 Token
浅层独立、深层混合
```

### 15.2 训练消融

```text
无 Stage 1
Stage 1 仅区域对齐
Stage 1 仅关系对齐
Stage 1 关系对齐 + 弱区域对齐（正式版本）
Stage 1 强逐 Patch 对齐
无退化训练
无整模态 dropout
无 Dense EMA
无 L_compression
无 L_robust
无有标签 logit distillation
不同 K/N
不同融合 Block 位置 R
```

### 15.3 必须报告的效率指标

- 参数量；
- 理论 FLOPs；
- 实际进入后续 ViT 的 Token 数量；
- batch size 1 和实际部署 batch 下的 GPU latency；
-峰值显存；
- clean mIoU、corrupted mIoU、missing-modality mIoU；
- 精度—延迟和精度—Token 预算曲线。

仅将 Token 乘零或加入 Attention Mask 不计为真实剪枝。必须在 QKV 和 FFN 前通过 `gather` 或等价的 packed sequence 实现缩短后的序列。

## 16. 推荐实现顺序

1. 实现 DINO RGB-T 双输入、模态 PatchEmbed 和浅层 Adapter；
2. 实现每位置 `2N -> N` 的 Anchor-only Dense baseline；
3. 训练并验证 Dense Robust 模型；
4. 增加一个 Extra，使 Dense 路径保持 `2N`；
5. 增加 Utility Router 和固定 Top-K `gather`；
6. 实现 Frozen Dense Teacher 到 Sparse Student 的特征蒸馏；
7. 最后再尝试多个 Router 和逐层降低 `K`。

在 Anchor-only baseline 未稳定前，不同时引入多次 Router、复杂退化组合和更多模态。

