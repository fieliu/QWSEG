# SAPT：面向鲁棒 RGB-T 语义分割的空间锚点保留式 Token 压缩

> **中文同步草稿 / 2026-09-19**<br>
> 英文投稿主稿见 `main.tex`。本文严格按照当前代码与配置描述方法。仓库中尚无完整训练日志和正式实验结果，因此所有数值均以 **TBD** 标记，不能作为论文结论引用。

## 摘要

RGB-热红外（RGB-T）语义分割能够在低照度等困难环境中提升场景理解能力，但实际部署同时面临两个相互制约的要求：当任一传感器退化或完全缺失时，模型仍需保持可靠；同时，双模态密集 Token 带来的计算开销必须可控。通用 Token 剪枝虽然可以减少计算，却可能直接删除语义分割所必需的空间位置；现有可靠性融合方法通常保留全部模态 Token，难以获得真实的序列级加速。

本文提出一种空间锚点保留式表示，将每个对齐位置的 RGB 与 Thermal Token 重参数化为一个必保留的 **Anchor Token** 和一个可选的 **Extra Token**。全部 Anchor 始终保留，以维持完整的二维分割网格；任务效用路由器则在进入后续深层 Transformer Block 之前，通过真实的 `gather` 操作选择固定预算的 Top-K Extra。由此，深层序列长度从 `2N` 缩短为 `N+K`，同时不会删除任何空间位置。为使稀疏模型具备鲁棒性，本文采用分阶段训练：首先将 RGB 预训练的 DINOv3 主干适配到 Thermal；随后在干净、程序化退化及整模态缺失视图上训练 Dense 模型，并使用 EMA 目标网络进行弱—强视图自蒸馏；最后冻结鲁棒 Dense Teacher，将其知识蒸馏到固定 Token 预算的 Sparse Student。无标签 RGB-T 数据只参与特征一致性学习，不生成语义伪标签。

最终实验将覆盖 MFNet、FMB、SemanticRT 和 PST900，并同时评价干净精度、退化鲁棒性、模态缺失鲁棒性及真实推理延迟。当前版本尚未完成正式实验，本文不作任何性能领先声明。

## 1. 引言

部署于自动驾驶、机器人和监控场景的语义分割模型，需要在照明变化和传感器质量波动下对每个像素进行分类。Thermal 在夜间或视觉信息模糊时可以补充 RGB，因此 MFNet、PST900、FMB 和 SemanticRT 等数据集推动了 RGB-T 语义分割的发展。

但是，增加一个模态并不自然等价于获得鲁棒性。RGB 可能出现低照度、雾、噪声、运动模糊或过曝；热红外可能受到传感器噪声、条纹噪声、模糊和量化误差影响；任一模态还可能被局部遮挡或完全失效。只在干净配对数据上训练的网络容易永久依赖某一个模态，并在真实故障条件下失效。

近期方法开始关注任意模态组合、可靠性感知融合和视觉基础模型适配。CMNeXt/DeLiVER、StitchFusion 和 CrossWeaver 已经探索了任意可用模态输入；RSGMamba 与 RTFDNet 进一步关注可靠性或鲁棒融合。与此同时，DINO 和 DINOv3 为密集视觉任务提供了强大的自监督特征。尤其是 SpectraDINO 已经使用模态输入层、模态嵌入和 Adapter，将 RGB DINO 表征迁移到多个红外频段。因此，**“共享 DINO + 模态 Adapter”只能作为强基线，不能作为本文的主要创新点。**

另一个尚未解决的问题是计算效率。若每个模态产生 `N` 个 Patch Token，直接拼接 RGB 与 Thermal 会在融合后形成 `2N` 长度的序列。后续自注意力计算量随序列长度平方增长。DynamicViT、TokenLearner、Token Merging 等通用压缩方法可以删除、学习或合并 Token，但对于语义分割而言，任意删除空间 Token 可能导致某些位置完全失去直接表征。仅将 Token 乘零或设置 Attention Mask 又不会改变张量形状，因而难以产生真实的稠密算子加速。

本文采用一个明确的结构约束：**每个空间位置至少保留一个 Token，只压缩跨模态冗余。** 在浅层融合点处，每个 RGB/T Token 对被映射成一个 Anchor 和一个 Extra。Dense 模型保留 `N` 个 Anchor 与 `N` 个 Extra；Sparse 模型保留所有 Anchor，只根据任务效用选择 `K` 个 Extra。硬选择通过真实 `gather` 完成，使后续全部深层 Block 的输入从 `2N` 变成 `N+K`。

鲁棒压缩还需要一个合适的教师。干净数据教师可能把脆弱行为传递给学生；尚未学会多模态分割时就建立 EMA 教师，也会给出不稳定目标。为此，本文将训练分为三个概念阶段，其中第二阶段包含两个工程子阶段：

1. 使用配对 RGB-T 数据进行 Thermal 模态适配；
2. 先训练不剪枝的 Dense 鲁棒分割模型，再建立 Dense EMA Target；
3. 冻结通过验证的 Dense Teacher，并蒸馏固定预算 Sparse Student。

本文的主要贡献为：

1. 提出 Anchor–Extra 重参数化：每个空间位置保留一个不可剪枝 Anchor，仅将跨模态互补信息放入可剪枝 Extra。
2. 提出固定预算的任务效用 Top-K 路由，并用真实 `gather` 缩短深层 ViT 序列，而不是仅做掩码稀疏。
3. 提出 Dense-Teacher / Sparse-Student 多阶段训练流程，将模态适配、鲁棒 Dense 表征学习和压缩学习分离，并允许无标签配对数据仅通过特征一致性参与训练。
4. 建立同时覆盖干净输入、模态特定退化、整模态缺失、真实恶劣场景、Token 数量、显存和延迟的评价协议。当前草稿仅定义协议，不提前声明性能优势。

## 2. 相关工作

### 2.1 RGB-T 与任意模态语义分割

MFNet、PST900、FMB 和 SemanticRT 提供了不同规模、不同场景和类别体系的 RGB-T 像素级标注。早期方法通常采用双分支编码器和多尺度融合；近期 CMNeXt/DeLiVER、StitchFusion 和 CrossWeaver 将研究范围扩展到 RGB-X 或任意可用模态。Open-RGBT 则研究开放词汇 RGB-T 分割。

本文当前定位是 **闭集 RGB-T 鲁棒分割与推理效率**，而不是开放词汇，也不宣称已经支持任意原始传感器。当前网络假设两种输入都能表示为大致对齐的二维图像网格。

### 2.2 退化与模态缺失下的鲁棒融合

多模态网络通常默认所有传感器同时存在且质量稳定。CMNeXt 关注任意模态可用性，RSGMamba 与 RTFDNet 关注可靠性感知或鲁棒融合。本文进一步追问：在真正缩短融合后 Transformer 序列的同时，能否维持退化和缺失条件下的分割性能？

程序化退化仅用于提供可控且保持标签的训练覆盖，不能替代真实传感器验证。因此，实验必须分别报告：

- 固定参数的合成退化；
- RGB 完全缺失与 Thermal 完全缺失；
- 未参与训练的退化类别；
- 数据集中真实夜间或恶劣场景。

### 2.3 跨模态基础模型适配

DINO 系列通过自蒸馏获得了可迁移的视觉表征，DINOv3 进一步扩大了模型与数据规模。RADIOv2.5 研究了多教师视觉基础模型及固定 Token 数量压缩。与本文关系最直接的是 SpectraDINO：其已经证明 RGB 预训练 DINO 可以通过模态相关输入组件和 Adapter 适配到红外频段。

因此，本文将共享 ViT、独立 PatchEmbed 和浅层 Adapter 视为基础组件。真正需要实验验证的贡献是：完整 Anchor 网格、可压缩 Extra、Dense 鲁棒教师、固定预算任务路由，以及由真实序列缩短带来的延迟收益。

### 2.4 Token 压缩

DynamicViT 动态删除低重要度 Token，TokenLearner 学习少量紧凑 Token，Token Merging 合并相似 Token。这些方法主要面向分类或一般视觉推理。语义分割要求密集输出，如果某个位置的所有表示都被删除，解码器只能从邻域间接恢复该位置。

本文不剪枝空间支撑：每个 Patch 的 Anchor 始终存在。路由器只作用于每个位置对应的跨模态 Extra，并在昂贵的深层 Attention 与 FFN 之前物理缩短张量。

## 3. 方法

### 3.1 问题定义

给定配准的 RGB 图像

\[
x^r\in\mathbb{R}^{H\times W\times3}
\]

和 Thermal 图像

\[
x^t\in\mathbb{R}^{H\times W\times C_t},
\]

模型输出像素级语义图

\[
y\in\{1,\ldots,C\}^{H\times W}.
\]

测试时任一模态都可能干净、局部退化或完全缺失。Patch 大小为 `P` 时，特征网格大小为

\[
H_p=H/P,\qquad W_p=W/P,\qquad N=H_pW_p.
\]

本文的核心不变量是：直到解码阶段，`N` 个空间位置中的每个位置都至少拥有一个可学习 Token。

### 3.2 模态条件共享 ViT

主干从 RGB 预训练的 DINOv3 ViT 初始化。RGB 与 Thermal 分别使用 PatchEmbed `E_r`、`E_t` 和模态嵌入。前 `R` 个 Transformer Block 共享 Attention 和 FFN 参数，但两个模态作为独立序列计算注意力。实现上可以将模态维并入 Batch 维，因此浅层不会产生 RGB/T 互注意。

对于模态 `m∈{r,t}` 的第 `l` 个浅层 Block：

\[
u_l^m=h_{l-1}^m+
\mathrm{Attn}_l(\mathrm{LN}(h_{l-1}^m)),
\]

\[
h_l^m=u_l^m+
\mathrm{FFN}_l(\mathrm{LN}(u_l^m))+
\gamma_l^m A_l^m(\mathrm{LN}(u_l^m)).
\]

其中，`A_l^m` 是模态独立的低秩 Adapter，`γ_l^m` 零初始化。Adapter 是共享 FFN 的残差补充，而不是替代。融合后 Token 不再具有唯一模态身份，因此深层 Block 不再使用模态 Adapter。

整模态缺失时，可用性标记会在浅层处理前将对应 Token 流置零，使融合网络明确处理“传感器不可用”，而不是把全零图像误认为正常观测。

### 3.3 Anchor–Extra 重参数化

在融合位置，第 `i` 个位置有 RGB Token `r_i` 和 Thermal Token `t_i`。先构造描述子：

\[
c_i=[r_i;t_i;|r_i-t_i|;r_i\odot t_i].
\]

融合权重为：

\[
[\alpha_i^r,\alpha_i^t]=\mathrm{softmax}(F_\alpha(c_i)).
\]

Anchor Token 定义为：

\[
a_i=\alpha_i^rV_r(r_i)+\alpha_i^tV_t(t_i).
\]

Extra Token 定义为：

\[
e_i=F_e([r_i-t_i;r_i;t_i;a_i])+p_i^e+q^e,
\]

其中 `p_i^e` 是 Extra 的空间位置编码，`q^e` 是 Extra 类型编码；Anchor 另外加入 `p_i^a`。

设计动机可以由线性特例解释：若

\[
a_i=\alpha_i r_i+(1-\alpha_i)t_i,\qquad e_i=r_i-t_i,
\]

且 `α_i` 已知，则原来的 `r_i,t_i` 可以恢复。这说明“共同成分 + 差异成分”有机会比直接 `2N→N` 融合保留更多信息。实际实现包含 MLP，因此**不声明严格可逆性**。

Dense 序列为：

\[
z_{dense}=[a_1,\ldots,a_N,e_1,\ldots,e_N].
\]

它进入第 `R+1` 到第 `L` 个共享 ViT Block，执行跨模态全局自注意力。由于联合序列不再是单一规则栅格，且 Extra 在稀疏模式下会被重排，深层 Block 关闭原始栅格 RoPE，改用上述 Anchor/Extra 位置与类型编码。最终仅取前 `N` 个 Anchor，恢复成 `D×H_p×W_p` 特征图供解码器使用。

### 3.4 任务效用 Extra 路由

路由器根据 Anchor 与 Extra 的关系预测每个 Extra 的任务效用：

\[
u_i=F_u([a_i;e_i;|a_i-e_i|;a_i\odot e_i]).
\]

固定预算为 `K` 时：

\[
\mathcal I_K=\mathrm{TopK}(\{u_i\}_{i=1}^N,K),
\]

\[
z_{sparse}=[a_1,\ldots,a_N;\mathrm{Gather}(E,\mathcal I_K)].
\]

`Gather` 会生成真实长度为 `N+K` 的张量，而不是将未选 Token 乘零。即使 `K=0`，全部 Anchor 仍然存在，因此解码器仍能得到完整 `H_p×W_p` 网格。

稀疏训练初期使用可微 Soft Gate：

\[
\tilde e_i=e_i\,\sigma((u_i-\tau_K)/\tau),
\]

其中 `τ_K` 是当前样本第 `K` 大的效用值。Soft Gate 保持 `N+N` 长度，只用于路由预热，不计为真实加速。预热结束后，训练和测试都切换到 Hard Top-K `gather`。

忽略线性投影等常数后，融合后深层 Attention 从

\[
O((2N)^2D)
\]

变为

\[
O((N+K)^2D),
\]

FFN 则从 `O(2ND²)` 变为 `O((N+K)D²)`。当 `K=0.5N` 时，深层 Attention Score 计算约为 Dense 路径的 `56.25%`，深层 FFN Token 计算约为 `75%`。端到端收益仍必须实测，因为浅层双流计算、融合、`gather` 和 Decoder 不会同步消失。

### 3.5 多阶段训练

#### Stage 1：跨模态适配

冻结：

- RGB PatchEmbed 与 RGB Modality Embedding；
- 共享 DINOv3 主干；
- RGB Adapter；
- Fusion、Router、Neck 和 Decoder；
- RGB Teacher Alignment Projector。

训练：

- Thermal PatchEmbed；
- Thermal Modality Embedding；
- 前 `R` 个 Thermal Adapter；
- Thermal Student Alignment Projector。

对投影并 L2 归一化后的特征，使用：

\[
\mathcal L_{patch}=\frac{1}{|\Omega|}\sum_{i\in\Omega}
\left(1-\cos(z_i^t,\mathrm{sg}[z_i^r])\right),
\]

\[
\mathcal L_{region}=\frac{1}{|\Omega'|}\sum_{j\in\Omega'}
\left(1-\cos(\bar z_j^t,\mathrm{sg}[\bar z_j^r])\right),
\]

\[
\mathcal L_1=\lambda_p\mathcal L_{patch}+
\lambda_r\mathcal L_{region}.
\]

区域池化用于容忍轻微配准误差；若数据无法可靠逐 Patch 对齐，则令 `λ_p=0`。教师主干与投影器固定，因此当前实现不需要额外防坍塌损失。此阶段可以使用去重后的 LLVIP、KAIST 等无标签 RGB-T 对，但不生成语义标签。

#### Stage 2A：Dense 鲁棒预热

该阶段首先在没有教师的情况下训练完整 Dense 分割模型。对于有标注样本 `(x,y)`，构造：

- 干净输入 `x`；
- 保持标签和坐标不变的程序化退化 `C(x)`；
- 随机删除一个完整模态的 `D(x)`。

使用 Mask2Former 分类、Mask BCE 与 Dice 损失：

\[
\mathcal L_{2A}=\mathcal L_{seg}(f(x),y)
+\lambda_d\mathcal L_{seg}(f(C(x)),y)
+\lambda_m\mathcal L_{seg}(f(D(x)),y).
\]

当前配置的三项 Mask2Former 权重为 `2:5:5`。退化不改变场景语义和像素坐标，因此三个视图复用真实标注，不属于伪标签训练。

#### Stage 2B：Dense EMA 鲁棒自蒸馏

只有 Stage 2A 模型达到可接受的干净和故障条件性能后，才从在线模型复制 EMA Target：

\[
\theta_{ema}\leftarrow\theta_{online}.
\]

在线模型接收强退化视图 `x_s`，EMA 模型接收弱退化视图 `x_w`：

\[
\mathcal L_{anchor}=\frac{1}{N}\sum_i
\left(1-\cos(a_i^{online}(x_s),
\mathrm{sg}[a_i^{ema}(x_w)])\right).
\]

EMA 更新为：

\[
\theta_{ema}\leftarrow m\theta_{ema}+(1-m)\theta_{online},
\]

其中 `m` 从 `0.996` 增加到 `0.9999`。有标签样本优化 `L_2A+λ_aL_anchor`；无标签样本只优化 `λ_aL_anchor`。当前代码虽然保留了 DINO 风格全局损失函数接口，但默认权重为 `0`，训练路径实际只使用逐位置 Anchor 一致性；论文不能声称使用了完整原始 DINO 目标。

#### Stage 3：Dense-to-Sparse 蒸馏

选取 Stage 2B 的最佳 Dense EMA 权重并完全冻结。Sparse Student 从 Teacher 初始化，并在退化输入 `C(x)` 上只保留 `K` 个 Extra。

压缩损失使用 Teacher 和 Student 的同一退化输入，以隔离 Token 压缩误差：

\[
\mathcal L_{comp}=\frac{1}{N}\sum_i
\left(1-\cos(a_i^s(C(x)),\mathrm{sg}[a_i^d(C(x))])\right).
\]

鲁棒损失要求退化 Sparse Student 接近干净 Dense Teacher：

\[
\mathcal L_{rob}=\frac{1}{N}\sum_i
\left(1-\cos(a_i^s(C(x)),\mathrm{sg}[a_i^d(x)])\right).
\]

有标注数据再加入像素级 Logit 蒸馏：

\[
\mathcal L_{logit}=T^2\mathrm{KL}
\left(p_d(C(x);T)\,\|\,p_s(C(x);T)\right).
\]

总目标为：

\[
\mathcal L_3=\lambda_c\mathcal L_{comp}
+\lambda_b\mathcal L_{rob}
+\mathbb 1_y\left(\mathcal L_{seg}+\lambda_l\mathcal L_{logit}\right).
\]

无标签样本只参与前两个特征蒸馏项。整个流程都不会把教师预测写回成语义 Mask。

### 3.6 程序化退化

退化在图像空间在线施加，再恢复到模型归一化空间。

Stage 2A 重点模拟传感器中断：每个样本随机选择一个模态，将其完整置零，或将随机矩形区域置零；另外再构造一个必然整模态缺失的监督视图。

Stage 2B 使用 13 种模态特定退化构造弱—强视图：

- RGB：高斯噪声、散粒噪声、运动模糊、离焦模糊、雾、低照度、RGB 缺失；
- Thermal：高斯噪声、条纹噪声、运动模糊、离焦模糊、量化、Thermal 缺失。

严重度从 `0–5` 采样，`0` 表示干净。局部退化区域比例从 `[0.3,0.8]` 采样，弱视图和强视图使用相同区域、不同严重度。这里的等级只是可控训练参数，并非经过物理标定的传感器质量分数。

当前 Stage 3 配置复用 Stage 2A 的整模态缺失/局部缺失生成器进行 Dense-to-Sparse 蒸馏；将 Stage 3 扩展到完整 13 类退化库属于待做消融，而不是当前默认实现。

## 4. 实验设计

### 4.1 当前状态

当前仓库已经包含模型主体、四阶段配置和 MFNet 数据路径，但没有可用于论文的最终 Checkpoint、训练曲线和测试日志。因此：

- 所有结果表必须保持 `TBD`，直到实验真实完成；
- 不能提前写“达到 SOTA”“显著优于”等结论；
- 最终数值必须能够追溯到配置、Checkpoint、Commit/工作区快照和原始日志；
- 来自论文的数值与统一环境复现数值必须分开标记。

### 4.2 数据集

计划分别在以下有标注数据集上训练或微调数据集专属 Head：

| 数据集 | 用途 | 注意事项 |
|---|---|---|
| MFNet | 当前主实现与首个完整实验 | 当前配置为 `480×640`、9 类 |
| FMB | 白天/夜间与全时段 RGB-T 分割 | 使用官方类别与划分 |
| SemanticRT | 更大规模鲁棒训练与评估 | 11,371 对；与 LLVIP/OSU/INO 有来源重叠 |
| PST900 | 小规模跨场景验证 | 防止只在单一交通场景得出结论 |

LLVIP 和 KAIST 可用于 Stage 1，以及 Stage 2B/3 的无标签特征一致性。由于 SemanticRT 主要来自 LLVIP、OSU 和 INO，在使用 LLVIP 预训练时必须按视频/场景 ID、精确 Hash 和感知 Hash 去重，避免 SemanticRT 验证或测试泄漏。

当前 Stage 1 配置暂时使用 MFNet Train 作为可运行替代，最终论文实验必须换成明确记录并去重的无标签配对集。

### 4.3 评价指标

精度指标：

- Clean mIoU、mAcc、各类别 IoU；
- 每种退化、每个严重度的 mIoU；
- Corruption mean mIoU 与相对 Clean 的下降；
- RGB Missing、Thermal Missing 及两者最差值；
- 真实夜间/恶劣条件子集结果。

效率指标：

- 参数量和理论 FLOPs；
- 实际进入深层 ViT 的 Token 数；
- Batch Size 1 和部署 Batch 下的 GPU 延迟；
- 峰值显存；
- mIoU–Latency 与 mIoU–Token Budget 曲线。

延迟测试必须固定硬件、精度、输入尺寸、Warm-up 次数、测试迭代次数和同步方式。

### 4.4 实现细节

- Backbone：DINOv3 ViT-B/16；
- 深度：`L=12`；
- 特征维度：`D=768`；
- 融合位置：`R=3`；
- 输入：`480×640`；
- Patch 网格：`30×40`，因此 `N=1200`；
- Dense 深层序列：`2400`；
- 默认 Sparse 预算：`K=600=0.5N`，深层序列 `1800`；
- Adapter Bottleneck：`64`；
- Neck：Feature2Pyramid；
- Decoder：Mask2Former，`100` Queries。

优化配置：

- Stage 1：AdamW，学习率 `1e-4`，100 Epoch；
- Stage 2A/2B/3：AdamW，学习率 `3e-5`，Weight Decay `0.05`，Layer Decay `0.9`，各 200 Epoch；
- Stage 3 前 30 Epoch：预算按 `N→0.75N→0.5N` 下降，并使用 Soft Gate；之后使用 Hard Gather；
- 默认损失权重：`λ_d=λ_m=λ_a=λ_c=λ_b=λ_l=1`；
- Logit 蒸馏温度：`T=2`。

这些参数只是当前实现默认值，尚不能称为最优超参数。

### 4.5 主结果表模板

| 方法 | Backbone | MFNet Clean | FMB Clean | SemanticRT Clean | PST900 Clean | Corr. | Missing-Worst | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RGB-only | DINOv3-B/16 | TBD | TBD | TBD | TBD | TBD | — | TBD |
| Thermal-only | DINOv3-B/16 | TBD | TBD | TBD | TBD | TBD | — | TBD |
| CMNeXt | MiT-B2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| StitchFusion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dense Anchor–Extra Teacher | DINOv3-B/16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 本文，`K/N=0.5` | DINOv3-B/16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.6 鲁棒性表模板

| 方法 | Clean | RGB Missing | T Missing | Seen Corruption | Unseen Corruption |
|---|---:|---:|---:|---:|---:|
| Dense Teacher | TBD | TBD | TBD | TBD | TBD |
| Sparse `K/N=0.75` | TBD | TBD | TBD | TBD | TBD |
| Sparse `K/N=0.50` | TBD | TBD | TBD | TBD | TBD |
| Sparse `K/N=0.25` | TBD | TBD | TBD | TBD | TBD |

### 4.7 效率表模板

| 模型 | 深层 Token | FLOPs | BS=1 延迟 | 峰值显存 | Clean mIoU |
|---|---:|---:|---:|---:|---:|
| Dense Teacher | 2400 | TBD | TBD | TBD | TBD |
| `K/N=0.75` | 2100 | TBD | TBD | TBD | TBD |
| `K/N=0.50` | 1800 | TBD | TBD | TBD | TBD |
| `K/N=0.25` | 1500 | TBD | TBD | TBD | TBD |
| Anchor-only，`K=0` | 1200 | TBD | TBD | TBD | TBD |

只有使用 Hard `gather` 后测得的结果才能写入效率表。Soft Gate、Token 乘零或 Attention Mask 仍维持 Dense 张量，不应被宣传为真实 Token 剪枝。

### 4.8 必要消融

| 变体 | Clean | Corr. | RGB Missing | T Missing | 深层 Token | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 完整模型，`K/N=0.5` | TBD | TBD | TBD | TBD | 1800 | TBD |
| 无 Stage 1 | TBD | TBD | TBD | TBD | 1800 | TBD |
| 无 Stage 2B EMA | TBD | TBD | TBD | TBD | 1800 | TBD |
| 无 `L_comp` | TBD | TBD | TBD | TBD | 1800 | TBD |
| 无 `L_rob` | TBD | TBD | TBD | TBD | 1800 | TBD |
| 无 Logit 蒸馏 | TBD | TBD | TBD | TBD | 1800 | TBD |
| 随机选择 `K` 个 Extra | TBD | TBD | TBD | TBD | 1800 | TBD |
| 只 Mask、不 Gather | TBD | TBD | TBD | TBD | 2400 | TBD |
| Anchor-only，`K=0` | TBD | TBD | TBD | TBD | 1200 | TBD |

还应单独扫描：

- `K/N∈{0,0.25,0.5,0.75,1}`；
- 融合深度 `R`；
- Adapter 宽度；
- 无 Adapter、独立 Backbone、从输入起直接混合；
- 直接 `2N→N` 与 `N Anchor+N Extra`；
- 不同 Soft-to-Hard 切换时间；
- 仅缺失训练、仅连续退化训练、两者同时训练。

### 4.9 路由器分析

需要可视化被保留的 Extra 位置，并分析：

- 是否更常出现在语义边界、弱模态区域或局部退化区域；
- 同一场景不同增强下的选择稳定性；
- 空间选择熵及覆盖率；
- 是否长期偏向某一模态或固定位置；
- 高效用 Extra 被删除后性能是否显著下降。

需要避免把 Utility Score 描述为“显式质量分数”。它没有质量等级监督，而是通过分割与蒸馏目标端到端学习的任务效用。

## 5. 讨论与局限

### 5.1 尚不是任意传感器模型

当前模型包含 RGB/T 专属 PatchEmbed、Modality Embedding 和 Adapter，并假设两个输入是空间对齐的图像网格。Depth 或 Event Frame 可以按相同流程扩展，但必须新增输入层与 Adapter，重新训练 Fusion、Dense Teacher 和 Sparse Student。只完成 RGB-T 实验时，不能声称模型已经统一支持任意模态。

### 5.2 合成退化不等于真实传感器退化

程序退化便于控制类型、严重度和区域，同时能够复用真实像素标签，但它们不代表完整的相机与热红外成像物理过程。论文结论必须由真实夜间、天气、噪声和缺失条件补充验证。

### 5.3 配准误差与信息损失

Anchor–Extra 融合默认 `r_i` 与 `t_i` 大致对应同一位置。明显视差、不同步采集和大范围配准错误可能破坏逐位置融合。虽然 Anchor 保证空间位置不消失，但 Extra 剪枝仍可能删除关键模态证据，因此必须报告预算—精度曲线和失败案例。

### 5.4 训练成本

Stage 2B 同时维护 Online 与 EMA 网络；Stage 3 同时运行 Dense Teacher 和 Sparse Student。方法减少的是最终推理成本，而不是训练成本。多阶段 Checkpoint 选择也增加了工程复杂度。

### 5.5 与 VLM / 开放词汇方法的关系

大型视觉语言模型和 Open-RGBT 解决的是类别开放性与文本语义泛化。本文解决的是闭集密集预测在传感器退化下的稳定性和显式计算预算，两者并非相互替代。没有开放词汇实验时，不应在标题或贡献中加入开放词汇声明。

## 6. 结论

本文提出空间锚点保留式 RGB-T Token 压缩框架。模型把每个对齐模态 Token 对转换为必保留 Anchor 和可选 Extra，在完整保留分割网格的前提下，通过固定预算 Hard Top-K `gather` 缩短深层 Transformer 序列。多阶段 Dense-Teacher / Sparse-Student 训练进一步将模态适配、鲁棒 Dense 表征、EMA 一致性和压缩蒸馏解耦，并允许无标签 RGB-T 对只参与特征学习而不生成语义伪标签。

当前架构和训练路径已经实现，但实验结论尚未建立。下一步必须完成跨数据集的干净、合成退化、整模态缺失和真实恶劣场景测试，并给出可复现的 FLOPs、显存和实际延迟结果。

## 参考文献入口

完整 BibTeX 见 `references.bib`。关键工作包括：

- DINOv3: <https://arxiv.org/abs/2508.10104>
- SpectraDINO: <https://arxiv.org/abs/2605.02258>
- DeLiVER / CMNeXt: <https://arxiv.org/abs/2303.01480>
- RSGMamba: <https://arxiv.org/abs/2604.12319>
- RTFDNet: <https://arxiv.org/abs/2603.09149>
- CrossWeaver: <https://arxiv.org/abs/2604.02948>
- StitchFusion: <https://arxiv.org/abs/2408.01343>
- Open-RGBT: <https://arxiv.org/abs/2410.06626>
- SemanticRT: <https://github.com/jiwei0921/SemanticRT>
- FMB / SegMiF: <https://github.com/JinyuanLiu-CV/SegMiF>
- MFNet: <https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/>
- PST900: <https://arxiv.org/abs/1909.10980>
- LLVIP: <https://arxiv.org/abs/2108.10831>
