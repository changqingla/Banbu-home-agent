# VisPCO：通过预算感知的帕累托前沿学习实现视觉语言模型的视觉令牌剪枝配置优化

Huawei Ji1, Yuanhao $\mathbf { S u n 1 }$ , Yuan $\mathbf { { J i n ^ { 1 } } }$ , Cheng Deng2, Jiaxin Ding1* Luoyi $\mathbf { F u 1 }$ , Xinbing Wang1 

1上海交通大学，上海，中国

2爱丁堡大学，爱丁堡，英国

{sjtu3365981, h_iden, lemon0703, jiaxinding, yiluofu, xwang8}@sjtu.edu.cn 

# 摘要

视觉令牌剪枝方法能够有效缓解视觉语言模型（VLMs）在处理高分辨率图像或长视频帧时所引发的二次方计算成本增长问题。然而，现有方法依赖于预定义的剪枝配置，而未能确定这些配置是否实现了计算与性能的最优平衡。在本工作中，我们提出 VisPCO，一个新颖的框架，将视觉令牌剪枝表述为帕累托配置优化问题，以自动识别最优配置。我们的方法采用连续松弛和直通估计器来实现基于梯度的搜索，并通过增广拉格朗日方法进行求解。在8个视觉基准上的大量实验表明，VisPCO 能够有效逼近通过网格搜索获得的经验帕累托前沿，并在各种剪枝方法和 VLM 架构上具有良好的泛化能力。此外，通过可学习核函数，我们研究了逐层剪枝模式，发现多步渐进式剪枝能够捕捉 VLMs 的层次化压缩结构，相比单层方法实现了更优的计算-性能权衡。

# 1 引言

大规模视觉语言模型（LVLMs）将视觉特征和文本特征作为输入进行处理，使其能够学习统一的多模态表示并进行跨模态推理。近期研究表明，更高分辨率的图像输入能够有效提升模型的理解和生成性能（Guo et al., 2024；An et al., 2025；Chen et al., 2024b）。与此同时，视频理解等任务要求模型处理大量帧以捕捉时序连续性和动态语义（Lin et al., 2024；Xu et al., 2025）。这两种场景都会显著增加视觉令牌的数量，导致计算成本呈二次方增长。
![[Pasted image 20260429135539.png]]

图1：不同配置下的帕累托前沿：(a) 帕累托前沿连接最优剪枝配置；(b) 不同 VLM 之间的帕累托前沿；(c) 不同剪枝方法之间的帕累托前沿；(d) 不同数据集之间的帕累托前沿。


为解决这一问题，面向 VLMs 的各种视觉剪枝算法应运而生。这些方法主要侧重于设计不同的重要性评分机制，以在单层或多层对冗余视觉令牌进行剪枝。例如，FastV（Chen et al., 2024a）、Dynamic-LLaVA（Huang et al., 2024）、VTW（Lin et al., 2025）和 TOPV（Yang et al., 2025）使用预定义的剪枝比例在特定 LLM 层对视觉令牌进行剪枝。相比之下，ATP-LLaVA（Ye et al., 2025b）、HiMAP（Zhou et al., 2024）和 SparseVLM（Zhang et al., 2024）在多个选定层上应用动态剪枝比例。所有这些工作都旨在降低计算成本（例如，浮点运算次数 FLOPs）的同时保持模型性能。

然而，现有工作中仍有两个关键问题尚未被探索。首先，目前尚不清楚当前的剪枝配置（即剪枝位置和比例）是否实现了最优的计算-性能权衡。其次，如何高效地调整这些配置以达到最优依然是一个开放性问题。解决这两个问题对于 VLMs 在现实应用中的高效部署至关重要，例如边缘设备部署和移动视觉系统。
![[Pasted image 20260429135809.png]]
图2：VisPCO 框架示意图。（左）视觉令牌剪枝过程概览。在每个 Transformer 块之后，根据重要性分数对视觉令牌进行排序，并过滤掉低分令牌。（右）上图：VisPCO 的整体架构，其中可训练的比例预测器（一个轻量级代理网络）决定每层的剪枝比例以指导令牌压缩。下图：端到端训练比例预测器时遇到的梯度断裂问题及本文提出的解决方案。


为此，我们引入帕累托最优性的概念来表征最优的计算-性能权衡（Arrow and Debreu, 2024）。帕累托最优性是多目标优化中的经典概念，描述一种在多个相互冲突的目标中无法进一步改善任何目标的状态。在本工作中，如果我们无法同时进一步降低计算成本并提升性能，则将该点定义为帕累托最优。连接这些点的曲线称为帕累托前沿。

在图1中，我们将不同剪枝配置在计算-性能空间中进行可视化，并展示其对应的帕累托前沿。每个点代表一个配置的实验结果。图1(a)揭示了在相同计算预算下，不同剪枝配置呈现出显著的性能差异。例如，在1.7 TFLOPs 的计算预算下，不同配置之间的性能差距可高达 $2 3 . 6 \%$。此外，帕累托最优前沿并非固定不变，而是随 VLM 架构、剪枝方法和图像复杂度的变化而变化，如图1(b-d)所示。在

实践中，为了在固定计算预算下确定最佳剪枝配置，我们通常需要对配置搜索空间进行网格采样。这涉及进行大量实验来测量不同配置下的性能和计算成本，然后选择最优配置进行部署。然而，这一过程耗时且资源密集。

在本文中，我们提出了一种计算预算感知的视觉令牌剪枝配置优化方法，称为 VisPCO。该方法采用可学习的代理模型，在给定计算预算的条件下自动预测帕累托前沿上的剪枝配置，从而实现最优的模型性能。如图2所示，与传统网格搜索方法不同，VisPCO 采用高效的梯度下降进行搜索，显著降低了搜索成本。为解决视觉令牌剪枝的离散性和不可微性，我们引入连续松弛技术和直通估计器以实现端到端优化。对于优化目标，我们将其表述为带有非凸不等式约束的帕累托优化问题，并使用增广拉格朗日方法（Nocedal and Wright, 2006）进行求解。此外，我们研究了 VLMs 中的逐层剪枝模式。具体而言，我们探索了在逐步压缩视觉令牌时最优剪枝比例如何在不同层之间变化。为了对这种变化建模，我们使用可学习核函数来参数化各层间的剪枝比例分布。通过评估不同核函数对帕累托前沿的逼近程度，我们确定了导致最优计算-性能权衡的内在剪枝模式。我们的贡献总结如下：

• 我们提出 VisPCO，一个可微分框架，通过基于梯度的方法自动寻找帕累托最优配置，消除了穷举网格搜索的高昂代价。

• 在8个基准上的实验表明，VisPCO 能够有效逼近经验帕累托前沿，并在各种剪枝方法和 VLM 架构上具有良好的泛化能力。

• 我们通过可学习剪枝核揭示了各层间非均匀的视觉令牌冗余性，表明多步剪枝在预算紧张时最为有效。我们的代码可在 https://github.com/JHW5981/VisPCO 获取。

# 2 相关工作

# 2.1 视觉令牌剪枝

视觉令牌剪枝通过减少处理数百个视觉令牌的计算成本来加速 VLMs。单层方法在特定层进行一次性压缩：FastV（Chen et al., 2024a）使用注意力分数在第2层后剪枝令牌，Dynamic-LLaVA（Huang et al., 2024）根据输入特征动态调整令牌保留比例，VTW（Lin et al., 2025）在充分吸收后撤回所有令牌，TopV（Yang et al., 2025）在推理时优化配置。多层渐进式方法将压缩分布在多个层上：ATP-LLaVA（Ye et al., 2025b）以层特定的比例自适应地在不同深度剪枝令牌，SparseVLM（Zhang et al., 2024）通过回收机制在各层渐进式地减少令牌，PyramidDrop（Xing et al., 2024）实现了阶段性金字塔式压缩，在浅层保留更多令牌。这些策略在保持竞争性能的同时实现了 $40\%$ 的计算节省。然而，这些方法中的剪枝配置要么是预定义的，要么是启发式确定的，它们是否实现了最优的计算-性能权衡尚不明确。

# 2.2 剪枝配置优化

尽管大多数剪枝方法依赖于预定义配置，但近期工作探索了自适应策略以优化各层的剪枝比例。FitPrune（Ye et al., 2025a）采用对注意力统计量的二值搜索来最小化分布散度，并生成逐层剪枝方案。G-Search（Zhao et al., 2025）将贪婪搜索与贝叶斯优化的sigmoid函数相结合，以逼近最优保留比例。ATP-LLaVA（Ye et al., 2025b）引入了可学习模块，用于基于训练的层特定稀疏性优化。SparseVLM（Zhang et al., 2024）采用基于排名的方法自适应确定每层稀疏化比例。更近期的方法探索了输入自适应配置：AIM（Zhong et al., 2025）开发了带有可调参数的调度器控制剪枝，MADTP（Cao et al., 2024）利用可学习阈值实现实例级自适应剪枝。尽管取得了这些进展，现有方法主要关注性能保持而非预算感知优化，缺乏在不同计算约束下系统识别近最优配置的机制。

# 3 VisPCO

# 3.1 帕累托优化

我们将视觉剪枝配置优化问题表述为寻找最优逐层剪枝比例 $\mathbf { r } = [ r _ { 1 } , r _ { 2 } , \ldots , r _ { L } ] \in [ 0 , 1 ] ^ { L }$，其中 $L$ 表示层数，$r _ { i }$ 表示第 $i$ 层的令牌保留比例（相对于原始视觉令牌数量）。我们的目标是在给定计算预算下，在帕累托前沿上识别配置 $\bar{r}$，以实现最优的模型性能。

为量化性能退化，我们将剪枝后的 VLM 输出 logits 定义为 $\hat { l }$，原始输出 logits 定义为 $l$，并使用 KL 散度来衡量二者的差异：

$$
\mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) = D _ {K L} \left(\operatorname {s o f t m a x} (\hat {l}) \| \operatorname {s o f t m a x} (l)\right). \tag {1}
$$

${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ 的值越小，表示剪枝对模型性能的影响越小。同时，我们定义计算成本函数为：

$$
F (\mathbf {r}) = \sum_ {i = 1} ^ {L} \left[ 2 4 \left(N _ {t} + r _ {i} N _ {v}\right) D ^ {2} + 4 \left(N _ {t} + r _ {i} N _ {v}\right) ^ {2} D \right] (2)
$$

其中 $N _ { t }$ 为文本令牌数量，$N _ { v }$ 为视觉令牌数量，$D$ 为隐藏维度。FLOPs 计算的详细推导见附录A。因此，计算-性能权衡的帕累托优化问题可以表述为如下约束优化问题：

$$
\min  _ {\mathbf {r} \in \mathcal {R}} \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) \tag {3}
$$

$$
\begin{array}{l l} \text {s . t .} & F (\mathbf {r}) \leq B, \end{array}
$$

其中 $B$ 为给定的计算预算。考虑到目标函数 ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ 通常是非凸的，我们采用增广拉格朗日方法进行数值迭代求解。

**定义 1（增广拉格朗日方法）。** 考虑如下等式约束优化问题：

$$
\min  f (\mathbf {x})
$$

$$
\begin{array}{l l} s. t. & h _ {j} (\mathbf {x}) = 0, \quad j = 1, \ldots , l \end{array}
$$

其中 $f ( \mathbf { x } ) : \mathbb { R } ^ { n }  \mathbb { R }$ 为待最小化的目标函数，$h _ { j } ( \mathbf { x } ) : \mathbb { R } ^ { n }  \mathbb { R }$ 为等式约束函数。增广拉格朗日函数定义为：

$$
\phi (\mathbf {x}, \mathbf {v}, \lambda) = f (\mathbf {x}) - \sum_ {j = 1} ^ {l} v _ {j} h _ {j} (\mathbf {x}) + \frac {\lambda}{2} \sum_ {j = 1} ^ {l} h _ {j} ^ {2} (\mathbf {x}), \tag {4}
$$

其中 $\mathbf { v } = [ v _ { 1 } , \dots , v _ { l } ]$ 为拉格朗日乘子向量，$\lambda > 0$ 为惩罚参数。

**定理 1（改编自 Bertsekas, 2014）。** 设 $\bar{x}$ 和 $\bar { \bf v }$ 满足该问题局部最优解的二阶条件。则存在 $\lambda ^ { \prime } \geq 0$，使得对所有 $\lambda > \lambda ^ { \prime }$，$\bar{x}$ 是 $\phi ( \mathbf { x } , \bar { \mathbf { v } } , \lambda )$ 的严格局部最小值点。

定理1的证明见附录B.1。基于此定理，我们可以开发一种具有有界 $\lambda$ 的迭代算法，从而避免二次惩罚方法的病态性（Nocedal and Wright, 2006）和标准拉格朗日方法的收敛困难（Bertsekas, 2014）。具体地，在第 $k$ 次迭代时，设 $\mathbf { x } ^ { ( k ) }$ 为关于 $\mathbf { x }$ 最小化式(4)的解，乘子更新规则为：

$$
v _ {j} ^ {(k + 1)} = v _ {j} ^ {(k)} - \lambda h _ {j} (\mathbf {x} ^ {(k)}), \quad j = 1, \dots , l \tag {5}
$$

通过这种迭代更新，$\mathbf { v } ^ { ( k ) } \to \bar { \mathbf { v } }$，$\mathbf { x } ^ { ( k ) }  \bar { \mathbf { x } }$，收敛速率通常由 $\| h ( \mathbf { x } ^ { ( k ) } ) \| / \| h ( \mathbf { \bar { x } } ^ { ( k - 1 ) } ) \|$ 衡量。

回到我们的优化问题(3)，我们引入辅助变量 $y$ 将不等式约束转化为等式约束：

$$
\min  \quad \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) \tag {6}
$$

$$
\begin{array}{l l} \text {s . t .} & B - F (\mathbf {r}) - y ^ {2} = 0. \end{array}
$$

对应的增广拉格朗日函数定义为：

$$
\begin{array}{l} \tilde {\phi} (\mathbf {r}, y, w, \lambda) = \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) - w (B - F (\mathbf {r}) - y ^ {2}) \\ + \frac {\lambda}{2} (B - F (\mathbf {r}) - y ^ {2}) ^ {2}. \tag {7} \\ \end{array}
$$

对 $y$ 进行配方，我们可以消除对 $y$ 的依赖，并得到简化的增广拉格朗日函数（见附录B.2）：

$$
\phi (\mathbf {r}, w, \lambda) = \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) + \frac {1}{2 \lambda} (z ^ {2} - w ^ {2}), \tag {8}
$$

其中 $z \ = \ \operatorname* { m a x } \bigl ( 0 , \ w - \lambda \bigl ( B - F ( \mathbf { r } ) \bigr ) \bigr )$。问题由此转化为最小化无约束目标 $\phi ( \mathbf { r } , w , \lambda )$。利用算法1中的迭代算法，我们可以得到最优剪枝配置 $\bar{r}$。

# 3.2 可微配置搜索

尽管我们已有帕累托优化目标函数式(8)和迭代算法1，但计算 $\nabla _ { \mathbf { r } } { \mathcal { L } } _ { \mathrm { d i s t i l l } } ( \mathbf { r } )$ 面临两个关键的不可微挑战，如图2右下方所示。第一，保留令牌数的离散化引入了不可微性。对于具有剪枝比例 $r _ { i }$ 和 $N _ { v }$ 个视觉令牌的第 $i$ 层，保留令牌数 $k _ { i } = \lfloor r _ { i } \cdot N _ { v } \rfloor$ 涉及取整操作，导致梯度消失，阻止了对 $r _ { i }$ 进行基于反向传播的更新。第二，根据重要性分数选择前 $k _ { i }$ 个令牌涉及阻断梯度流的离散操作。我们分别提出以下两种方法来解决这两个挑战：

**连续松弛。** 为解决第一个挑战，我们采用连续松弛策略。我们保留浮点形式 $\tilde { k } _ { i } = r _ { i } \cdot N _ { v }$，并设计一种使用高斯核的软插值方法来估计选择阈值。具体地，我们首先将所有视觉令牌的重要性分数按降序排列，得到 $\{ s _ { i 1 } , s _ { i 2 } , \ldots , s _ { i N _ { v } } \}$，其中 $s _ { i j }$ 表示排序后第 $i$ 层第 $j$ 个令牌的分数。传统的硬阈值处理直接使用位置

$\lfloor \tilde { k } _ { i } \rfloor$ 处的分数作为阈值，这会导致梯度消失。相反，我们采用基于高斯核的软插值来保持可微性：

$$
w _ {i j} = \exp \left(- \frac {\left(j - \tilde {k} _ {i}\right) ^ {2}}{2 \sigma^ {2}}\right), \quad \tau_ {i} = \frac {\sum_ {j = 1} ^ {N _ {v}} w _ {i j} s _ {i j}}{\sum_ {j = 1} ^ {N _ {v}} w _ {i j}}, \tag {9}
$$

其中 $w _ { i j }$ 是第 $i$ 层第 $j$ 个令牌的高斯权重。参数 $\sigma$ 控制核宽度，平衡逼近精度和梯度稳定性。当 $\sigma  0$ 时，软阈值 $\tau _ { i }$ 收敛到硬阈值 $s _ { i \left\lfloor \tilde { k } _ { i } \right\rfloor }$。$\tau _ { i }$ 关于 $r _ { i }$ 的梯度可以表示为：

$$
\frac {\partial \tau_ {i}}{\partial r _ {i}} = N _ {v} \sum_ {j = 1} ^ {N _ {v}} \frac {w _ {i j} (j - \tilde {k} _ {i})}{\sigma^ {2} \sum_ {l = 1} ^ {N _ {v}} w _ {i l}} \left(s _ {i j} - \tau_ {i}\right), \tag {10}
$$

对所有 $r _ { i } \in ( 0 , 1 )$ 均有良好定义，使得梯度能够通过反向传播顺畅地流动。

**直通估计器。** 为解决第二个挑战，我们采用直通估计器（STE）策略。受 Gumbel-Softmax（Jang et al., 2016）启发，我们在前向传播中使用离散的硬决策，在反向传播中使用连续的软近似。给定阈值 $\tau _ { i }$，对于第 $i$ 层的每个视觉令牌 $j$，我们同时计算硬选择掩码和软掩码：

$$
m _ {i j} = \mathbb {I} [ s _ {i j} \geq \tau_ {i} ], \quad \tilde {m} _ {i j} = \operatorname {s i g m} \left(\frac {s _ {i j} - \tau_ {i}}{T}\right), \tag {11}
$$

其中 $\mathbb { I } [ \cdot ]$ 为指示函数，$\mathrm { s i g m } ( \cdot )$ 为sigmoid函数，$T$ 为温度参数。软掩码 $\tilde { m } _ { i j }$ 提供了一个平滑的近似：当 $T  0 ^ { + }$ 时，$\tilde { m } _ { i j }  m _ { i j }$。STE 通过以下方式结合两种掩码：

$$
\hat {m} _ {i j} = \tilde {m} _ {i j} + \operatorname {s g} \left(m _ {i j} - \tilde {m} _ {i j}\right), \tag {12}
$$

其中 $\operatorname { s g } ( \cdot )$ 表示停止梯度操作。这一公式确保 $\hat { m } _ { i j } = m _ { i j }$ 在前向传播中成立，而反向传播的梯度满足：

$$
\frac {\partial \mathcal {L}}{\partial \tau_ {i}} = \sum_ {j} \frac {\partial \mathcal {L}}{\partial \hat {m} _ {i j}} \cdot \frac {\partial \tilde {m} _ {i j}}{\partial \tau_ {i}} = - \sum_ {j} \frac {\partial \mathcal {L}}{\partial \hat {m} _ {i j}} \cdot \frac {\tilde {m} _ {i j} (1 - \tilde {m} _ {i j})}{T}, \tag {13}
$$

其中 $\frac { \partial \mathcal { L } } { \partial \hat { m } _ { i j } }$ ∂mˆ ij 表示从后续层传播的上游梯度，由于 $\hat { m } _ { i j }$ 在反向传播中被视为 $m _ { i j }$ 的可微代理，因此有良好定义。这些提供了有偏但低方差的梯度估计器，使得令牌分数和自适应阈值的端到端优化成为可能。

# 算法 1 VisPCO 训练流程

1: 输入：初始配置 $\mathbf { r } ^ { ( 0 ) }$，初始拉格朗日乘子 $w ^ { ( 1 ) }$，惩罚参数 $\lambda$，收敛阈值 $\epsilon > 0$，更新系数 $\alpha > 1$，$\beta \in ( 0 , 1 )$

2: 输出：局部最优配置 $\bar{r}$ 和最优乘子 $\bar { w }$

3: 令 $k = 1$

4: while True do

5:     从 $\mathbf { r } ^ { ( k - 1 ) }$ 出发，求解优化问题 $\min \phi ( \mathbf { r } , w , \lambda )$

6:     /* 使用梯度下降进行训练 */

7:     得到解 $\mathbf { r } ^ { ( k ) }$

8:     if $\| B - F ( \mathbf { r } ^ { ( k ) } ) \| < \epsilon$ then

9:         /* 约束已满足，训练收敛 */

10:        break

11:    end if

12:    if $\begin{array} { r } { \frac { \| B - F ( { \bf r } ^ { ( k ) } ) \| } { \| B - F ( { \bf r } ^ { ( k - 1 ) } ) \| } \ge \beta } \end{array}$ then

13:        /* 更新惩罚参数 */

14:        $\lambda  \alpha \lambda$

15:    end if

16:    /* 更新拉格朗日乘子 */

17:    $w ^ { ( k + 1 ) } \gets w ^ { ( k ) } - \lambda ( B - F ( { \bf r } ^ { ( k ) } ) )$

18:    k ← k + 1

19: end while

20: return $\bar{r} = r^{(k)}$，$\bar{w} = w^{(k)}$

# 3.3 可学习核函数

借助 VisPCO，我们可以以可微的方式自动搜索帕累托前沿上的配置。为进一步研究不同剪枝模式如何影响帕累托前沿，我们对剪枝配置搜索空间施加结构约束。在实践中，视觉令牌剪枝在各层中呈现出单调不增的模式：较深的层倾向于保留更少的令牌。我们利用这一先验知识，引入可学习核函数来参数化逐层剪枝比例。这一设计有两个关键优势：(1) 为视觉令牌剪枝行为提供可解释性；(2) 减少参数搜索空间，确保优化稳定性和计算效率。

我们考虑两种剪枝场景：单层剪枝和多层剪枝。对于第一种情况，我们采用参数化的 p-sigmoid 核来模拟在第 $k$ 层的急剧转变：

$$
\mathcal {K} _ {\mathrm {s}} (i; k, r, \gamma) = 1 + (r - 1) \cdot \operatorname {s i g m} (\gamma (i - k)), \tag {14}
$$

其中 $i$ 为层索引，$k$ 为剪枝位置，$r \in ( 0 , 1 ]$ 控制最终保留比例，$\gamma >$ 0 为锐度参数，$\mathrm { s i g m } ( \cdot )$ 为sigmoid函数。当 $\gamma$ 足够大时，转变趋近于阶跃函数，在第 $k$ 层之前保留所有令牌，之后应用保留比例 $r$，从而有效近似单层剪枝同时保持可微性。

对于第二种情况，我们探索多种核函数来捕捉多样化的剪枝模式。首先，受认知心理学中艾宾浩斯遗忘曲线的启发（Fuchs, 2000），我们设计了一个指数衰减核，以研究 VLMs 是否在各层中对视觉令牌表现出类似的注意力衰减模式：

$$
\mathcal {K} _ {\mathrm {e}} (i; k, r) = r \cdot e ^ {- k \cdot i}. \tag {15}
$$

其次，我们考虑线性衰减核来模拟均匀、渐进的令牌减少：

$$
\mathcal {K} _ {1} (i; k, r) = - k \cdot i + r. \tag {16}
$$

第三，受（Zhao et al., 2025）中发现注意力分数排名在各层保持相似并遵循sigmoid曲线的研究启发，我们采用温和的 p-sigmoid 核。依照式(14)，我们使用较小的 $\gamma$ 来捕捉平滑的渐进剪枝转变。最后，受深度网络在不同层次进行特征提取的层次表示学习的启发，我们引入多步sigmoid核来建模 VLMs 在多个关键层压缩信息的假说：

$$
\mathcal {K} _ {\mathrm {m s}} (i; k, r, M) = 1 - \sum_ {j = 1} ^ {M} \frac {1 - r}{M} \cdot \sigma \left(k \left(i - \frac {(2 j - 1) L}{2 M}\right)\right), \tag {17}
$$

其中 $M$ 为剪枝步数，第 $j$ 步的中心位于第 $\frac { ( 2 j - 1 ) L } { 2 M }$ 层，将各步均匀分布在各层间。这一设计创建了 $M$ 个均匀分布的决策点，用于在各层中渐进式地压缩信息。总体而言，这些可学习核函数涵盖了广泛的剪枝模式，使模型能够发现任务特定的压缩策略。

参数 $k$ 和 $r$ 由轻量级代理神经网络 $f _ { \theta }$ 动态预测。如图2右上方所示，该网络以视觉和文本嵌入的拼接及计算预算 $B$ 为输入，并根据所选剪枝模式计算逐层保留比例 $r _ { i } = \mathcal { K } ( i ; k , r )$。所有输出 $r _ { i }$ 被截断到 $[0, 1]$。通过基于梯度的优化，代理网络学习哪些层和令牌是关键的，从而提供了 VLMs 如何在各层中优先处理视觉信息的机制性见解。

# 4 实验

# 4.1 实现细节

我们使用 Qwen2.5VL-3B（Bai et al., 2025）作为基础模型，通过从 LLaVA-Instruct-150K（Liu et al., 2023）中下采样3万条样本构建训练集。为缓解图像分辨率的长尾分布问题，我们采用基于分辨率的重采样，以防止在罕见图像尺寸上的性能下降。我们的评估涵盖三类基准：视觉问答（A-OKVQA（Schwenk et al., 2022）、VizWiz（Bigham et al., 2010）、SEED-Bench（Li et al., 2023））、多模态推理（MMBench（Liu et al., 2024a）、MME（Fu et al., 2025））以及图表理解（ChartQA（Masry et al., 2022）、OCRBench（Liu et al., 2024b）、TextVQA（Singh et al., 2019））。

对于单层剪枝，我们设置惩罚参数 $\lambda = 1 0 0$，收敛阈值 $\epsilon = 0 . 0 1$，更新系数 $\alpha = 2$，$\beta \ : = \ : 0 . 5$。高斯核宽度 $\sigma = 1 0$ 和温度 $T ~ = ~ 0 . 1$ 分别控制连续松弛和直通估计器。我们使用 AdamW 优化器，学习率为 $4 \times 1 0 ^ { - 4 }$，批大小为16。所有实验在8块 NVIDIA H20 GPU（每块96GB）上进行。其他训练配置见附录C.1。

# 4.2 帕累托前沿近似

我们将 VisPCO 应用于三种具有代表性的剪枝方法：FastV（Chen et al., 2024a）、FitPrune（Ye et al., 2025a）和 SparseVLM（Zhang et al., 2024），这三种方法采用不同的视觉令牌重要性评分机制。表1比较了在不同 FLOPs 预算下应用 VisPCO 前后的性能。对于每种方法，未使用 VisPCO 的结果通过对多个满足预算约束的配置进行采样并取均值得到（报告标准差 $\pm$ std）。图3（左）展示了通过网格搜索获得的经验帕累托前沿与 VisPCO 预测的帕累托前沿，x轴为计算预算，y轴为8个视觉基准上的平均准确率。


表1：在不同预算下，各剪枝方法在八个基准上使用 VisPCO 前后的对比。未使用 VisPCO 的结果为满足预算约束的多个采样配置的均值（$\pm$ std）。

<table><tr><td>方法</td><td>AOKVQA</td><td>VizWiz</td><td>SEED</td><td>MMB</td><td>\(MME^†\)</td><td>ChartQA</td><td>OCRB</td><td>TextVQA</td><td>Avg (%)</td></tr><tr><td colspan="10">上界，100% 预算，~3.56 TFLOPs</td></tr><tr><td>Qwen2.5VL-3B</td><td>90.2</td><td>75.1</td><td>75.6</td><td>79.8</td><td>84.2</td><td>64.1</td><td>74.6</td><td>81.3</td><td>78.1</td></tr><tr><td colspan="10">FLOPs 预算降低至 90%，~3.20 TFLOPs</td></tr><tr><td>FastV</td><td>88.2 ± 0.4</td><td>72.9 ± 0.9</td><td>72.4 ± 0.9</td><td>76.4 ± 0.5</td><td>81.3 ± 0.5</td><td>62.2 ± 0.8</td><td>71.6 ± 0.7</td><td>79.1 ± 0.6</td><td>75.5 ± 0.7</td></tr><tr><td>+ VisPCO</td><td>88.4</td><td>73.8</td><td>73.2</td><td>76.9</td><td>81.7</td><td>62.9</td><td>72.3</td><td>79.5</td><td>76.1</td></tr><tr><td>SparseVLM</td><td>88.5 ± 0.3</td><td>73.1 ± 0.5</td><td>73.4 ± 0.4</td><td>76.9 ± 0.6</td><td>82.1 ± 0.3</td><td>62.2 ± 0.7</td><td>71.9 ± 0.6</td><td>79.5 ± 0.6</td><td>76.0 ± 0.5</td></tr><tr><td>+ VisPCO</td><td>88.6</td><td>73.5</td><td>73.8</td><td>77.5</td><td>82.4</td><td>62.9</td><td>72.5</td><td>80.0</td><td>76.4</td></tr><tr><td>FitPrune</td><td>89.1 ± 0.5</td><td>73.9 ± 0.4</td><td>74.2 ± 0.5</td><td>77.6 ± 0.4</td><td>82.5 ± 0.6</td><td>63.1 ± 0.6</td><td>72.5 ± 0.5</td><td>79.9 ± 0.3</td><td>76.2 ± 0.5</td></tr><tr><td>+ VisPCO</td><td>89.6</td><td>74.1</td><td>74.6</td><td>77.9</td><td>82.8</td><td>63.5</td><td>72.9</td><td>81.2</td><td>77.1</td></tr><tr><td colspan="10">FLOPs 预算降低至 50%，~3.56 TFLOPs</td></tr><tr><td>FastV</td><td>74.7 ± 10.1</td><td>60.3 ± 9.6</td><td>61.5 ± 8.1</td><td>62.4 ± 9.3</td><td>68.8 ± 9.1</td><td>51.6 ± 9.9</td><td>59.2 ± 9.1</td><td>65.9 ± 10.8</td><td>63.1 ± 9.5</td></tr><tr><td>+ VisPCO</td><td>84.8</td><td>69.4</td><td>67.6</td><td>71.2</td><td>77.1</td><td>58.1</td><td>67.8</td><td>75.9</td><td>71.5</td></tr><tr><td>SparseVLM</td><td>75.9 ± 9.8</td><td>62.6 ± 8.2</td><td>63.1 ± 7.2</td><td>63.9 ± 8.6</td><td>69.9 ± 8.2</td><td>51.9 ± 8.3</td><td>62.4 ± 6.9</td><td>66.6 ± 9.8</td><td>64.5 ± 8.4</td></tr><tr><td>+ VisPCO</td><td>85.2</td><td>69.0</td><td>68.1</td><td>71.9</td><td>77.6</td><td>58.4</td><td>67.9</td><td>76.3</td><td>71.8</td></tr><tr><td>FitPrune</td><td>77.1 ± 8.7</td><td>63.4 ± 7.7</td><td>63.9 ± 6.8</td><td>64.5 ± 8.2</td><td>70.8 ± 7.9</td><td>52.8 ± 8.1</td><td>63.3 ± 6.4</td><td>67.6 ± 9.4</td><td>65.4 ± 7.9</td></tr><tr><td>+ VisPCO</td><td>85.9</td><td>69.4</td><td>68.4</td><td>72.4</td><td>77.9</td><td>58.8</td><td>68.2</td><td>76.6</td><td>72.2</td></tr><tr><td colspan="10">FLOPs 预算降低至 10%，~0.36 TFLOPs</td></tr><tr><td>FastV</td><td>33.3 ± 2.3</td><td>30.4 ± 1.6</td><td>44.5 ± 2.7</td><td>33.0 ± 2.5</td><td>39.7 ± 1.4</td><td>29.8 ± 4.1</td><td>8.3 ± 2.1</td><td>33.7 ± 2.8</td><td>31.6 ± 2.4</td></tr><tr><td>+ VisPCO</td><td>35.5</td><td>31.7</td><td>46.9</td><td>35.5</td><td>40.1</td><td>33.2</td><td>10.1</td><td>36.1</td><td>33.6</td></tr><tr><td>SparseVLM</td><td>33.6 ± 2.1</td><td>31.2 ± 1.3</td><td>44.9 ± 2.5</td><td>33.9 ± 2.3</td><td>40.3 ± 1.1</td><td>30.5 ± 3.7</td><td>9.1 ± 2.0</td><td>34.4 ± 2.2</td><td>32.2 ± 2.2</td></tr><tr><td>+ VisPCO</td><td>35.5</td><td>31.5</td><td>47.1</td><td>35.8</td><td>40.4</td><td>33.3</td><td>10.2</td><td>36.3</td><td>33.8</td></tr><tr><td>FitPrune</td><td>33.8 ± 2.1</td><td>31.5 ± 1.1</td><td>45.3 ± 2.4</td><td>34.2 ± 2.2</td><td>40.6 ± 1.0</td><td>30.9 ± 3.5</td><td>9.6 ± 1.9</td><td>34.6 ± 2.1</td><td>32.6 ± 2.0</td></tr><tr><td>+ VisPCO</td><td>35.6</td><td>31.6</td><td>47.3</td><td>35.8</td><td>40.9</td><td>33.5</td><td>10.4</td><td>36.4</td><td>33.9</td></tr></table>


**中等预算下的性能提升。** 如表1所示，VisPCO 的收益在不同计算预算下差异显著。在极端预算下，配置选择的影响有限。例如，在90%预算下，由于资源充足，不同配置之间的性能差异不足1个百分点。类似地，在极低预算下，严重的资源限制制约了所有配置。相比之下，中等预算（例如50%）是一个关键区间，配置选择对性能的影响显著——不同配置之间的性能差距可高达19个百分点。这一显著的性能差距证明了原则性配置优化的重要性，也验证了 VisPCO 等方法的必要性。

**前沿近似质量。** 如图3（左）所示，预测的帕累托前沿与经验前沿几乎完美吻合。这验证了我们基于核的近似方法在捕捉真实计算-性能权衡格局方面的有效性。表2与现有方法进行了全面比较，包括预定义剪枝配置策略、基于训练的方法和随机搜索基线（Random-N表示从N个随机样本中选取最优）。我们同时评估搜索效率和性能。VisPCO 优于所有基线


表2：配置搜索方法对比。时间表示识别最优配置所需的搜索时间（对 VisPCO 而言为训练时间）。Random-N 表示对N个配置进行随机采样。所有方法目标计算预算相近。


<table><tr><td>方法</td><td>FLOPs (T) ↓</td><td>时间 (h) ↓</td><td>MMB ↑</td><td>SEED ↑</td><td>TQA ↑</td></tr><tr><td>VTW</td><td>2.34</td><td>1+</td><td>36.5</td><td>44.1</td><td>73.2</td></tr><tr><td>G-Search</td><td>2.58</td><td>-</td><td>42.1</td><td>47.5</td><td>80.2</td></tr><tr><td>ATP-LLaVA</td><td>2.23</td><td>48+</td><td>37.2</td><td>45.5</td><td>77.2</td></tr><tr><td>MADTP</td><td>3.91</td><td>6+</td><td>42.4</td><td>46.8</td><td>79.1</td></tr><tr><td>AIM</td><td>2.33</td><td>-</td><td>39.5</td><td>43.8</td><td>74.6</td></tr><tr><td>Random-40</td><td>2.18</td><td>12+</td><td>31.2</td><td>36.6</td><td>70.9</td></tr><tr><td>Random-80</td><td>2.20</td><td>24+</td><td>42.8</td><td>47.7</td><td>80.1</td></tr><tr><td>Random-160</td><td>2.20</td><td>48+</td><td>44.0</td><td>48.1</td><td>82.4</td></tr><tr><td>VisPCO</td><td>2.20</td><td>1+</td><td>43.6</td><td>47.5</td><td>81.3</td></tr></table>

方法，包括 VTW（Lin et al., 2025）、G-Search（Zhao et al., 2025）、ATP-LLaVA（Ye et al., 2025b）、MADTP（Cao et al., 2024）、AIM（Zhong et al., 2025）以及中等规模的随机搜索变体，而仅需1小时的训练时间。值得注意的是，MADTP 需要额外训练 MAG 和 DTP 模块（6+小时），且产生显著更高的 FLOPs（3.91T）。AIM 是免训练的，采用固定的逐层剪枝策略；在相近 FLOPs 下（2.20T vs. 2.33T），VisPCO 取得了明显更优的性能（MMB：43.6 vs. 39.5；TQA：81.3 vs. 74.6；SEED：47.5 vs. 43.8）。

# 4.3 跨模型泛化

为验证 VisPCO 在不同 VLM 架构上的泛化能力，我们将其应用于 Gemma3-4B（Team et al., 2025）和 LLaVA-v1.5-7B（Liu et al., 2023）。表3展示了使用 FastV（Chen et al., 2024a）在 $50\%$ FLOPs 预算下的代表性结果。

![[Pasted image 20260429140039.png]]

图3：VisPCO 的实验结果。（左）经验帕累托前沿与预测帕累托前沿的对比。（中）不同 VLM 架构下经验帕累托前沿与预测帕累托前沿的对比。（右）不同剪枝模式下帕累托前沿的对比。



表3：不同 VLM 在 $50\%$ FLOPs 预算下使用 VisPCO 前后的性能对比（$\pm$ std）。

<table><tr><td>模型</td><td>AOKVQA</td><td>MMBench</td><td>TextVQA</td><td>Avg (%)</td></tr><tr><td>LLaVA-7B</td><td>50.3 ± 10.7</td><td>26.5 ± 11.5</td><td>64.2 ± 11.4</td><td>47.0 ± 11.2</td></tr><tr><td>+ VisPCO</td><td>60.8</td><td>38.0</td><td>75.2</td><td>58.0</td></tr><tr><td>Gemma3-4B</td><td>41.2 ± 12.4</td><td>44.7 ± 12.9</td><td>33.6 ± 12.3</td><td>39.8 ± 12.5</td></tr><tr><td>+ VisPCO</td><td>53.4</td><td>57.3</td><td>45.5</td><td>52.1</td></tr><tr><td>Qwen2.5VL-3B</td><td>74.7 ± 10.1</td><td>62.4 ± 9.3</td><td>65.9 ± 10.8</td><td>67.7 ± 10.1</td></tr><tr><td>+ VisPCO</td><td>84.8</td><td>71.2</td><td>75.9</td><td>77.3</td></tr></table>


表4：在 $50\%$ 计算预算下各模型的硬件性能测量。TTFT：首字生成时间。吞吐量：每秒生成的令牌数。平均性能：各基准上的平均准确率。

<table><tr><td>方法</td><td>预算</td><td>TTFT (ms) ↓</td><td>吞吐量 (tokens/s) ↑</td><td>平均性能 ↑</td></tr><tr><td>Qwen2.5VL-3B</td><td>100%</td><td>83 ± 3</td><td>18 ± 2</td><td>78.1</td></tr><tr><td>+ FastV</td><td>50%</td><td>74 ± 2</td><td>20 ± 3</td><td>63.1</td></tr><tr><td>+ FastV + VisPCO</td><td>50%</td><td>76 ± 3</td><td>20 ± 2</td><td>71.5</td></tr><tr><td>LLaVA-v1.5-7B</td><td>100%</td><td>114 ± 12</td><td>14 ± 6</td><td>63.9</td></tr><tr><td>+ FastV</td><td>50%</td><td>96 ± 10</td><td>16 ± 6</td><td>41.4</td></tr><tr><td>+ FastV + VisPCO</td><td>50%</td><td>97 ± 8</td><td>16 ± 4</td><td>52.3</td></tr><tr><td>Gemma3-4B</td><td>100%</td><td>94 ± 6</td><td>16 ± 4</td><td>68.9</td></tr><tr><td>+ FastV</td><td>50%</td><td>81 ± 7</td><td>18 ± 5</td><td>38.8</td></tr><tr><td>+ FastV + VisPCO</td><td>50%</td><td>81 ± 6</td><td>18 ± 5</td><td>50.8</td></tr></table>


**架构间的一致性。** 尽管 Qwen2.5VL 相比 Gemma3 和 LLaVA 表现出更宽的性能范围，VisPCO 始终能够选择最优配置。图3（中）表明，在所有架构上，预测的帕累托前沿与经验前沿紧密吻合，展示了强大的泛化能力。值得注意的是，Qwen2.5VL 的前沿位于左上区域，表明其具有更优的精度-效率特性。这一优势源于其原生图像分辨率，而 Gemma3 和 LLaVA 则将输入调整为固定尺寸。这表明，对于高效 VLMs 而言，保留原始尺寸并结合学习式剪枝可能比激进的预处理更有效。

**硬件效率。** 为验证 FLOPs 的降低能够转化为实际加速，我们在 NVIDIA H20 GPU 上测量了首字生成时间（TTFT）和吞吐量。如表4所示，在 $50\%$ 计算预算下，应用 VisPCO 的 FastV 与单独使用 FastV 保持相同的硬件效率——相近的 TTFT 和吞吐量——同时显著恢复了因剪枝损失的性能。例如，在 Qwen2.5VL-3B 上，VisPCO 将平均性能从 $6 3 . 1 \%$ 提升至 $7 1 . 5 \%$，而延迟开销可忽略不计（TTFT 为76 ms vs. 74 ms）。LLaVA-v1.5-7B 和 Gemma3-4B 上也呈现相似规律，证实了 VisPCO 在显著提升任务性能的同时，保留了底层剪枝方法的硬件效率。

# 4.4 剪枝模式分析

我们研究了单层与多层剪枝策略之间的性能差异，以及多层配置中不同核选择的影响。表5展示了在 $50\%$ 计算预算下不同剪枝模式的帕累托最优结果。图3（右）展示并比较了不同剪枝模式下的帕累托前沿。

**剪枝模式的策略性选择。** 如表5所示，采用多步核的多层剪枝在 $50\%$ 预算下取得了最佳性能，优于单层剪枝和其他核变体（线性、指数、sig-

moid）。图3（右）揭示了这一优势与预算密切相关。当计算预算超过 $50\%$ 时，所有剪枝模式收敛到相近的性能，并与经验帕累托前沿紧密吻合，此时策略选择影响较小。然而，当预算低于 $50\%$ 时，各剪枝模式之间出现显著差异，多步核展现出明显的优势。

**对 VLM 设计的启示。** 多步核在低预算下的优越性能揭示了重要的架构洞见。视觉令牌冗余性在特定层中出现，而非在网络中均匀分布。某些层通过注意力或特征变换引入冗余，而其他层则保留关键表示。多步核能够识别这些关键压缩点，在保留关键信息的同时实现有针对性的剪枝。这些发现提供了实践指导：当资源充足时（预算 $50\%$），简单的单层剪枝即可实现接近最优的性能；在资源紧张时（预算 ${ < } 5 0 \%$），推荐采用多步逐层剪枝，以更好地利用 VLMs 的层次化压缩结构。


表5：在 $50\%$ FLOPs 预算下不同剪枝模式的性能对比。

<table><tr><td>模式</td><td>核函数</td><td>AOK</td><td>MMB</td><td>TQA</td><td>Avg (%)</td></tr><tr><td>单层</td><td>-</td><td>84.8</td><td>71.2</td><td>75.9</td><td>77.3</td></tr><tr><td rowspan="4">多层</td><td>线性</td><td>82.6</td><td>70.9</td><td>74.9</td><td>76.1</td></tr><tr><td>指数</td><td>82.2</td><td>70.4</td><td>74.4</td><td>75.7</td></tr><tr><td>P-Sigmoid</td><td>81.9</td><td>69.6</td><td>74.1</td><td>75.2</td></tr><tr><td>多步</td><td>84.9</td><td>71.8</td><td>76.7</td><td>77.8</td></tr></table>

# 5 结论

在本文中，我们提出了 VisPCO，一个新颖的计算预算感知框架，用于自动优化视觉语言模型中的视觉令牌剪枝配置。通过将问题表述为带连续松弛的帕累托优化，VisPCO 实现了高效的端到端基于梯度的训练，能够针对任意给定的计算预算自动识别最优剪枝配置。与传统的穷举网格搜索方法相比，该方法显著降低了搜索成本。在8个视觉基准上的大量实验表明，我们的方法在各种剪枝策略和 VLM 架构上均具有良好的泛化能力。此外，通过可学习核函数的探究，我们发现渐进式多步剪枝在性能上始终优于单层方法和其他多层核方法，为资源受限部署场景下高效 VLM 设计提供了有价值的见解。

# 局限性

尽管我们的框架在多样化基准和模型架构上表现出强劲的性能，但仍有若干局限性有待

在未来工作中加以解决。第一，我们的实验主要聚焦于单图像任务，尚需进一步验证优化后的剪枝配置在多图像和视频输入上的泛化效果——这些场景涉及更为复杂的时空冗余。第二，我们提出的核函数以结构化、可解释的方式对剪枝分布进行建模。未来工作可探索将该方法扩展至学习更灵活的非参数或输入自适应模式，从而潜在地捕捉更细粒度的任务特定剪枝策略。

# 伦理声明

本工作聚焦于优化视觉语言模型的视觉令牌剪枝配置以提升计算效率。我们的方法不涉及私人或敏感数据的收集或使用；所有实验均在公开可用的基准上进行。我们预见本研究不会产生直接的负面社会影响。通过降低 VLMs 的计算成本，本工作有望助力减少大规模模型推理所带来的能耗和碳排放，从而推动更可持续、更普惠的 AI 部署。

# 致谢

我们衷心感谢上海交通大学数据智能研究中心的同学和工程师们在本工作研发过程中提供的支持与帮助。本工作受国家自然科学基金资助，项目编号为 T2421002、92579104、62525209、T2542021。

# References

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, and 1 others. 2025. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661. 




Kenneth J Arrow and Gerard Debreu. 2024. Existence of an equilibrium for a competitive economy. In The Foundations of Price Theory Vol 5, pages 289–316. Routledge. 




Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 




2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. 




Dimitri P Bertsekas. 2014. Constrained optimization and Lagrange multiplier methods. Academic press. 




Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, and 1 others. 2010. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23rd annual ACM symposium on User interface software and technology, pages 333–342. 




Jianjian Cao, Peng Ye, Shengze Li, Chong Yu, Yansong Tang, Jiwen Lu, and Tao Chen. 2024. Madtp: Multimodal alignment-guided dynamic token pruning for accelerating vision-language transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15710–15719. 




Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer. 




Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, and 1 others. 2024b. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271. 




Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2025. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The 39th Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 




Thomas Fuchs. 2000. Das gedächtnis des leibes. Phänomenologische Forschungen, 5(1):71–89. 




Zonghao Guo, Ruyi Xu, Yuan Yao, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. 2024. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In European Conference on Computer Vision, pages 390–406. Springer. 




Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaosheng Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, Yao Hu, and Shaohui Lin. 2024. Dynamic-llava: Efficient multimodal large language models via dynamic vision-language context sparsification. arXiv preprint arXiv:2412.00876. 




Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144. 




Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125. 




Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5971–5984. 




Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. 2025. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 5334–5342. 




Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916. 




Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024a. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer. 




Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. 2024b. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102. 




Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263– 2279. 




Jorge Nocedal and Stephen J Wright. 2006. Numerical optimization. Springer. 




Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer. 




Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326. 




Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786. 


Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and 1 others. 2024. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247. 

Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. 2025. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608. 

Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, and Bo Yuan. 2025. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19803–19813. 

Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. 2025a. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 22128–22136. 

Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. 2025b. Atp-llava: Adaptive token pruning for large vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24972–24982. 

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. 2024. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417. 

Shiyu Zhao, Zhenting Wang, Felix Juefei-Xu, Xide Xia, Miao Liu, Xiaofang Wang, Mingfu Liang, Ning Zhang, Dimitris N Metaxas, and Licheng Yu. 2025. Accelerating multimodal large language models by searching optimal vision token reduction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29869–29879. 

Yiwu Zhong, Zhuoming Liu, Yin Li, and Liwei Wang. 2025. Aim: Adaptive inference of multi-modal llms via token merging and pruning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20180–20192. 

Yi Zhou, Hui Zhang, Jiaqian Yu, Yifan Yang, Sangil Jung, Seung-In Park, and ByungIn Yoo. 2024. Himap: Hybrid representation learning for end-toend vectorized hd map construction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15396–15406. 

# A FLOPs 计算

本节对视觉语言模型中 Transformer 层的浮点运算次数（FLOPs）计算进行详细推导。

对于标准 Transformer 层，主要计算成本来自自注意力机制和前馈网络（FFN）。给定序列长度 $N$ 和隐藏维度 $D$，我们分别计算每个组件的 FLOPs。

# A.1 自注意力机制

自注意力机制由以下操作组成：

(1) 线性投影：三个投影矩阵 $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { D \times D }$ 将输入映射为 Query、Key 和 Value 表示。每次矩阵乘法需要 $2 N D ^ { 2 }$ 次浮点运算（将大小为 $N \times D$ 的输入与大小为 $D \times D$ 的权重相乘），因此：

$$
\mathrm {F L O P s} _ {\mathrm {Q K V}} = 3 \times 2 N D ^ {2} = 6 N D ^ {2}. \tag {18}
$$

(2) 注意力分数：计算 $\begin{array} { r l } { \mathbf { Q K } ^ { T } } & { { } \in } \end{array}$ $\mathbb { R } ^ { N \times N }$，其中 $\mathbf { Q } , \mathbf { K } \in \mathbb { R } ^ { N \times D }$：

$$
\mathrm {F L O P s} _ {\text {S c o r e}} = 2 N ^ {2} D. \tag {19}
$$

(3) 注意力加权：计算 Softmax $( \mathbf { Q K } ^ { T } / \sqrt { D } ) \mathbf { V }$，即将 $\mathbb { R } ^ { N \times N }$ 与 $\mathbb { R } ^ { N \times D }$ 相乘：

$$
\mathrm {F L O P s} _ {\text {W e i g h t}} = 2 N ^ {2} D. \tag {20}
$$

注：Softmax 操作的 FLOPs 相对较小，通常忽略不计。

(4) 输出投影：通过 $\mathbf { W } _ { O } \in \mathbb { R } ^ { D \times D }$ 投影回原始维度：

$$
\mathrm {F L O P s} _ {\text {O u t p u t}} = 2 N D ^ {2}. \tag {21}
$$

因此，自注意力机制的总 FLOPs 为：

$$
\mathrm {F L O P s} _ {\text {A t t e n t i o n}} = 8 N D ^ {2} + 4 N ^ {2} D. \tag {22}
$$

# A.1.1 前馈网络

标准 FFN 由两个线性层组成，中间维度为 $D _ { \mathrm { f f n } }$：

$$
\operatorname {F F N} (\mathbf {x}) = \mathbf {W} _ {2} \cdot \operatorname {G E L U} \left(\mathbf {W} _ {1} \cdot \mathbf {x}\right). \tag {23}
$$

其中 $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { D \times D _ { \mathrm { f f n } } }$，$\mathbf { W } _ { 2 } \in \mathbb { R } ^ { D _ { \mathrm { f f n } } \times D }$。FFN 的总 FLOPs 为：

$$
\mathrm {F L O P s} _ {\mathrm {F F N}} = 4 N D _ {\mathrm {f f n}} D. \tag {24}
$$

# A.1.2 单层总 FLOPs

将自注意力与 FFN 合并，单个 Transformer 层的总 FLOPs 为：

$$
\mathrm {F L O P s} _ {\text {l a y e r}} = 8 N D ^ {2} + 4 N ^ {2} D + 4 N D _ {\mathrm {f f n}} D. \tag {25}
$$

在标准 Transformer 架构中，$D _ { \mathrm { f f n } } =$ $4 D$，由此得：

$$
\mathrm {F L O P s} _ {\text {l a y e r}} = 2 4 N D ^ {2} + 4 N ^ {2} D. \tag {26}
$$

LayerNorm 和残差连接等计算成本相对较小，忽略不计。

# A.2 视觉语言模型的总 FLOPs

对于视觉语言模型，输入序列由文本令牌和视觉令牌组成。设 $N _ { t }$ 为文本令牌数量，$N _ { v }$ 为初始视觉令牌数量。第 $i$ 层的令牌总数为：

$$
N _ {i} = N _ {t} + r _ {i} N _ {v} \tag {27}
$$

其中 $r _ { i } \in [ 0 , 1 ]$ 表示第 $i$ 层视觉令牌的保留比例。

对于具有 $L$ 层的 Transformer 模型，总计算成本为：

$$
F (\mathbf {r}) = \sum_ {i = 1} ^ {L} \left[ 2 4 \left(N _ {t} + r _ {i} N _ {v}\right) D ^ {2} + 4 \left(N _ {t} + r _ {i} N _ {v}\right) ^ {2} D \right] \tag {28}
$$

其中：

• 第一项 $2 4 ( N _ { t } + r _ { i } N _ { v } ) D ^ { 2 }$ 对应自注意力和 FFN 中的线性投影

• 第二项 $4 ( N _ { t } { + } r _ { i } N _ { v } ) ^ { 2 } D$ 对应注意力矩阵计算的二次方复杂度

• $\mathbf { r } ~ = ~ \left[ r _ { 1 } , r _ { 2 } , \ldots , r _ { L } \right]$ 为各层视觉令牌保留比例向量

该公式表明，随着视觉令牌被剪枝（$r _ { i }$ 减小），模型的计算成本显著降低，尤其是二次方复杂度项。

# B 理论分析

# B.1 定理1的证明

考虑如下等式约束优化问题：

$$
\begin{array}{l} \min  f (\mathbf {x}) \tag {29} \\ \begin{array}{l l} \text {s . t .} & h _ {j} (\mathbf {x}) = 0, \quad j = 1, \ldots , l, \end{array} \\ \end{array}
$$

其中 $f , h _ { j } : \mathbb { R } ^ { n }  \mathbb { R }$ 为二阶连续可微函数。该问题的增广拉格朗日函数由式(4)给出。

设 $\bar{x}$ 为问题(29)满足二阶充分条件的局部最优解，即存在拉格朗日乘子向量 $\bar { \mathbf { v } } = [ \bar { v } _ { 1 } , \dots , \bar { v } _ { l } ] ^ { T }$，使得：

$$
\nabla f (\bar {\mathbf {x}}) - \mathbf {A} \bar {\mathbf {v}} = 0, \tag {30}
$$

$$
h _ {j} (\bar {\mathbf {x}}) = 0, \quad j = 1, \dots , l, \tag {31}
$$

且对每个满足 $\mathbf { d } ^ { T } \nabla h _ { j } ( \bar { \mathbf { x } } ) = \bar { 0 }$（$j = 1 , \dots , l$）的非零向量 $\mathbf{d}$，有：

$$
\mathbf {d} ^ {T} \nabla_ {\mathbf {x}} ^ {2} \mathcal {L} (\bar {\mathbf {x}}, \bar {\mathbf {v}}) \mathbf {d} > 0, \tag {32}
$$

其中

$$
\mathbf {A} = \left[ \nabla h _ {1} (\bar {\mathbf {x}}), \dots , \nabla h _ {l} (\bar {\mathbf {x}}) \right], \tag {33}
$$

且 ${ \mathcal { L } } ( \mathbf { x } , \mathbf { v } ) = f ( \mathbf { x } ) - \mathbf { v } ^ { T } \mathbf { h } ( \mathbf { x } )$ 为标准拉格朗日函数。

由假设，$\bar { \bf x }$ 是问题(29)的 Karush-Kuhn-Tucker（KKT）点，因此：

$$
\nabla_ {\mathbf {x}} \phi (\bar {\mathbf {x}}, \bar {\mathbf {v}}, \lambda) = 0. \tag {34}
$$

下面证明 Hessian 矩阵 $\nabla _ { \mathbf { x } } ^ { 2 } \phi ( \bar { \mathbf { x } } , \bar { \mathbf { v } } , \lambda )$ 在 $\bar { \bf x }$ 处对足够大的 $\lambda$ 是正定的。

由式(4)可推导出：

$$
\begin{array}{l} \nabla_ {\mathbf {x}} ^ {2} \phi (\mathbf {x}, \bar {\mathbf {v}}, \lambda) = \nabla^ {2} f (\mathbf {x}) - \sum_ {j = 1} ^ {l} \bar {v} _ {j} \nabla^ {2} h _ {j} (\mathbf {x}) \\ + \sigma \sum_ {j = 1} ^ {l} h _ {j} (\mathbf {x}) \nabla^ {2} h _ {j} (\mathbf {x}) + \lambda \sum_ {j = 1} ^ {l} \nabla h _ {j} (\mathbf {x}) \nabla h _ {j} (\mathbf {x}) ^ {T} \tag {35} \\ = \nabla^ {2} f (\mathbf {x}) - \sum_ {j = 1} ^ {l} (\bar {v} _ {j} - \lambda h _ {j} (\mathbf {x})) \nabla^ {2} h _ {j} (\mathbf {x}) \\ + \lambda \sum_ {j = 1} ^ {l} \nabla h _ {j} (\mathbf {x}) \nabla h _ {j} (\mathbf {x}) ^ {T} = \mathbf {Q} + \lambda \mathbf {A A} ^ {T}, \\ \end{array}
$$

其中

$$
\begin{array}{l} \mathbf {Q} = \nabla^ {2} f (\mathbf {x}) - \sum_ {j = 1} ^ {l} \left(\bar {v} _ {j} - \lambda h _ {j} (\mathbf {x})\right) \nabla^ {2} h _ {j} (\mathbf {x}), (36) \\ \mathbf {A} = \left[ \nabla h _ {1} (\mathbf {x}), \dots , \nabla h _ {l} (\mathbf {x}) \right]. (37) \\ \end{array}
$$

在点 $\bar{x}$ 处，有：

$$
\nabla_ {\mathbf {x}} ^ {2} \phi (\bar {\mathbf {x}}, \bar {\mathbf {v}}, \lambda) = \bar {\mathbf {Q}} + \lambda \bar {\mathbf {A}} \bar {\mathbf {A}} ^ {T}, \tag {38}
$$

其中 $\bar { \mathbf { Q } }$ 和 $\bar { \mathbf A }$ 表示在 $\bar{x}$ 处的取值。

设 $\mathrm{rank}( \bar { \mathbf { A } } ) = r \leq l$，令 $\mathbf { B } \in \mathbb { R } ^ { n \times r }$ 为 $\bar { \mathbf A }$ 的标准正交基矩阵（即 $\mathbf { B } ^ { T } \mathbf { B } = \mathbf { I } _ { r }$），即 $\mathbf{B}$ 的 $r$ 列构成 $\bar { \mathbf A }$ 的 $l$ 列所张成子空间的标准正交基。因此有：

$$
\bar {\mathbf {A}} = \mathbf {B} \mathbf {C}, \tag {39}
$$

其中 $\mathbf { C } = \mathbf { B } ^ { T } \bar { \mathbf { A } }$ 的秩为 $r$。

对任意非零向量 $\mathbf { u } \in \mathbb { R } ^ { n }$，将其分解为：

$$
\mathbf {u} = \mathbf {p} + \mathbf {B q}, \tag {40}
$$

其中 $\mathbf{p}$ 满足 $\mathbf { B } ^ { T } \mathbf { p } = \mathbf { 0 }$。显然 $\bar { \mathbf { A } } ^ { T } \mathbf { p } = \mathbf { 0 }$，即：

$$
\nabla h _ {j} (\bar {\mathbf {x}}) ^ {T} \mathbf {p} = 0, \quad j = 1, \dots , l. \tag {41}
$$

由此可写出 $\mathbf { u } ^ { T } \nabla _ { \mathbf { x } } ^ { 2 } \phi ( \bar { \mathbf { x } } , \bar { \mathbf { v } } , \lambda ) \mathbf { u }$ 为：

$$
\begin{array}{l} \mathbf {u} ^ {T} \nabla_ {\mathbf {x}} ^ {2} \phi (\bar {\mathbf {x}}, \bar {\mathbf {v}}, \lambda) \mathbf {u} \\ = (\mathbf {p} + \mathbf {B q}) ^ {T} \left(\bar {\mathbf {Q}} + \lambda \bar {\mathbf {A}} \bar {\mathbf {A}} ^ {T}\right) (\mathbf {p} + \mathbf {B q}) \tag {42} \\ = \mathbf {p} ^ {T} \bar {\mathbf {Q}} \mathbf {p} + 2 \mathbf {p} ^ {T} \bar {\mathbf {Q}} \mathbf {B} \mathbf {q} + \mathbf {q} ^ {T} \mathbf {B} ^ {T} \bar {\mathbf {Q}} \mathbf {B} \mathbf {q} \\ + \lambda \mathbf {q} ^ {T} \mathbf {C} \mathbf {C} ^ {T} \mathbf {q}. \\ \end{array}
$$

由于 $\bar{x}$ 是问题(29)满足二阶充分条件的局部最优解，存在常数 $\alpha > 0$ 使得：

$$
\mathbf {p} ^ {T} \bar {\mathbf {Q}} \mathbf {p} \geq \alpha \| \mathbf {p} \| ^ {2}. \tag {43}
$$

设 $b$ 为 $\bar { \mathbf { Q B } }$ 的最大奇异值，$e \ = \ \lVert \mathbf { B } ^ { T } \bar { \mathbf { Q } } \mathbf { B } \rVert _ { 2 }$，$\mu > 0$ 为 $\mathbf { C } \mathbf { C } ^ { T }$ 的最小特征值。则：

$$
\mathbf {u} ^ {T} \nabla_ {\mathbf {x}} ^ {2} \phi (\bar {\mathbf {x}}, \bar {\mathbf {v}}, \lambda) \mathbf {u} \geq \alpha \| \mathbf {p} \| ^ {2} - 2 b \| \mathbf {p} \| \| \mathbf {q} \| + (\lambda \mu - e) \| \mathbf {q} \| ^ {2}. \tag {44}
$$

由于 $\mathbf u \neq \mathbf 0$，向量 $\mathbf{p}$ 和 $\mathbf{q}$ 不能同时为零。因此，若选取足够大的 $\lambda$ 使得：

$$
\lambda \mu - e - \frac {b ^ {2}}{\alpha} > 0, \tag {45}
$$

即

$$
\lambda > \frac {b ^ {2} + \alpha e}{\alpha \mu}, \tag {46}
$$

则恒有：

$$
\mathbf {u} ^ {T} \nabla_ {\mathbf {x}} ^ {2} \phi (\bar {\mathbf {x}}, \bar {\mathbf {v}}, \lambda) \mathbf {u} > 0. \tag {47}
$$

因此，存在：

$$
\lambda^ {\prime} = \frac {b ^ {2} + \alpha e}{\alpha \mu}. \tag {48}
$$

当惩罚参数 $\lambda > \lambda ^ { \prime }$ 时，矩阵 $\nabla _ { \mathbf { x } } ^ { 2 } \phi ( \bar { \mathbf { x } } , \bar { \mathbf { v } } , \lambda )$ 是正定的。结合式(34)和式(47)，可得 $\bar { \bf x }$ 是 $\phi ( \mathbf { x } , \bar { \mathbf { v } } , \lambda )$ 的严格局部最小值点。证明完毕。

# B.2 通过配方消去 $y$

从式(7)的增广拉格朗日函数出发，对辅助变量 $y$ 进行配方以消去它。

设 $g ( \mathbf { r } ) = B - F ( \mathbf { r } )$ 为约束函数。将式(7)改写为：

$$
\begin{array}{l} \tilde {\phi} (\mathbf {r}, y, w, \lambda) \\ = \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) - w \left(g (\mathbf {r}) - y ^ {2}\right) + \frac {\lambda}{2} \left(g (\mathbf {r}) - y ^ {2}\right) ^ {2} \tag {49} \\ = \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) + \left[ - w (g (\mathbf {r}) - y ^ {2}) + \frac {\lambda}{2} (g (\mathbf {r}) - y ^ {2}) ^ {2} \right]. \\ \end{array}
$$

对 $y ^ { 2 }$ 进行配方，有：

$$
\begin{array}{l} - w (g (\mathbf {r}) - y ^ {2}) + \frac {\lambda}{2} (g (\mathbf {r}) - y ^ {2}) ^ {2} \\ = \frac {\lambda}{2} \left[ (g (\mathbf {r}) - y ^ {2}) - \frac {w}{\lambda} \right] ^ {2} - \frac {w ^ {2}}{2 \lambda} \\ = \frac {\lambda}{2} \left[ y ^ {2} - \left(g (\mathbf {r}) - \frac {w}{\lambda}\right) \right] ^ {2} - \frac {w ^ {2}}{2 \lambda}. \tag {50} \\ \end{array}
$$

为关于 $y$ 最小化 $\tilde { \phi }$，分析 $y ^ { 2 }$ 的最优值。当满足以下条件时，项 $\begin{array} { r } { \frac { \lambda } { 2 } [ y ^ { 2 } - ( g ( \mathbf { r } ) - \frac { w } { \lambda } ) ] ^ { 2 } } \end{array}$ 取最小值：

$$
y ^ {2} = g (\mathbf {r}) - \frac {w}{\lambda} = \frac {1}{\lambda} (\lambda g (\mathbf {r}) - w). \tag {51}
$$

然而，由于 $y \in \mathbb { R }$，必须有 $y ^ { 2 } \geq 0$，因此最优值为：

$$
y ^ {2} = \max  \left\{0, \frac {1}{\lambda} (\lambda g (\mathbf {r}) - w) \right\}. \tag {52}
$$

等价地表示为：

$$
y ^ {2} = \left\{ \begin{array}{l l} \frac {1}{\lambda} (\lambda g (\mathbf {r}) - w), & \text {若} \lambda g (\mathbf {r}) - w \geq 0, \\ 0, & \text {若} \lambda g (\mathbf {r}) - w <   0. \end{array} \right. \tag {53}
$$

将最优 $y ^ { 2 }$ 代回式(50)，得：

$$
- w \left(g (\mathbf {r}) - y ^ {2}\right) + \frac {\lambda}{2} \left(g (\mathbf {r}) - y ^ {2}\right) ^ {2} = \frac {1}{2 \lambda} \left(z ^ {2} - w ^ {2}\right), \tag {54}
$$

其中

$$
\begin{array}{r l} z & = \max  \left\{0, w - \lambda (B - F (\mathbf {r})) \right\} \\ & = \max  \left\{0, w - \lambda g (\mathbf {r}) \right\}. \end{array} \tag {55}
$$

因此，消去 $y$ 后的简化增广拉格朗日函数为：

$$
\phi (\mathbf {r}, w, \lambda) = \mathcal {L} _ {\text {d i s t i l l}} (\mathbf {r}) + \frac {1}{2 \lambda} (z ^ {2} - w ^ {2}), \tag {56}
$$

其中 $z = \operatorname* { m a x } \{ 0 , w - \lambda ( B - F ( \mathbf { r } ) ) \}$，$g ( \mathbf { r } ) =$ $B - F ( \mathbf { r } )$ 表示约束满足程度。

# C 实验细节

# C.1 实验设置

# C.1.1 训练数据集

我们发现直接在原始数据集上训练会导致次优性能：VisPCO 对高分辨率图像预测的帕累托前沿集中于低计算预算区域，偏离了高预算区间的经验前沿。经分析，训练数据集中图像面积的分布呈现出显著的偏斜，高度集中于较小面积，如图4左图所示。这种不平衡不利于学习合适的剪枝比例，导致对高分辨率图像的泛化能力较差。

为解决这一问题，我们使用直方图均衡化对训练图像进行预处理，以平衡面积分布。具体而言，我们将图像面积范围划分为均匀的区间，并应用分层采样以确保各面积区间的均衡表示。对每个区间，我们对欠表示区间的图像进行过采样，对过表示区间的图像进行欠采样，使各区间的样本数量大致相等。这一重平衡过程确保训练分布均匀覆盖全谱图像分辨率，使 VisPCO 能够为低分辨率和高分辨率图像均学习到鲁棒的剪枝配置。图4左图为原始偏斜分布，右图为均衡化后的平衡分布。

# C.1.2 评测数据集

我们使用 VLMEvalKit 提供的评测数据集，其中包含来自各视觉语言基准的精心整理的问答对和图像。评测涵盖三类任务：视觉问答、多模态推理和图表理解。对于 MME，为保持与其他基准的可比性，我们报告正确答案与总问题数之比，将最终评测结果归一化到 $[0, 1]$ 区间。图5展示了一个评测样例。

**A-OKVQA**（Schwenk et al., 2022）是一个基于知识的视觉问答数据集，要求模型利用超越视觉内容的外部常识和世界知识。它包含1,145个涵盖多样图像类型的问题，挑战模型执行结合视觉理解与事实知识的推理。

**VizWiz**（Bigham et al., 2010）是一个由盲人用户拍摄图片并提问所收集的视觉问答数据集。该数据集包含超过4,319个图文问答对，场景自然真实，通常包含图像质量差、模糊或视角异常等具有挑战性的条件，对评估模型鲁棒性尤为有价值。

**SEEDBench**（Li et al., 2023）是一个用于多维度评估多模态大语言模型的综合基准。它包含14,232道多项选择题，涵盖九个评估维度，包括场景理解、实例识别、空间关系和视觉推理，提供对模型能力的全面评估。

**MMBench**（Liu et al., 2024a）（多模态基准）是一个系统设计的客观基准，用于评估视觉语言模型的各项能力。它涵盖组织为三大类的20个能力维度：感知（如目标定位、OCR）、推理（如社会推理、物理常识）和知识（如名人识别、地标辨识）。

**MME**（Fu et al., 2025）（多模态评估）是一个综合评估基准，衡量感知和认知能力。它由14个子任务组成，包括存在性、计数、位置、颜色、海报、名人、场景、地标、艺术品、OCR、常识推理、数值计算、文本翻译和代码推理。我们将分数归一化到 $[0, 1]$ 以与其他基准保持一致。

**ChartQA**（Masry et al., 2022）专注于统计图表的问答。该数据集包含超过2,000道人工编写的问题，涵盖条形图、折线图和饼图，要求模型对图表图像进行视觉推理、数据提取和数值计算。

**OCRBench**（Liu et al., 2024b）是一个综合基准，用于评估视觉语言模型的光学字符识别和文本理解能力。它涵盖多样的文字识别场景，如场景文字、手写文字、文档文字和多语言文字，评估基础OCR准确率和基于文本的推理能力。

**TextVQA**（Singh et al., 2019）要求模型通过读取和推理图像中的文字来回答问题。该数据集包含来自 OpenImages 的1,000张图像，回答问题需要读取并理解场景文字，对于评估文本感知型视觉推理能力至关重要。
![[Pasted image 20260429141007.png]]

图4：对训练数据集中图像面积分布的直方图，展示了应用直方图均衡化以平衡面积多样性前后的对比。左图为原始分布，高度集中于较小图像面积；右图为均衡化后更加平衡的分布。

![[Pasted image 20260429141113.png]]

图5：来自 VLMEvalKit 基准的一个评测样例。该图展示了一个典型的图文问答对，呈现了模型如何处理视觉和文本输入以生成评测响应。


# C.1.3 剪枝配置采样

为确定作为 VisPCO 评估基准真值的经验帕累托前沿，我们采用基于综合采样的方法。该方法包含三个步骤：(1) 在搜索空间中系统采样大量剪枝配置；(2) 评估每个配置在多个基准上的性能并测量其 FLOPs 计算成本；(3) 从评估结果中提取帕累托最优配置。

**采样策略。** 我们的采样策略在层级操作，以捕捉细粒度剪枝模式。对于具有 $L$ 个 Transformer 层的视觉语言模型，我们对第1层到第 $L$ 层的每一层独立采样视觉令牌保留比例。具体地，第 $i$ 层的保留比例 $r _ { i }$ 从离散集合 $\{ 0 . 0 1 , 0 . 0 6 , 0 . 1 1 , \ldots , 0 . 9 6 , 0 . 9 9 \}$ 中采样，步长为0.05。这一粒度在配置空间的全面覆盖与计算可行性之间取得平衡。对于具有 $_ { \mathrm { L } = 3 6 }$ 层的 Qwen2.5-VL-3B 模型，该采样方案共生成700个涵盖不同计算预算的独立剪枝配置。

**评估协议。** 对每个采样配置，我们进行完整评估以获得其性能和计算成本。性能通过对八个评测基准的准确率取平均得到，提供对模型能力的综合评估。计算成本使用附录A中推导的 FLOPs 公式计算，涵盖所有层的注意力机制和前馈网络操作。

**帕累托前沿提取。** 给定已评估配置的集合 $\mathcal { C } = \{ ( p _ { i } , f _ { i } ) \} _ { i = 1 } ^ { N }$，其中 $p _ { i } \in [ 0 , 1 ]$ 为归一化平均性能（越高越好），$f _ { i }$ 为配置 $i$ 的 TFLOPs 计算成本（越低越好），我们利用帕累托支配准则识别帕累托前沿。形式上，若且仅若满足以下条件，称配置 $( p _ { i } , f _ { i } )$ 支配另一配置 $( p _ { j } , f _ { j } )$：

$$
p _ {i} \geq p _ {j} \quad \text {且} \quad f _ {i} \leq f _ {j}. \tag {57}
$$

帕累托前沿 $\mathcal { P }$ 由所有非被支配配置组成：

$$
\mathcal {P} = \left\{\left(p _ {i}, f _ {i}\right) \in \mathcal {C} \mid \not \exists (p _ {j}, f _ {j}) \in \mathcal {C} \right. \tag {58}
$$

使得 $( p _ { j } , f _ { j } )$ 支配 $( p _ { i } , f _ { i } ) \}$。

这些帕累托最优配置代表性能与计算效率之间可达的最优权衡，构成用于评估 VisPCO 预测的经验前沿。这一大规模采样与评估过程需要大量计算资源（700个配置约需48+GPU小时），凸显了 VisPCO 等高效优化方法的实际必要性。

# C.1.4 超参数设置

我们在表6和表7中提供了实验的详细超参数配置。主要超参数及其作用如下：

$\lambda$ 为增广拉格朗日方法中的惩罚参数，控制约束执行的强度。$\epsilon$ 为收敛阈值，决定优化何时终止。$\alpha$ 和 $\beta$ 分别为拉格朗日乘子和惩罚参数的更新系数，控制收敛动态。$\sigma$ 控制用于离散剪枝决策连续松弛的高斯核宽度，


表6：不同 VLM 主实验中各方法的超参数设置。


<table><tr><td>模型 + 方法</td><td>λ</td><td>α</td><td>ε</td><td>β</td><td>σ</td><td>T</td><td>lr</td><td>B</td></tr><tr><td>Qwen2.5-VL-3B + FastV</td><td>100</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>1e-4</td><td>16</td></tr><tr><td>Qwen2.5-VL-3B + SparseVLM</td><td>100</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>1e-4</td><td>16</td></tr><tr><td>Qwen2.5-VL-3B + FitPrune</td><td>100</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>1e-4</td><td>16</td></tr><tr><td>Gemma3-4B + FastV</td><td>1</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-4</td><td>16</td></tr><tr><td>Gemma3-4B + SparseVLM</td><td>1</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-4</td><td>16</td></tr><tr><td>Gemma3-4B + FitPrune</td><td>1</td><td>5</td><td>0.005</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-4</td><td>16</td></tr><tr><td>LLaVA-v1.5-7B + FastV</td><td>100</td><td>10</td><td>0.01</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-5</td><td>16</td></tr><tr><td>LLaVA-v1.5-7B + SparseVLM</td><td>100</td><td>10</td><td>0.01</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-5</td><td>16</td></tr><tr><td>LLaVA-v1.5-7B + FitPrune</td><td>100</td><td>10</td><td>0.01</td><td>0.5</td><td>10</td><td>0.1</td><td>5e-5</td><td>16</td></tr></table>


表7：不同剪枝调度策略消融实验的超参数设置。


<table><tr><td>模型 + 策略</td><td>λ</td><td>α</td><td>ε</td><td>β</td><td>σ</td><td>T</td><td>lr</td><td>B</td></tr><tr><td>Qwen2.5-VL-3B + Linear</td><td>100</td><td>5</td><td>0.01</td><td>0.5</td><td>1</td><td>0.1</td><td>1e-4</td><td>16</td></tr><tr><td>Qwen2.5-VL-3B + Exponential</td><td>100</td><td>10</td><td>0.005</td><td>0.5</td><td>1</td><td>0.1</td><td>5e-5</td><td>16</td></tr><tr><td>Qwen2.5-VL-3B + P-sigmoid</td><td>100</td><td>5</td><td>0.01</td><td>0.5</td><td>1</td><td>0.1</td><td>1e-4</td><td>16</td></tr><tr><td>Qwen2.5-VL-3B + Multi-step</td><td>1</td><td>10</td><td>0.05</td><td>0.5</td><td>5</td><td>0.1</td><td>1e-4</td><td>16</td></tr></table>

较大的值对应更平滑的近似。$T$ 为直通估计器的温度参数，在训练中平衡梯度流与离散化锐度。lr 为 AdamW 优化器的学习率，$B$ 为批大小（每次训练迭代的样本数量）。

# C.2 更多实验结果

本节提供更详细的实验结果。首先，表8给出了不同剪枝方法在各计算预算下应用 VisPCO 前后的结果。其次，表9展示了将 VisPCO 应用于不同基础 VLM 的结果。第三，表10报告了 VisPCO 在不同剪枝模式下的结果。

# C.3 预测剪枝配置案例研究

我们展示 VisPCO 在 Qwen2.5-VL-3B 上预测的剪枝配置案例研究，以揭示其在不同计算预算下的行为规律。图7展示了 VisPCO 在不同预算约束下预测的逐层剪枝曲线，以及不同层中对应的视觉令牌保留模式。这些可视化结果揭示了 VisPCO 如何根据不同资源约束自适应调整其剪枝策略。

可视化结果揭示了若干关键规律。第一，随着计算预算趋于紧张，VisPCO 采用越来越激进的剪枝策略，剪枝在网络中更早发生，并达到更低的保留比例。这证明了模型根据预算约束自适应分配计算资源的能力。第二，预测配置在各层间呈现平滑过渡，验证了我们连续松弛方法的有效性。

此外，图6对比了在 $50\%$ 计算预算下多层剪枝的不同核函数（线性、指数、P-Sigmoid、多步）。逐层剪枝比例揭示了各自独特的模式：线性核产生渐进过渡，指数核将剪枝集中在较深层，P-sigmoid核生成平滑的S形曲线，多步核产生渐进式离散过渡。这些多样化的模式使 VisPCO 能够探索不同的权衡策略。
![[Pasted image 20260429141227.png]]

图6：在 $50\%$ 计算预算下不同核函数的逐层剪枝比例对比。线性核在各层产生渐进过渡，指数核将剪枝集中在较深层，P-Sigmoid核生成平滑的S形曲线，多步核产生渐进式离散过渡。这些多样化的模式使得对不同剪枝策略的全面探索成为可能。



图7：VisPCO 在不同计算预算下预测的逐层剪枝配置。左图展示了预算从 $10\%$ 到 $90\%$ 时各层的保留比例曲线。右图（从上到下依次对应 $90\%$、$50\%$、$10\%$ 预算）可视化了选定层中实际视觉令牌的保留情况，展示了激进剪枝（较低预算）如何导致更早且更大规模的令牌移除。


表8：在不同计算预算下，各剪枝方法在八个视觉语言基准上使用 VisPCO 前后的详细对比。未使用 VisPCO 的结果为满足预算约束的多个随机采样配置的均值，括号内为标准差。


<table><tr><td>Method</td><td>AOKVQA</td><td>VizWiz</td><td>SEED</td><td>MMB</td><td>\(MME^†\)</td><td>ChartQA</td><td>OCRB</td><td>TextVQA</td><td>Avg (%)</td></tr><tr><td colspan="10">Upper Bound, 100% Budget, ~3.56 TFLOPs</td></tr><tr><td>Qwen2.5VL-3B</td><td>90.2</td><td>75.1</td><td>75.6</td><td>79.8</td><td>84.2</td><td>64.1</td><td>74.6</td><td>81.3</td><td>78.1</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 90%, ~3.20 TFLOPs</td></tr><tr><td>FastV</td><td>88.2 ± 0.4</td><td>72.9 ± 0.9</td><td>72.4 ± 0.9</td><td>76.4 ± 0.5</td><td>81.3 ± 0.5</td><td>62.2 ± 0.8</td><td>71.6 ± 0.7</td><td>79.1 ± 0.6</td><td>75.5 ± 0.7</td></tr><tr><td>+ VisPCO</td><td>88.4</td><td>73.8</td><td>73.2</td><td>76.9</td><td>81.7</td><td>62.9</td><td>72.3</td><td>79.5</td><td>76.1</td></tr><tr><td>SparseVLM</td><td>88.5 ± 0.3</td><td>73.1 ± 0.5</td><td>73.4 ± 0.4</td><td>76.9 ± 0.6</td><td>82.1 ± 0.3</td><td>62.2 ± 0.7</td><td>71.9 ± 0.6</td><td>79.5 ± 0.6</td><td>76.0 ± 0.5</td></tr><tr><td>+ VisPCO</td><td>88.6</td><td>73.5</td><td>73.8</td><td>77.5</td><td>82.4</td><td>62.9</td><td>72.5</td><td>80.0</td><td>76.4</td></tr><tr><td>FitPrune</td><td>89.1 ± 0.5</td><td>73.9 ± 0.4</td><td>74.2 ± 0.5</td><td>77.6 ± 0.4</td><td>82.5 ± 0.6</td><td>63.1 ± 0.6</td><td>72.5 ± 0.5</td><td>79.9 ± 0.3</td><td>76.2 ± 0.5</td></tr><tr><td>+ VisPCO</td><td>89.6</td><td>74.1</td><td>74.6</td><td>77.9</td><td>82.8</td><td>63.5</td><td>72.9</td><td>81.2</td><td>77.1</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 80%, ~2.84 TFLOPs</td></tr><tr><td>FastV</td><td>87.0 ± 1.6</td><td>71.6 ± 2.1</td><td>71.2 ± 2.1</td><td>75.1 ± 1.8</td><td>80.1 ± 1.6</td><td>61.0 ± 2.2</td><td>70.2 ± 1.9</td><td>77.7 ± 2.2</td><td>74.2 ± 1.9</td></tr><tr><td>+ VisPCO</td><td>88.3</td><td>73.7</td><td>73.1</td><td>76.9</td><td>81.6</td><td>62.8</td><td>72.0</td><td>79.4</td><td>75.7</td></tr><tr><td>SparseVLM</td><td>87.1 ± 1.8</td><td>71.7 ± 2.4</td><td>71.4 ± 2.1</td><td>75.3 ± 2.1</td><td>80.2 ± 2.2</td><td>61.6 ± 2.2</td><td>70.5 ± 2.1</td><td>78.3 ± 2.0</td><td>74.5 ± 2.1</td></tr><tr><td>+ VisPCO</td><td>88.4</td><td>73.9</td><td>73.5</td><td>77.4</td><td>82.1</td><td>62.6</td><td>72.4</td><td>79.6</td><td>76.2</td></tr><tr><td>FitPrune</td><td>87.3 ± 2.2</td><td>72.3 ± 2.1</td><td>72.5 ± 2.3</td><td>75.8 ± 2.2</td><td>80.6 ± 2.2</td><td>61.6 ± 2.5</td><td>70.6 ± 2.5</td><td>78.4 ± 1.6</td><td>74.9 ± 2.2</td></tr><tr><td>+ VisPCO</td><td>89.3</td><td>73.8</td><td>74.3</td><td>77.4</td><td>82.6</td><td>63.3</td><td>72.5</td><td>79.9</td><td>76.6</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 70%, ~2.50 TFLOPs</td></tr><tr><td>FastV</td><td>83.9 ± 4.5</td><td>69.2 ± 4.4</td><td>68.4 ± 4.8</td><td>73.1 ± 3.8</td><td>76.2 ± 4.5</td><td>59.3 ± 4.3</td><td>66.2 ± 4.9</td><td>75.5 ± 4.1</td><td>71.5 ± 4.4</td></tr><tr><td>+ VisPCO</td><td>88.0</td><td>73.4</td><td>72.8</td><td>76.7</td><td>80.9</td><td>62.4</td><td>71.1</td><td>78.9</td><td>75.5</td></tr><tr><td>SparseVLM</td><td>84.1 ± 4.7</td><td>69.4 ± 4.6</td><td>68.5 ± 5.1</td><td>73.3 ± 4.0</td><td>76.3 ± 4.5</td><td>59.5 ± 4.5</td><td>66.3 ± 5.0</td><td>75.7 ± 4.2</td><td>71.6 ± 4.6</td></tr><tr><td>+ VisPCO</td><td>88.1</td><td>73.6</td><td>72.9</td><td>76.7</td><td>81.0</td><td>62.5</td><td>71.1</td><td>79.1</td><td>75.6</td></tr><tr><td>FitPrune</td><td>84.2 ± 4.6</td><td>69.5 ± 4.7</td><td>68.6 ± 5.2</td><td>73.5 ± 4.1</td><td>76.6 ± 4.5</td><td>59.4 ± 4.7</td><td>66.5 ± 5.2</td><td>75.8 ± 4.3</td><td>71.8 ± 4.7</td></tr><tr><td>+ VisPCO</td><td>88.2</td><td>73.7</td><td>73.1</td><td>76.7</td><td>81.1</td><td>62.8</td><td>71.0</td><td>79.3</td><td>75.7</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 60%, ~2.14 TFLOPs</td></tr><tr><td>FastV</td><td>80.2 ± 5.9</td><td>68.4 ± 5.4</td><td>66.6 ± 6.8</td><td>69.4 ± 5.3</td><td>74.5 ± 5.5</td><td>56.4 ± 5.3</td><td>65.2 ± 5.1</td><td>71.7 ± 5.1</td><td>69.2 ± 5.4</td></tr><tr><td>+ VisPCO</td><td>86.0</td><td>71.4</td><td>70.8</td><td>74.7</td><td>78.9</td><td>60.4</td><td>69.1</td><td>76.9</td><td>73.5</td></tr><tr><td>SparseVLM</td><td>80.3 ± 6.2</td><td>68.5 ± 5.5</td><td>66.7 ± 7.1</td><td>69.6 ± 5.6</td><td>74.8 ± 5.8</td><td>56.4 ± 5.6</td><td>65.7 ± 5.5</td><td>71.8 ± 5.3</td><td>69.4 ± 5.7</td></tr><tr><td>+ VisPCO</td><td>86.2</td><td>71.6</td><td>70.9</td><td>74.9</td><td>79.2</td><td>60.6</td><td>69.3</td><td>77.1</td><td>73.7</td></tr><tr><td>FitPrune</td><td>80.4 ± 6.1</td><td>68.6 ± 5.4</td><td>66.8 ± 7.0</td><td>69.8 ± 5.8</td><td>74.9 ± 5.9</td><td>56.7 ± 5.9</td><td>65.9 ± 5.7</td><td>71.9 ± 5.3</td><td>69.5 ± 5.8</td></tr><tr><td>+ VisPCO</td><td>86.3</td><td>71.7</td><td>71.0</td><td>75.1</td><td>79.4</td><td>60.9</td><td>69.6</td><td>77.1</td><td>73.9</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 50%, ~1.78 TFLOPs</td></tr><tr><td>FastV</td><td>74.7 ± 10.1</td><td>60.3 ± 9.6</td><td>61.5 ± 8.1</td><td>62.4 ± 9.3</td><td>68.8 ± 9.1</td><td>51.6 ± 9.9</td><td>59.2 ± 9.1</td><td>65.9 ± 10.8</td><td>63.1 ± 9.5</td></tr><tr><td>+ VisPCO</td><td>84.8</td><td>69.4</td><td>67.6</td><td>71.2</td><td>77.1</td><td>58.1</td><td>67.8</td><td>75.9</td><td>71.5</td></tr><tr><td>SparseVLM</td><td>75.9 ± 9.8</td><td>62.6 ± 8.2</td><td>63.1 ± 7.2</td><td>63.9 ± 8.6</td><td>69.9 ± 8.2</td><td>51.9 ± 8.3</td><td>62.4 ± 6.9</td><td>66.6 ± 9.8</td><td>64.5 ± 8.4</td></tr><tr><td>+ VisPCO</td><td>85.2</td><td>69.0</td><td>68.1</td><td>71.9</td><td>77.6</td><td>58.4</td><td>67.9</td><td>76.3</td><td>71.8</td></tr><tr><td>FitPrune</td><td>77.1 ± 8.7</td><td>63.4 ± 7.7</td><td>63.9 ± 6.8</td><td>64.5 ± 8.2</td><td>70.8 ± 7.9</td><td>52.8 ± 8.1</td><td>63.3 ± 6.4</td><td>67.6 ± 9.4</td><td>65.4 ± 7.9</td></tr><tr><td>+ VisPCO</td><td>85.9</td><td>69.4</td><td>68.4</td><td>72.4</td><td>77.9</td><td>58.8</td><td>68.2</td><td>76.6</td><td>72.2</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 40%, ~1.42 TFLOPs</td></tr><tr><td>FastV</td><td>65.6 ± 12.2</td><td>50.3 ± 11.4</td><td>52.6 ± 10.2</td><td>53.3 ± 11.3</td><td>61.4 ± 11.2</td><td>42.5 ± 11.7</td><td>51.4 ± 11.2</td><td>57.5 ± 12.7</td><td>54.3 ± 11.5</td></tr><tr><td>+ VisPCO</td><td>77.6</td><td>61.7</td><td>62.6</td><td>64.4</td><td>72.6</td><td>54.1</td><td>62.5</td><td>69.9</td><td>65.7</td></tr><tr><td>SparseVLM</td><td>66.8 ± 12.6</td><td>50.9 ± 11.8</td><td>53.2 ± 10.9</td><td>53.5 ± 11.8</td><td>62.3 ± 11.3</td><td>42.9 ± 11.9</td><td>51.8 ± 11.3</td><td>58.1 ± 12.9</td><td>54.9 ± 11.8</td></tr><tr><td>+ VisPCO</td><td>77.9</td><td>61.9</td><td>62.7</td><td>64.9</td><td>72.9</td><td>54.4</td><td>62.6</td><td>70.2</td><td>65.9</td></tr><tr><td>FitPrune</td><td>66.9 ± 12.5</td><td>50.7 ± 12.2</td><td>53.0 ± 11.1</td><td>53.7 ± 11.9</td><td>62.7 ± 11.5</td><td>42.8 ± 12.2</td><td>51.7 ± 11.5</td><td>58.3 ± 12.6</td><td>55.0 ± 11.9</td></tr><tr><td>+ VisPCO</td><td>78.4</td><td>62.2</td><td>62.8</td><td>65.0</td><td>73.1</td><td>54.6</td><td>62.6</td><td>70.4</td><td>66.1</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 30%, ~1.06 TFLOPs</td></tr><tr><td>FastV</td><td>62.5 ± 8.1</td><td>46.4 ± 8.2</td><td>48.7 ± 7.6</td><td>49.4 ± 7.5</td><td>57.2 ± 8.9</td><td>38.8 ± 8.1</td><td>46.7 ± 8.3</td><td>53.6 ± 9.1</td><td>50.4 ± 8.2</td></tr><tr><td>+ VisPCO</td><td>70.2</td><td>54.6</td><td>55.5</td><td>54.5</td><td>65.7</td><td>46.4</td><td>54.4</td><td>62.8</td><td>58.0</td></tr><tr><td>SparseVLM</td><td>62.6 ± 8.3</td><td>46.5 ± 8.3</td><td>48.9 ± 7.7</td><td>49.5 ± 7.7</td><td>57.4 ± 9.0</td><td>38.9 ± 8.3</td><td>46.9 ± 8.5</td><td>53.8 ± 9.3</td><td>50.6 ± 8.4</td></tr><tr><td>+ VisPCO</td><td>70.5</td><td>54.8</td><td>55.9</td><td>54.8</td><td>65.8</td><td>46.9</td><td>55.1</td><td>63.4</td><td>58.4</td></tr><tr><td>FitPrune</td><td>62.7 ± 8.2</td><td>46.7 ± 8.2</td><td>49.0 ± 7.6</td><td>49.7 ± 7.9</td><td>57.3 ± 9.1</td><td>39.1 ± 8.1</td><td>46.7 ± 8.6</td><td>53.9 ± 9.4</td><td>50.6 ± 8.4</td></tr><tr><td>+ VisPCO</td><td>70.6</td><td>54.9</td><td>56.0</td><td>54.7</td><td>65.8</td><td>47.1</td><td>55.3</td><td>63.1</td><td>58.4</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 20%, ~0.72 TFLOPs</td></tr><tr><td>FastV</td><td>42.5 ± 4.1</td><td>39.4 ± 4.2</td><td>46.7 ± 3.6</td><td>39.4 ± 4.5</td><td>47.2 ± 4.9</td><td>34.8 ± 4.1</td><td>12.7 ± 5.3</td><td>43.6 ± 5.1</td><td>38.3 ± 4.5</td></tr><tr><td>+ VisPCO</td><td>46.6</td><td>43.1</td><td>50.1</td><td>43.9</td><td>51.2</td><td>38.9</td><td>17.8</td><td>48.5</td><td>42.5</td></tr><tr><td>SparseVLM</td><td>42.6 ± 4.2</td><td>39.5 ± 4.3</td><td>46.9 ± 3.7</td><td>39.6 ± 4.6</td><td>47.4 ± 5.0</td><td>34.9 ± 4.2</td><td>12.8 ± 5.4</td><td>43.7 ± 5.3</td><td>38.4 ± 4.6</td></tr><tr><td>+ VisPCO</td><td>46.7</td><td>43.2</td><td>50.2</td><td>43.9</td><td>51.3</td><td>39.1</td><td>17.9</td><td>48.6</td><td>42.6</td></tr><tr><td>FitPrune</td><td>42.5 ± 4.3</td><td>39.4 ± 4.4</td><td>46.7 ± 3.9</td><td>39.7 ± 4.7</td><td>47.6 ± 5.1</td><td>34.7 ± 4.1</td><td>12.6 ± 5.2</td><td>43.8 ± 5.4</td><td>38.3 ± 4.6</td></tr><tr><td>+ VisPCO</td><td>46.8</td><td>43.3</td><td>50.4</td><td>44.1</td><td>51.4</td><td>39.2</td><td>18.0</td><td>48.7</td><td>42.7</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 10%, ~0.36 TFLOPs</td></tr><tr><td>FastV</td><td>33.3 ± 2.3</td><td>30.4 ± 1.6</td><td>44.5 ± 2.7</td><td>33.0 ± 2.5</td><td>39.7 ± 1.4</td><td>29.8 ± 4.1</td><td>8.3 ± 2.1</td><td>33.7 ± 2.8</td><td>31.6 ± 2.4</td></tr><tr><td>+ VisPCO</td><td>35.5</td><td>31.7</td><td>46.9</td><td>35.5</td><td>40.1</td><td>33.2</td><td>10.1</td><td>36.1</td><td>33.6</td></tr><tr><td>SparseVLM</td><td>33.6 ± 2.1</td><td>31.2 ± 1.3</td><td>44.9 ± 2.5</td><td>33.9 ± 2.3</td><td>40.3 ± 1.1</td><td>30.5 ± 3.7</td><td>9.1 ± 2.0</td><td>34.4 ± 2.2</td><td>32.2 ± 2.2</td></tr><tr><td>+ VisPCO</td><td>35.5</td><td>31.5</td><td>47.1</td><td>35.8</td><td>40.4</td><td>33.3</td><td>10.2</td><td>36.3</td><td>33.8</td></tr><tr><td>FitPrune</td><td>33.8 ± 2.1</td><td>31.5 ± 1.1</td><td>45.3 ± 2.4</td><td>34.2 ± 2.2</td><td>40.6 ± 1.0</td><td>30.9 ± 3.5</td><td>9.6 ± 1.9</td><td>34.6 ± 2.1</td><td>32.6 ± 2.0</td></tr><tr><td>+ VisPCO</td><td>35.6</td><td>31.6</td><td>47.3</td><td>35.8</td><td>40.9</td><td>33.5</td><td>10.4</td><td>36.4</td><td>33.9</td></tr></table>


表9：VisPCO 应用于不同基础视觉语言模型在不同计算预算下八个基准上的性能。结果展示了 VisPCO 在不同模型架构和规模上的泛化能力与有效性。


<table><tr><td>Method</td><td>AOKVQA</td><td>VizWiz</td><td>SEED</td><td>MMB</td><td>MME†</td><td>ChartQA</td><td>OCRB</td><td>TextVQA</td><td>Avg (%)</td></tr><tr><td colspan="10">Upper Bound, 100% Budget</td></tr><tr><td>LLaVA-7B</td><td>72.3</td><td>93.1</td><td>52.1</td><td>48.2</td><td>50.4</td><td>42.3</td><td>64.3</td><td>88.2</td><td>63.9</td></tr><tr><td>Gemma3-4B</td><td>80.1</td><td>61.2</td><td>69.9</td><td>72.3</td><td>79.3</td><td>53.8</td><td>64.2</td><td>70.3</td><td>68.9</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 90%</td></tr><tr><td>LLaVA-7B</td><td>71.3 ± 0.5</td><td>91.5 ± 0.8</td><td>50.3 ± 0.6</td><td>47.3 ± 0.6</td><td>48.4 ± 0.8</td><td>40.9 ± 0.7</td><td>62.8 ± 0.7</td><td>86.2 ± 0.5</td><td>62.3 ± 0.7</td></tr><tr><td>+ VisPCO</td><td>71.8</td><td>92.3</td><td>50.8</td><td>47.7</td><td>49.2</td><td>41.5</td><td>63.5</td><td>86.5</td><td>62.9</td></tr><tr><td>Gemma3-4B</td><td>78.8 ± 0.5</td><td>59.8 ± 0.9</td><td>67.7 ± 0.7</td><td>70.5 ± 0.6</td><td>77.6 ± 0.9</td><td>51.8 ± 0.8</td><td>62.3 ± 0.7</td><td>58.8 ± 0.5</td><td>65.9 ± 0.7</td></tr><tr><td>+ VisPCO</td><td>79.3</td><td>60.5</td><td>68.4</td><td>71.1</td><td>78.5</td><td>52.5</td><td>62.9</td><td>59.2</td><td>66.6</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 80%</td></tr><tr><td>LLaVA-7B</td><td>70.1 ± 1.6</td><td>89.7 ± 1.7</td><td>48.7 ± 1.3</td><td>45.4 ± 1.3</td><td>46.2 ± 1.7</td><td>38.5 ± 1.6</td><td>60.6 ± 1.7</td><td>84.1 ± 1.6</td><td>60.2 ± 1.9</td></tr><tr><td>+ VisPCO</td><td>71.7</td><td>91.4</td><td>49.9</td><td>46.7</td><td>47.6</td><td>40.0</td><td>62.3</td><td>85.6</td><td>61.9</td></tr><tr><td>Gemma3-4B</td><td>76.6 ± 1.4</td><td>57.9 ± 1.7</td><td>65.6 ± 1.6</td><td>68.4 ± 1.6</td><td>75.3 ± 1.8</td><td>49.8 ± 1.8</td><td>60.2 ± 1.6</td><td>56.6 ± 1.5</td><td>63.8 ± 1.6</td></tr><tr><td>+ VisPCO</td><td>77.8</td><td>59.5</td><td>67.2</td><td>69.9</td><td>77.0</td><td>51.5</td><td>61.8</td><td>57.9</td><td>65.3</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 70%</td></tr><tr><td>LLaVA-7B</td><td>66.2 ± 4.7</td><td>85.7 ± 4.5</td><td>44.6 ± 4.3</td><td>41.3 ± 4.4</td><td>42.3 ± 4.6</td><td>34.6 ± 4.7</td><td>56.3 ± 4.5</td><td>80.3 ± 4.1</td><td>56.4 ± 4.5</td></tr><tr><td>+ VisPCO</td><td>70.7</td><td>89.3</td><td>48.9</td><td>45.7</td><td>46.6</td><td>39.1</td><td>60.7</td><td>84.4</td><td>60.7</td></tr><tr><td>Gemma3-4B</td><td>70.2 ± 5.1</td><td>51.7 ± 5.4</td><td>59.5 ± 5.5</td><td>62.3 ± 5.2</td><td>69.1 ± 4.7</td><td>43.7 ± 5.4</td><td>54.3 ± 5.4</td><td>50.6 ± 6.2</td><td>57.7 ± 5.4</td></tr><tr><td>+ VisPCO</td><td>75.3</td><td>56.9</td><td>65.1</td><td>67.5</td><td>73.5</td><td>58.9</td><td>59.7</td><td>56.7</td><td>64.2</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 60%</td></tr><tr><td>LLaVA-7B</td><td>62.1 ± 6.6</td><td>82.9 ± 6.2</td><td>41.7 ± 6.4</td><td>38.4 ± 5.6</td><td>38.1 ± 7.0</td><td>31.6 ± 6.5</td><td>53.5 ± 6.1</td><td>76.1 ± 6.3</td><td>53.1 ± 6.3</td></tr><tr><td>+ VisPCO</td><td>68.1</td><td>88.3</td><td>47.5</td><td>43.6</td><td>44.1</td><td>37.0</td><td>59.6</td><td>81.3</td><td>58.7</td></tr><tr><td>Gemma3-4B</td><td>54.1 ± 6.1</td><td>46.6 ± 6.3</td><td>54.4 ± 6.6</td><td>57.2 ± 7.3</td><td>65.1 ± 6.8</td><td>38.6 ± 6.2</td><td>49.4 ± 6.5</td><td>45.6 ± 7.2</td><td>53.1 ± 6.6</td></tr><tr><td>+ VisPCO</td><td>60.2</td><td>52.9</td><td>60.9</td><td>64.1</td><td>71.1</td><td>44.3</td><td>54.9</td><td>51.3</td><td>57.5</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 50%</td></tr><tr><td>LLaVA-7B</td><td>50.3 ± 10.7</td><td>70.8 ± 10.3</td><td>30.8 ± 11.3</td><td>26.5 ± 11.5</td><td>26.1 ± 11.4</td><td>21.3 ± 10.6</td><td>41.2 ± 11.6</td><td>64.2 ± 11.4</td><td>41.4 ± 11.1</td></tr><tr><td>+ VisPCO</td><td>60.8</td><td>80.5</td><td>42.1</td><td>38.0</td><td>37.4</td><td>31.6</td><td>52.8</td><td>75.2</td><td>52.3</td></tr><tr><td>Gemma3-4B</td><td>41.2 ± 12.4</td><td>33.5 ± 12.6</td><td>41.8 ± 11.5</td><td>44.7 ± 12.9</td><td>52.3 ± 11.8</td><td>25.7 ± 12.3</td><td>37.6 ± 12.7</td><td>33.6 ± 12.3</td><td>38.8 ± 12.3</td></tr><tr><td>+ VisPCO</td><td>53.4</td><td>45.9</td><td>53.1</td><td>57.3</td><td>62.8</td><td>37.9</td><td>50.3</td><td>45.5</td><td>50.8</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 40%</td></tr><tr><td>LLaVA-7B</td><td>48.4 ± 8.6</td><td>68.5 ± 8.4</td><td>29.7 ± 8.8</td><td>24.1 ± 9.4</td><td>24.2 ± 9.8</td><td>19.5 ± 8.6</td><td>39.3 ± 9.4</td><td>62.7 ± 9.9</td><td>39.6 ± 9.1</td></tr><tr><td>+ VisPCO</td><td>56.8</td><td>76.9</td><td>38.4</td><td>33.5</td><td>34.0</td><td>27.9</td><td>48.7</td><td>71.7</td><td>48.5</td></tr><tr><td>Gemma3-4B</td><td>39.4 ± 8.1</td><td>31.1 ± 9.4</td><td>39.8 ± 9.7</td><td>42.8 ± 8.9</td><td>50.4 ± 9.5</td><td>23.6 ± 8.4</td><td>35.8 ± 8.9</td><td>31.7 ± 8.3</td><td>36.8 ± 8.9</td></tr><tr><td>+ VisPCO</td><td>47.5</td><td>40.1</td><td>48.7</td><td>51.5</td><td>59.9</td><td>31.4</td><td>44.2</td><td>39.9</td><td>45.4</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 30%</td></tr><tr><td>LLaVA-7B</td><td>41.3 ± 6.7</td><td>61.6 ± 6.1</td><td>22.8 ± 6.5</td><td>17.2 ± 6.5</td><td>17.4 ± 6.9</td><td>14.6 ± 6.7</td><td>36.4 ± 6.3</td><td>60.8 ± 8.2</td><td>34.0 ± 6.7</td></tr><tr><td>+ VisPCO</td><td>47.8</td><td>67.7</td><td>29.2</td><td>23.6</td><td>24.2</td><td>20.9</td><td>42.5</td><td>68.9</td><td>40.6</td></tr><tr><td>Gemma3-4B</td><td>31.4 ± 6.1</td><td>24.1 ± 6.4</td><td>31.8 ± 6.7</td><td>36.8 ± 6.9</td><td>44.4 ± 6.5</td><td>20.6 ± 6.4</td><td>31.8 ± 6.9</td><td>25.7 ± 6.3</td><td>30.8 ± 6.5</td></tr><tr><td>+ VisPCO</td><td>37.5</td><td>30.1</td><td>38.5</td><td>43.5</td><td>50.2</td><td>27.0</td><td>38.3</td><td>31.9</td><td>37.1</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 20%</td></tr><tr><td>LLaVA-7B</td><td>40.8 ± 2.8</td><td>60.9 ± 2.2</td><td>21.7 ± 2.6</td><td>17.1 ± 2.9</td><td>16.9 ± 3.1</td><td>13.8 ± 2.5</td><td>34.6 ± 2.5</td><td>60.1 ± 2.3</td><td>37.2 ± 2.6</td></tr><tr><td>+ VisPCO</td><td>43.6</td><td>63.1</td><td>24.3</td><td>19.9</td><td>19.8</td><td>16.3</td><td>37.1</td><td>62.4</td><td>39.8</td></tr><tr><td>Gemma3-4B</td><td>31.2 ± 2.2</td><td>23.8 ± 2.4</td><td>31.9 ± 2.2</td><td>36.4 ± 2.5</td><td>44.6 ± 2.6</td><td>20.3 ± 2.6</td><td>31.7 ± 2.3</td><td>25.6 ± 2.3</td><td>30.7 ± 2.4</td></tr><tr><td>+ VisPCO</td><td>33.4</td><td>26.2</td><td>34.0</td><td>38.9</td><td>47.2</td><td>22.9</td><td>34.0</td><td>27.9</td><td>33.1</td></tr><tr><td colspan="10">Reduce FLOPs Budget to 10%</td></tr><tr><td>LLaVA-7B</td><td>40.1 ± 0.7</td><td>60.8 ± 0.8</td><td>21.4 ± 0.6</td><td>17.2 ± 0.4</td><td>16.7 ± 0.2</td><td>13.5 ± 0.3</td><td>34.3 ± 0.9</td><td>60.2 ± 0.5</td><td>36.8 ± 0.6</td></tr><tr><td>+ VisPCO</td><td>40.7</td><td>61.5</td><td>22.0</td><td>17.6</td><td>16.8</td><td>13.8</td><td>35.0</td><td>60.6</td><td>33.5</td></tr><tr><td>Gemma3-4B</td><td>30.1 ± 0.3</td><td>22.8 ± 0.5</td><td>27.4 ± 0.6</td><td>33.3 ± 0.8</td><td>41.5 ± 0.7</td><td>19.2 ± 0.6</td><td>30.6 ± 0.6</td><td>23.5 ± 0.4</td><td>28.6 ± 0.6</td></tr><tr><td>+ VisPCO</td><td>30.4</td><td>23.2</td><td>27.9</td><td>34.1</td><td>42.2</td><td>19.7</td><td>31.2</td><td>23.9</td><td>29.1</td></tr></table>


表10：VisPCO 在不同剪枝模式下跨视觉语言基准的性能对比。表格展示了单层剪枝及各多层剪枝策略（线性、指数、P-Sigmoid、多步）的详细结果。


<table><tr><td>Method</td><td>Kernels</td><td>AOKVQA</td><td>VizWiz</td><td>SEED</td><td>MMB</td><td>\(MME^†\)</td><td>ChartQA</td><td>OCRB</td><td>TextVQA</td><td>Avg (%)</td></tr><tr><td colspan="11">Upper Bound, 100% Budget, ~3.56 TFLOPs</td></tr><tr><td>Qwen2.5VL-3B</td><td>-</td><td>90.2</td><td>75.1</td><td>75.6</td><td>79.8</td><td>84.2</td><td>64.1</td><td>74.6</td><td>81.3</td><td>78.1</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 90%, ~3.20 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>88.4</td><td>73.8</td><td>73.2</td><td>76.9</td><td>81.7</td><td>62.9</td><td>72.3</td><td>79.5</td><td>76.1</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>88.2</td><td>73.1</td><td>72.5</td><td>76.4</td><td>81.3</td><td>61.8</td><td>72.0</td><td>79.1</td><td>75.6</td></tr><tr><td>Exponential</td><td>87.9</td><td>72.8</td><td>72.3</td><td>76.4</td><td>81.3</td><td>61.7</td><td>71.9</td><td>79.0</td><td>75.4</td></tr><tr><td>P-Sigmoid</td><td>87.5</td><td>72.3</td><td>72.1</td><td>76.1</td><td>81.0</td><td>61.3</td><td>71.8</td><td>78.6</td><td>75.1</td></tr><tr><td>Multi-Step</td><td>88.5</td><td>73.9</td><td>73.3</td><td>76.9</td><td>81.9</td><td>62.9</td><td>72.2</td><td>79.8</td><td>76.2</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 80%, ~2.84 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>88.3</td><td>73.7</td><td>73.1</td><td>76.9</td><td>81.6</td><td>62.8</td><td>72.0</td><td>79.4</td><td>75.7</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>87.7</td><td>73.2</td><td>72.6</td><td>76.3</td><td>81.4</td><td>62.4</td><td>71.3</td><td>78.8</td><td>75.5</td></tr><tr><td>Exponential</td><td>87.4</td><td>72.7</td><td>72.2</td><td>75.9</td><td>81.0</td><td>62.2</td><td>70.9</td><td>78.8</td><td>75.1</td></tr><tr><td>P-Sigmoid</td><td>87.1</td><td>72.3</td><td>71.8</td><td>74.9</td><td>80.2</td><td>61.3</td><td>69.9</td><td>77.7</td><td>74.4</td></tr><tr><td>Multi-Step</td><td>88.5</td><td>73.9</td><td>73.4</td><td>77.2</td><td>81.9</td><td>63.2</td><td>72.4</td><td>79.7</td><td>76.3</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 70%, ~2.50 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>88.0</td><td>73.4</td><td>72.8</td><td>76.7</td><td>80.9</td><td>62.4</td><td>71.1</td><td>78.9</td><td>75.5</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>87.4</td><td>72.8</td><td>72.2</td><td>76.3</td><td>80.5</td><td>62.1</td><td>70.8</td><td>78.3</td><td>75.1</td></tr><tr><td>Exponential</td><td>87.2</td><td>72.5</td><td>72.0</td><td>76.1</td><td>80.4</td><td>61.8</td><td>70.6</td><td>78.2</td><td>74.9</td></tr><tr><td>P-Sigmoid</td><td>86.8</td><td>72.3</td><td>71.8</td><td>75.7</td><td>80.1</td><td>61.6</td><td>70.4</td><td>78.0</td><td>74.6</td></tr><tr><td>Multi-Step</td><td>87.7</td><td>72.9</td><td>72.5</td><td>76.6</td><td>80.8</td><td>62.0</td><td>70.6</td><td>78.7</td><td>75.2</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 60%, ~2.14 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>86.0</td><td>71.4</td><td>70.8</td><td>74.7</td><td>78.9</td><td>60.4</td><td>69.1</td><td>76.9</td><td>73.5</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>84.5</td><td>69.7</td><td>68.9</td><td>72.7</td><td>77.1</td><td>58.1</td><td>68.1</td><td>74.4</td><td>71.7</td></tr><tr><td>Exponential</td><td>83.2</td><td>68.5</td><td>67.7</td><td>71.4</td><td>76.3</td><td>56.8</td><td>67.5</td><td>73.1</td><td>70.6</td></tr><tr><td>P-Sigmoid</td><td>82.9</td><td>68.1</td><td>66.5</td><td>70.1</td><td>75.3</td><td>56.2</td><td>66.5</td><td>72.4</td><td>69.8</td></tr><tr><td>Multi-Step</td><td>86.3</td><td>71.7</td><td>71.1</td><td>74.9</td><td>79.2</td><td>60.6</td><td>69.3</td><td>77.3</td><td>73.8</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 50%, ~1.78 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>84.8</td><td>69.4</td><td>67.6</td><td>71.2</td><td>77.1</td><td>58.1</td><td>67.8</td><td>75.9</td><td>71.5</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>82.6</td><td>67.7</td><td>65.4</td><td>70.9</td><td>76.2</td><td>57.5</td><td>66.5</td><td>74.9</td><td>70.2</td></tr><tr><td>Exponential</td><td>82.2</td><td>67.3</td><td>65.1</td><td>70.4</td><td>75.3</td><td>57.1</td><td>65.5</td><td>74.4</td><td>69.7</td></tr><tr><td>P-Sigmoid</td><td>81.9</td><td>67.0</td><td>64.8</td><td>69.6</td><td>74.7</td><td>56.8</td><td>65.2</td><td>74.1</td><td>69.3</td></tr><tr><td>Multi-Step</td><td>84.9</td><td>69.5</td><td>68.6</td><td>71.8</td><td>77.9</td><td>59.2</td><td>68.5</td><td>76.7</td><td>72.1</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 40%, ~1.42 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>77.6</td><td>61.7</td><td>62.6</td><td>64.4</td><td>72.6</td><td>54.1</td><td>62.5</td><td>69.9</td><td>65.7</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>76.2</td><td>59.4</td><td>60.3</td><td>62.6</td><td>70.3</td><td>52.7</td><td>61.3</td><td>67.7</td><td>63.8</td></tr><tr><td>Exponential</td><td>76.1</td><td>59.2</td><td>59.6</td><td>62.3</td><td>69.6</td><td>52.3</td><td>50.7</td><td>67.4</td><td>62.2</td></tr><tr><td>P-Sigmoid</td><td>73.3</td><td>56.3</td><td>56.7</td><td>60.5</td><td>66.9</td><td>49.1</td><td>47.5</td><td>65.7</td><td>59.5</td></tr><tr><td>Multi-Step</td><td>77.8</td><td>62.1</td><td>62.9</td><td>64.8</td><td>73.3</td><td>55.3</td><td>62.6</td><td>66.1</td><td>65.7</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 30%, ~1.06 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>70.2</td><td>54.6</td><td>55.5</td><td>54.5</td><td>65.7</td><td>46.4</td><td>54.4</td><td>62.8</td><td>58.0</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>65.5</td><td>49.6</td><td>50.8</td><td>49.3</td><td>61.3</td><td>41.8</td><td>49.4</td><td>57.8</td><td>53.2</td></tr><tr><td>Exponential</td><td>65.3</td><td>49.5</td><td>50.6</td><td>49.2</td><td>60.9</td><td>41.6</td><td>49.3</td><td>57.7</td><td>53.0</td></tr><tr><td>P-Sigmoid</td><td>60.9</td><td>45.3</td><td>47.2</td><td>44.7</td><td>55.9</td><td>38.3</td><td>45.9</td><td>52.4</td><td>48.8</td></tr><tr><td>Multi-Step</td><td>71.3</td><td>55.7</td><td>56.2</td><td>55.6</td><td>66.2</td><td>47.6</td><td>55.6</td><td>63.2</td><td>58.9</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 20%, ~0.72 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>46.6</td><td>43.1</td><td>50.1</td><td>43.9</td><td>51.2</td><td>38.9</td><td>17.8</td><td>48.5</td><td>42.5</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>44.8</td><td>41.7</td><td>48.2</td><td>41.4</td><td>48.7</td><td>36.4</td><td>15.3</td><td>46.4</td><td>40.4</td></tr><tr><td>Exponential</td><td>43.9</td><td>40.2</td><td>47.2</td><td>40.4</td><td>47.4</td><td>35.8</td><td>14.5</td><td>45.4</td><td>39.4</td></tr><tr><td>P-Sigmoid</td><td>42.1</td><td>39.2</td><td>46.1</td><td>39.7</td><td>46.6</td><td>34.3</td><td>13.8</td><td>44.5</td><td>38.3</td></tr><tr><td>Multi-Step</td><td>46.7</td><td>43.2</td><td>50.5</td><td>43.2</td><td>51.4</td><td>38.8</td><td>18.6</td><td>47.9</td><td>42.5</td></tr><tr><td colspan="11">Reduce FLOPs Budget to 10%, ~0.36 TFLOPs</td></tr><tr><td>Single-Layer</td><td>-</td><td>35.5</td><td>31.7</td><td>46.9</td><td>35.5</td><td>40.1</td><td>33.2</td><td>10.1</td><td>36.1</td><td>33.6</td></tr><tr><td rowspan="4">Multi-Layer</td><td>Linear</td><td>35.2</td><td>31.4</td><td>46.7</td><td>35.2</td><td>39.8</td><td>32.9</td><td>9.9</td><td>35.7</td><td>33.6</td></tr><tr><td>Exponential</td><td>34.4</td><td>30.9</td><td>46.4</td><td>34.9</td><td>39.1</td><td>31.9</td><td>9.3</td><td>35.2</td><td>32.8</td></tr><tr><td>P-Sigmoid</td><td>34.6</td><td>31.3</td><td>46.8</td><td>35.2</td><td>39.4</td><td>32.3</td><td>9.8</td><td>35.5</td><td>33.1</td></tr><tr><td>Multi-Step</td><td>35.4</td><td>31.1</td><td>46.3</td><td>35.3</td><td>39.7</td><td>32.8</td><td>10.0</td><td>35.8</td><td>33.3</td></tr></table>