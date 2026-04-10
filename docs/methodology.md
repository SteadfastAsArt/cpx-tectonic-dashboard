# 单斜辉石构造背景分类：方法论

本文档系统阐述本项目所用的机器学习方法：从问题形式化、多分类 XGBoost 的损失构造，到类别权重、交叉验证、评价指标选择、缺失值处理与 SHAP 解释的数学机制。读者对象为具备扎实 ML/统计背景的研究人员，因此我们尽量给出「为什么这样做」而非仅停留在「这样做」。

## 1. 问题定义

本项目的任务是依据单斜辉石（clinopyroxene, Cpx）的化学成分推断其母岩浆的**构造背景**，属于一个 11 类的多分类（multi-class classification）问题。类别涵盖从 MORB、IAB 到大陆裂谷、板内火山、洋底溢流玄武岩等地球动力学环境。样本总数约 17 万，但类别分布极度不平衡：最大类 **INTRAPLATE VOLCANICS 有 84,835 个样本**，最小类 **OCEAN-BASIN FLOOD BASALT 仅 207 个**，最大最小比达到 **409.8:1**。这种长尾结构直接决定了后续的权重、评价指标与解释策略。

特征空间为 50 维，由三部分组成：11 个主量氧化物（如 $\text{SiO}_2, \text{Al}_2\text{O}_3, \text{K}_2\text{O}$ 等）、32 个微量元素（REE、HFSE、大离子亲石元素与部分过渡金属）以及 7 个端员组分（Wo、En、Fs、Jd 等计算量）。一个关键的数据结构性事实是：**约 81% 的样本整行缺失全部 32 个微量元素**——这是因为许多经典文献只做电子探针主量分析而不做 LA-ICP-MS 微量分析。缺失并非随机（MCAR），而是由分析仪器与文献时代系统性产生的 MNAR（Missing Not At Random）结构。这一点决定了我们既不能简单丢弃含缺失的样本，也不能用朴素插补掩盖掉「缺失本身是信号」这一信息。

## 2. 多分类 XGBoost

我们选择 XGBoost 作为核心模型，在 softmax 目标下进行多分类。对 $K=11$ 个类别，模型在每轮 boosting 中为每一类**同时**拟合一棵回归树，第 $t$ 轮后样本 $i$ 在第 $k$ 类上的累计 logit 记为 $z_{k,i}^{(t)}$，预测概率为

$$p_{k,i} = \frac{\exp(z_{k,i})}{\sum_{j=1}^{K}\exp(z_{j,i})}.$$

损失为多类交叉熵（`mlogloss`）：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{k,i}\log p_{k,i},$$

其中 $y_{k,i}\in\{0,1\}$ 为 one-hot 标签。XGBoost 对该损失做二阶泰勒展开，得到每个叶节点的最优增量。参考代码 `10_cv_confusion.py:151`，我们设定 `eval_metric="mlogloss"`，超参数取 `max_depth=12, learning_rate=0.08, n_estimators=400`，即较深树 + 中等学习率 + 四百轮，较大的深度是为了捕捉主量-微量之间的高阶非线性交互。

之所以不采用 One-vs-Rest（OvR）或 One-vs-One（OvO），是因为这两者在每个子问题上独立训练，会导致类间决策边界**不协调**：一个样本可能同时被两个二分类器认领，或同时被全部拒绝。softmax 则在概率单纯形（simplex）上联合优化，各类共享同一组树结构并通过 $\sum_k p_k=1$ 的约束相互制衡，这在极度不平衡场景下能显著稳定小类的边界。

## 3. 类别权重处理

为了抵消不平衡损失被大类主导的效应，我们在样本级别赋权。权重公式为

$$w_i = \frac{N}{K \cdot n_{y_i}},$$

其中 $N$ 为总样本数，$K=11$，$n_{y_i}$ 为样本 $i$ 所在类的样本数。可以验证 $\sum_i w_i = N$，即权重是「类别平权」的归一化：所有类在加权损失中贡献相同的总质量，无论其样本数多寡。代码实现见 `10_cv_confusion.py:139-142`。

与重采样（over/under-sampling）的关键差别在于：**重采样改变了经验分布**，从而改变了模型看到的「似然」；而权重法保留原始分布、仅重新分配损失贡献，更符合最大似然框架下的加权 ERM（Empirical Risk Minimization）。我们也明确不使用 SMOTE 或类似合成样本方法：地球化学数据的 50 维流形由复杂的相平衡与分配系数约束决定，任意两个同类样本的线性插值很可能落在物理上不可能的成分组合上（例如违反电荷守恒或固溶体限制），反而引入噪声。

## 4. 5 折分层交叉验证

模型验证采用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，分层保证每一折内的类别比例与整体一致，避免小类在某一折被整窝抽走。我们不使用一次性的 train/test split，原因是在 409.8:1 的不平衡下，单次划分的小类方差极大——比如 207 个样本的最小类若按 80/20 划分，测试集只有 41 个样本，单个误分类就会使 recall 抖动 2.4 个百分点。五折交叉不仅提供 F1-macro 的均值，还天然给出 $\pm\text{std}$ 作为稳定性估计。

一个容易被忽视的细节是：全局混淆矩阵的构造方式。我们采用 **OOF（out-of-fold）累积**——每个样本 $i$ 恰好在一个 fold 中作为验证集出现，由该 fold 的模型给出其预测 $\hat{y}_i$。把所有 OOF 预测拼起来得到长度为 $N$ 的预测向量，再与真实标签一次性构造混淆矩阵。这**不是** 5 个 fold 各自混淆矩阵的平均：前者是一个严格的概率矩阵（每行和为该类样本数），后者在数值上等价但在小类上方差更大，不便于 F1 的稳健估计。实现见 `10_cv_confusion.py:118-190`。

## 5. F1-macro 作为主指标

我们以 **F1-macro** 为主评价指标。其定义为每类 F1 的算术平均：

$$F_1^{\text{macro}} = \frac{1}{K}\sum_{k=1}^{K} F_1^{(k)}, \quad F_1^{(k)} = \frac{2\,\text{Prec}^{(k)}\,\text{Rec}^{(k)}}{\text{Prec}^{(k)} + \text{Rec}^{(k)}}.$$

注意这里对类别是**等权**求和，而 F1-weighted 则按类别样本数 $n_k$ 加权：$F_1^{\text{w}} = \sum_k (n_k/N) F_1^{(k)}$。在本项目 409.8:1 的不平衡下，weighted 会被最大类（INTRAPLATE VOLCANICS 占 50% 以上）几乎完全主导，任何小类上的失败都被稀释到看不见。

从几何视角理解：F1-macro 可视为小类与大类在一个共同 simplex 上的「最差类约束」——任何一个类的 F1 掉下来都会线性拉低整体。本项目最终结果为 $F_1^{\text{macro}}=0.9090\pm 0.005$，而 $F_1^{\text{weighted}}=0.9526$，两者相差约 4.4 个百分点，这个 gap 正是不平衡对小类性能折扣的定量体现，也是我们选择前者作为主指标的直接依据。

## 6. 原生 NaN 处理

XGBoost 的一个被低估的特性是：在每次寻找 split 的时候，它会**学习缺失值的默认方向**。具体地，对候选分裂 $(x_j\le s)$，算法会把 NaN 样本分别尝试划入左子与右子，取使 gain 最大化的方向作为该节点的「缺失默认路径」，并把这条路径存进模型。这意味着缺失值本身成为了一个可学习的信号——即便两个样本的 $x_j$ 均未观测，模型依然能把它们推向对任务最有用的分支。

这与中位数插补形成鲜明对比：中位数会把「未测量」与「恰好等于总体中位数」两种完全不同的状态混同，抹掉了「未被测量」这条信息。而在地球化学中，缺失绝非 MCAR：一条样本若缺失全部 32 个微量，它几乎必然来自仅做电子探针的老文献；而这类样本又非均匀地落在某些构造背景（如经典 ophiolite 报道）上。中位数插补把这种相关性熨平了。我们的对照实验证实了这一点：排除「整行微量缺失」行 + 保留原生 NaN 的方案给出 $F_1^{\text{macro}}=0.9125$，而对同一数据做中位数插补只得到 $0.732$，差距接近 18 个百分点。实现细节见 `08_train_no_full_trace_missing.py`。

## 7. 多分类 SHAP 分析

SHAP（SHapley Additive exPlanations）把每个样本的预测分解为各特征的边际贡献，满足唯一性公理（local accuracy, missingness, consistency）。对于 $K$ 类问题，`TreeExplainer` 返回 $K$ 个形状为 $(n, p)$ 的数组，我们把它堆叠成 3D 张量 $\text{SHAP}\in\mathbb{R}^{K\times n\times p}$，其中 $\text{SHAP}_{k,i,j}$ 是特征 $j$ 对样本 $i$ 在第 $k$ 类 logit 上的贡献。

### Softmax 平移不变性与零和

一个对多分类 SHAP 解释至关重要但常被忽略的性质是：**对于任意样本 $i$ 和特征 $j$，所有类的 SHAP 值之和为零**：

$$\sum_{k=1}^{K}\text{SHAP}_{k,i,j} = 0.$$

原因是 softmax 对 logit 加一个常数 $c$ 不变：$\text{softmax}(z_k+c)=\text{softmax}(z_k)$，因此「绝对贡献」这一概念在 softmax 输出层不存在，只有「相对贡献」才有意义。任何把特征 $j$ 的总贡献推高 $\Delta$ 的类，必然伴随其他类总共减少 $\Delta$。在解释 SHAP 结果时必须牢记这一点：一个特征「把样本推向某类」总是相对其他类而言。

### 三种聚合方式

基于张量 $\text{SHAP}_{k,i,j}$，我们采用三种聚合。

**全局重要性**：$\text{Imp}_j^{\text{global}}=\text{mean}_{k,i}\,|\text{SHAP}_{k,i,j}|$，衡量特征 $j$ 在所有样本、所有类上的平均贡献幅度——与「影响谁」无关，只看「是否有影响」。

**每类重要性**：$\text{Imp}_{k,j}^{\text{class}}=\text{mean}_i\,\text{SHAP}_{k,i,j}$，带符号，反映特征 $j$ 系统性地把样本推向（正值）或推离（负值）第 $k$ 类。

**beeswarm 图**：在单类视角下，把每个样本的 $(\text{SHAP}_{k,i,j}, x_{i,j})$ 散点展示，可同时观察值的分布与贡献的分布，便于识别非线性阈值（如某元素超过某浓度才开始起作用）。实现见 `09_shap_analysis.py:145-211`。

## 8. Gain 与 SHAP 的差异

一个有趣的现象是：XGBoost 的内置 **gain** 排序与 SHAP 排序给出显著不同的特征重要性榜单。Gain 前列是 NiO、Ferrosilite、Zn，而 SHAP 前列是 Sr、Sm、Zr、Sc。这并非某一方错误，而是两个量衡量的是**不同对象**。

Gain 是**训练时**的量：每次 split 时该特征带来的损失下降，按特征累加。如果某特征在 boosting 早期就被用来做根节点分裂，即便后续层用其他特征做了更精细的修正，它也已获得巨大的 gain。SHAP 是**推理时**的量：基于 Shapley 值公理分解每个样本的**最终预测**，只反映决定性的边际贡献。这两者的差异可以用一个直觉类比：gain 像「谁打开了房门」，SHAP 像「谁在屋里最终拿到了奖」。

这也解释了 $\text{K}_2\text{O}$ 的「反常」：在单变量统计判别力上（见 `viz_data.json` 的 discrimination 指标，CV $=1.753$）它最强，但在 SHAP 全局榜上并不突出。原因是它的信息被其他相关特征「吸收」了——$\text{Na}_2\text{O}$、Rb、Ba、Cs 等碱金属与大离子亲石元素与 $\text{K}_2\text{O}$ 高度相关，在分裂时 XGBoost 挑了这些相关特征之一即可。这在 SHAP 语境下叫做 **correlated feature attribution dilution**，提醒我们单一的特征重要性数字从来不是因果强度的可靠代理，而只是模型在特定相关性结构下做出的会计分配。

## 参考文献与扩展阅读

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.

Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*, 2(1), 56–67.

He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

Krawczyk, B. (2016). Learning from Imbalanced Data: Open Challenges and Future Directions. *Progress in Artificial Intelligence*, 5(4), 221–232.

Sokolova, M., & Lapalme, G. (2009). A Systematic Analysis of Performance Measures for Classification Tasks. *Information Processing & Management*, 45(4), 427–437.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Shapley, L. S. (1953). A Value for n-Person Games. *Contributions to the Theory of Games*, 2(28), 307–317.
