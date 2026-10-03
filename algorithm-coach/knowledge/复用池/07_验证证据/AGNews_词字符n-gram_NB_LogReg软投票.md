# AG News 四分类 × 词/字符 n-gram 软投票（ComplementNB ∪ LogisticRegression）

## 能力说明

本卡记录「逐行规范化 → 词 1/2-gram TF-IDF ∪ 词内字符 3/5-gram TF-IDF → ComplementNB(α=0.3) 与 LogisticRegression(C=8) 软投票集成」这条浅层分类链路在 AG News 四分类采样集上的真实验证结果。分层随机 80/20、seed=42、测试 800 行下，测试准确率 **0.890**、宏 F1 **0.889**，高于同划分的 harness 自带参考基线（TF-IDF(1,2)+LogReg(C=4)：0.875 / 0.874）约 1.5 个宏 F1 百分点，也略高于同体系单模型 ComplementNB（0.885 / 0.884）。训练集准确率 0.976，训练/测试差 **+8.6 个百分点**。harness 分类检查 19 项通过 / 0 未通过 / 0 跳过，反泄漏与确定性检查均通过。价值在于：给"小样本、无 GPU、均衡四分类英文短文本"场景一条**有真实数字、有跨切分种子复核、有强基线对照**的集成基线，同时把集成的真实增益量级与过拟合代价一起钉住——相对单模型宏 F1 仅约 +0.6 个百分点，而训练/测试差反而更大。

## 输入契约

- 对象：AG News 四分类（Business / SciTech / Sports / World）。
- 数据：项目内 `examples/text_cls_demo/data/agnews_sample.csv`，4000 行，四类各 1000（均衡，多数类占比 0.250），平均文本约 80 词；**是 4000 行采样集，不是 AG News 全量（全量约 127k，量级锚点见《文本分类基准数据集与划分纪律》）**。
- 切分由 harness 控制，算法不自行切分：分层随机 80/20、seed=42，训练 3200 行 / 测试 800 行。
- `build_features(df)` 只做逐行文本规范化：小写 → 去 HTML 标签与实体 → 非字母数字换空格 → 压缩空白；不引用任何跨行统计量，全量运行与仅训练段运行的逐行输出一致。

## 输出契约

- `predict(model, test_df)` 返回长度等于测试集行数（800）的类别标签数组，覆盖 4 类；`test_df` 已剥掉 `label` 列。
- 评测输出：测试准确率、宏 F1、逐类 F1、混淆矩阵、多数类基线、harness 参考基线对照、训练/测试差距。
- 交付物（`generated/`）：`algorithm.py`（三契约函数）、`manifest.json`、可独立运行的 `run.py`、`README.md`；验证报告在 `validation/report.md` 与 `report.json`。

## 调用方式

- 独立运行入口：`python run.py --data <csv> --out <结果csv>`（读数据 → 训练 → 预测 → 写结果，不依赖 harness）。
- 本项目 harness 验收：
  `D:/environment/miniconda3/envs/math/python.exe -m harness validate knowledge/任务池/2026-10-02_agnews_四分类_softvote/generated --data examples/text_cls_demo/data/agnews_sample.csv --out knowledge/任务池/2026-10-02_agnews_四分类_softvote/validation/`
- 产物路径：`knowledge/任务池/2026-10-02_agnews_四分类_softvote/generated/`；选型与稳健性实验脚本在该任务 `experiments/select_model.py` 与 `experiments/select_alpha.py`。

## 关键参数

**特征**：词 TF-IDF(`analyzer='word'`, `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `strip_accents='unicode'`) ∪ 词内字符 TF-IDF(`analyzer='char_wb'`, `ngram_range=(3,5)`, `sublinear_tf=True`, `min_df=3`)；词表与 IDF 只在训练段 fit。

**分类器**：`sklearn.ensemble.VotingClassifier(voting='soft', weights=[1,1])`，成员 `ComplementNB(alpha=0.3)` 与 `LogisticRegression(max_iter=1000, C=8)`。

**真实数字（seed=42，测试 800 行）**：

- 本方法：准确率 0.890、宏 F1 0.889；训练集准确率 0.976，训练/测试差 **+8.6 个百分点**。
- 逐类 F1：Sports 0.958、World 0.913、Business 0.844、SciTech 0.843。
- 混淆矩阵（行=真实，列=预测，序 Business/SciTech/Sports/World）：`[[165,23,4,8],[18,169,4,9],[1,3,195,1],[7,6,4,183]]`；最大混淆块 Business↔SciTech 双向共 41 例（真实 Business→SciTech 23、真实 SciTech→Business 18）。
- 多数类基线：准确率 0.250、宏 F1 0.100。
- harness 参考基线（同划分，TF-IDF(1,2)+LogReg(C=4)）：准确率 0.875、宏 F1 0.874。
- 单划分选型对照（`experiments/select_model.py`，测试准确率 / 宏 F1）：ComplementNB(α=0.1) 0.8900/0.8892、ComplementNB(α=0.3) 0.8850/0.8844、LogReg(C=8) 0.8800/0.8796、LinearSVC(C=1) 0.8738/0.8731、软投票 NB1:LR1 0.8900/0.8894、NB2:LR1 0.8912/0.8907、NB1:LR2 0.8900/0.8896。
- 跨 5 个切分种子（42/7/2024/1/99，`experiments/select_alpha.py`）复核（宏 F1 均值 ± 标准差 / 最低 / 训练−测试均值 / 最大）：
  - 软投票 NB1:LR1（本卡配置）：**0.8856 ± 0.0054 / 最低 0.8783 / +0.0895 / +0.0988**
  - 软投票 NB2:LR1：0.8855 ± 0.0080 / 0.8755 / +0.0780 / +0.0903
  - 单模型 ComplementNB(α=0.3)：0.8796 ± 0.0066 / 0.8715 / +0.0669 / +0.0759
  - 单模型 ComplementNB(α=0.1)：0.8798 ± 0.0090 / 0.8672 / +0.0734 / +0.0853
- 运行开销：fit/predict 3.3s、build_features 0.6s，预算 300s。

## 依赖

- Python 3.9（项目 `D:\environment\miniconda3\envs\math\python.exe`）。
- `scikit-learn`：`TfidfVectorizer`、`ComplementNB`、`LogisticRegression`、`VotingClassifier`、`FeatureUnion`、`make_pipeline`。
- `pandas`、`numpy`。
- 本项目 `harness` 模块仅用于验证，交付物不依赖它（`run.py` 自包含）。

## 适用条件

- 英文短文本、单标签、类别较均衡（本卡 4 类各 25%）、标注量在数千行量级。
- 无 GPU 或低算力预算，需要在一条便宜基线上再挤少量收益；训练成本预算宽裕（本卡约 3.3s）。
- 手头已有可用的浅层单模型基线（如 ComplementNB），想用软投票做小幅提升且能接受拟合差距变大。

## 不适用条件

- 不可外推到 AG News 全量（127k）或其它语言/领域：本卡只跑一个 4000 行英文采样集。
- 未覆盖类别不平衡、多标签、长文档场景。
- 追求显著提升时不适用：本卡集成相对单模型 ComplementNB 的宏 F1 增益仅约 +0.6 个百分点（跨种子 0.8856 vs 0.8796），过拟合差距反而更大（+8.95% vs +6.69%，仍在 harness 15% 上限内）。
- char 特征与集成权重未做严格单项消融，不能据此给出字符特征或权重比的独立贡献。

## 验证状态

已验证（限度内）：有真实运行结果与 harness 报告（通过 19 / 未通过 0 / 跳过 0）。可信度评估：

- **有真实数字**：测试 800 行上准确率 0.890、宏 F1 0.889，逐类 F1 与混淆矩阵齐备。
- **有同划分强基线对照**：harness 参考基线（TF-IDF(1,2)+LogReg(C=4)）0.875 / 0.874，多数类 0.250 / 0.100。
- **有反泄漏检查**：训练段 3200 行逐行一致，偏差 0；词表/IDF 只在训练段拟合。
- **有确定性检查**：同种子两次运行的 800 个预测完全一致。
- **有跨种子复核**：5 个切分种子上软投票 NB1:LR1 宏 F1 0.8856 ± 0.0054、最低 0.8783，比单模型更稳（标准差更小）但训练/测试差更大。
- **有限度**：只跑一个 4000 行采样集、单标签、英文；未覆盖不平衡/多标签/长文档；集成增益小且未做严格单项消融；跨种子复核用的切分种子与 harness 验收的 seed=42 并非同一批协议（前者是独立实验脚本），两组数字只能各自内部比。

## 来源

- `提炼池/线上博客/文本分类专题/scikit-learn_文本特征提取文档.md`：提供词袋 / TF-IDF / 字符 n-gram（`char_wb`）的接口与参数口径，以及"向量化器必须在训练段 fit、测试段只 transform"这一反泄漏依据；本卡的词表与 IDF 拟合范围按此实现，`VotingClassifier` 软投票接口亦来自 scikit-learn 文档。
- `提炼池/线上博客/文本分类专题/文本分类综述_从浅层到深度学习.md`：提供"数据量小、算力受限时传统/浅层模型常优于深度模型"的选型画像，以及单标签评测用 Accuracy 与宏 F1 的口径；本卡同口径报告准确率与宏 F1。
- `提炼池/线上博客/文本分类专题/文本分类是否真的进步了_对比综述.md`：提供基线纪律（必须对照调好的经典词袋线性基线）与"自跑实验换种子重复报均值与标准差"的可复现要求；本卡据此对照 harness 参考基线并跨 5 个切分种子复核集成配置。
