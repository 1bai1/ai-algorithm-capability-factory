# AG News 四分类 × 特征哈希 + TF-IDF 加权 + 线性 SVM

## 能力说明

本卡记录「逐行规范化 → 词 1/2-gram 与词内字符 3/5-gram 各经 HashingVectorizer 哈希到 2^18 维 → 拼接 → TfidfTransformer(sublinear_tf=True) 补 idf → LinearSVC(C=0.5)」这条**不建词表**的浅层分类链路在 AG News 四分类采样集上的真实验证结果。分层随机 80/20、seed=42、测试 800 行下，测试准确率 **0.890**、宏 F1 **0.890**，高于同划分的 harness 自带参考基线（词表 TF-IDF(1,2)+LogReg(C=4)：0.875 / 0.874）。训练集准确率 0.990，训练/测试差 **+10.0 个百分点**。harness 分类检查 **19 项通过 / 0 未通过 / 0 跳过**，反泄漏与确定性检查均通过。价值在于：这是复用池第一条「哈希表示 + TF-IDF 加权 + 线性 SVM」的实证记录，与既有词表 TF-IDF 路线证据卡构成**同一 4000 行采样集上的哈希 vs 词表同口径对照**；同时记录了跨种子相对参考基线增益很小（约 +0.6 个百分点）这一证据边界。

## 输入契约

- 对象：AG News 四分类（Business / SciTech / Sports / World）。
- 数据：项目内 `examples/text_cls_demo/data/agnews_sample.csv`，4000 行、2 列（`text` / `label`），四类各 1000（均衡，多数类占比 0.250），平均文本长度约 80 词；**是 4000 行采样集，不是 AG News 全量（全量约 127k，量级锚点见《文本分类基准数据集与划分纪律》）**。
- 切分由 harness 控制，算法不自行切分：分层随机（stratified random）80/20、seed=42，训练 3200 行 / 测试 800 行。
- `build_features(df)` 只做逐行文本规范化：小写 → 去 HTML 标签 → 非字母数字换空格 → 压缩空白；不引用任何跨行统计量，因此全量运行与仅训练段运行的逐行输出一致。

## 输出契约

- `predict(model, test_df)` 返回长度等于测试集行数（800）的类别标签数组，覆盖 4 类；`test_df` 已剥掉 `label` 列。
- 评测输出：测试准确率、宏 F1、逐类 precision/recall/F1、混淆矩阵、多数类基线、harness 参考基线对照、训练/测试差距。
- 交付物（`generated/`）：`algorithm.py`（三契约函数）、`manifest.json`、可独立运行的 `run.py`、`README.md`；验证报告在 `validation/report.md` 与 `report.json`。

## 调用方式

- 独立运行入口：`python run.py --data <csv> --out <结果csv>`（读数据 → 训练 → 预测 → 写结果，不依赖 harness）。
- 本项目 harness 验收：
  `D:/environment/miniconda3/envs/math/python.exe -m harness validate knowledge/任务池/2026-10-03_agnews_四分类_哈希线性SVM/generated --data examples/text_cls_demo/data/agnews_sample.csv --out knowledge/任务池/2026-10-03_agnews_四分类_哈希线性SVM/validation/`
- 产物路径：`knowledge/任务池/2026-10-03_agnews_四分类_哈希线性SVM/generated/`；选型与稳健性实验脚本在该任务 `experiments/select_hashed.py` 与 `experiments/select_robust.py`。

## 关键参数

**特征**：词 `HashingVectorizer(analyzer='word', ngram_range=(1,2), n_features=2**18, alternate_sign=False, norm=None)` ∪ 词内字符 `HashingVectorizer(analyzer='char_wb', ngram_range=(3,5), n_features=2**18, alternate_sign=False, norm=None)`，两支拼接后经 `TfidfTransformer(sublinear_tf=True)` 补 idf；哈希器无状态（无需 fit），idf 只在训练段拟合。

**分类器**：`LinearSVC(C=0.5)`。

**真实数字（seed=42，测试 800 行）**：

- 本方法：准确率 0.890、宏 F1 0.890；训练集准确率 0.990，训练/测试差 **+10.0 个百分点**（harness 上限 15）。
- 逐类 F1：Sports 0.951、World 0.913、SciTech 0.854、Business 0.841；最大混淆为 Business↔SciTech（真实 Business→SciTech 23 例、真实 SciTech→Business 18 例）。
- 多数类基线：准确率 0.250、宏 F1 0.100。
- harness 参考基线（同划分、同文本列，词表 TF-IDF(1,2)+LogisticRegression(C=4)）：准确率 0.875、宏 F1 0.874。
- 单划分选型（`experiments/select_hashed.py`，测试准确率 / 宏 F1 / 训练减测试准确率）：
  - 参考基线 词表 TF-IDF(1,2)+LogReg(C=4)：0.8750 / 0.8745 / +0.1219
  - 哈希 2^18 word(1,2)+char(3,5) LinearSVC(C=1)：0.8888 / 0.8882 / +0.1100
  - 同特征 + LogReg(C=4)：0.8862 / 0.8858 / +0.1059
  - 同特征 + ComplementNB(alpha=0.3)：0.8775 / 0.8766 / +0.0775
  - 哈希 2^20 与 2^18 结果一致（均为 0.8888 / 0.8882），说明 2^18 维碰撞已可忽略
  - 纯词哈希 LinearSVC(C=1)：0.8825 / 0.8819（字符支整体约 +0.6 个百分点）
- 跨 5 个切分种子（42/7/2024/1/99，`experiments/select_robust.py`）宏 F1 均值±标准差 / 最低 / 训练−测试均值 / 最大：
  - 哈希 word+char LinearSVC(**C=0.5**，本卡配置)：0.8759 ± 0.0111 / 0.8585 / +0.1145 / +0.1338
  - 哈希 word+char LinearSVC(C=1)：0.8762 ± 0.0133 / 0.8525 / +0.1220 / +0.1462
  - 哈希 word+char LinearSVC(C=2)：0.8712 ± 0.0123 / 0.8524 / +0.1283 / +0.1475
  - 纯词哈希 LinearSVC(C=1)：0.8743 ± 0.0111 / 0.8568 / +0.1248 / +0.1425
  - 参考基线 词表 TF-IDF(1,2)+LogReg(C=4)：0.8700 ± 0.0101 / 0.8530 / +0.1261 / +0.1434
- C 的选择：C=0.5 与 C=1 宏 F1 基本持平，但 C=0.5 的最大训练/测试差距更小（+0.1338 vs +0.1462），故选 0.5。
- 反泄漏：全量与仅训练段两次 `build_features` 在训练集 3200 行逐行一致，最大偏差 0。
- 确定性：同种子两次运行 800 个预测完全一致。
- 运行开销：`build_features` 0.4s、`fit/predict` 1.2s，预算 300s。

## 依赖

- Python 3.9（项目 `D:\environment\miniconda3\envs\math\python.exe`）。
- `scikit-learn`：`HashingVectorizer`、`TfidfTransformer`、`LinearSVC`、`FeatureUnion`、`Pipeline`。
- `pandas`、`numpy`。
- 本项目 `harness` 模块仅用于验证，交付物不依赖它（`run.py` 自包含）。

## 适用条件

- 英文短文本、单标签、类别较均衡（本卡 4 类各 25%）、标注量在数千行量级。
- 无 GPU 或低算力预算，需要一条便宜、确定性强、内存可控的基线。
- 语料会持续出现未见词、或希望流式处理、或不愿维护词表与 idf 持久化时，哈希路线天然接受新词、加载快。
- 已在用词表 TF-IDF 路线、想在同口径下比较内存更省的哈希替代。

## 不适用条件

- 不可外推到 AG News 全量（127k）或其它语言/领域：本卡只跑一个 4000 行英文采样集。
- 未覆盖类别不平衡、多标签、长文档、领域术语（金融/医疗）场景。
- 追求相对参考基线**明显**提升时不适用：跨 5 个切分种子上，本方法宏 F1 0.8759 ± 0.0111 对参考基线 0.8700 ± 0.0101，增益约 +0.6 个百分点、很小；单划分 seed=42 上的 +1.5 个百分点有切分运气成分，且两组数字口径不同（单划分 vs 跨种子协议），只能各自内部比。
- 字符支与哈希维度**未做严格单项消融**：选型实验中「词 vs 词∪字符」「2^18 vs 2^20」是不同配置的整体对比，不是固定其余超参的单项消融，不能给出各自独立贡献。
- 哈希碰撞只是被 2^18 维压低、并未消除，理论上仍可能有两个词元落到同一桶。
- 需要语义理解、追求最优效果时应转向预训练模型微调，本卡只是浅层基线，不能作为上限。

## 验证状态

已验证（限度内）：有真实运行结果与 harness 报告（通过 19 / 未通过 0 / 跳过 0）。可信度评估：

- **有真实数字**：测试 800 行上准确率 0.890、宏 F1 0.890，逐类 F1 与混淆矩阵齐备。
- **有同划分强基线对照**：harness 参考基线（词表 TF-IDF(1,2)+LogReg(C=4)）0.875 / 0.874，多数类基线 0.250 / 0.100。
- **有反泄漏检查**：全量与仅训练段两次 `build_features` 在训练集 3200 行逐行一致，最大偏差 0；哈希器无状态、idf 只在训练段拟合。
- **有确定性检查**：同种子两次运行 800 个预测完全一致。
- **有跨种子复核**：5 个切分种子（42/7/2024/1/99）上，C=0.5 宏 F1 0.8759 ± 0.0111、最低 0.8585，训练/测试差均值 +0.1145、最大 +0.1338；相对参考基线增益小但方向一致。
- **有限度**：只跑一个 4000 行采样集、单标签、英文；未覆盖不平衡/多标签/长文档；字符支与哈希维度未做严格单项消融；跨种子复核用的切分种子与 harness 验收的 seed=42 并非同一批协议（前者为独立实验脚本），两组数字只能各自内部比。

## 来源

- `提炼池/线上博客/文本分类专题/scikit-learn_文本特征提取文档.md`：提供 `HashingVectorizer` 的无词表/流式路线与 `n_features`、`alternate_sign`、`norm` 参数口径，以及「哈希本身不提供 idf、要加权须另接 `TfidfTransformer`」与「向量化器只在训练段 fit、测试段只 transform」的反泄漏依据；本卡的哈希双支与 idf 拟合范围按此实现。
- `提炼池/线上博客/文本分类专题/文本分类综述_从浅层到深度学习.md`：提供「数据量小、算力受限时传统/浅层模型常优于深度模型」的选型画像，以及单标签评测用 Accuracy 与宏 F1 的口径；本卡据此选 `LinearSVC(C=0.5)` 并同口径报告准确率与宏 F1。
- `提炼池/线上博客/文本分类专题/文本分类是否真的进步了_对比综述.md`：提供基线纪律（必须对照调好的经典词袋线性基线）与「自跑实验换种子重复报均值与标准差」的可复现要求；本卡据此对照 harness 参考基线并跨 5 个切分种子复核。
- 本卡的运行记录派生于任务 `knowledge/任务池/2026-10-03_agnews_四分类_哈希线性SVM/`（`report.md`、`validation/report.json`、`generated/algorithm.py`、`experiments/select_hashed.py`、`experiments/select_robust.py`），数字均取自该任务的 harness 报告与实验脚本输出。
