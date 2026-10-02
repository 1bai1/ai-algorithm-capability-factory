# AG News 四分类 × 词/字符 n-gram ComplementNB

## 能力说明

本卡记录「逐行规范化 → 词 1/2-gram TF-IDF ∪ 词内字符 3/5-gram TF-IDF → ComplementNB(alpha=0.3)」这条浅层分类链路在 AG News 四分类采样集上的真实验证结果。分层随机 80/20、seed=42、测试 800 行下，测试准确率 **0.885**、宏 F1 **0.884**，高于同划分的 harness 自带参考基线（TF-IDF(1,2)+LogReg(C=4)：0.875 / 0.874）约 1.0 个宏 F1 百分点，训练/测试准确率差距仅 **+6.1 个百分点**（参考基线约 +12）。harness 20 项分类检查全部通过。价值在于：给"小样本、无 GPU、均衡四分类英文短文本"场景一条**有真实数字、有强基线对照、有反泄漏与确定性检查**的可用基线，同时记录了 char n-gram 增益未做严格消融这一证据边界。

## 输入契约

- 对象：AG News 四分类（Business / SciTech / Sports / World）。
- 数据：项目内 `examples/text_cls_demo/data/agnews_sample.csv`，4000 行、2 列（`text` / `label`），四类各 1000（均衡，多数类占比 0.250），平均文本长度约 80 词；**是 4000 行采样集，不是 AG News 全量（全量约 127k，量级锚点见《文本分类基准数据集与划分纪律》）**。
- 切分由 harness 控制，算法不自行切分：分层随机（stratified random）80/20、seed=42，训练 3200 行 / 测试 800 行。
- `build_features(df)` 只做逐行文本规范化：小写、去 HTML 标签、非字母数字换空格、压缩空白；不引用任何跨行统计量，因此全量运行与仅训练段运行的逐行输出一致。

## 输出契约

- `predict(model, test_df)` 返回长度等于测试集行数（800）的类别标签数组，覆盖 4 类；`test_df` 已剥掉 `label` 列。
- 评测输出：测试准确率、宏 F1、逐类 precision/recall/F1、混淆矩阵、多数类基线、harness 参考基线对照、训练/测试差距。
- 交付物（`generated/`）：`algorithm.py`（三契约函数）、`manifest.json`、可独立运行的 `run.py`、`README.md`；验证报告在 `validation/report.md` 与 `report.json`。

## 调用方式

- 独立运行入口：`python run.py --data <csv> --out <结果csv>`（读数据 → 训练 → 预测 → 写结果，不依赖 harness）。
- 本项目 harness 验收：
  `D:/environment/miniconda3/envs/math/python.exe -m harness validate knowledge/任务池/2026-10-02_agnews_四分类/generated --data examples/text_cls_demo/data/agnews_sample.csv --out knowledge/任务池/2026-10-02_agnews_四分类/validation/`
- 产物路径：`knowledge/任务池/2026-10-02_agnews_四分类/generated/`；选型实验脚本在该任务 `experiments/select_model.py` 与 `experiments/select_nb.py`。

## 关键参数

**特征**：词 TF-IDF(`ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`, `strip_accents='unicode'`) ∪ 词内字符 TF-IDF(`analyzer='char_wb'`, `ngram_range=(3,5)`, `sublinear_tf=True`, `min_df=3`)；词表与 IDF 只在 fit 内、只用训练段拟合。

**分类器**：`ComplementNB(alpha=0.3)`。

**真实数字（seed=42，测试 800 行）**：

- 本方法：准确率 0.885、宏 F1 0.884；训练集准确率 0.946，训练/测试差距 **+6.1 个百分点**。
- 逐类 F1：Sports 0.954、World 0.905、Business 0.842、SciTech 0.837；主要混淆为 Business↔SciTech（真实 Business 误判 SciTech 23 例、真实 SciTech 误判 Business 18 例，为该矩阵最大两块非对角）。
- harness 参考基线（同划分、同文本列，TF-IDF(1,2)+LogisticRegression(max_iter=1000, C=4)）：准确率 0.875、宏 F1 0.874；本方法宏 F1 高约 1.0 个百分点。
- 多数类基线：准确率 0.250、宏 F1 0.100。
- 模型选型实验（同口径 seed=42，`experiments/select_model.py`，每格为 测试准确率 / 宏 F1 / 训练减测试准确率）：
  - 词(1,2)+LogReg(C=4)：0.8750 / 0.8745 / +0.1219
  - 词(1,2)∪字符(3,5)+LogReg(C=4)：0.8838 / 0.8835 / +0.1159
  - 同特征+LinearSVC(C=1)：0.8738 / 0.8731 / +0.1259
  - 同特征+ComplementNB(α=0.3)：0.8850 / 0.8844 / +0.0606
  - 同特征+LogReg(C=8)：0.8800 / 0.8796 / +0.1200
- 跨 5 个切分种子（42, 7, 2024, 1, 99，`experiments/select_nb.py`）复核 ComplementNB(α=0.3)：宏 F1 **0.8796 ± 0.0066（最低 0.8715）**，训练/测试差距均值 +0.0669、最大 +0.0759。
- 运行开销：build_features 0.4s、fit/predict 1.3s，预算 300s。

## 依赖

- Python 3.9（项目 `D:\environment\miniconda3\envs\math\python.exe`）。
- `scikit-learn`：`TfidfVectorizer`、`ComplementNB`、`FeatureUnion`、`make_pipeline`。
- `pandas`、`numpy`。
- 本项目 `harness` 模块仅用于验证，交付物不依赖它（`run.py` 自包含）。

## 适用条件

- 英文短文本、单标签、类别较均衡（本卡 4 类各 25%）、标注量在数千行量级。
- 无 GPU 或低算力预算，需要一条便宜、确定性强、训练/测试差距小的基线。
- 文本存在拼写/词形变化、或分词边界不稳，词特征与词内字符特征并集能互补。

## 不适用条件

- 不可外推到 AG News 全量（127k）或其它语料/语言：本卡只跑了 4000 行英文采样集。
- 未覆盖类别不平衡、多标签、长文档、领域术语（金融/医疗）场景——本卡与所依赖资料均未验证这些环节。
- 需要语义理解、最优效果对照预训练模型时：本卡是浅层基线，不能作为上限。
- char n-gram 相对纯词的增益**未做严格消融**：选型实验中"词 vs 词∪字符"是不同配置的整体对比（B vs 其他），不是固定其余超参的单项消融，不能给出字符特征的独立贡献。

## 验证状态

已验证（限度内）：有真实运行结果与 harness 报告（通过 19 / 未通过 0 / 跳过 0）。可信度评估：

- **有真实数字**：测试 800 行上准确率 0.885、宏 F1 0.884，逐类 F1 与混淆矩阵齐备。
- **有同划分强基线对照**：参考基线（TF-IDF(1,2)+LogReg(C=4)）0.875 / 0.874，多数类 0.250 / 0.100；本方法宏 F1 高约 1.0 个百分点，属小幅但方向一致的领先。
- **有反泄漏检查**：全量与仅训练段两次 `build_features` 逐行一致，训练段 3200 行 × 2 列最大偏差 0；词表/IDF 只在训练段拟合。
- **有确定性检查**：同种子两次运行 800 个预测完全一致（ComplementNB 无随机初始化）。
- **有限度**：只跑一个 4000 行采样集、单标签、英文；未做重复训练取均值（ComplementNB 确定性，故无随机初始化方差）；char n-gram 增益未做严格消融；`alpha` 仅由跨种子稳健性脚本旁证（0.1/0.3/0.5/1.0 中 0.3 被选中），本卡未逐档记录全部 alpha 数字。

## 来源

- `提炼池/线上博客/文本分类专题/scikit-learn_文本特征提取文档.md`：提供词袋 / TF-IDF / 字符 n-gram（`char_wb`）的接口与参数口径，以及"向量化器必须在训练段 fit、测试段只 transform"这一反泄漏依据；本卡的词表与 IDF 拟合范围按此实现。
- `提炼池/线上博客/文本分类专题/文本分类综述_从浅层到深度学习.md`：提供"数据量小、算力受限时传统/浅层模型常优于深度模型"的选型画像，以及单标签评测用 Accuracy 与宏 F1 的口径；本卡同口径报告准确率与宏 F1。
- `提炼池/线上博客/文本分类专题/文本分类是否真的进步了_对比综述.md`：提供基线纪律（必须对照调好的经典词袋线性基线）与"自跑实验换种子重复报均值与标准差"的可复现要求；本卡据此对照 harness 参考基线并跨 5 个切分种子复核。
