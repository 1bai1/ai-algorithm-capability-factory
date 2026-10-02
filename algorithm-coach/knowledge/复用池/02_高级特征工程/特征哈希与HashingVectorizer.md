# 特征哈希与 HashingVectorizer

## 能力说明
用有符号哈希函数把词元直接映射到固定维度的列号，跳过词表构建：内存占用与语料规模解耦、无状态、
可流式处理，适合大语料与 out-of-core 训练。代价是不可解释、不可逆、本身不带 idf。

## 输入契约
- `HashingVectorizer`：文本（自带 preprocessor/tokenizer/analyzer）。
- `FeatureHasher`：按 `input_type` 接受 dict、`(feature, value)` 对或字符串序列（字符串隐式值为 1）。
- 必须事先固定 `n_features`，训练与推理用同一个值，否则输入维度对不上。

## 输出契约
- 始终是 CSR 格式的 `scipy.sparse` 矩阵，列数 = `n_features`，与语料规模无关。
- 没有 `inverse_transform`，无法从列号还原词元（单向哈希）。
- `alternate_sign=True`（默认）时矩阵含负值；关闭后输出非负。

## 调用方式
```python
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

hv = HashingVectorizer(n_features=2**18)   # 无状态：不需要 fit
X = hv.transform(texts)

# 需要 idf 时在流水线里补一步（TfidfTransformer 会把状态带进来）
Pipeline([("hash", HashingVectorizer()),
          ("tfidf", TfidfTransformer()),
          ("clf", LinearSVC())])

# out-of-core：按 mini-batch transform 后喂 partial_fit 类估计器，内存上限 = 批次大小
```

## 关键参数
- `n_features`：默认 `2**20`；建议取 2 的幂（实现用简单取模定列号，否则列分布不均）；
  内存紧张可降到 `2**18`。
- `alternate_sign=True`（默认）：有符号哈希让碰撞正负抵消，小哈希表（< 10000）尤其有用。
- `alternate_sign=False`：输出非负，可接 `MultinomialNB`、`chi2` 特征选择这类要求非负输入的估计器。
- `input_type`（`FeatureHasher`）：`'dict'` / `'pair'` / `'string'`。
- `norm='l2'`、`binary=False`、`ngram_range`、`analyzer` 等与 `CountVectorizer` 同参。

## 依赖
scikit-learn（`HashingVectorizer` / `FeatureHasher`）、scipy.sparse、numpy；
哈希实现为 MurmurHash3 的有符号 32 位变体，最大特征数 2^31 − 1。

## 适用条件
- 语料大到词表放不下，或需要在线/流式增量学习（无状态，不必 fit）。
- 只关心模型效果、不需要按词解释特征。
- out-of-core：数据不必进内存，按 mini-batch 向量化后喂 `partial_fit` 类估计器。

## 不适用条件
- 需要特征可解释、要反查词元或输出特征重要性清单时（无 `inverse_transform`）。
- 需要 idf 时不能单用它，必须在流水线里另接 `TfidfTransformer`。
- 下游是 `MultinomialNB` / `chi2` 等要求非负输入的估计器时必须显式 `alternate_sign=False`，
  否则负值输入不合法。
- `n_features` 太小会碰撞：原文 toy 例中 `n_features=10` 时 19 个非零被压成 16 个。
- 原文未给出哈希向量化与词表向量化在同等任务上的精度/效率对比数字。

## 验证状态
- 已验证（机制层面）：原文给出可复现输出——同一 4 句 toy 语料在 `HashingVectorizer(n_features=10)`
  下得到 16 个非零（对照 `CountVectorizer` 的 19 个，说明发生了碰撞）；用默认 `n_features` 时为
  19 个非零、形状 (4, 1048576)；`FeatureHasher(input_type='string')` 对 (token, pos) 特征窗口的
  示例也给出了运行结果（1×6 的非零向量与特征名列表）。
- 未给出数字：与 `CountVectorizer`/`TfidfVectorizer` 在分类精度或训练耗时上的对比（原文只给了
  "FeatureHasher and DictVectorizer Comparison" 的示例入口，本页内没有数字）。

## 来源
- `scikit-learn_文本特征提取文档.md`：`FeatureHasher` 与 `HashingVectorizer` 的接口与参数、
  有符号哈希的动机、`n_features` 取 2 的幂与取值建议、碰撞非零数示例、无状态/不可逆/无 idf 的限制、
  out-of-core 策略、MurmurHash3 实现细节、与 `MultinomialNB`/`chi2` 的非负约束。
