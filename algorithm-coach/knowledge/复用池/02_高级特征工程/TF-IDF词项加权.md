# TF-IDF 词项加权

## 能力说明
在词频上乘逆文档频率：压低在所有文档里都高频出现的词、抬高只在少数文档出现的词，把计数矩阵变成
更适合分类器（尤其线性模型）的浮点权重矩阵。常用于高维稀疏的文本分类与聚类。

## 输入契约
- 计数矩阵（文档 × 词项，非负），或原始文本（用 `TfidfVectorizer` 时直接给文本）。
- idf 由 fit 语料的文档频率决定，因此训练段与测试段必须分开：训练段 fit（或 fit_transform），
  测试段只 transform。

## 输出契约
- 稀疏浮点矩阵；每行默认按 L2 归一化（`norm='l2'`）。
- `idf_` 属性保存每个词项的 idf 值；`vocabulary_` 同词袋，可反查列名。

## 调用方式
```python
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

# 文本 → tf-idf（= CountVectorizer + TfidfTransformer 合并成一个类）
vec = TfidfVectorizer()
X_train = vec.fit_transform(train_texts)
X_test = vec.transform(test_texts)          # 同一个已 fit 的对象

# 已有计数矩阵时只做变换
tfidf = TfidfTransformer(smooth_idf=False).fit_transform(counts)
tfidf.idf_                                   # 每列一个 idf
```

## 关键参数
- `smooth_idf=True`（默认）：idf(t) = log((1+n)/(1+df(t))) + 1，防止 df=0 时除零。
- `smooth_idf=False`：idf(t) = log(n/df(t)) + 1（原文特别说明「1」加在分子而不是分母）。
- `norm='l2'`（默认）：行向量按欧氏范数归一化；`norm=None` 可关。
- `use_idf=True`、`sublinear_tf=False`（默认）、`binary`（TfidfVectorizer 侧只记出现与否）。

## 依赖
scikit-learn（`TfidfVectorizer` / `TfidfTransformer`）、scipy.sparse、numpy。

## 适用条件
- 词频分布长尾、需要压低常见虚词影响的语料（信息检索的传统用法，也用于文档分类与聚类）。
- 与线性模型 / SVM 配合的高维稀疏文本分类。
- 需要审计 idf 值判断词重要性的场合（`idf_` 可直接看）。

## 不适用条件
- 短文本：tf-idf 值噪声大，原文建议改用二值出现特征（`binary=True`）。
- BernoulliNB 这类显式建模布尔变量的分类器：配二值计数而不是 tf-idf。
- 跨语料/跨时间复用同一套权重：idf 依赖拟合语料的 df，语料构成一变权重就不可比。
- 原文未给出 tf-idf 与纯计数在分类指标上的对比实验，收益只能按任务自测。

## 验证状态
- 已验证（机制层面）：原文给出可手算复核的完整数值例——counts 6×3 在 `smooth_idf=False` 下得到
  `[[0.81940995, 0, 0.57320793], ...]`（term3 的原始 tf-idf ≈ 2.0986，L2 归一化后 0.573）；
  同一个元素在 `smooth_idf=True` 下变为 1.8473 / 0.5243；`idf_` 打印为 `[1., 2.25, 1.84]`。
- 未给出数字：tf-idf 相对纯计数的分类收益（原文仅定性说明高频词会 "shadow" 稀有词），
  也没有任何准确率、F1 或交叉验证结果。

## 来源
- `scikit-learn_文本特征提取文档.md`：tf-idf 定义、两套 idf 公式（smooth 与否）、L2 归一化过程与
  全部数值示例、`idf_` 属性、`TfidfVectorizer` 与 `TfidfTransformer` 的分工、短文本宜用 binary 的提示。
- `csdncopy.md`：TF 与 IDF 的直观含义（文档内重要性 vs 文档间区分度），以及 IDF 分母 +1 即拉普拉斯平滑。
