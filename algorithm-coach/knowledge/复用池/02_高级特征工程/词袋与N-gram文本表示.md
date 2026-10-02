# 词袋与 N-gram 文本表示

## 能力说明
用词或字符 n-gram 的出现次数把变长文本压成定长的稀疏向量：每篇文档一行、每个词元一列，值是计数
（或二值）。它是文本分类最省算力的基线表示，也是与线性模型/朴素贝叶斯配合的默认路线；换字符 n-gram
后还能容忍拼写与词形变化。

## 输入契约
- 已分词的文档序列，或原始字符串配自己的 tokenizer/preprocessor。
- 语料要能放进内存以构建 `vocabulary_`；放不下时改用哈希路线。
- 训练段与测试段必须分开：词表只能来自训练段（fit 一次，后续只 transform）。

## 输出契约
- `scipy.sparse` 稀疏矩阵，行 = 文档、列 = 词元（或 n-gram），值为计数；`binary=True` 时值为 0/1。
- `vocabulary_`：词元 → 列号；`get_feature_names_out()`：列号 → 词元，可据此审计进模型的词。
- transform 阶段未见过的词元被整体忽略，新文档可能得到全零行。

## 调用方式
```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()                  # 默认：至少两个字母的词元 + 小写化
X_train = vec.fit_transform(train_texts) # 词表只从训练段来
X_test = vec.transform(test_texts)       # 测试段只做变换

# 词 + 2-gram：保留部分局部顺序信息
CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b')
# 抗拼写/词形变化：词内字符 n-gram（词边界补空格）
CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))

# 与分类器装进同一 pipeline 再交叉验证，避免全量语料 fit 词表
Pipeline([("vec", CountVectorizer()), ("clf", LinearSVC())])
```

## 关键参数
- `ngram_range`：默认 (1,1)；要局部顺序信息取 (1,2)，词表会显著变大。
- `analyzer`：`'word'`（默认）/ `'char'`（跨词字符 n-gram）/ `'char_wb'`（词内字符 n-gram，词边界补空格）。
- `token_pattern`：默认 `\b\w\w+\b`（丢单字符词元）；配 n-gram 又要保留单字时改 `r'\b\w+\b'`。
- `binary`：True 时只记出现与否——短文本与 BernoulliNB 口径。
- `min_df` / `max_df` / `max_features`：裁剪词表（原文示例用 `min_df=1`）。
- `dtype`：默认 int64。

## 依赖
scikit-learn（`CountVectorizer`）、scipy.sparse（存储）、numpy；在 pipeline 中通常与线性模型/朴素贝叶斯配合。

## 适用条件
- 需要极快、可解释、低算力的文本分类基线（稀疏特征 + 线性模型/NB）。
- 短文本、主题/情感这类词特征足够强的任务。
- 需要按列名审计「哪些词进了模型」的场合。
- 拼写错误与词形派生多的语料：改用 `char_wb` 字符 n-gram。

## 不适用条件
- 依赖词序或句法结构的任务：词袋丢掉词序，toy 语料里 "This is the first document." 与
  "Is this the first document?" 被编码成完全相同的向量。
- 拼写变体：'words' 与 'wprds' 在词袋下没有任何共同特征，分类器无从知道二者相关。
- 语料装不进内存、或需要严格在线学习时：词表（vocabulary_）是内存与并行瓶颈，应改用哈希向量化。
- 需要保留未登录词信息时：transform 对未见词直接忽略（全零行），没有回退机制。
- 原文未给出中文按字/按词切入词袋的任何实测，中文场景要自行验证。
- 本卡没有任何分类精度保证：原文只演示了表示的数值结果，未给准确率。

## 验证状态
- 已验证（机制层面）：原文给出可复现的 doctest 输出——4 句 toy 语料得到 4×9 稀疏矩阵（19 个非零）、
  词表 `['and','document','first','is','one','second','the','third','this']`；2-gram 下 "is this"
  只出现在最后一句（特征列取值为 [0,0,0,1]）；'words'/'wprds' 的 `char_wb` 2-gram 矩阵为 2×8
  且 4 列完全相同。
- 未给出数字：词袋/字符 n-gram 与分类精度的关系（原文只定性称 char_wb "can increase accuracy and
  convergence speed"），没有任何分类指标或消融。

## 来源
- `scikit-learn_文本特征提取文档.md`：词袋定义与稀疏性数据、`CountVectorizer` 接口与全部 doctest 输出、
  n-gram 与 `char_wb`/`char` 的数值示例、短文本宜用 `binary` 的建议、pipeline + 网格搜索的调参建议。
- `csdncopy.md`：词袋的直观定义（无序去重 + 计数向量）与三大缺点（维度大、稀疏、丢词序），
  以及「已基本淘汰」的定性判断（与 sklearn 口径不一致，见该提炼的局限与风险）。
