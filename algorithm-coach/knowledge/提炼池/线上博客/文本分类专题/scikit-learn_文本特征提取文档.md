---
source: 原始池/线上博客/文本分类专题/scikit-learn_文本特征提取文档.md
column: 线上博客
title: scikit-learn 文本特征提取文档
distilled: 2026-10-01
relevance: 高
---

# scikit-learn 文本特征提取文档

## 筛选结论
保留 —— 本份是文本分类「特征提取」环节的官方实现文档，接口、参数与数值示例可直接对照写代码，
对应流水线中「文本 → 数值特征」这一步。原文后两节（图像 patch 抽取、图像连通图）与文本分类无关，
已在核心内容中略去。

## 核心内容
1. 文本不能直接进模型，必须先转成定长的数值向量；scikit-learn 给出三条路径——词表计数（词袋）、
   tf-idf 加权、特征哈希（无词表）。
2. 词袋把每个词元的出现次数当特征、每篇文档当一个样本，得到「文档 × 词」矩阵；真实语料中该矩阵
   99% 以上为零（原文举例：1 万篇短文本共用约 10 万词表，单篇只用到 100–1000 个词），必须用稀疏矩阵存。
3. 词袋丢失词序：同一份 toy 语料里 "This is the first document." 与 "Is this the first document?"
   被编码成完全相同的向量，只有加上 2-gram（"is this"）才区分得开。
4. 词表只由 fit 决定：transform 阶段没见过的词被整体忽略，新文档可能得到全零向量——所以向量化器
   必须在训练段 fit、测试段只 transform。
5. tf-idf = tf × idf；默认 `smooth_idf=True` 时 idf(t)=log((1+n)/(1+df(t)))+1，再按 L2 归一化。
   原文给出可手算复核的数值例与 `idf_` 属性（[1., 2.25, 1.84]）。
6. 哈希技巧用有符号哈希直接定列号：省内存、可流式、无状态（不必 fit），代价是不可逆（无
   `inverse_transform`）、不提供 idf；`n_features` 建议取 2 的幂，否则列分布不均。
7. 大语料上词表（`vocabulary_`）是瓶颈：内存随语料增长、fit 要全量过一遍、pickle 慢、难并行；
   改用哈希向量化 + mini-batch 可做 out-of-core 训练。
8. 字符 n-gram 抗拼写与词形变化：'words' 与 'wprds' 在词袋下无共同特征，在 `char_wb` 的 2-gram 下
   8 个特征里有 4 个相同；`char_wb` 只切词内字符（词边界补空格），比跨词的 `char` 噪声小。
9. 停用词表必须与向量化器用同一套预处理与分词：默认 tokenizer 把 "we've" 拆成 we 与 ve，
   若停用词表只有 "we've" 而没有 "ve"，"ve" 会残留成特征。
10. 特征提取参数的正确调法是「向量化器 + 分类器」装进同一个 pipeline，再做交叉验证网格搜索。

## 可复用要点
- 基线表示：`CountVectorizer`（计数）→ 需要权重换 `TfidfVectorizer`；已有计数矩阵时用
  `TfidfTransformer` 单独做变换。
- 抗拼写：`analyzer='char_wb'` + `ngram_range=(2,2)`；要兼顾局部顺序：`ngram_range=(1,2)`。
- 短文本：tf-idf 值噪声大，改用 `binary=True` 的出现/不出现特征（也是 BernoulliNB 的口径）。
- 大语料/流式：`HashingVectorizer(n_features=2**20)`（内存紧张可 2**18，务必取 2 的幂）；
  需要 idf 时在 pipeline 里追加 `TfidfTransformer`。
- 停用词：可传 `'english'` 或自定义表，但必须与 tokenizer 同源；风格/人格类任务不应去停用词
  （原文指出停用词在文体分类中是有信息的）。
- 维度与算法：稀疏矩阵下维度不影响 CSR 类算法（`LinearSVC(dual=True)`、`Perceptron`、
  `SGDClassifier`）的 CPU 训练时间，但影响 CSC 类算法（`LinearSVC(dual=False)`、`Lasso`）。
- 自定义预处理：传 `preprocessor` / `tokenizer` / `analyzer` 可调用对象，或继承并覆写
  `build_preprocessor` / `build_tokenizer` / `build_analyzer`；外部分词结果用空白连接后传 `analyzer=str.split`。
- 调参：`Pipeline([("vec", TfidfVectorizer()), ("clf", ...)])` + 交叉验证网格搜索。

## 关键实现
- 模块：`sklearn.feature_extraction.text`（`CountVectorizer`、`TfidfTransformer`、`TfidfVectorizer`、
  `HashingVectorizer`）、`sklearn.feature_extraction`（`DictVectorizer`、`FeatureHasher`）。
- 关键参数：`ngram_range`、`analyzer`（'word'/'char'/'char_wb'）、`token_pattern`（默认 `\b\w\w+\b`，
  即至少两个字母）、`min_df`/`max_df`/`max_features`、`binary`、`stop_words`、`smooth_idf`、`norm`、
  `n_features`、`alternate_sign`、`input_type`、`encoding`/`decode_error`。
- 产物属性：`vocabulary_`（词元→列号）、`get_feature_names_out()`、`TfidfTransformer.idf_`；
  输出为 `scipy.sparse` 矩阵（`FeatureHasher` 固定 CSR）。
- 实现细节：`FeatureHasher` 用 MurmurHash3 有符号 32 位变体，最大特征数 2^31−1。

## 数据与假设
- 文档内示例数据：4 句 toy 语料、6×3 计数矩阵、'words'/'wprds' 两文档、'jumpy fox' 字符 n-gram；
  真实语料只提到 20 Newsgroups（编码混杂，建议退回 latin-1）。原文未给任何真实语料的规模、时间范围与类别分布。
- 前提假设：文本能正确解码为字符串（否则 `UnicodeDecodeError`）；`char_wb` 只适合用空白分词的语言；
  词袋/哈希都不建模语序。

## 局限与风险
- 全文没有分类任务的真实准确率：所有输出都是 API 数值示例；「char_wb 可提高准确率与收敛速度」
  「词袋配合线性模型效果不错」均为定性表述，没有指标支撑。
- 词袋与 n-gram 都会破坏文档内部结构，承担不了需要句子/段落结构的任务（原文明确说这类 structured
  output 超出 scikit-learn 范围）。
- 停用词表是语言相关的通用清单，原文自承有已知问题，且可能删掉对某些任务高信息量的词（点名 "computer"），
  不能当 one-size-fits-all。
- 哈希在 `n_features` 偏小时碰撞明显（原文 toy 例：19 个非零被压成 16 个），且不可反查特征名，解释性差。
- tf-idf 的 idf 由拟合语料的 df 决定，语料构成一变权重就变，跨语料不可比。
- 原文未涉及类别不平衡、多标签与判定阈值——本份资料覆盖不到这些环节。
- 覆盖范围提示：可复用的部分集中在 8.2.1（DictVectorizer）、8.2.2（Feature hashing）与 8.2.3
  （词袋/tf-idf/字符 n-gram/解码）；8.2.4 图像特征提取与本主题无关。
