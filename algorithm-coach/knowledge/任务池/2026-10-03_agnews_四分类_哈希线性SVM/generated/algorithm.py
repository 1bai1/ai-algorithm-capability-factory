"""AG News 四分类 —— 特征哈希 + TF-IDF 加权 + 线性 SVM。

与「词表 TF-IDF」路线的区别在于：本算法**不建词表**，而是用 scikit-learn 的
``HashingVectorizer`` 把词 1/2-gram 与词内字符 3/5-gram 直接哈希到固定维度的
稀疏空间，再套一层 ``TfidfTransformer`` 补上 idf 加权，最后用 ``LinearSVC``
分类。哈希向量化器无状态（不需要 fit），天然规避「词表在全量语料上先 fit」
这类反泄漏；代价是哈希碰撞，所以维度取得较大（每支 2^18）。

契约三条（见项目 ``harness/contract.py``）：
- ``build_features`` 只做**逐行**文本规范化，不学任何东西（不哈希、不算 idf）；
- ``fit`` 把「学」的动作全放这里：idf 统计与分类器都只在训练段上拟合；
- ``predict`` 只做变换与判决，不再拟合。

依据卡片（项目复用池）：
- 《特征哈希与HashingVectorizer》：哈希路线省内存、无需词表，但要另接
  ``TfidfTransformer`` 才能拿到 idf；本算法的字/词双支哈希即此路线。
- 《TF-IDF词项加权》：``sublinear_tf`` 抑制高频项、``min_df`` 去稀有项；
  词表/IDF 必须只在训练段拟合、测试段只 transform。
- 《浅层文本分类_朴素贝叶斯与SVM》：中小样本、无 GPU 时线性 SVM 是浅层强基线。
- 《文本分类评估与交叉验证》：特征提取要随折重训，不能在全量语料上先 fit。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NORM_COLUMN = "text_norm"

# 哈希空间大小：每支 2^18 = 262144 维。碰撞率统计实验显示 2^18 与 2^20 的
# 宏 F1 一致（见任务 experiments/select_hashed.py），故取更省的 2^18。
HASH_DIM = 2 ** 18
# 正则化强度：跨 5 个切分种子的稳健性实验（experiments/select_robust.py）中，
# C=0.5 的宏 F1 与 C=1 持平而训练/测试差距更小，故选 0.5。
SVM_C = 0.5


def _normalize(s: object) -> str:
    """逐行文本规范化：小写 → 去 HTML 标签 → 非字母数字换空格 → 压缩空白。

    只依赖当前这一行，不引用任何跨行统计量，因此全量运行与只喂训练段运行
    在训练行上的输出逐行一致（反泄漏检查据此通过）。
    """
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """契约函数一：加一列逐行规范化文本；行数、行序、原始文本列均不变。"""
    out = df.copy()
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


class Model:
    """打包「哈希 → TF-IDF → 线性 SVM」流水线；predict 阶段只读不拟合。"""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline


def fit(train_df: pd.DataFrame) -> Model:
    """契约函数二：只用训练段拟合 idf（哈希器无状态）与线性 SVM。"""
    pipeline = Pipeline([
        ("hash", FeatureUnion([
            ("word", HashingVectorizer(
                n_features=HASH_DIM, analyzer="word", ngram_range=(1, 2),
                alternate_sign=False, norm=None)),
            ("char", HashingVectorizer(
                n_features=HASH_DIM, analyzer="char_wb", ngram_range=(3, 5),
                alternate_sign=False, norm=None)),
        ])),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("clf", LinearSVC(C=SVM_C)),
    ])
    pipeline.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(pipeline)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    """契约函数三：输出长度等于 test_df 行数的类别标签数组。"""
    return np.asarray(model.pipeline.predict(test_df[NORM_COLUMN].astype(str)))
