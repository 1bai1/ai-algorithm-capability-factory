"""AG News 四分类 —— 词 + 字符 n-gram 特征联合的 ComplementNB 分类器。

设计（每处选择都能指回知识库能力卡）
------------------------------------
文本规范化（逐行、无跨行统计）
    只做解码/大小写/标点与空白归一，供下游两个向量化器共用。依据
    复用池《文本预处理与分词》：预处理不得依赖全样本统计量。

特征表示：词 n-gram TF-IDF ∪ 词内字符 n-gram TF-IDF
    - 词 TF-IDF(1,2)：`sublinear_tf` + `min_df`，依据《TF-IDF 词项加权》
      与《词袋与 N-gram 文本表示》——线性/NB 类分类器配高维稀疏加权特征的默认路线。
    - 字符 char_wb(3,5)：依据《词袋与 N-gram 文本表示》——`char_wb` 对拼写、
      词形派生与未见词更稳健，且不依赖分词正确性；与词特征拼接取并集，
      让"整词匹配"与"词内片段匹配"互补。
    两个向量化器都只在 fit 内、只用训练段拟合词表与 IDF（反泄漏）。依据
    《文本分类评估与交叉验证》：特征提取必须随折重训，不能在全量语料上先 fit 词表。

分类器：ComplementNB(alpha=0.3)
    依据《浅层文本分类：朴素贝叶斯与 SVM》与《文本分类模型谱系与选择》：
    小数据、低算力场景浅层族占优；本任务 4000 行、4 类、无 GPU，
    属于典型的"稀疏特征 + 浅层分类器"。在 experiment/select_model.py 的同口径
    对比中，ComplementNB 相对 TF-IDF+逻辑回归参考基线在宏 F1 与训练/测试
    差距两项上都更优（见任务报告）。

契约三条（见 harness/contract.py）：
- build_features 只做逐行规范化，不学任何东西
- fit 里才建词表、算 IDF、训练分类器，且只吃训练段
- predict 用 fit 出的向量化器做 transform，只读不拟合
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NORM_COLUMN = "text_norm"

# 与 manifest.label.classes 对齐；仅用于文档，不参与逻辑
CLASSES = ["Business", "SciTech", "Sports", "World"]

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
_WS = re.compile(r"\s+")


def _normalize(text: object) -> str:
    """逐行文本规范化：大小写归一、去 HTML 标签、非字母数字换空格、压缩空白。

    纯逐行变换，不引用任何全局统计量，因此 cut/truncation 与全量运行结果一致。
    """
    s = str(text).lower()
    s = _HTML_TAG.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _vectorizer() -> FeatureUnion:
    """词 TF-IDF 与词内字符 TF-IDF 的并集。

    新建实例、无共享状态，保证 fit 之间互不污染（同一进程多次调用安全）。
    """
    word = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        dtype=np.float64,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
        dtype=np.float64,
    )
    return FeatureUnion([("word", word), ("char_wb", char)])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """契约入口：逐行规范化，返回同长度、同顺序、保留标签列的特征表。"""
    out = df.copy()
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


class Model:
    """把训练段拟合出的（向量化器 + 分类器）整体打包；predict 阶段只读。"""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, texts) -> np.ndarray:
        return np.asarray(self.pipeline.predict(texts))


def fit(train_df: pd.DataFrame) -> Model:
    """只在训练段上建词表、算 IDF 并训练 ComplementNB。"""
    pipeline = make_pipeline(_vectorizer(), ComplementNB(alpha=0.3))
    pipeline.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(pipeline)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    """对测试文本预测类别标签；test_df 已剥掉 label 列。"""
    return model.predict(test_df[NORM_COLUMN].astype(str))
