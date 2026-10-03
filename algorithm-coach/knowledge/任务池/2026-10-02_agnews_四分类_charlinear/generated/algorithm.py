"""AG News 四分类 —— 词/词内字符 n-gram TF-IDF ∪ + ComplementNB。

harness 契约三条（见项目 `harness/contract.py`）：
- ``build_features``：只做**逐行**文本规范化，不学任何跨行统计量；
- ``fit``：把"学"的动作全部放在这里——词表、IDF、分类器都只用训练段拟合；
- ``predict``：复用 fit 出的向量化器做 transform，只读不拟合。

依据知识卡（复用池）：
- 《TF-IDF 词项加权》：idf 依赖拟合语料的 df，训练段 fit、测试段只 transform；
- 《词袋与 N-gram 文本表示》：词 1/2-gram ∪ 词内字符 3/5-gram（``char_wb``）互补；
- 《浅层文本分类：朴素贝叶斯与 SVM》：小样本无 GPU 时浅层族占优，NB 是廉价强基线；
- 《AG News 四分类 × 词/字符 n-gram ComplementNB》：本任务选型实验在
  seed=42 同划分下复核了该链路的成绩，故采用 ``ComplementNB(alpha=0.3)``。
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


def _normalize(value: object) -> str:
    """逐行规范化：小写 → 去 HTML 标签 → 非字母数字换空格 → 压缩空白。

    只依赖当前这一行，不引用任何跨行统计量，因此全量运行与仅训练段运行
    对同一行的输出完全一致（反泄漏检查的要求）。
    """
    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """输入原始表，输出同长度、同顺序的特征表（含 label 列）。"""
    out = df.copy()                       # 行数不变，标签列原样保留
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


def _build_pipeline():
    """词 TF-IDF ∪ 词内字符 TF-IDF，接 ComplementNB。"""
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            sublinear_tf=True, min_df=2, ngram_range=(1, 2),
            strip_accents="unicode")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=3)),
    ])
    return make_pipeline(features, ComplementNB(alpha=0.3))


class Model:
    """打包训练段拟合出的向量化器 + 分类器；predict 阶段只读不拟合。"""

    def __init__(self, pipeline):
        self.pipeline = pipeline


def fit(train_df: pd.DataFrame) -> Model:
    pipeline = _build_pipeline()
    pipeline.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(pipeline)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.pipeline.predict(test_df[NORM_COLUMN].astype(str)))
