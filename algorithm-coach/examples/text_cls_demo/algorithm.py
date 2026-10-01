"""AG News 四分类 —— 按 harness 契约整理的文本分类基线算法。

契约三条（见 `harness/contract.py`）：
- `build_features` 只做**逐行**的文本规范化，不学任何东西（不建词表、不算 IDF）
- `fit` 把"学"的动作全放在这里：TF-IDF 词表与 IDF 只在训练段上拟合，分类器同理
- `predict` 用 fit 时的向量化器变换测试文本，输出类别标签

依据卡片：复用池《TF-IDF 词项加权》《词袋与 N-gram 文本表示》
《文本分类评估与交叉验证》（特征提取必须随折重训，不能在全量语料上先 fit 词表）。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NORM_COLUMN = "text_norm"


def _normalize(s: object) -> str:
    """逐行文本规范化：小写、去掉标点与多余空白。不做任何跨行统计。"""
    text = str(s).lower()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()                      # 行数不变，标签列保留
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


class Model:
    """把训练段拟合出来的向量化器与分类器打包，predict 阶段只读不拟合。"""

    def __init__(self, pipeline):
        self.pipeline = pipeline


def fit(train_df: pd.DataFrame) -> Model:
    pipeline = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, C=4.0),
    )
    pipeline.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(pipeline)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.pipeline.predict(test_df[NORM_COLUMN].astype(str)))
