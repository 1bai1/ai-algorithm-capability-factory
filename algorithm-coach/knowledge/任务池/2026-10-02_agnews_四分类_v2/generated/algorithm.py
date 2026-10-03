"""AG News 四分类 —— 词/字符 n-gram 稀疏特征 + ComplementNB。

独立实现（不复制示例），依据复用池已验证能力卡：
- 《TF-IDF 词项加权》`status: 已验证`：sublinear_tf / min_df 的加权口径。
- 《词袋与 N-gram 文本表示》`status: 已验证`：词 n-gram 与 `char_wb` 词内字符 n-gram
  各自的分词边界，并集互补。
- 《浅层文本分类：朴素贝叶斯与 SVM》`status: 已验证`：小样本、无 GPU 时浅层族占优。
- 《文本分类评估与交叉验证》`status: 待验证`：词表 / IDF 必须随训练段重训，
  不得在全量语料上先 fit 再切分（反泄漏）。
- 《AG News 四分类 × 词/字符 n-gram ComplementNB》`status: 已验证`：本数据上的
  真实验证结果（宏 F1 0.884），给出本条链路的参数与证据边界。

契约三条（`harness/contract.py`）：
- `build_features` 只做**逐行**文本规范化，不建词表、不算 IDF、不引用跨行统计量。
- `fit` 把"学"的动作全部放在训练段：两个 TF-IDF 的词表 / IDF 与分类器都只在 train 上拟合。
- `predict` 只用 fit 时学到的对象变换测试文本，输出类别标签。
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

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: object) -> str:
    """逐行文本规范化：小写 → 去 HTML 标签/实体 → 非字母数字换空格 → 压缩空白。

    全程只看当前这一行，不依赖任何跨行统计量，因此全量运行与仅训练段运行
    对同一行给出完全相同的输出（反泄漏检查 `feature_fit_scope` 直接抓这条）。
    """
    s = str(text).lower()
    s = _HTML_TAG.sub(" ", s)
    s = _HTML_ENTITY.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    return _WHITESPACE.sub(" ", s).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """行数不变的逐行变换：新增规范化文本列，保留原列与标签列。"""
    out = df.copy()
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


def _build_pipeline():
    """词 1/2-gram TF-IDF 并集词内字符 3/5-gram TF-IDF，接 ComplementNB。

    - 词特征：`ngram_range=(1, 2)`、`sublinear_tf=True`、`min_df=2`、
      `strip_accents='unicode'`，捕捉词形与二元搭配。
    - 字符特征：`analyzer='char_wb'`（词内字符，不跨词边界）、`ngram_range=(3, 5)`、
      `sublinear_tf=True`、`min_df=3`，对拼写 / 词形变化更鲁棒，与词特征互补。
    - 两个向量化器只在 fit 内、只用训练段拟合词表与 IDF。
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, strip_accents="unicode")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)),
    ])
    return make_pipeline(features, ComplementNB(alpha=0.3))


class Model:
    """把训练段拟合出的向量化器与分类器打包；predict 阶段只读、不再拟合。"""

    def __init__(self, pipeline):
        self.pipeline = pipeline


def fit(train_df: pd.DataFrame) -> Model:
    pipeline = _build_pipeline()
    pipeline.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(pipeline)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.pipeline.predict(test_df[NORM_COLUMN].astype(str)))
