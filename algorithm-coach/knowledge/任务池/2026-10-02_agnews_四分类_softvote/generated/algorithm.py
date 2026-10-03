"""AG News 四分类 —— 词/字符 n-gram TF-IDF 稀疏特征 + ComplementNB·LogReg 软投票集成。

独立实现（不复制示例，也不沿用已有任务的单模型 ComplementNB 路线）。依据复用池：

- 《TF-IDF 词项加权》`status: 已验证`：idf 依赖拟合语料的文档频率，训练段 fit、
  测试段只 transform；`sublinear_tf` 抑制高频词。
- 《词袋与 N-gram 文本表示》`status: 已验证`：词 1/2-gram 与词内字符 3/5-gram
  （`analyzer='char_wb'`，词边界补空格）互补，后者抗拼写 / 词形变化。
- 《浅层文本分类：朴素贝叶斯与 SVM》`status: 已验证`：小数据、无 GPU 时浅层族
  （NB / SVM / **集成方法**）占优；NB 参数少、对缺失不敏感，是廉价强基线。
- 《AG News 四分类 × 词/字符 n-gram ComplementNB》`status: 已验证`：本数据上
  ComplementNB(α=0.3) 的真实成绩（宏 F1 0.884），本条以其为集成成员之一。
- 《文本分类评估与交叉验证》`status: 待验证`：词表 / IDF 必须随训练段重训，
  不得在全量语料上先 fit 再切分。

路线差异：把概率型 ComplementaryNB 与判别型 LogisticRegression 用
`VotingClassifier(voting='soft')` 等权平均。两者在同一稀疏表示上误差互补，
本任务 `experiments/` 中跨 5 个切分种子的复核显示：集成宏 F1 0.8856 ± 0.0054
（最低 0.8783），优于单模型 ComplementNB 的 0.8796 ± 0.0066（最低 0.8715）。

契约三条（见 `harness/contract.py`）：
- `build_features` 只做**逐行**文本规范化，不建词表、不算 IDF、不引用跨行统计量；
- `fit` 把"学"的动作全部放在这里：两路 TF-IDF 的词表 / IDF 与两个分类器
  都只用训练段拟合；
- `predict` 只用 fit 时学到的对象变换测试文本，输出类别标签。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    """词 TF-IDF ∪ 词内字符 TF-IDF，接 ComplementNB + LogReg 等权软投票。

    - 词特征：`ngram_range=(1, 2)`、`sublinear_tf=True`、`min_df=2`、
      `strip_accents='unicode'`，捕捉关键词与二元搭配。
    - 字符特征：`analyzer='char_wb'`（词内字符、不跨词边界）、`ngram_range=(3, 5)`、
      `sublinear_tf=True`、`min_df=3`，对词形变化与拼写差异更鲁棒，与词特征互补。
    - 分类器：`ComplementNB(alpha=0.3)`（概率型、小数据稳）与
      `LogisticRegression(C=8)`（判别型、边界更细）软投票等权平均。
      两者都输出 `predict_proba`，`voting='soft'` 才能平均概率而非硬标签。
    - 两个向量化器只在 fit 内、只用训练段拟合词表与 IDF；集成成员共享同一份特征。
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, strip_accents="unicode")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)),
    ])
    voters = VotingClassifier(
        estimators=[
            ("nb", ComplementNB(alpha=0.3)),
            ("lr", LogisticRegression(max_iter=1000, C=8.0)),
        ],
        voting="soft", weights=[1, 1])
    return make_pipeline(features, voters)


class Model:
    """把训练段拟合出的向量化器与集成分类器打包；predict 阶段只读、不再拟合。"""

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
