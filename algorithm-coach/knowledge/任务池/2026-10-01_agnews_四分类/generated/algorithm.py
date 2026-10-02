"""AG News 四分类：TF-IDF(词 1/2-gram) × 互补朴素贝叶斯 + 线性 SVM 硬投票集成。

本实现由知识库分层检索得到，未照抄 `examples/text_cls_demo` 标定样例
（标定样例等价于 harness 自带的 TF-IDF(1,2)+LogReg(C=4) 参考基线）。

检索路径（复用池 → 提炼池）
------------------------------------------------------------
- 《浅层文本分类：朴素贝叶斯与 SVM》`已验证`
  小数据集上浅层模型（NB/SVM）通常优于深度模型，且 NB 在少量标注数据上表现稳定。
  本任务 4000 行、4 类，正落在该卡「小数据集」区间，故选 NB 与 SVM 两个浅层族。
- 《TF-IDF 词项加权》`已验证`
  `sublinear_tf=True` 压缩高频词、`min_df=2` 裁词表、L2 归一化；训练段 fit、测试段 transform。
- 《词袋与 N-gram 文本表示》`已验证`
  用 `ngram_range=(1,1)` 与 `(1,2)` 两套词表制造特征多样性，供集成投票。
- 《文本分类评估与交叉验证》`待验证`
  向量化器必须随折在训练段 fit；据此 `build_features` 只做逐行规范化，词表/IDF 全放进 `fit`。
- 《基线未调优导致虚假提升》`有缺陷`（反例）
  不得拿未调参的基线当陪衬：harness 自带 TF-IDF(1,2)+LogReg(C=4) 强基线，
  本算法必须在同划分、同文本列上不劣于它，并披露自己的选型依据。
- 《停用词表与分词器不一致》`有缺陷`（反例）
  不做停用词过滤，避免词表词形与分词器输出口径错位污染 IDF。

选型依据（离线多 seed 扫参，详见任务报告 report.md）
------------------------------------------------------------
在 8 个分层随机划分（test_size=0.2）上比较 ComplementNB / LinearSVC / LogReg 及其组合，
硬投票 [CNB word(1,2), CNB word(1,1), LinearSVC word(1,2) C=0.5] 的平均宏 F1 最高（0.8825）、
最差 seed 也最好（0.8729）。三个成员都属上述浅层族，`fit` 秒级完成，远低于深度模型成本。

契约三条（见 `harness/contract.py`）
------------------------------------------------------------
- `build_features`：只做逐行文本规范化（小写 + 空白折叠 + 去首尾空白），不建词表、不算 IDF；
- `fit`：只在传入的训练段上 fit 向量化器与三个分类器，再装进硬投票集成；
- `predict`：只做 transform + vote，输出类别标签，长度与测试集行数一致。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEXT_COLUMN = "text"          # 原始文本列（manifest.text_column）
LABEL_COLUMN = "label"        # 标签列（manifest.label.column）
NORM_COLUMN = "text_norm"     # 逐行规范化后的文本列，供向量化器消费

_WHITESPACE = re.compile(r"\s+")


def _normalize(value: object) -> str:
    """逐行文本规范化：仅小写 + 折叠空白，不做任何跨行统计（反泄漏要求）。"""
    return _WHITESPACE.sub(" ", str(value).lower()).strip()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """行数、行序不变；标签列原样保留；只新增逐行变换得到的 `text_norm`。"""
    out = df.copy()
    out[NORM_COLUMN] = out[TEXT_COLUMN].map(_normalize)
    return out


def _tfidf(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """词级 TF-IDF：sublinear_tf 压缩高频词，min_df=2 裁掉只出现一次的词。"""
    return TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        ngram_range=ngram_range,
        strip_accents="unicode",
    )


def _members() -> list[tuple[str, Pipeline]]:
    """三个浅层成员：两种 n-gram 的 ComplementNB + 一种 LinearSVC。

    用两套 n-gram 词表（1,1）与（1,2）制造表示多样性，再配一个判别式 SVM，
    硬投票（多数票）合并；成员数取奇数，避免平票。
    """
    return [
        ("cnb_bigram", Pipeline([("tfidf", _tfidf((1, 2))),
                                 ("clf", ComplementNB(alpha=1.0))])),
        ("cnb_unigram", Pipeline([("tfidf", _tfidf((1, 1))),
                                  ("clf", ComplementNB(alpha=1.0))])),
        ("svc_bigram", Pipeline([("tfidf", _tfidf((1, 2))),
                                 ("clf", LinearSVC(C=0.5))])),
    ]


class Model:
    """把在训练段上拟合好的硬投票集成打包，预测阶段只读不拟合。"""

    def __init__(self, ensemble: VotingClassifier) -> None:
        self.ensemble = ensemble


def fit(train_df: pd.DataFrame) -> Model:
    ensemble = VotingClassifier(estimators=_members(), voting="hard")
    ensemble.fit(train_df[NORM_COLUMN].astype(str),
                 train_df[LABEL_COLUMN].astype(str))
    return Model(ensemble)


def predict(model: Model, test_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.ensemble.predict(test_df[NORM_COLUMN].astype(str)))
