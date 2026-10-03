"""模型选型实验：在 harness 同口径（分层随机 8:2, seed=42）下比较若干浅层配置。

与交付算法解耦：直接用算法目录的 `algorithm.build_features` 做逐行规范化，
再在同一划分上训练 / 测试不同「特征 ∪ 分类器」组合，报告测试准确率、宏 F1
与训练-测试差距。目的有二：
1. 用本数据上的实测数字支撑 `generated/algorithm.py` 的选型；
2. 补一条**字符特征消融**（固定分类器，只切换词特征 / 词∪字符特征），
   区分字符 n-gram 的独立贡献——这是知识卡「未做严格消融」的缺口。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "generated"
sys.path.insert(0, str(GEN))
import algorithm  # noqa: E402

DATA = HERE.parents[3] / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
NORM = algorithm.NORM_COLUMN


def word_features(min_df=2):
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                           min_df=min_df, strip_accents="unicode")


def char_features():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True, min_df=3)


def union_features():
    return FeatureUnion([("word", word_features()), ("char", char_features())])


CONFIGS = {
    "A 词(1,2)+LogReg(C=4)（参考基线口径）": (word_features, LogisticRegression(max_iter=1000, C=4.0)),
    "B 词(1,2)∪字符(3,5)+LogReg(C=4)": (union_features, LogisticRegression(max_iter=1000, C=4.0)),
    "C 词(1,2)∪字符(3,5)+LinearSVC(C=1)": (union_features, LinearSVC(C=1.0)),
    "D 词(1,2)∪字符(3,5)+ComplementNB(α=0.3)": (union_features, ComplementNB(alpha=0.3)),
    "E 词(1,2)∪字符(3,5)+ComplementNB(α=0.1)": (union_features, ComplementNB(alpha=0.1)),
    "F 词(1,2)+ComplementNB(α=0.3)（字符消融）": (word_features, ComplementNB(alpha=0.3)),
}


def main() -> int:
    raw = pd.read_csv(DATA)
    feats = algorithm.build_features(raw)
    idx = np.arange(len(feats))
    y = feats["label"].astype(str)
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_train = feats.iloc[idx_train][NORM].astype(str)
    X_test = feats.iloc[idx_test][NORM].astype(str)
    y_train = y.iloc[idx_train]
    y_test = y.iloc[idx_test].astype(str)

    rows = []
    for name, (feat_factory, clf) in CONFIGS.items():
        pipe = make_pipeline(feat_factory(), clf)
        pipe.fit(X_train, y_train)
        te_pred = pipe.predict(X_test)
        tr_pred = pipe.predict(X_train)
        rows.append({
            "config": name,
            "accuracy": float(accuracy_score(y_test, te_pred)),
            "macro_f1": float(f1_score(y_test, te_pred, average="macro", zero_division=0)),
            "train_accuracy": float(accuracy_score(y_train, tr_pred)),
            "gap": float(accuracy_score(y_train, tr_pred) - accuracy_score(y_test, te_pred)),
        })
        print(f"{name:44s} acc={rows[-1]['accuracy']:.4f} "
              f"macroF1={rows[-1]['macro_f1']:.4f} gap={rows[-1]['gap']:+.4f}")

    (HERE / "select_model_results.json").write_text(
        json.dumps({"data": str(DATA), "rows": int(len(feats)), "seed": 42,
                    "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
