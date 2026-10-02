"""模型选型实验（与 harness 同一切分口径：分层随机, test_size=0.2, seed=42）。

只用于挑配置，不是交付物。对比：
  A. 参考基线（harness 自带）：word TF-IDF(1,2) + LogReg(C=4)
  B. word TF-IDF(1,2) + char_wb TF-IDF(3,5) + LogReg(C=4)
  C. 同上 + LinearSVC(C=1)
  D. word TF-IDF(1,2) + char_wb(3,5) + ComplementNB
  E. 只用 word(1,2)+char_wb(3,5) + LogReg(C=8)
报告 训练/测试准确率、宏 F1，以及训练-测试差距（harness 上限 0.15）。
"""
from __future__ import annotations

import re
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

ROOT = Path(r"D:\awork\akf\llmagent\code\algorithm-coach")
DATA = ROOT / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipe(kind: str):
    word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                           min_df=2, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True, min_df=3, strip_accents="unicode")
    if kind == "A":
        return make_pipeline(word, LogisticRegression(max_iter=1000, C=4.0))
    if kind == "B":
        union = FeatureUnion([("w", word), ("c", char)])
        return make_pipeline(union, LogisticRegression(max_iter=2000, C=4.0))
    if kind == "C":
        union = FeatureUnion([("w", word), ("c", char)])
        return make_pipeline(union, LinearSVC(C=1.0, random_state=0))
    if kind == "D":
        union = FeatureUnion([("w", word), ("c", char)])
        return make_pipeline(union, ComplementNB(alpha=0.3))
    if kind == "E":
        union = FeatureUnion([("w", word), ("c", char)])
        return make_pipeline(union, LogisticRegression(max_iter=2000, C=8.0))
    raise ValueError(kind)


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["text_norm"] = df["text"].map(normalize)
    X = df["text_norm"].astype(str).to_numpy()
    y = df["label"].astype(str).to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2,
                              random_state=42, stratify=y)
    for kind in ("A", "B", "C", "D", "E"):
        pipe = build_pipe(kind)
        pipe.fit(X[tr], y[tr])
        p_tr = pipe.predict(X[tr])
        p_te = pipe.predict(X[te])
        acc_tr = accuracy_score(y[tr], p_tr)
        acc = accuracy_score(y[te], p_te)
        f1 = f1_score(y[te], p_te, average="macro", zero_division=0)
        print(f"{kind}: test_acc={acc:.4f} macroF1={f1:.4f} "
              f"train_acc={acc_tr:.4f} gap={acc_tr - acc:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
