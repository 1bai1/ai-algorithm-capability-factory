"""AG News 四分类 —— 模型/特征选型实验。

口径严格对齐 harness：分层随机 80/20、seed=42、文本规范化与 build_features 一致，
只在训练段拟合词表/IDF（反泄漏）。候选模型全部来自复用池已核对的浅层族。

运行:
    D:/environment/miniconda3/envs/math/python.exe experiments/select_model.py
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

DATA = Path(__file__).resolve().parents[4] / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
SEED = 42


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_feats():
    return TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                           strip_accents="unicode")


def union_feats():
    return FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                 strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True, min_df=3)),
    ])


def main() -> None:
    df = pd.read_csv(DATA)
    df["text_norm"] = df["text"].map(normalize)
    idx_tr, idx_te = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=SEED,
        stratify=df["label"].astype(str).to_numpy())
    tr, te = df.loc[idx_tr], df.loc[idx_te]
    y_tr = tr["label"].astype(str)
    y_te = te["label"].astype(str)

    candidates = {
        "word(1,2)+LogReg(C=4)  [harness 参考基线口径]":
            make_pipeline(word_feats(), LogisticRegression(max_iter=1000, C=4.0)),
        "word+char_union+LogReg(C=8)":
            make_pipeline(union_feats(), LogisticRegression(max_iter=1000, C=8.0)),
        "word+char_union+LinearSVC(C=1)":
            make_pipeline(union_feats(), LinearSVC(C=1.0)),
        "word+char_union+ComplementNB(0.3)":
            make_pipeline(union_feats(), ComplementNB(alpha=0.3)),
    }

    print(f"{'配置':<44}{'测试acc':>9}{'宏F1':>9}{'训练acc':>9}{'差距':>8}")
    rows = []
    for name, pipe in candidates.items():
        pipe.fit(tr["text_norm"], y_tr)
        pred_te = pipe.predict(te["text_norm"])
        pred_tr = pipe.predict(tr["text_norm"])
        acc = accuracy_score(y_te, pred_te)
        f1 = f1_score(y_te, pred_te, average="macro")
        acc_tr = accuracy_score(y_tr, pred_tr)
        gap = acc_tr - acc
        rows.append((name, acc, f1, acc_tr, gap))
        print(f"{name:<44}{acc:>9.4f}{f1:>9.4f}{acc_tr:>9.4f}{gap:>+8.4f}")

    best = max(rows, key=lambda r: r[2])
    print("\n宏 F1 最高:", best[0], f"{best[2]:.4f}")


if __name__ == "__main__":
    sys.exit(main())
