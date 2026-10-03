"""ComplementNB 的 alpha 扫档 + 跨切分种子稳健性（词+字符 union 特征）。

运行:
    D:/environment/miniconda3/envs/math/python.exe experiments/select_alpha.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

DATA = Path(__file__).resolve().parents[4] / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
SEEDS = [42, 7, 2024, 1, 99]
ALPHAS = [0.1, 0.3, 0.5, 1.0]


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_pipe(alpha: float):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                                 strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True, min_df=3)),
    ])
    return make_pipeline(feats, ComplementNB(alpha=alpha))


def main() -> None:
    df = pd.read_csv(DATA)
    df["text_norm"] = df["text"].map(normalize)

    print("== alpha 扫档 (seed=42) ==")
    idx_tr, idx_te = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=42,
        stratify=df["label"].astype(str).to_numpy())
    tr, te = df.loc[idx_tr], df.loc[idx_te]
    for a in ALPHAS:
        pipe = make_pipe(a).fit(tr["text_norm"], tr["label"].astype(str))
        pred = pipe.predict(te["text_norm"])
        print(f"  alpha={a:<4} acc={accuracy_score(te['label'], pred):.4f} "
              f"macroF1={f1_score(te['label'], pred, average='macro'):.4f}")

    print("\n== 跨 5 个切分种子（alpha=0.1 / 0.3 对照） ==")
    for alpha in (0.1, 0.3):
        f1s, gaps = [], []
        for seed in SEEDS:
            idx_tr, idx_te = train_test_split(
                df.index.to_numpy(), test_size=0.2, random_state=seed,
                stratify=df["label"].astype(str).to_numpy())
            tr, te = df.loc[idx_tr], df.loc[idx_te]
            pipe = make_pipe(alpha).fit(tr["text_norm"], tr["label"].astype(str))
            f1 = f1_score(te["label"], pipe.predict(te["text_norm"]), average="macro")
            acc = accuracy_score(te["label"], pipe.predict(te["text_norm"]))
            acc_tr = accuracy_score(tr["label"], pipe.predict(tr["text_norm"]))
            f1s.append(f1)
            gaps.append(acc_tr - acc)
            print(f"  alpha={alpha:<4} seed={seed:<5} acc={acc:.4f} macroF1={f1:.4f} "
                  f"训练测试差距={acc_tr - acc:+.4f}")
        print(f"  alpha={alpha}: macroF1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
              f"(min {np.min(f1s):.4f})；差距均值 {np.mean(gaps):+.4f} 最大 {np.max(gaps):+.4f}")


if __name__ == "__main__":
    sys.exit(main())
