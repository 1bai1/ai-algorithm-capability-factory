"""ComplementNB 稳健性：跨切分种子 × alpha，看宏 F1 与训练/测试差距。

harness 只用 seed=42 的一个分层随机划分；这里额外换 5 个种子检查结论不是
单次划分的偶然。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(r"D:\awork\akf\llmagent\code\algorithm-coach")
DATA = ROOT / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pipe(alpha: float):
    word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           sublinear_tf=True, min_df=3)
    union = FeatureUnion([("w", word), ("c", char)])
    return make_pipeline(union, ComplementNB(alpha=alpha))


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    X = df["text"].map(normalize).astype(str).to_numpy()
    y = df["label"].astype(str).to_numpy()
    for alpha in (0.1, 0.3, 0.5, 1.0):
        f1s, gaps = [], []
        for seed in (42, 7, 2024, 1, 99):
            tr, te = train_test_split(np.arange(len(df)), test_size=0.2,
                                      random_state=seed, stratify=y)
            m = pipe(alpha)
            m.fit(X[tr], y[tr])
            f1s.append(f1_score(y[te], m.predict(X[te]), average="macro", zero_division=0))
            gaps.append(accuracy_score(y[tr], m.predict(X[tr]))
                        - accuracy_score(y[te], m.predict(X[te])))
        print(f"alpha={alpha}: macroF1 {np.mean(f1s):.4f}±{np.std(f1s):.4f} "
              f"(min {min(f1s):.4f})  gap mean {np.mean(gaps):+.4f} max {max(gaps):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
