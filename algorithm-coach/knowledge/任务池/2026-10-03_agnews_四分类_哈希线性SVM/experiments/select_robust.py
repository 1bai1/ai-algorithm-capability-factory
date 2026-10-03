"""稳健性实验：在 5 个切分种子上复核哈希 LinearSVC 配置，并与参考基线对照。

依据卡片：《文本分类评测口径与可复现性》《文本分类评估与交叉验证》——
自跑实验要换种子重复，报均值与标准差，不能只报单次划分。

运行：D:/environment/miniconda3/envs/math/python.exe experiments/select_robust.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
SEEDS = (42, 7, 2024, 1, 99)


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hashed_pipe(c: float, char: bool = True) -> Pipeline:
    parts = [("word", HashingVectorizer(
        n_features=2 ** 18, analyzer="word", ngram_range=(1, 2),
        alternate_sign=False, norm=None))]
    if char:
        parts.append(("char", HashingVectorizer(
            n_features=2 ** 18, analyzer="char_wb", ngram_range=(3, 5),
            alternate_sign=False, norm=None)))
    return Pipeline([
        ("hash", FeatureUnion(parts)),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ("clf", LinearSVC(C=c)),
    ])


def ref_pipe() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, C=4.0))


def run() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    norm = df["text"].map(normalize)
    y = df["label"].astype(str).to_numpy()

    configs = {
        "hash 2^18 word+char LinearSVC(C=0.5)": hashed_pipe(0.5),
        "hash 2^18 word+char LinearSVC(C=1)": hashed_pipe(1.0),
        "hash 2^18 word+char LinearSVC(C=2)": hashed_pipe(2.0),
        "hash 2^18 word LinearSVC(C=1)": hashed_pipe(1.0, char=False),
        "ref word-tfidf(1,2)+LogReg(C=4)": ref_pipe(),
    }
    stats = {k: {"f1": [], "acc": [], "gap": []} for k in configs}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2,
                                  random_state=seed, stratify=y)
        Xtr, Xte = norm.iloc[tr], norm.iloc[te]
        ytr, yte = y[tr], y[te]
        for name, pipe in configs.items():
            pipe.fit(Xtr, ytr)
            tr_acc = accuracy_score(ytr, pipe.predict(Xtr))
            pred = pipe.predict(Xte)
            stats[name]["f1"].append(f1_score(yte, pred, average="macro",
                                              zero_division=0))
            stats[name]["acc"].append(accuracy_score(yte, pred))
            stats[name]["gap"].append(tr_acc - accuracy_score(yte, pred))

    print(f"{'config':42s} {'macroF1 mean±sd':>18s} {'minF1':>7s} "
          f"{'meanGap':>8s} {'maxGap':>7s}")
    for name, s in stats.items():
        f1 = np.array(s["f1"])
        gap = np.array(s["gap"])
        print(f"{name:42s} {f1.mean():.4f}±{f1.std():.4f}      "
              f"{f1.min():.4f} {gap.mean():+8.4f} {gap.max():+7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
