"""选型实验：对比「特征哈希 + TF-IDF 加权 + 线性分类器」若干配置。

目的：为 AG News 四分类选一条与既有词表 TF-IDF 路线不同的哈希路线，
并在同一划分口径下与参考基线（词表 TF-IDF(1,2)+LogReg(C=4)）对照。

依据卡片：《特征哈希与HashingVectorizer》《TF-IDF词项加权》《浅层文本分类_朴素贝叶斯与SVM》。
运行：D:/environment/miniconda3/envs/math/python.exe experiments/select_hashed.py
"""
from __future__ import annotations

import re
import sys
import time
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
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"


def normalize(s: object) -> str:
    text = str(s).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def hashed_union(n_features: int, char_features: int) -> FeatureUnion:
    return FeatureUnion([
        ("word", HashingVectorizer(
            n_features=n_features, analyzer="word", ngram_range=(1, 2),
            alternate_sign=False, norm=None)),
        ("char", HashingVectorizer(
            n_features=char_features, analyzer="char_wb", ngram_range=(3, 5),
            alternate_sign=False, norm=None)),
    ])


def evaluate(name: str, model, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    return {"name": name, "acc": acc, "f1": f1, "gap": tr_acc - acc,
            "fit_s": fit_s}


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"data: {len(df)} rows, classes={sorted(df['label'].unique())}")
    norm = df["text"].map(normalize)
    y = df["label"].astype(str).to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42,
        stratify=y)
    Xtr_txt, Xte_txt = norm.iloc[train_idx], norm.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    results = []

    # 参考基线（与 harness 自带一致）
    ref = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, C=4.0))
    results.append(evaluate("ref: word-tfidf(1,2)+LogReg(C=4)", ref,
                            Xtr_txt, ytr, Xte_txt, yte))

    # 哈希路线：word ∪ char_wb -> TfidfTransformer -> 线性分类器
    for n_feat in (2 ** 18, 2 ** 20):
        for clf_name, clf in (
            ("LinearSVC(C=1)", LinearSVC(C=1.0)),
            ("LogReg(C=4)", LogisticRegression(max_iter=2000, C=4.0)),
            ("ComplementNB(0.3)", ComplementNB(alpha=0.3)),
        ):
            pipe = Pipeline([
                ("hash", hashed_union(n_feat, n_feat)),
                ("tfidf", TfidfTransformer(sublinear_tf=True)),
                ("clf", clf),
            ])
            results.append(evaluate(
                f"hash{n_feat//1024}k word(1,2)+char(3,5) {clf_name}", pipe,
                Xtr_txt, ytr, Xte_txt, yte))

    # 对照：纯词哈希
    for clf_name, clf in (("LinearSVC(C=1)", LinearSVC(C=1.0)),):
        pipe = Pipeline([
            ("hash", HashingVectorizer(
                n_features=2 ** 20, analyzer="word", ngram_range=(1, 2),
                alternate_sign=False, norm=None)),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
            ("clf", clf),
        ])
        results.append(evaluate(
            f"hash1m word(1,2) {clf_name}", pipe, Xtr_txt, ytr, Xte_txt, yte))

    print(f"\n{'model':52s} {'acc':>7s} {'macroF1':>8s} {'gap':>7s} {'fit_s':>7s}")
    for r in results:
        print(f"{r['name']:52s} {r['acc']:7.4f} {r['f1']:8.4f} "
              f"{r['gap']:+7.4f} {r['fit_s']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
