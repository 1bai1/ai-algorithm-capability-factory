"""选型实验：在 harness 同口径（seed=42 分层随机 80/20）下比较候选链路。

候选覆盖复用池已验证的两条浅层路线：
- 《AG News 四分类 × 词/字符 n-gram ComplementNB》记录 ComplementNB(α=0.3) 为强基线；
- 《浅层文本分类：朴素贝叶斯与 SVM》把 NB、SVM 与**集成方法**列为同一浅层族，
  集成可作为 NB 与线性模型的互补增强。

输出每格：测试准确率 / 宏 F1 / 训练减测试准确率。并发跑 alpha 与权重档。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # 仓库根
DATA = ROOT / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
SEED = 42


def normalize(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=2, strip_accents="unicode")),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)),
    ])


def make_model(kind: str):
    if kind == "nb0.1":
        return Pipeline([("f", features()), ("c", ComplementNB(alpha=0.1))])
    if kind == "nb0.3":
        return Pipeline([("f", features()), ("c", ComplementNB(alpha=0.3))])
    if kind == "nb0.5":
        return Pipeline([("f", features()), ("c", ComplementNB(alpha=0.5))])
    if kind == "lr8":
        return Pipeline([("f", features()),
                         ("c", LogisticRegression(max_iter=1000, C=8.0))])
    if kind == "svc1":
        return Pipeline([("f", features()), ("c", LinearSVC(C=1.0))])
    if kind.startswith("vote"):
        # vote_nb1_lr1 / vote_nb2_lr1 / vote_nb1_lr2
        parts = kind.split("_")
        w_nb, w_lr = int(parts[1][2]), int(parts[2][2])
        clf = VotingClassifier(
            estimators=[
                ("nb", ComplementNB(alpha=0.3)),
                ("lr", LogisticRegression(max_iter=1000, C=8.0)),
            ],
            voting="soft", weights=[w_nb, w_lr])
        return Pipeline([("f", features()), ("c", clf)])
    raise ValueError(kind)


def evaluate(kind: str, texts, train_idx, test_idx, y) -> dict:
    model = make_model(kind)
    t0 = time.perf_counter()
    model.fit(texts.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(texts.iloc[test_idx])
    train_pred = model.predict(texts.iloc[train_idx])
    elapsed = time.perf_counter() - t0
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "kind": kind,
        "accuracy": float(accuracy_score(y.iloc[test_idx], pred)),
        "macro_f1": float(f1_score(y.iloc[test_idx], pred, average="macro")),
        "train_accuracy": float(accuracy_score(y.iloc[train_idx], train_pred)),
        "gap": float(accuracy_score(y.iloc[train_idx], train_pred)
                     - accuracy_score(y.iloc[test_idx], pred)),
        "seconds": round(elapsed, 2),
    }


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    texts = df[text_col].astype(str).map(normalize)
    y = df[label_col].astype(str)

    idx_train, idx_test = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=SEED, stratify=y)
    print(f"训练 {len(idx_train)} / 测试 {len(idx_test)}")

    kinds = ["nb0.1", "nb0.3", "nb0.5", "lr8", "svc1",
             "vote_nb1_lr1", "vote_nb2_lr1", "vote_nb1_lr2"]
    rows = []
    for kind in kinds:
        r = evaluate(kind, texts, idx_train, idx_test, y)
        rows.append(r)
        print(f"{kind:14s} acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} "
              f"gap={r['gap']:+.4f} ({r['seconds']}s)")

    out = HERE / "select_model_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
