"""alpha 与划分种子稳健性：对 ComplementNB 的正则强度 alpha 跨多个划分种子复核，
避免用单次划分的偶然数字定参（依据《文本分类评测口径与可复现性》）。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline

HERE = Path(__file__).resolve().parent
GEN = HERE.parent / "generated"
sys.path.insert(0, str(GEN))
import algorithm  # noqa: E402

DATA = HERE.parents[3] / "examples" / "text_cls_demo" / "data" / "agnews_sample.csv"
SEEDS = [42, 7, 2024, 1, 99]
ALPHAS = [0.1, 0.3, 0.5, 1.0]


def features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=2, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True, min_df=3)),
    ])


def main() -> int:
    feats = algorithm.build_features(pd.read_csv(DATA))
    y = feats["label"].astype(str)
    X = feats[algorithm.NORM_COLUMN].astype(str)
    idx = np.arange(len(feats))

    table = {}
    for alpha in ALPHAS:
        f1s, gaps = [], []
        for seed in SEEDS:
            tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
            pipe = make_pipeline(features(), ComplementNB(alpha=alpha))
            pipe.fit(X.iloc[tr], y.iloc[tr])
            te_p = pipe.predict(X.iloc[te])
            tr_p = pipe.predict(X.iloc[tr])
            f1s.append(float(f1_score(y.iloc[te], te_p, average="macro", zero_division=0)))
            gaps.append(float(accuracy_score(y.iloc[tr], tr_p)
                              - accuracy_score(y.iloc[te], te_p)))
        table[str(alpha)] = {
            "macro_f1_mean": statistics.mean(f1s),
            "macro_f1_std": statistics.pstdev(f1s),
            "macro_f1_min": min(f1s),
            "gap_mean": statistics.mean(gaps),
            "gap_max": max(gaps),
            "per_seed_f1": [round(v, 4) for v in f1s],
        }
        t = table[str(alpha)]
        print(f"alpha={alpha:<4} macroF1 {t['macro_f1_mean']:.4f} ± {t['macro_f1_std']:.4f} "
              f"(min {t['macro_f1_min']:.4f}) gap_mean {t['gap_mean']:+.4f} gap_max {t['gap_max']:+.4f}")

    (HERE / "select_alpha_results.json").write_text(
        json.dumps({"seeds": SEEDS, "alphas": ALPHAS, "table": table},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
