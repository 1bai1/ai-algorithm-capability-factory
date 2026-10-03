"""稳健性复核：跨 5 个切分种子比较候选，避免以单次划分的偶然数字定参。

依据《文本分类是否真的进步了_对比综述》：自跑实验须换种子重复报均值与标准差，
并看最差情形，而不是只报一次划分的最好数字。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_model import DATA, make_model, normalize  # noqa: E402

SEEDS = [42, 7, 2024, 1, 99]
KINDS = ["nb0.1", "nb0.3", "vote_nb1_lr1", "vote_nb2_lr1", "vote_nb1_lr2"]


def main() -> int:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    texts = df["text"].astype(str).map(normalize)
    y = df["label"].astype(str)

    summary = {}
    for kind in KINDS:
        accs, f1s, gaps = [], [], []
        for seed in SEEDS:
            idx_train, idx_test = train_test_split(
                df.index.to_numpy(), test_size=0.2, random_state=seed, stratify=y)
            model = make_model(kind)
            model.fit(texts.iloc[idx_train], y.iloc[idx_train])
            pred = model.predict(texts.iloc[idx_test])
            tr = model.predict(texts.iloc[idx_train])
            from sklearn.metrics import accuracy_score
            a = accuracy_score(y.iloc[idx_test], pred)
            accs.append(a)
            f1s.append(f1_score(y.iloc[idx_test], pred, average="macro"))
            gaps.append(accuracy_score(y.iloc[idx_train], tr) - a)
        summary[kind] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "f1_min": float(np.min(f1s)),
            "acc_mean": float(np.mean(accs)),
            "gap_mean": float(np.mean(gaps)),
            "gap_max": float(np.max(gaps)),
            "f1_per_seed": [round(x, 4) for x in f1s],
        }
        s = summary[kind]
        print(f"{kind:14s} F1 {s['f1_mean']:.4f} ± {s['f1_std']:.4f} "
              f"(min {s['f1_min']:.4f}) | acc {s['acc_mean']:.4f} "
              f"| gap mean {s['gap_mean']:+.4f} max {s['gap_max']:+.4f}")

    out = HERE / "select_alpha_results.json"
    out.write_text(json.dumps({"seeds": SEEDS, "summary": summary},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
