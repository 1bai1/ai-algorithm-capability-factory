"""运行入口：读 CSV → 训练 → 预测 → 写结果 CSV。

用法::

    python run.py --data 数据.csv [--out 预测结果.csv] [--train-ratio 0.8] [--seed 42]

只依赖 pandas / numpy / scikit-learn，**不依赖生成它的项目或 harness**。
切分口径与 harness 一致：分层随机 8:2（seed 取自 manifest，默认 42），
训练段拟合词表与分类器，测试段只做 transform。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import algorithm  # noqa: E402


def load_manifest() -> dict:
    path = HERE / "manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="AG News 四分类：TF-IDF(词+字符 n-gram) + ComplementNB，"
                    "读数据 → 训练 → 预测 → 写结果")
    ap.add_argument("--data", required=True, help="输入 CSV，需含 text 与 label 两列")
    ap.add_argument("--out", default="predictions.csv", help="预测结果输出路径")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练段占比，默认 0.8")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        print("--train-ratio 必须在 (0, 1) 之间", file=sys.stderr)
        return 2

    manifest = load_manifest()
    label_col = (manifest.get("label") or {}).get("column", algorithm.LABEL_COLUMN)
    seed = args.seed if args.seed is not None else manifest.get("seed", 42)

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    if algorithm.TEXT_COLUMN not in raw.columns or label_col not in raw.columns:
        print(f"输入数据需同时包含 {algorithm.TEXT_COLUMN} 与 {label_col} 两列；"
              f"实际列：{list(raw.columns)}", file=sys.stderr)
        return 2

    feats = algorithm.build_features(raw)
    usable = feats[label_col].notna() & (feats[algorithm.NORM_COLUMN] != "")
    positions = np.flatnonzero(usable.to_numpy())
    if len(positions) < 20:
        print(f"可用训练行只有 {len(positions)} 行，无法训练", file=sys.stderr)
        return 2

    from sklearn.model_selection import train_test_split

    y = feats[label_col].astype(str).iloc[positions]
    stratify = y if int(y.value_counts().min()) >= 2 else None
    tr, te = train_test_split(positions, test_size=1.0 - args.train_ratio,
                              random_state=seed, stratify=stratify)

    model = algorithm.fit(feats.iloc[np.sort(tr)])
    test_df = feats.iloc[np.sort(te)]
    preds = np.asarray(algorithm.predict(model, test_df.drop(columns=[label_col])))

    out = pd.DataFrame({"row": np.sort(te), "text": test_df[algorithm.TEXT_COLUMN].to_numpy(),
                        "prediction": preds})
    truth = test_df[label_col].to_numpy()
    out["actual"] = truth
    out["has_truth"] = ~pd.isna(truth)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"数据 {len(feats)} 行 ｜ 训练 {len(tr)} / 测试 {len(te)} ｜ seed={seed}")
    known = ~pd.isna(truth)
    if known.any():
        from sklearn.metrics import accuracy_score, f1_score

        t = np.asarray(truth)[known].astype(str)
        p = np.asarray(preds)[known].astype(str)
        print(f"测试段准确率 {accuracy_score(t, p):.3f} ｜ "
              f"宏 F1 {f1_score(t, p, average='macro', zero_division=0):.3f}"
              f"（{int(known.sum())} 行有真值）")
    print(f"预测结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
