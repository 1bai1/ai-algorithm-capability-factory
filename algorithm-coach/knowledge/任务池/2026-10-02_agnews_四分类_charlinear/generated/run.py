"""运行入口：读数据 → 训练 → 预测 → 写结果。

不做任何项目内依赖，只用到 pandas / numpy / scikit-learn 与同目录的 `algorithm.py`。
harness 只做质检；用户直接跑本文件即可。

用法::

    python run.py --data 数据.csv --out 预测结果.csv
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


def _manifest() -> dict:
    path = HERE / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="AG News 四分类：词/字符 n-gram TF-IDF + ComplementNB。"
                    "读入带 label 的 CSV，训练后输出测试段预测。")
    ap.add_argument("--data", required=True, help="输入 CSV（含 text 与 label 两列）")
    ap.add_argument("--out", default="predictions.csv", help="预测结果输出路径")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练段占比，默认 0.8")
    ap.add_argument("--seed", type=int, default=None, help="随机种子，默认取 manifest")
    args = ap.parse_args(argv)

    meta = _manifest()
    label_col = (meta.get("label") or {}).get("column", "label")
    seed = args.seed if args.seed is not None else int(meta.get("seed", 42))

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    feats = algorithm.build_features(raw)
    if label_col not in feats.columns:
        print(f"输入缺少标签列 {label_col}，无法训练。", file=sys.stderr)
        return 2

    labeled = feats[label_col].notna()
    keep = feats[labeled]
    if len(keep) < 10:
        print(f"有标签的行只有 {len(keep)} 行，太少，无法训练。", file=sys.stderr)
        return 2

    from sklearn.model_selection import train_test_split

    y = keep[label_col].astype(str)
    stratify = y if int(y.value_counts().min()) >= 2 else None
    train_idx, test_idx = train_test_split(
        keep.index.to_numpy(), test_size=1.0 - args.train_ratio,
        random_state=seed, stratify=stratify)

    train_df = keep.loc[train_idx]
    test_df = keep.loc[test_idx]
    model = algorithm.fit(train_df)
    preds = np.asarray(algorithm.predict(model, test_df.drop(columns=[label_col])))

    out = pd.DataFrame({"row": test_df.index.to_numpy(), "prediction": preds})
    if algorithm.TEXT_COLUMN in test_df.columns:
        out.insert(1, algorithm.TEXT_COLUMN, test_df[algorithm.TEXT_COLUMN].to_numpy())
    out["actual"] = test_df[label_col].to_numpy()
    out["has_truth"] = True
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    from sklearn.metrics import accuracy_score, f1_score
    truth = test_df[label_col].astype(str).to_numpy()
    print(f"数据 {len(feats)} 行 ｜ 训练 {len(train_df)} / 预测 {len(test_df)} ｜ seed={seed}")
    print(f"测试段 准确率 {accuracy_score(truth, preds):.3f} ｜ "
          f"宏 F1 {f1_score(truth, preds, average='macro'):.3f}")
    print(f"预测结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
