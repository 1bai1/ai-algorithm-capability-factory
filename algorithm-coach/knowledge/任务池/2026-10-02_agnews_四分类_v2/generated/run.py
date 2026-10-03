"""运行入口：读数据 → 训练 → 预测 → 写结果。

用法::

    python run.py --data 数据.csv --out 预测.csv

只依赖 pandas / numpy / scikit-learn 与同目录的 ``algorithm.py``，
不依赖生成它的项目、不需要 harness。按 `manifest.json` 的声明读取文本列与标签列，
做分层随机切分（默认 8:2，seed 取 manifest，默认 42），在训练段上训练，
对测试段预测并输出结果；有真值时顺带打印准确率与宏 F1。
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

import algorithm  # noqa: E402  同目录算法核心


def load_manifest() -> dict:
    path = HERE / "manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="AG News 四分类：读数据 → 训练 → 预测 → 写结果")
    ap.add_argument("--data", required=True, help="输入数据 CSV（需含文本列与标签列）")
    ap.add_argument("--out", default="predictions.csv", help="预测结果输出路径")
    ap.add_argument("--test-size", type=float, default=0.2, help="测试段占比，默认 0.2")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")
    ap.add_argument("--text-col", default=None, help="文本列名，默认取 manifest")
    ap.add_argument("--label-col", default=None, help="标签列名，默认取 manifest")
    args = ap.parse_args()

    manifest = load_manifest()
    label_col = args.label_col or (manifest.get("label") or {}).get("column", "label")
    text_col = args.text_col or manifest.get("text_column", "text")
    seed = args.seed if args.seed is not None else manifest.get("seed", 42)

    if not Path(args.data).is_file():
        print(f"数据文件不存在: {args.data}", file=sys.stderr)
        return 2

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    for col in (text_col, label_col):
        if col not in raw.columns:
            print(f"输入数据缺少必需列 {col!r}；实际列: {list(raw.columns)}", file=sys.stderr)
            return 2

    feats = algorithm.build_features(raw)
    y = feats[label_col].astype(str)
    if y.nunique() < 2:
        print("标签只有一类，无法训练分类器。", file=sys.stderr)
        return 2

    from sklearn.model_selection import train_test_split

    idx = np.arange(len(feats))
    stratify = y if y.value_counts().min() >= 2 else None
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=seed, stratify=stratify)

    train_df = feats.iloc[idx_train]
    model = algorithm.fit(train_df)

    test_df = feats.iloc[idx_test]
    test_features = test_df.drop(columns=[label_col])
    preds = np.asarray(algorithm.predict(model, test_features))

    out = pd.DataFrame({
        "row": idx_test,
        "prediction": preds,
        "actual": test_df[label_col].to_numpy(),
    })
    if text_col in test_df.columns:
        out.insert(1, text_col, test_df[text_col].to_numpy())
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    from sklearn.metrics import accuracy_score, f1_score

    truth = test_df[label_col].astype(str).to_numpy()
    print(f"数据 {len(feats)} 行 ｜ 训练 {len(train_df)} / 测试 {len(test_df)} ｜ seed={seed}")
    print(f"测试段 准确率 {accuracy_score(truth, preds):.3f} ｜ "
          f"宏 F1 {f1_score(truth, preds, average='macro', zero_division=0):.3f}")
    print(f"预测结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
