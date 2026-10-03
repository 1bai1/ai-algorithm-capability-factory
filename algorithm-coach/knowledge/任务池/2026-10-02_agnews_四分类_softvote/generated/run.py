"""本算法的运行入口：读数据 → 训练 → 预测 → 写结果。

用法::

    python run.py --data 你的数据.csv [--out 预测结果.csv]

自带数据集时（本仓库内）::

    python run.py --data ../data/agnews_sample.csv --out predictions.csv

拿到这个目录就能跑，只依赖 pandas / numpy / scikit-learn，不依赖本项目 harness。
切分是**分层随机**（保证训练段与预测段类别比例一致），随机种子取自 manifest.json。
输出预测类别，并打印预测段的准确率与宏 F1。
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

import algorithm  # noqa: E402  （同目录的算法核心）


def load_manifest() -> dict:
    path = HERE / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_split(feats: pd.DataFrame, label_col: str, train_ratio: float, seed: int):
    """分层随机切分出训练行与预测行；类别比例与训练段一致。"""
    from sklearn.model_selection import train_test_split

    keep = feats[label_col].notna().to_numpy()
    positions = np.flatnonzero(keep)
    if len(positions) < 10:
        raise SystemExit(f"可用训练行只有 {len(positions)} 行，无法训练")

    labels = feats[label_col].astype(str).iloc[positions]
    stratify = labels if labels.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        positions, test_size=1 - train_ratio, random_state=seed, stratify=stratify)
    return feats.index[np.sort(train_idx)], feats.index[np.sort(test_idx)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="AG News 四分类（TF-IDF 词/字符 n-gram + NB·LogReg 软投票）："
                    "读数据 → 训练 → 预测 → 写结果")
    ap.add_argument("--data", required=True, help="输入 CSV（需含标签列，用于训练）")
    ap.add_argument("--out", default="predictions.csv", help="预测结果输出路径")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练段占比，默认 0.8")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")
    args = ap.parse_args()

    manifest = load_manifest()
    label_col = (manifest.get("label") or {}).get("column", "label")
    seed = args.seed if args.seed is not None else manifest.get("seed", 42)

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    feats = algorithm.build_features(raw)
    if label_col not in feats.columns:
        print(f"输入数据缺少标签列 {label_col}——本算法需要先在你自己的数据上训练。",
              file=sys.stderr)
        return 2

    idx_train, idx_test = make_split(feats, label_col, args.train_ratio, seed)
    if len(idx_test) == 0:
        print("没有可预测的行。", file=sys.stderr)
        return 2

    model = algorithm.fit(feats.loc[idx_train])
    test_rows = feats.loc[idx_test]
    preds = np.asarray(algorithm.predict(model, test_rows.drop(columns=[label_col])))

    out = pd.DataFrame({"row": idx_test, "prediction": preds})
    if "text" in test_rows.columns:
        out.insert(1, "text", test_rows["text"].to_numpy())
    truth = test_rows[label_col].to_numpy()
    out["actual"] = truth
    out["has_truth"] = ~pd.isna(truth)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"数据 {len(feats)} 行 ｜ 训练 {len(idx_train)} / 预测 {len(idx_test)} ｜ seed={seed}")
    known = ~pd.isna(truth)
    if known.any():
        from sklearn.metrics import accuracy_score, f1_score

        t = np.asarray(truth)[known].astype(str)
        p = np.asarray(preds)[known].astype(str)
        print(f"预测段 准确率 {accuracy_score(t, p):.3f} ｜ "
              f"宏 F1 {f1_score(t, p, average='macro', zero_division=0):.3f}"
              f"（{int(known.sum())} 行有真值）")
    print(f"预测结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
