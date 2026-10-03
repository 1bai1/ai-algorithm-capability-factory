"""本算法的运行入口：读数据 → 训练 → 预测 → 写结果。

用法::

    python run.py --data 你的数据.csv [--out 预测结果.csv] [--train-ratio 0.8]

设计原则：**拿到这个目录就能跑**。只依赖 pandas / numpy / scikit-learn，
不依赖生成它的项目的任何代码。

切分是**分层随机**（按标签分层，保证训练段与预测段的类别比例一致），
随机种子取自 manifest.json。算法用「特征哈希（词 1/2-gram ∪ 词内字符 3/5-gram）
→ TF-IDF 加权 → 线性 SVM」，不需要词表、不需要 GPU。
输出预测类别，并打印预测段的准确率与宏 F1。

行级卫生与 harness 保持一致：标签缺失、或特征全为 NaN 的行不进训练集。
更严格的检查（切分控制、反泄漏、基线对照）由项目的 harness 负责，
见本目录 README 的「完整验证」一节。
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

NON_FEATURE = ("text",)


def load_manifest() -> dict:
    path = HERE / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_rows(feats: pd.DataFrame, label_col: str,
              train_ratio: float, seed: int):
    """挑出训练行与预测行；分层随机切分，类别比例与训练段一致。"""
    feature_cols = [c for c in feats.columns if c not in (label_col, *NON_FEATURE)]
    usable = feats[label_col].notna().to_numpy().copy()
    if feature_cols:
        usable &= ~feats[feature_cols].isna().any(axis=1).to_numpy()
    positions = np.flatnonzero(usable)
    if len(positions) < 10:
        raise SystemExit(f"可用训练行只有 {len(positions)} 行（标签缺失或特征全 NaN），无法训练")

    from sklearn.model_selection import train_test_split

    y = feats[label_col].astype(str).iloc[positions]
    stratify = y if y.value_counts().min() >= 2 else None
    tr, te = train_test_split(positions, test_size=1 - train_ratio,
                              random_state=seed, stratify=stratify)
    return feats.index[np.sort(tr)], feats.index[np.sort(te)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="运行本目录的算法：读数据 → 训练 → 预测 → 写结果")
    ap.add_argument("--data", required=True, help="输入数据 CSV（需含标签列，用于训练）")
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

    idx_train, idx_test = pick_rows(feats, label_col, args.train_ratio, seed)
    if len(idx_test) == 0:
        print("没有可预测的行。", file=sys.stderr)
        return 2

    model = algorithm.fit(feats.loc[idx_train])
    test_df = feats.loc[idx_test]
    preds = np.asarray(algorithm.predict(model, test_df.drop(columns=[label_col])))

    out = pd.DataFrame({"row": idx_test, "prediction": preds})
    for col in ("text",):
        if col in test_df.columns:
            out.insert(1, col, test_df[col].to_numpy())
    truth = test_df[label_col].to_numpy()
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
    print("完整验证（切分控制 / 反泄漏 / 基线对照）见本项目 harness："
          "python -m harness validate <本目录> --data <csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
