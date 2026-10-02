"""本算法的运行入口：读数据 → 训练 → 预测 → 写结果。

用法::

    python run.py --data 你的数据.csv [--out 预测结果.csv] [--train-ratio 0.8]

设计原则：**拿到这个目录就能跑**。只依赖 pandas / numpy / scikit-learn，
不依赖生成它的项目的任何代码。

按 manifest.json 声明的任务类型自动分派：
  - classification：分层随机切分，输出预测类别，打印准确率与宏 F1
  - time_series   ：按时间切分；末尾若干行标签未知，它们的预测就是"对未来若干期的预测"

行级卫生与 harness 保持一致：标签缺失、或特征全为 NaN 的行不进训练集
（预热期本来就是 NaN，末尾若干行标签天然未知）。
更严格的检查（切分控制、反泄漏、成本扫描、基线对照）由项目的 harness 负责，
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

NON_FEATURE = ("date", "text", "symbol", "code")


def load_manifest() -> dict:
    path = HERE / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_rows(feats: pd.DataFrame, label_col: str, task: str,
              train_ratio: float, seed: int):
    """挑出训练行 / 预测行 / 末尾无真值的行（真正的"未来"）。"""
    feature_cols = [c for c in feats.columns
                    if c not in (label_col, *NON_FEATURE)]
    usable = feats[label_col].notna().to_numpy().copy()
    if feature_cols:
        usable &= ~feats[feature_cols].isna().any(axis=1).to_numpy()
    positions = np.flatnonzero(usable)
    if len(positions) < 10:
        raise SystemExit(f"可用训练行只有 {len(positions)} 行（标签缺失或特征全 NaN），无法训练")

    if task == "classification":
        from sklearn.model_selection import train_test_split

        y = feats[label_col].astype(str).iloc[positions]
        stratify = y if y.value_counts().min() >= 2 else None
        tr, te = train_test_split(positions, test_size=1 - train_ratio,
                                  random_state=seed, stratify=stratify)
        return feats.index[np.sort(tr)], feats.index[np.sort(te)], feats.index[:0]

    n_train = max(1, int(len(positions) * train_ratio))
    train_idx = feats.index[positions[:n_train]]
    test_idx = feats.index[positions[n_train:]]
    tail_idx = feats.index[positions[-1] + 1:]      # 末尾标签未知的行 = 要预测的未来
    return train_idx, test_idx, tail_idx


def main() -> int:
    ap = argparse.ArgumentParser(
        description="运行本目录的算法：读数据 → 训练 → 预测 → 写结果")
    ap.add_argument("--data", required=True, help="输入数据 CSV（需含标签列，用于训练）")
    ap.add_argument("--out", default="predictions.csv", help="预测结果输出路径")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="训练段占比，默认 0.8")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")
    args = ap.parse_args()

    manifest = load_manifest()
    task = manifest.get("task", "time_series")
    label_col = (manifest.get("label") or {}).get("column", "label")
    horizon = (manifest.get("label") or {}).get("horizon")
    seed = args.seed if args.seed is not None else manifest.get("seed", 42)

    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    feats = algorithm.build_features(raw)
    if label_col not in feats.columns:
        print(f"输入数据缺少标签列 {label_col}——本算法需要先在你自己的数据上训练。",
              file=sys.stderr)
        return 2

    idx_train, idx_test, idx_tail = pick_rows(feats, label_col, task, args.train_ratio, seed)
    train_df = feats.loc[idx_train]
    predict_idx = idx_test.append(idx_tail)
    if len(predict_idx) == 0:
        print("没有可预测的行。", file=sys.stderr)
        return 2

    model = algorithm.fit(train_df)
    test_df = feats.loc[predict_idx]
    preds = np.asarray(algorithm.predict(model, test_df.drop(columns=[label_col])))

    out = pd.DataFrame({"row": predict_idx, "prediction": preds})
    for col in ("date", "symbol", "text"):
        if col in test_df.columns:
            out.insert(1, col, test_df[col].to_numpy())
    truth = test_df[label_col].to_numpy()
    out["actual"] = truth
    out["has_truth"] = ~pd.isna(truth)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"任务类型 {task} ｜ 数据 {len(feats)} 行 ｜ 训练 {len(train_df)} / "
          f"预测 {len(test_df)}（其中 {int((~out['has_truth']).sum())} 行无真值）｜ seed={seed}")
    known = ~pd.isna(truth)
    if known.any():
        if task == "classification":
            from sklearn.metrics import accuracy_score, f1_score
            t = np.asarray(truth)[known].astype(str)
            p = np.asarray(preds)[known].astype(str)
            print(f"预测段准确率 {accuracy_score(t, p):.3f} ｜ "
                  f"宏 F1 {f1_score(t, p, average='macro', zero_division=0):.3f}"
                  f"（{int(known.sum())} 行有真值）")
        else:
            err = np.asarray(preds)[known] - np.asarray(truth)[known]
            hit = np.mean(np.sign(np.asarray(preds)[known])
                          == np.sign(np.asarray(truth)[known]))
            print(f"预测段 RMSE {np.sqrt(np.mean(err ** 2)):.4f} ｜ 方向准确率 {hit:.3f}"
                  f"（{int(known.sum())} 行有真值）")
    if task == "time_series" and int((~known).sum()) > 0:
        print(f"末尾 {int((~known).sum())} 行的预测即对未来"
              f"{horizon if horizon else '若干'}期的预测。")
    print(f"预测结果已写入 {args.out}")
    print("完整验证（切分控制 / 反泄漏 / 成本与基线对照）见本项目 harness："
          "python -m harness validate <本目录> --data <csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
