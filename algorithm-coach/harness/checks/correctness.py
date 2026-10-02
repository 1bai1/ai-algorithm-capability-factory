"""功能正确性检查：算法算出来的东西对不对。

检查项
------
- 行数守恒        build_features 不得增删行、不得改动文本列
- 标签列有效性    标签列存在、无缺失、至少两类，manifest 声明的类别要对得上
- 拟合范围一致性  全量跑 vs 只喂训练集跑，训练集行上的特征必须逐行一致（反泄漏）
- 预测输出有效性  非空、无 NaN、非常数、长度与测试集一致、只输出见过的类别
- 多数类基线对比  明显不如「全猜多数类」才算没学到东西（严格判据在指标表现模块）

没有覆盖到的失败模式：随机切分、切分前重采样、全样本训练……
这些由 harness 自己控制切分与训练数据，结构上就不可能发生。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import contract, metrics, runner
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "correctness"

CHECKS: list[tuple[str, str, str | None]] = [
    ("correctness.row_conservation", "行数守恒", None),
    ("correctness.label_validity", "标签列有效性", "文本分类基准数据集与划分纪律"),
    ("correctness.feature_fit_scope", "拟合范围一致性（反泄漏）", "文本分类评估与交叉验证"),
    ("correctness.prediction_validity", "预测输出有效性", "文本分类评估与交叉验证"),
    ("correctness.naive_baseline", "多数类基线对比", "基线未调优导致虚假提升"),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}

TOL = 1e-9                 # 特征逐行比对的容差
BASELINE_MARGIN = 0.05     # 比多数类基线低这么多才算"明显没学到东西"


def _result(cid: str, passed: bool, detail: str = "", location=None,
            elapsed: float | None = None) -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult(id=cid, module=MODULE, name=name, passed=passed, detail=detail,
                       location=location, kb_card=card, elapsed=elapsed)


def _skip(cid: str, reason: str) -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult.skipped(cid, MODULE, name, reason, kb_card=card)


def skipped_all(reason: str) -> list[CheckResult]:
    return [_skip(cid, reason) for cid, _, _ in CHECKS]


def run(v: "Validator") -> list[CheckResult]:
    results = [
        _check_row_conservation(v),
        _check_label_validity(v),
        _check_feature_fit_scope(v),
        _check_prediction_validity(v),
    ]
    results.append(_check_majority_baseline(v, results))
    return results


# ------------------------------------------------------------------ 行数守恒
def _check_row_conservation(v: "Validator") -> CheckResult:
    cid = "correctness.row_conservation"
    res = v.features_result
    if res is None:
        return _skip(cid, "build_features 未执行")
    if not res.ok:
        if res.status == "timeout":
            return _result(cid, False, "build_features 超时")
        return _result(cid, False, f"build_features 报错: {_tail(res.error)}", res.location)

    frame = v.features
    if not isinstance(frame, pd.DataFrame):
        return _result(cid, False,
                       f"build_features 返回 {type(frame).__name__}，契约要求 DataFrame")
    if len(frame) != len(v.raw):
        return _result(cid, False,
                       f"输入 {len(v.raw)} 行，输出 {len(frame)} 行（差 {len(frame) - len(v.raw):+d} 行）；"
                       f"契约要求行数不变", res.location)

    text_col = v.manifest.text_column if v.manifest else contract.DEFAULT_TEXT_COLUMN
    if text_col not in frame.columns:
        return _result(cid, False,
                       f"特征表丢了文本列 {text_col}——契约要求原样保留", res.location)
    same = (frame[text_col].astype(str).reset_index(drop=True)
            .equals(v.raw[text_col].astype(str).reset_index(drop=True)))
    if not same:
        return _result(cid, False,
                       f"文本列 {text_col} 被就地改动——规范化结果应写进新列，"
                       f"原始文本要留痕", res.location)
    return _result(cid, True,
                   f"{len(frame)} 行进 {len(frame)} 行出，文本列 {text_col} 未改动",
                   elapsed=res.elapsed)


# -------------------------------------------------------------- 标签列有效性
def _check_label_validity(v: "Validator") -> CheckResult:
    """标签列必须存在、无缺失、至少两类；manifest 声明的类别要与数据对得上。

    依据：评测口径卡要求「数据集标识与类数」必须披露，划分纪律卡要求用标准类数。
    """
    cid = "correctness.label_validity"
    if v.features is None or not isinstance(v.features, pd.DataFrame):
        return _skip(cid, "特征表不可用")
    col = v.manifest.label.column
    if col not in v.features.columns:
        return _result(cid, False, f"特征表缺少标签列 {col}")
    series = v.features[col]
    missing = int(series.isna().sum() + series.astype(str).str.strip().isin(["", "nan"]).sum())
    if missing:
        return _result(cid, False, f"标签列有 {missing} 处缺失/空值")
    values = series.astype(str)
    classes = sorted(values.unique())
    if len(classes) < 2:
        return _result(cid, False, f"标签只有一类：{classes}")
    declared = v.manifest.label.classes
    detail = (f"{len(classes)} 类，共 {len(values)} 行；"
              f"多数类占比 {float(values.value_counts(normalize=True).iloc[0]):.3f}")
    if declared:
        unexpected = sorted(set(classes) - set(declared))
        unused = sorted(set(declared) - set(classes))
        if unexpected:
            return _result(cid, False,
                           f"出现未声明的类别 {unexpected}（manifest 声明 {declared}）")
        detail += "；manifest 声明的类别全部出现" + (f"，未出现：{unused}" if unused else "")
    return _result(cid, True, detail)


# ------------------------------------------------------- 拟合范围一致性（反泄漏）
def _check_feature_fit_scope(v: "Validator") -> CheckResult:
    """全量拟合 vs 只用训练集拟合，在训练集行上必须给出相同的特征。

    这是文本分类最重要的一条反泄漏检查：把向量化器（词表/IDF）在全量语料上
    先 fit 再切分，是最常见的隐性泄漏——本检查直接抓它。
    依据：评估与交叉验证卡「让特征提取也随折重训，而不是在全量语料上先 fit 词表再切分」。
    """
    cid = "correctness.feature_fit_scope"
    if v.features is None or not isinstance(v.features, pd.DataFrame):
        return _skip(cid, "全量特征表不可用")
    if len(v.features) != len(v.raw):
        return _skip(cid, "行数不一致，比对无意义")
    if v.split is None:
        return _skip(cid, "切分不可用")

    train_positions = np.setdiff1d(np.arange(len(v.raw)), v.split.test_positions)
    subset = v.raw.iloc[train_positions].reset_index(drop=True)
    res = runner.run_features(v.algo_dir, v.module_name, subset, v.budget)
    if not res.ok:
        if res.status == "timeout":
            return _result(cid, False, f"只喂训练集 {len(subset)} 行时 build_features 超时")
        return _result(cid, False,
                       f"只喂训练集时 build_features 报错: {_tail(res.error)}", res.location)
    frame = res.value
    if not isinstance(frame, pd.DataFrame):
        return _result(cid, False, f"训练集返回 {type(frame).__name__}")
    if len(frame) != len(subset):
        return _result(cid, False, f"训练集输入 {len(subset)} 行，返回 {len(frame)} 行")

    label_col = v.manifest.label.column
    skip_cols = (label_col, v.manifest.text_column)
    columns = [c for c in v.features.columns if c in frame.columns and c not in skip_cols]
    extra = [c for c in v.features.columns if c not in frame.columns and c not in skip_cols]
    if extra:
        return _result(cid, False, f"只喂训练集时缺少列: {', '.join(extra[:5])}")
    if not columns:
        return _result(cid, False, "没有可比对的列")

    max_diff, worst = _compare_frames(v.features.iloc[train_positions], frame, columns)
    if max_diff > TOL:
        hint = ("；特征变换用到了训练集以外的信息——典型原因是向量化器/词表/IDF "
                "在全量语料上 fit，而不是只在训练集上 fit")
        return _result(cid, False,
                       f"训练集 {len(train_positions)} 行特征不一致：{worst} "
                       f"偏差 {max_diff:.6g}{hint}", elapsed=res.elapsed)
    return _result(cid, True,
                   f"全量与仅训练集两次运行，训练集 {len(train_positions)} 行 × "
                   f"{len(columns)} 列逐行一致（最大偏差 {max_diff:.1e}）",
                   elapsed=res.elapsed)


# -------------------------------------------------------------- 预测输出有效性
def _check_prediction_validity(v: "Validator") -> CheckResult:
    cid = "correctness.prediction_validity"
    if v.split is None:
        return _skip(cid, "切分不可用")
    chain = v.chain
    if chain is None:
        return _skip(cid, "fit/predict 未执行")
    if not chain.ok:
        if chain.status == "timeout":
            return _result(cid, False, f"fit/predict 超过时间预算 {v.budget:.0f}s")
        return _result(cid, False, f"fit/predict 报错: {_tail(chain.error)}", chain.location)

    preds = np.asarray(chain.value).astype(str)
    n_expected = len(v.split.test_features)
    if len(preds) != n_expected:
        return _result(cid, False,
                       f"predict 返回 {len(preds)} 个值，测试集 {n_expected} 行，长度不一致")
    if any(x in ("", "nan", "None") for x in preds):
        return _result(cid, False, "预测里有空值/nan")
    allowed = set(v.manifest.label.classes or []) | set(v.split.test_truth.unique()) \
        | set(v.split.train_df[v.manifest.label.column].astype(str).unique())
    outside = sorted(set(preds) - allowed)
    if outside:
        return _result(cid, False,
                       f"预测出现训练与测试都没见过的类别 {outside}（契约要求输出类别标签）")
    if len(set(preds)) == 1:
        return _result(cid, False, f"预测恒为同一类 {preds[0]}（模型未产生有效输出）")
    dist = pd.Series(preds).value_counts(normalize=True)
    detail = (f"{len(preds)} 个预测，覆盖 {len(set(preds))} 类；"
              f"占比 " + "、".join(f"{k} {v_:.2f}" for k, v_ in dist.items()))
    return _result(cid, True, detail, elapsed=chain.elapsed)


# ---------------------------------------------------------------- 多数类基线
def _check_majority_baseline(v: "Validator", previous: list[CheckResult]) -> CheckResult:
    """功能层的松判据：明显不如「全猜多数类」才算没学到东西。

    严格判据（宏 F1 对标参考基线）在指标表现模块。
    """
    cid = "correctness.naive_baseline"
    validity = next((r for r in previous if r.id == "correctness.prediction_validity"), None)
    if validity is None or not validity.passed:
        return _skip(cid, "预测输出无效，不进行比较")
    preds = np.asarray(v.chain.value).astype(str)
    truth = v.split.test_truth.to_numpy().astype(str)
    acc = metrics.accuracy(truth, preds)
    majority = metrics.majority_share(v.split.train_df[v.manifest.label.column])
    detail = (f"准确率 模型 {acc:.3f} / 多数类基线 {majority:.3f}"
              f"（样本 {len(truth)} 行）；宏 F1 {metrics.macro_f1(truth, preds):.3f}")
    if acc < majority - BASELINE_MARGIN:
        return _result(cid, False,
                       f"{detail}；低于多数类基线 {100 * (majority - acc):.1f} 个百分点")
    return _result(cid, True, detail)


# -------------------------------------------------------------------- 工具
def _compare_frames(a: pd.DataFrame, b: pd.DataFrame,
                    columns: list[str]) -> tuple[float, str]:
    """逐列比对两帧（同长度），返回 (最大绝对偏差, 偏差最大的列名)。"""
    worst_col, max_diff = "", 0.0
    for c in columns:
        x = pd.to_numeric(a[c], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b[c], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(x) - np.nan_to_num(y)))
        nan_mismatch = int((np.isnan(x) ^ np.isnan(y)).sum())
        d = float(np.max(diff)) if len(diff) else 0.0
        if nan_mismatch:
            d = max(d, 1.0)
        if d > max_diff:
            worst_col, max_diff = c, d
    return max_diff, worst_col


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return "（无错误信息）"
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " ⏎ ".join(parts[-lines:])
