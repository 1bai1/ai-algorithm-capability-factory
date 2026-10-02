"""运行稳定性检查：换种情况还撑不撑得住。

检查项
------
- 时间预算        fit/predict 是否在预算内完成
- 确定性          同种子两次运行，预测值必须逐个一致
- 小样本不崩      只喂 150 行最短数据，全链路仍能跑完
- 坏数据不崩      注入缺失值 + 删几行（模拟停牌/缺日），全链路仍能跑完
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import data, runner
from ..split import split_classification
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "stability"
TITLE = "运行稳定性"
ORDER = 3
GATE = False
NEEDS_CHAIN = True

CHECKS: list[tuple[str, str, str | None]] = [
    ("stability.timeout", "时间预算", None),
    ("stability.determinism", "确定性（同种子可复现）", "过拟合与结果不稳定"),
    ("stability.small_sample", "小样本不崩", None),
    ("stability.bad_data", "坏数据不崩", None),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}
SMALL_ROWS = 150


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
    return [
        _check_timeout(v),
        _check_determinism(v),
        _check_small_sample(v),
        _check_bad_data(v),
    ]


# ---------------------------------------------------------------- 时间预算
def _check_timeout(v: "Validator") -> CheckResult:
    cid = "stability.timeout"
    chain = v.chain
    if chain is None:
        return _skip(cid, "fit/predict 未执行")
    feat_txt = ""
    if v.features_result is not None and v.features_result.ok:
        feat_txt = f"；build_features {v.features_result.elapsed:.1f}s"
    if chain.status == "timeout":
        return _result(cid, False, f"fit/predict 超过预算 {v.budget:.0f}s 被强制终止{feat_txt}")
    if chain.status == "error":
        return _skip(cid, f"主流程运行报错，耗时检查不适用{feat_txt}")
    return _result(cid, True,
                   f"fit/predict {chain.elapsed:.1f}s / 预算 {v.budget:.0f}s{feat_txt}",
                   elapsed=chain.elapsed)


# ---------------------------------------------------------------- 确定性
def _check_determinism(v: "Validator") -> CheckResult:
    cid = "stability.determinism"
    if v.split is None or v.chain is None or not v.chain.ok:
        return _skip(cid, "主流程未产生可比对的预测")

    first = runner.run_chain(v.algo_dir, v.module_name, v.split.train_df,
                             v.split.test_features, v.seed, v.budget)
    second = runner.run_chain(v.algo_dir, v.module_name, v.split.train_df,
                              v.split.test_features, v.seed, v.budget)
    if not (first.ok and second.ok):
        bad = first if not first.ok else second
        reason = "超时" if bad.status == "timeout" else "报错"
        return _skip(cid, f"重跑{reason}，无法比对两次结果")

    a = np.asarray(first.value)
    b = np.asarray(second.value)
    if len(a) != len(b):
        return _result(cid, False, f"两次运行长度不同：{len(a)} vs {len(b)}")
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":     # 分类：类别标签逐条比对
        mismatch = int(np.sum(a.astype(str) != b.astype(str)))
        if mismatch:
            return _result(cid, False,
                           f"同种子两次运行 {len(a)} 个预测中 {mismatch} 个类别不一致；"
                           f"算法内部疑似有未受种子控制的随机源",
                           elapsed=first.elapsed + second.elapsed)
    else:
        a = a.astype(float)
        b = b.astype(float)
        mismatch = int(np.sum(~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)))
        if mismatch:
            diff = np.abs(a - b)
            diff = diff[np.isfinite(diff)]
            max_diff = float(diff.max()) if len(diff) else float("nan")
            return _result(cid, False,
                           f"同种子两次运行 {len(a)} 个预测中 {mismatch} 个不一致"
                           f"（最大偏差 {max_diff:.4g}）；"
                           f"算法内部疑似有未受种子控制的随机源"
                           f"（如未设种子的 np.random.default_rng()）",
                           elapsed=first.elapsed + second.elapsed)
    return _result(cid, True,
                   f"同种子两次运行 {len(a)} 个预测完全一致"
                   f"（各 {first.elapsed:.1f}s / {second.elapsed:.1f}s）",
                   elapsed=first.elapsed + second.elapsed)


# -------------------------------------------------------------- 小样本不崩
def _check_small_sample(v: "Validator") -> CheckResult:
    cid = "stability.small_sample"
    sub = v.raw.iloc[:min(SMALL_ROWS, len(v.raw))].reset_index(drop=True)
    return _run_subset(v, sub, cid, label=f"{len(sub)} 行数据")


# -------------------------------------------------------------- 坏数据不崩
def _check_bad_data(v: "Validator") -> CheckResult:
    cid = "stability.bad_data"
    bad = data.corrupt_text(v.raw, seed=v.seed,
                            text_column=v.manifest.text_column)
    return _run_subset(v, bad, cid, label="把部分文本置空、塞入无信息短文本并删掉若干行后",
                       is_bad=True)


def _run_subset(v: "Validator", sub, cid: str, label: str, is_bad: bool = False) -> CheckResult:
    """在小样本/坏数据上跑一遍全链路（build_features → fit → predict）。"""
    feats = runner.run_features(v.algo_dir, v.module_name, sub, v.budget)
    if not feats.ok:
        if feats.status == "timeout":
            return _result(cid, False, f"{label} build_features 超时")
        return _result(cid, False,
                       f"{label} build_features 报错: {_tail(feats.error)}", feats.location)
    if not isinstance(feats.value, pd.DataFrame):
        return _result(cid, False,
                       f"{label} build_features 返回 {type(feats.value).__name__}，不是 DataFrame")

    split, error = split_classification(
        feats.value, v.manifest.label.column, test_size=0.3, seed=v.seed,
        min_train=5, min_test=5)
    if split is None:
        return _result(cid, False, f"{label} 无法切分: {error}", feats.location)

    chain = runner.run_chain(v.algo_dir, v.module_name, split.train_df,
                             split.test_features, v.seed, v.budget)
    if not chain.ok:
        if chain.status == "timeout":
            return _result(cid, False, f"{label} fit/predict 超时")
        return _result(cid, False,
                       f"{label} fit/predict 报错: {_tail(chain.error)}", chain.location)

    raw_preds = np.asarray(chain.value)
    if len(raw_preds) != len(split.test_features):
        return _result(cid, False,
                       f"{label} predict 返回 {len(raw_preds)} 个值，"
                       f"测试集 {len(split.test_features)} 行")
    if raw_preds.dtype.kind in "OUS":                       # 分类：类别标签
        if len(set(raw_preds.astype(str))) == 1:
            note = "；预测为常数"
        else:
            note = ""
    else:
        preds = raw_preds.astype(float)
        if not np.isfinite(preds).all():
            n_bad = int((~np.isfinite(preds)).sum())
            return _result(cid, False, f"{label} 预测含 {n_bad} 个非有限值")
        note = "；预测为常数" if preds.max() - preds.min() <= 0 else ""
    return _result(cid, True,
                   f"{label} 全链路跑通（训练 {len(split.train_df)} 行，"
                   f"预测 {len(split.test_features)} 行）{note}",
                   elapsed=feats.elapsed + chain.elapsed)


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return "（无错误信息）"
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " ⏎ ".join(parts[-lines:])
