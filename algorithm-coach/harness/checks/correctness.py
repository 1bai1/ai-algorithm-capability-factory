"""功能正确性检查：算法算出来的东西对不对。

检查项
------
- 行数守恒        build_features 不得增删行、不得改动 date 列
- 标签口径对账    harness 用收盘价独立重算标签，与算法的 label 列逐行比对
- 截断一致性      全量跑 vs 只喂前 t 行跑，前 t 行特征必须完全一致（反前视）
- 预测输出有效性  非空、无 NaN、非常数、长度与测试集一致
- 朴素基线对比    方向准确率不得明显低于「全猜涨」「明日=今日」两个笨办法

没有覆盖到的失败模式：随机切分、切分前重采样、全样本训练……
这些由 harness 自己控制切分与训练数据，结构上就不可能发生。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .. import contract, data, runner
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "correctness"

CHECKS: list[tuple[str, str, str | None]] = [
    ("correctness.row_conservation", "行数守恒", None),
    ("correctness.label_reconcile", "标签口径对账", "评价指标口径与量纲误用"),
    ("correctness.truncation", "截断一致性（反前视）", "全样本统计量泄漏"),
    ("correctness.prediction_validity", "预测输出有效性", "预测输出有效性未校验"),
    ("correctness.naive_baseline", "朴素基线对比", "缺朴素基线导致预测力误判"),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}

TRUNCATION_TOL = 1e-9
LABEL_TOL = 1e-9
BASELINE_MARGIN = 0.03


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
        _check_label_reconcile(v),
        _check_truncation(v),
        _check_prediction_validity(v),
    ]
    results.append(_check_naive_baseline(v, results))
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
                       f"契约要求行数不变，预热期 NaN 行应保留", res.location)
    if "date" in frame.columns:
        same = (pd.to_datetime(frame["date"]).reset_index(drop=True)
                .equals(pd.to_datetime(v.raw["date"]).reset_index(drop=True)))
        if not same:
            return _result(cid, False, "date 列被改动或顺序被打乱", res.location)
        detail = f"{len(frame)} 行进 {len(frame)} 行出，date 列未改动"
    else:
        detail = f"{len(frame)} 行进 {len(frame)} 行出（未保留 date 列）"
    return _result(cid, True, detail, elapsed=res.elapsed)


# -------------------------------------------------------------- 标签口径对账
def _check_label_reconcile(v: "Validator") -> CheckResult:
    cid = "correctness.label_reconcile"
    if v.features is None or not isinstance(v.features, pd.DataFrame):
        return _skip(cid, "特征表不可用")
    if len(v.features) != len(v.raw):
        return _skip(cid, "行数不一致，无法逐行对齐")

    col = contract.LABEL_COLUMN
    if col not in v.features.columns:
        return _result(cid, False, f"特征表缺少标签列 {col}")

    have = pd.to_numeric(v.features[col], errors="coerce").reset_index(drop=True)
    ref = data.reference_label(v.raw, v.manifest.label)
    both = ref.notna() & have.notna()
    if int(both.sum()) < 50:
        return _result(cid, False, f"可逐行比对的标签只有 {int(both.sum())} 个，不足以核对")

    diff = (have[both] - ref[both]).abs()
    max_diff = float(diff.max())
    nan_mismatch = int((ref.isna() != have.isna()).sum())
    spec = v.manifest.label

    if max_diff > LABEL_TOL or nan_mismatch > 0:
        worst = int(diff.idxmax())
        date_txt = str(pd.Timestamp(v.raw["date"].iloc[worst]).date())
        detail = (f"算法 label 与契约口径（{spec.type}, horizon={spec.horizon}）不一致："
                  f"最大偏差 {max_diff:.6g}（{date_txt} 行）；缺失位置不符 {nan_mismatch} 处。"
                  f"harness 独立重算的公式："
                  f"{'log(close.shift(-h)/close)' if spec.type == 'log' else 'close.shift(-h)/close - 1'}")
        return _result(cid, False, detail)
    return _result(cid, True,
                   f"{int(both.sum())} 个标签逐行一致（最大偏差 {max_diff:.2e}），"
                   f"口径 {spec.type} horizon={spec.horizon}")


# ---------------------------------------------------------------- 截断一致性
def _check_truncation(v: "Validator") -> CheckResult:
    cid = "correctness.truncation"
    if v.features is None or not isinstance(v.features, pd.DataFrame):
        return _skip(cid, "全量特征表不可用")
    if len(v.features) != len(v.raw):
        return _skip(cid, "行数不一致，截断比对无意义")

    t = max(60, int(len(v.raw) * 0.7))
    cut = v.raw.iloc[:t].reset_index(drop=True)
    res = runner.run_features(v.algo_dir, v.module_name, cut, v.budget)
    if not res.ok:
        if res.status == "timeout":
            return _result(cid, False, f"只喂前 {t} 行时 build_features 超时")
        return _result(cid, False,
                       f"只喂前 {t} 行时 build_features 报错: {_tail(res.error)}", res.location)

    frame = res.value
    if not isinstance(frame, pd.DataFrame):
        return _result(cid, False, f"截断输入返回 {type(frame).__name__}")
    if len(frame) != t:
        return _result(cid, False,
                       f"截断输入 {t} 行，返回 {len(frame)} 行")

    columns = [c for c in v.features.columns
               if c in frame.columns and c not in (contract.LABEL_COLUMN, "date")]
    extra = [c for c in v.features.columns if c not in frame.columns
             and c not in (contract.LABEL_COLUMN, "date")]
    if extra:
        return _result(cid, False, f"截断输入下缺少列: {', '.join(extra[:5])}")
    if not columns:
        return _result(cid, False, "没有可比对的数值列")

    max_diff, worst = _compare_frames(v.features.iloc[:t], frame, columns)
    if max_diff > TRUNCATION_TOL:
        hint = "；特征疑似使用了全样本统计量（均值/标准差/分位数）或后向窗口" if np.isfinite(max_diff) \
            else "；一侧有值一侧为 NaN"
        return _result(cid, False,
                       f"前 {t} 行特征不一致：{worst} 偏差 {max_diff:.6g}{hint}",
                       elapsed=res.elapsed)
    return _result(cid, True,
                   f"全量与前 {t} 行两次运行，{len(columns)} 列特征逐行一致"
                   f"（最大偏差 {max_diff:.1e}）", elapsed=res.elapsed)


def _compare_frames(a: pd.DataFrame, b: pd.DataFrame,
                    columns: list[str]) -> tuple[float, str | None]:
    max_diff = 0.0
    worst: str | None = None
    for col in columns:
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        if len(av) != len(bv):
            return float("inf"), f"列 {col} 长度不一致"
        both_nan = np.isnan(av) & np.isnan(bv)
        diff = np.abs(av - bv)
        diff[both_nan] = 0.0
        diff[np.isnan(diff)] = np.inf       # 一边 NaN 一边有值 = 不一致
        idx = int(np.argmax(diff))
        if diff[idx] > max_diff:
            max_diff = float(diff[idx])
            worst = f"第 {idx} 行 {col}"
            if not np.isfinite(max_diff):
                worst += "（一侧为 NaN）"
    return max_diff, worst


# ------------------------------------------------------------ 预测输出有效性
def _check_prediction_validity(v: "Validator") -> CheckResult:
    cid = "correctness.prediction_validity"
    if v.split is None:
        return _skip(cid, "训练/样本外切分不可用（特征表或标签列有问题）")
    chain = v.chain
    if chain is None:
        return _skip(cid, "fit/predict 未执行")
    if not chain.ok:
        if chain.status == "timeout":
            return _result(cid, False, f"fit/predict 超过时间预算 {v.budget:.0f}s")
        return _result(cid, False, f"fit/predict 报错: {_tail(chain.error)}", chain.location)

    preds = np.asarray(chain.value, dtype=float)
    n_expected = len(v.split.test_features)
    if preds.ndim != 1:
        return _result(cid, False, f"predict 返回 {preds.ndim} 维数组，契约要求一维")
    if len(preds) != n_expected:
        return _result(cid, False,
                       f"predict 返回 {len(preds)} 个值，样本外 {n_expected} 行，长度不一致")
    n_nan = int(np.isnan(preds).sum())
    if n_nan:
        return _result(cid, False, f"预测值含 {n_nan} 个 NaN")
    n_inf = int(np.isinf(preds).sum())
    if n_inf:
        return _result(cid, False, f"预测值含 {n_inf} 个 inf")
    spread = float(preds.max() - preds.min())
    if spread <= 0:
        return _result(cid, False,
                       f"预测值恒为常数 {preds[0]:.6g}（极差 0），模型未产生有效输出")
    return _result(cid, True,
                   f"{len(preds)} 个预测全部有限，取值范围 [{preds.min():.4g}, {preds.max():.4g}]",
                   elapsed=chain.elapsed)


# ---------------------------------------------------------------- 朴素基线
def _check_naive_baseline(v: "Validator", previous: list[CheckResult]) -> CheckResult:
    cid = "correctness.naive_baseline"
    validity = next((r for r in previous if r.id == "correctness.prediction_validity"), None)
    if validity is None or not validity.passed:
        return _skip(cid, "预测输出无效，不进行比较")

    preds = np.asarray(v.chain.value, dtype=float)
    truth = v.split.test_truth.to_numpy(dtype=float)
    positions = v.split.test_positions
    r1 = v.raw["close"].astype(float).pct_change().to_numpy()
    r1_test = r1[positions]

    mask = ~np.isnan(truth) & ~np.isnan(preds) & ~np.isnan(r1_test)
    if int(mask.sum()) < 30:
        return _skip(cid, f"可用样本仅 {int(mask.sum())} 行，不足以比较方向准确率")

    t = truth[mask]
    p = preds[mask]
    model_acc = float(np.mean(np.sign(p) == np.sign(t)))
    up_acc = float(np.mean(t > 0))
    persist_acc = float(np.mean(np.sign(r1_test[mask]) == np.sign(t)))
    best = max(up_acc, persist_acc)

    detail = (f"方向准确率 模型 {model_acc:.3f} / 全猜涨 {up_acc:.3f} / 明日=今日 {persist_acc:.3f}"
              f"（样本 {int(mask.sum())} 行）")
    if model_acc < best - BASELINE_MARGIN:
        return _result(cid, False,
                       f"{detail}；低于最好基线 {100 * (best - model_acc):.1f} 个百分点，预测力为负")
    return _result(cid, True, detail)


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return "（无错误信息）"
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " ⏎ ".join(parts[-lines:])
