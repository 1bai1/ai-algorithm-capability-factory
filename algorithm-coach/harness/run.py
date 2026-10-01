#!/usr/bin/env python3
"""验证 harness 入口。

用法：
    python harness/run.py --task knowledge/任务池/<任务名>/ [--profile regression]
                          [--solution <path>] [--timeout 300]

产出：
    <task>/validation/report.json   给 agent 读，用于按错误信息修复
    <task>/validation/report.md     给人看，也是任务书要求的「验证报告样例」

退出码：
    0  通过
    1  有 error 级检查未通过（方法或结果有问题）
    2  执行中断（契约不满足、数据取不到、超时等，流程没跑完）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HARNESS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARNESS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness import contract, metrics, report  # noqa: E402
from harness.checks import all_checks, selected  # noqa: E402
from harness.context import CheckContext  # noqa: E402
from harness.splitter import time_split  # noqa: E402
from harness.checks import leakage  # noqa: E402

DEFAULT_PROFILE = HARNESS_DIR / "profiles" / "regression.json"
COST_SCAN_POINTS = (0.0, 0.0005, 0.001, 0.0015, 0.002)


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def load_profile(name: str) -> dict:
    path = DEFAULT_PROFILE if name in ("", "regression") else HARNESS_DIR / "profiles" / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"找不到验证配置: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_task_overrides(profile: dict, task_dir: Path) -> dict:
    """任务目录下的 validate.json 可覆盖配置，用于单任务定制。"""
    override = task_dir / "validate.json"
    if override.is_file():
        profile = {**profile, **json.loads(override.read_text(encoding="utf-8"))}
    return profile


# --------------------------------------------------------------------------
# 交易表现（harness 自己算，不由生成的代码提供）
# --------------------------------------------------------------------------

def compute_trading(predictions: np.ndarray, target_test: np.ndarray,
                    horizon: int, in_sample_predictions: np.ndarray | None = None) -> tuple[dict, dict, dict]:
    """按非重叠周期算策略表现与买入持有基准，并给出摩擦敏感性。

    为什么要非重叠：预测目标是"未来 h 日收益"，逐日计算时相邻样本的收益窗口
    互相重叠，逐日累乘得到的净值曲线是错的。按 h 为步长取样，各期互不重叠，
    累乘才有意义。

    为什么由 harness 算而不是让 solution 算：同一套策略口径下比较不同方案，
    差异才归因于模型本身而不是各自的回测实现——这也是任务书「多个候选算法方案
    自动比较」的前提。

    阈值标定：**在样本内（训练段）取中位数作为持仓阈值，再原样用于测试段。**
    不用固定的 0——知识库「预测阈值转持仓信号」卡明确把"阈值默认 0"列为反例：
    预测值系统性偏负时，固定 0 会让策略几乎全程空仓，交易指标失去意义。
    阈值取自训练段而非测试段，是为了避免"用测试集挑参数"这另一种泄漏。
    """
    step = max(1, int(horizon))
    pred = np.asarray(predictions, dtype=float).ravel()[::step]
    fut = np.asarray(target_test, dtype=float).ravel()[::step]
    mask = np.isfinite(pred) & np.isfinite(fut)
    pred, fut = pred[mask], fut[mask]

    if pred.size == 0:
        return {}, {}, {}

    threshold, threshold_source = _calibrate_threshold(pred, in_sample_predictions)

    # A 股不可做空：预测高于阈值满仓，否则空仓（1/0），不用 1/-1
    position = (pred > threshold).astype(float)
    strat_returns = position * fut

    trades = int(np.abs(np.diff(np.concatenate([[0.0], position]))).sum())
    trading = metrics.summarize_returns(strat_returns)
    trading["trades"] = trades
    trading["periods"] = int(pred.size)
    trading["threshold"] = float(threshold)
    trading["exposure"] = float(position.mean())
    trading["period_note"] = (
        f"按 {step} 日非重叠周期计算（预测目标为未来 {step} 日收益，"
        f"逐日累乘会因窗口重叠而失真）；策略为「预测值高于阈值则满仓、否则空仓」，"
        f"阈值为 {threshold:.6g}（{threshold_source}），持仓占比 {position.mean():.0%}，"
        f"未计交易成本。"
    )

    baseline = metrics.summarize_returns(fut)
    baseline["cum_return"] = metrics.cumulative_return(fut)

    # 摩擦敏感性：用 (1-f)^trades 给策略累计收益打上界，f 为单边成本率
    total = float(np.prod(1.0 + strat_returns))
    rows = [
        {
            "cost_rate": f,
            "trades": trades,
            "capped_return": float(total * (1.0 - f) ** trades - 1.0),
        }
        for f in COST_SCAN_POINTS
    ]
    sensitivity = {
        "note": (
            f"策略在测试段换手 {trades} 次。下表给出不同单边成本率下策略累计收益的"
            f"上界估算（按 (1-f)^换手次数 折算，参照知识库"
            f"「买入持有基准与摩擦敏感性检验」卡的做法）——摩擦率的细微变化即可"
            f"扭转结论时，不应把它当作稳健策略。"
        ),
        "rows": rows,
    }
    return trading, baseline, sensitivity


def _calibrate_threshold(pred: np.ndarray,
                         in_sample: np.ndarray | None) -> tuple[float, str]:
    """持仓阈值：优先用样本内预测的中位数，退回时用测试段中位数并标明。"""
    if in_sample is not None:
        samp = np.asarray(in_sample, dtype=float).ravel()
        samp = samp[np.isfinite(samp)]
        if samp.size:
            return float(np.median(samp)), "取训练段预测中位数"
    return float(np.median(pred)), "训练段预测不可用，退回测试段中位数（口径已注明）"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def run(task_dir: Path, profile: dict, solution_path: Path | None,
        timeout_sec: float | None) -> tuple[CheckContext, dict, str, int]:
    ctx = CheckContext(
        task_dir=str(task_dir),
        started_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        profile=profile,
    )
    t0 = time.monotonic()

    def fail_stage(stage: str, exc: BaseException) -> None:
        ctx.errors.append({
            "stage": stage,
            "error_type": type(exc).__name__,
            "detail": str(exc),
        })

    # --- 1. 定位并导入方案 ---
    if solution_path is None:
        candidates = sorted((task_dir / "generated").glob("*.py"))
        if not candidates:
            fail_stage("locate", FileNotFoundError(f"{task_dir}/generated/ 下没有 .py 文件"))
            return _finish(ctx, t0, timeout_sec)
        solution_path = candidates[0]
    ctx.solution_path = str(solution_path)

    try:
        ctx.module = contract.load_solution(solution_path)
    except Exception as exc:  # noqa: BLE001
        fail_stage("import", exc)
        return _finish(ctx, t0, timeout_sec)

    ctx.interface = contract.check_interface(ctx.module)

    # --- 2. 取数据（由方案提供） ---
    try:
        ctx.df = contract.call_load_data(ctx.module)
    except Exception as exc:  # noqa: BLE001
        fail_stage("load_data", exc)
        return _finish(ctx, t0, timeout_sec)

    # --- 3. 切分（harness 掌控） ---
    try:
        ctx.split = time_split(
            ctx.df,
            train_ratio=profile.get("split", {}).get("train_ratio", 0.8),
            min_test_rows=profile.get("split", {}).get("min_test_rows", 30),
        )
    except Exception as exc:  # noqa: BLE001
        fail_stage("split", exc)
        return _finish(ctx, t0, timeout_sec)

    # --- 4. 特征构造 + 截断一致性 ---
    try:
        ctx.features = contract.call_build_features(ctx.module, ctx.df)
    except Exception as exc:  # noqa: BLE001
        fail_stage("build_features", exc)
        return _finish(ctx, t0, timeout_sec)

    try:
        target_col_for_trunc = profile.get("target_column")
        ignore = (target_col_for_trunc,) if target_col_for_trunc else ()
        ctx.truncation = leakage.truncation_consistency(
            lambda d: contract.call_build_features(ctx.module, d), ctx.df,
            ignore_columns=ignore,
        )
    except Exception as exc:  # noqa: BLE001
        fail_stage("truncation_check", exc)

    # --- 5. 训练与预测（切分结果只按段给出） ---
    target_col = profile.get("target_column")
    if target_col and target_col not in ctx.features.columns:
        fail_stage("target", KeyError(
            f"特征表里找不到目标列 `{target_col}`；实际列：{list(ctx.features.columns)[:12]}"
        ))
        return _finish(ctx, t0, timeout_sec)

    n_train = ctx.split.n_train
    try:
        ctx.fitted = contract.call_fit(ctx.module, ctx.features.iloc[:n_train])
        ctx.predictions = contract.call_predict(
            ctx.module, ctx.fitted, ctx.features.iloc[n_train:]
        )
    except Exception as exc:  # noqa: BLE001
        fail_stage("fit_predict", exc)
        return _finish(ctx, t0, timeout_sec)

    # 样本内预测：只用于标定持仓阈值（知识库要求阈值取自样本内，不得用测试集挑）。
    # 拿不到就退回测试段中位数并在报告里注明，不因它中断整次验证。
    in_sample_predictions = None
    try:
        in_sample_predictions = contract.call_predict(
            ctx.module, ctx.fitted, ctx.features.iloc[:n_train]
        )
    except Exception:  # noqa: BLE001
        pass

    # --- 6. 指标 ---
    if target_col:
        y_test = ctx.features[target_col].to_numpy(dtype=float)[n_train:]
        ctx.test_target = y_test
        ctx.metrics = {
            "rmse": metrics.rmse(y_test, ctx.predictions),
            "mae": metrics.mae(y_test, ctx.predictions),
            "direction_accuracy": metrics.direction_accuracy(y_test, ctx.predictions),
            "n_test": int(ctx.split.n_test),
        }
        horizon = int(profile.get("horizon", 1))
        ctx.trading, ctx.baseline, sensitivity = compute_trading(
            ctx.predictions, y_test, horizon, in_sample_predictions
        )
        if ctx.baseline:
            ctx.baseline["excess_return"] = (
                _nan_safe(ctx.trading.get("cum_return")) - _nan_safe(ctx.baseline.get("cum_return"))
            )
        ctx.profile["_cost_sensitivity"] = sensitivity

    return _finish(ctx, t0, timeout_sec)


def _nan_safe(v):
    return float("nan") if v is None else float(v)


def _finish(ctx: CheckContext, t0: float, timeout_sec: float | None):
    ctx.duration_sec = time.monotonic() - t0
    if timeout_sec:
        ctx.profile.setdefault("timeout_sec", timeout_sec)

    names = ctx.profile.get("checks")
    records = run_checks(ctx, None if names is None else list(names))

    error_failed = any(r["level"] == "error" and r["result"] == "fail" for r in records)
    status = "error" if ctx.errors else ("failed" if error_failed else "passed")
    payload = report.build_payload(ctx, records, status)
    if ctx.profile.get("_cost_sensitivity"):
        payload["cost_sensitivity"] = ctx.profile["_cost_sensitivity"]
    md = report.render_markdown(ctx, payload)

    exit_code = {"passed": 0, "failed": 1, "error": 2}[status]
    return ctx, payload, md, exit_code


def run_checks(ctx: CheckContext, names: list[str] | None) -> list[dict]:
    """逐个跑检查。单个检查抛异常不影响其他检查——记成 error 级失败即可。"""
    records: list[dict] = []
    wanted = all_checks().values() if names is None else selected(names)

    for reg in wanted:
        if reg.requires and not ctx.has(*reg.requires):
            records.append({
                "name": reg.name, "level": reg.level, "description": reg.description,
                "result": "skip", "detail": "前置步骤未完成，无从判断",
                "location": None, "evidence": None,
            })
            continue
        try:
            res = reg.fn(ctx)
            records.append({
                "name": reg.name, "level": reg.level, "description": reg.description,
                "result": res.result, "detail": res.detail,
                "location": res.location, "evidence": res.evidence,
            })
        except Exception as exc:  # noqa: BLE001 — 检查自己出错也要如实记录
            records.append({
                "name": reg.name, "level": reg.level, "description": reg.description,
                "result": "fail",
                "detail": f"检查项自身执行失败：{type(exc).__name__}: {exc}",
                "location": None,
                "evidence": {"traceback": traceback.format_exc(limit=3)},
            })
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证生成的量化算法方案")
    parser.add_argument("--task", required=True, help="任务目录，如 knowledge/任务池/<任务名>/")
    parser.add_argument("--profile", default="regression", help="验证配置名（profiles/ 下）")
    parser.add_argument("--solution", default=None, help="方案文件路径，默认取任务目录 generated/ 下第一个 .py")
    parser.add_argument("--timeout", type=float, default=None, help="整次验证的时限（秒）")
    args = parser.parse_args(argv)

    task_dir = Path(args.task)
    if not task_dir.is_dir():
        print(f"找不到任务目录: {task_dir}", file=sys.stderr)
        return 2

    profile = merge_task_overrides(load_profile(args.profile), task_dir)
    solution_path = Path(args.solution) if args.solution else None

    ctx, payload, md, exit_code = run(task_dir, profile, solution_path, args.timeout)

    json_path, md_path = report.write_reports(
        ctx, payload, md, task_dir / "validation"
    )

    print(f"状态: {payload['status']}　耗时 {payload['duration_sec']} 秒")
    for r in payload["checks"]:
        mark = {"pass": "OK  ", "fail": "FAIL", "skip": "skip"}[r["result"]]
        print(f"  [{mark}] {r['level']:5s} {r['name']}")
    print(f"报告: {json_path}")
    print(f"      {md_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
