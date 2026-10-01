"""指标表现检查：跑出来的成绩单好不好。

两类产出：

1. **指标数值**（写进报告的「指标表现」表）：精度账 3 个、交易账 7 个、
   对照 2 个、稳健性 2 个，外加年化波动率与交易笔数两个辅助数字；
2. **判定**（过 / 不过）：精度必须不劣于基线、交易必须胜买入持有、
   回撤不得失控、成本翻倍后仍要胜、样本内外差距要披露。

策略规则由 harness 固定，算法只负责给预测值：
    阈值 = 训练段预测的中位数（只能用样本内信息定，避免同一样本上调参）
    信号 = 预测 > 阈值 → 次日建仓，持有 horizon 天，按持仓期分段不重叠
    成本 = 单边（手续费+滑点），进出各一次；买入持有同口径收两次
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import metrics
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "performance"

CHECKS: list[tuple[str, str, str | None]] = [
    ("performance.prediction_quality", "精度账（对标基线）", "缺朴素基线导致预测力误判"),
    ("performance.trading_vs_benchmark", "交易账（对标买入持有）", "回测无买入持有基准与风险指标"),
    ("performance.drawdown_constraint", "回撤约束（对标买入持有）", "回测无买入持有基准与风险指标"),
    ("performance.cost_sensitivity", "成本敏感度（×2 后仍胜基准）", "回测无交易成本与滑点"),
    ("performance.overfit_gap", "样本内外差距披露", "无独立验证集与样本外评估"),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}

DRAWDOWN_TOLERANCE = 1.5      # 策略最大回撤不得超过买入持有的 1.5 倍
OVERFIT_GAP_LIMIT = 0.15      # 样本内外方向准确率差距上限（15 个百分点）


def _result(cid: str, passed: bool, detail: str = "", location=None) -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult(id=cid, module=MODULE, name=name, passed=passed,
                       detail=detail, location=location, kb_card=card)


def _skip(cid: str, reason: str) -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult.skipped(cid, MODULE, name, reason, kb_card=card)


def skipped_all(reason: str) -> list[CheckResult]:
    return [_skip(cid, reason) for cid, _, _ in CHECKS]


def _pct(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{100 * x:+.{digits}f}%"


def _plain(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x:.{digits}f}"


def run(v: "Validator") -> list[CheckResult]:
    if v.split is None or v.chain is None or not v.chain.ok:
        return skipped_all("主流程未产生样本外预测，跳过")

    preds = np.asarray(v.chain.value, dtype=float)
    truth = v.split.test_truth.to_numpy(dtype=float)
    close = v.raw["close"].astype(float).to_numpy()
    window_start = int(v.split.test_positions[0])
    horizon = v.manifest.label.horizon
    cost = v.cost

    # ---- 精度账（不需要阈值，只要有预测与真值）
    mask = metrics.alignment_mask(preds, truth)
    preds_ok, truth_ok = preds[mask], truth[mask]
    r1 = v.raw["close"].astype(float).pct_change().to_numpy()[v.split.test_positions][mask]
    rmse_model = metrics.rmse(preds_ok, truth_ok)
    rmse_zero = float(np.sqrt(np.mean(truth_ok ** 2)))
    acc_model = metrics.direction_accuracy(preds_ok, truth_ok)
    acc_up = metrics.up_share(truth_ok)
    acc_persist = metrics.persistence_accuracy(truth_ok, r1)
    ic = metrics.information_coefficient(preds_ok, truth_ok)

    # ---- 样本内预测（用于过拟合差距披露）：同一模型在训练段上的表现
    insample_acc = insample_rmse = None
    if v.insample is not None and v.insample.ok:
        ins_pred = np.asarray(v.insample.value, dtype=float)
        ins_truth = v.split.train_df["label"].to_numpy(dtype=float)
        ins_mask = metrics.alignment_mask(ins_pred, ins_truth)
        if int(ins_mask.sum()) >= 30:
            insample_acc = metrics.direction_accuracy(ins_pred[ins_mask], ins_truth[ins_mask])
            insample_rmse = metrics.rmse(ins_pred[ins_mask], ins_truth[ins_mask])

    payload: dict = {
        "setup": {
            "horizon": horizon,
            "cost_per_side": cost,
            "window_start": window_start,
            "rule": "预测 > 训练段预测中位数 → 次日建仓，持有 horizon 天，非重叠",
        },
        "prediction": {
            "rows": int(mask.sum()),
            "rmse": rmse_model,
            "rmse_zero_baseline": rmse_zero,
            "direction_accuracy": acc_model,
            "baseline_up": acc_up,
            "baseline_persistence": acc_persist,
            "ic": ic,
        },
    }

    results = [_check_prediction_quality(payload)]

    # ---- 交易账需要阈值：只用训练段预测的中位数
    if v.insample is None or not v.insample.ok:
        reason = "取不到训练段预测，无法按样本内信息定阈值（不用默认 0 阈值）"
        results += [_skip(cid, reason) for cid, _, _ in CHECKS[1:]]
        v.metrics_payload = payload
        return results

    threshold = float(np.median(np.asarray(v.insample.value, dtype=float)))
    payload["setup"]["threshold"] = threshold

    signal_idx = v.split.test_positions[preds > threshold]
    positions, trades = metrics.build_positions(signal_idx, horizon, len(close))
    net_full = metrics.strategy_returns(close, positions, cost)
    net = net_full[window_start:]
    stats = metrics.equity_stats(net)
    trades_info = metrics.trade_stats(net_full, trades)
    turnover = metrics.turnover_annual(positions[window_start:], stats["years"])
    benchmark = metrics.buy_hold_stats(close, window_start, len(close) - 1, cost)
    scan = metrics.cost_scan(close, positions, cost, start=window_start)

    excess = stats["cum_return"] - benchmark["cum_return"]
    sharpe_diff = (stats["sharpe"] - benchmark["sharpe"]
                   if stats["sharpe"] is not None and benchmark["sharpe"] is not None else None)

    payload.update({
        "trading": {**{k: stats[k] for k in
                       ("cum_return", "annual_return", "annual_vol", "sharpe",
                        "max_drawdown", "days", "years")},
                    **trades_info,
                    "turnover_annual": turnover},
        "benchmark": {"buy_hold": benchmark,
                      "excess_return": excess,
                      "sharpe_diff": sharpe_diff},
        "robustness": {
            "cost_scan": scan,
            "in_sample": {"direction_accuracy": insample_acc, "rmse": insample_rmse},
            "out_of_sample": {"direction_accuracy": acc_model, "rmse": rmse_model},
            "gap": {
                "direction_accuracy": (None if insample_acc is None
                                       else insample_acc - acc_model),
                "rmse": (None if insample_rmse is None else insample_rmse - rmse_model),
            },
        },
    })
    v.metrics_payload = payload

    results += [
        _check_trading(stats, benchmark, trades_info, turnover, excess, sharpe_diff),
        _check_drawdown(stats, benchmark),
        _check_cost(scan, benchmark),
        _check_overfit(insample_acc, acc_model, insample_rmse, rmse_model),
    ]
    return results


# ---------------------------------------------------------------- 各项判定
def _check_prediction_quality(payload: dict) -> CheckResult:
    cid = "performance.prediction_quality"
    p = payload["prediction"]
    best_base = max(p["baseline_up"], p["baseline_persistence"])
    detail = (f"RMSE {p['rmse'] * 100:.2f}%（零预测基线 {p['rmse_zero_baseline'] * 100:.2f}%）；"
              f"方向准确率 {p['direction_accuracy']:.3f}，"
              f"基线 全猜涨 {p['baseline_up']:.3f} / 明日=今日 {p['baseline_persistence']:.3f}；"
              f"IC {_plain(p['ic'])}（样本 {p['rows']} 行）")
    problems = []
    if p["rmse"] > p["rmse_zero_baseline"]:
        problems.append("RMSE 高于零预测基线")
    if p["direction_accuracy"] < best_base:
        problems.append(f"方向准确率低于最好基线 {100 * (best_base - p['direction_accuracy']):.1f} 个百分点")
    if problems:
        return _result(cid, False, "；".join(problems) + "。" + detail)
    return _result(cid, True, detail)


def _check_trading(stats: dict, benchmark: dict, trades_info: dict,
                   turnover: float | None, excess: float,
                   sharpe_diff: float | None) -> CheckResult:
    cid = "performance.trading_vs_benchmark"
    detail = (
        f"策略 累计 {_pct(stats['cum_return'])}、年化 {_pct(stats['annual_return'])}、"
        f"夏普 {_plain(stats['sharpe'], 2)}、最大回撤 {_pct(stats['max_drawdown'])}、"
        f"胜率 {_plain(trades_info['win_rate'])}、盈亏比 {_plain(trades_info['profit_loss_ratio'], 2)}、"
        f"年化换手 {_plain(turnover, 0)}、{trades_info['trades']} 笔 ｜ "
        f"买入持有 累计 {_pct(benchmark['cum_return'])}、夏普 {_plain(benchmark['sharpe'], 2)}、"
        f"最大回撤 {_pct(benchmark['max_drawdown'])} ｜ "
        f"超额 {_pct(excess)}、夏普差 {_plain(sharpe_diff, 2)}"
    )
    problems = []
    if stats["cum_return"] <= benchmark["cum_return"]:
        problems.append("累计收益未跑赢买入持有")
    if (stats["sharpe"] is not None and benchmark["sharpe"] is not None
            and stats["sharpe"] <= benchmark["sharpe"]):
        problems.append("夏普未跑赢买入持有")
    if problems:
        return _result(cid, False, "；".join(problems) + "。" + detail)
    return _result(cid, True, detail)


def _check_drawdown(stats: dict, benchmark: dict) -> CheckResult:
    cid = "performance.drawdown_constraint"
    mdd, bench_mdd = stats["max_drawdown"], benchmark["max_drawdown"]
    limit = DRAWDOWN_TOLERANCE * abs(bench_mdd)
    detail = (f"策略最大回撤 {_pct(mdd)}，买入持有 {_pct(bench_mdd)}，"
              f"上限 {_pct(-limit)}（基准的 {DRAWDOWN_TOLERANCE:g} 倍）")
    if abs(bench_mdd) < 1e-9:
        return _result(cid, True, detail + "；基准无回撤，不设约束")
    if abs(mdd) > limit:
        return _result(cid, False, f"回撤超过上限 {_pct(-(abs(mdd) - limit))}。" + detail)
    return _result(cid, True, detail)


def _check_cost(scan: list[dict], benchmark: dict) -> CheckResult:
    cid = "performance.cost_sensitivity"
    parts = [f"×{s['multiplier']:g}（单边 {100 * s['cost_per_side']:.3f}%）"
             f" {_pct(s['cum_return'])}" for s in scan]
    base = f"买入持有 {_pct(benchmark['cum_return'])}"
    detail = "；".join(parts) + " ｜ " + base
    double = next((s for s in scan if s["multiplier"] == 2.0), None)
    if double is None:
        return _skip(cid, "成本扫描缺少 ×2 档")
    if double["cum_return"] <= benchmark["cum_return"]:
        return _result(cid, False, f"成本翻倍后累计收益 {_pct(double['cum_return'])}"
                                   f"已不敌买入持有。" + detail)
    return _result(cid, True, detail)


def _check_overfit(in_acc: float | None, out_acc: float | None,
                   in_rmse: float | None, out_rmse: float | None) -> CheckResult:
    cid = "performance.overfit_gap"
    if in_acc is None or out_acc is None:
        return _skip(cid, "取不到样本内预测，无法比较")
    gap = in_acc - out_acc
    detail = (f"方向准确率 样本内 {in_acc:.3f} / 样本外 {out_acc:.3f}，差 {100 * gap:+.1f} 个百分点"
              f"（上限 {100 * OVERFIT_GAP_LIMIT:.0f}）；"
              f"RMSE 样本内 {100 * (in_rmse or 0):.2f}% / 样本外 {100 * (out_rmse or 0):.2f}%")
    if gap > OVERFIT_GAP_LIMIT:
        return _result(cid, False, f"样本内外差距过大，过拟合风险高。" + detail)
    return _result(cid, True, detail)
