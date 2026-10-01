"""指标计算：精度账、交易账、对照、稳健性——全部是纯函数。

判据（过不过）在 checks/performance.py；这里只负责把数字算准。

口径约定（不写死就没法比较）
----------------------------
- **只在样本外区间计算**；样本内数字仅用于「样本内外差距」披露，不进结论。
- **夏普**：无风险利率取 0，日频标准差用样本标准差（ddof=1），年化乘 √252。
- **按笔 vs 按日**：胜率与盈亏比按「笔」算。持仓 h 天却按日统计，
  会把同一段行情数 h 遍，指标系统性虚高。
- **成本**：单边费率 = 手续费 + 滑点，进出各收一次；买入持有也按同样口径
  收两次，否则对照不成立。
- **非重叠**：信号按持仓期分段，上一段没结束就跳过新信号。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------------------------------------------ 精度账
def rmse(preds: np.ndarray, truth: np.ndarray) -> float:
    """均方根误差。"""
    err = np.asarray(preds, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def direction_accuracy(preds: np.ndarray, truth: np.ndarray) -> float:
    """方向准确率：预测与实际同号的比例。"""
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truth, dtype=float)
    return float(np.mean(np.sign(p) == np.sign(t)))


def up_share(truth: np.ndarray) -> float:
    """「全猜涨」基线的方向准确率 = 实际上涨比例。"""
    t = np.asarray(truth, dtype=float)
    return float(np.mean(t > 0))


def persistence_accuracy(truth: np.ndarray, prev_returns: np.ndarray) -> float:
    """「明日=今日」基线的方向准确率：拿上一日收益的符号当预测。"""
    t = np.asarray(truth, dtype=float)
    r = np.asarray(prev_returns, dtype=float)
    return float(np.mean(np.sign(r) == np.sign(t)))


def information_coefficient(preds: np.ndarray, truth: np.ndarray) -> float | None:
    """IC：预测与实际收益的 Spearman 秩相关。常数预测返回 None。"""
    p = np.asarray(preds, dtype=float)
    t = np.asarray(truth, dtype=float)
    if len(p) < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return None
    from scipy import stats
    value = float(stats.spearmanr(p, t).statistic)
    return None if np.isnan(value) else value


# ------------------------------------------------------------------ 交易账
def build_positions(signal_indices: np.ndarray, horizon: int,
                    n_total: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """把信号日转成持仓数组（非重叠）。

    信号在 t 日收盘产生 → t+1 日建仓，持有 horizon 天。
    上一段持仓没结束就跳过新信号，避免重叠计数。

    返回 (positions, trades)：positions 长度 n_total，取值 0/1；
    trades 是 [(建仓下标, 平仓下标)]，区间为半开 [entry, exit)。
    """
    positions = np.zeros(n_total, dtype=float)
    trades: list[tuple[int, int]] = []
    next_free = 0
    for idx in np.sort(np.asarray(signal_indices, dtype=int)):
        entry = int(idx) + 1
        if entry < next_free or entry >= n_total:
            continue
        exit_ = min(entry + horizon, n_total)
        positions[entry:exit_] = 1.0
        trades.append((entry, exit_))
        next_free = exit_
    return positions, trades


def strategy_returns(close: np.ndarray, positions: np.ndarray,
                     cost_per_side: float) -> np.ndarray:
    """策略日净收益：持仓收益 − 换手成本。

    持仓变化当天收一次单边成本；若最后一天仍持仓，按强制平仓再收一次。
    """
    close = np.asarray(close, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0

    change = np.abs(np.diff(np.concatenate([[0.0], positions])))
    net = positions * ret - change * cost_per_side
    if n and positions[-1] > 0:
        net[-1] -= cost_per_side          # 期末强制平仓
    return net


def equity_stats(net_returns: np.ndarray) -> dict:
    """由日净收益序列算收益性/风险性指标。"""
    net = np.asarray(net_returns, dtype=float)
    n = len(net)
    equity = np.cumprod(1.0 + net)
    cumulative = float(equity[-1] - 1.0) if n else 0.0
    years = n / TRADING_DAYS
    annual = float((1.0 + cumulative) ** (1.0 / years) - 1.0) if years > 0 and cumulative > -1 else None
    vol = float(np.std(net, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else None
    sharpe = None
    if n > 1 and np.std(net, ddof=1) > 0:
        sharpe = float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(TRADING_DAYS))
    running_max = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:] if n else equity
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if n else 0.0
    return {
        "cum_return": cumulative,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "days": n,
        "years": years,
    }


def turnover_annual(positions: np.ndarray, years: float) -> float | None:
    """年化双边换手 = Σ|仓位变化| / 年数。"""
    if years <= 0:
        return None
    change = np.abs(np.diff(np.asarray(positions, dtype=float)))
    return float(change.sum() / years)


def trade_stats(net_returns: np.ndarray, trades: list[tuple[int, int]]) -> dict:
    """按笔统计：笔数、胜率、盈亏比。

    每笔收益 = 该持仓区间内日净收益的复利，成本已含在内。
    """
    gross = np.asarray(net_returns, dtype=float)
    per_trade = []
    for entry, exit_ in trades:
        seg = gross[entry:exit_]
        if len(seg):
            per_trade.append(float(np.prod(1.0 + seg) - 1.0))
    wins = [r for r in per_trade if r > 0]
    losses = [r for r in per_trade if r < 0]
    win_rate = float(len(wins) / len(per_trade)) if per_trade else None
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(np.abs(losses))) if losses else None
    if avg_win is None:
        pl_ratio = None
    elif avg_loss in (None, 0.0):
        pl_ratio = float("inf")
    else:
        pl_ratio = float(avg_win / avg_loss)
    return {
        "trades": len(per_trade),
        "win_rate": win_rate,
        "profit_loss_ratio": pl_ratio,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


# ---------------------------------------------------------------- 对照/基准
def buy_hold_stats(close: np.ndarray, start: int, end: int,
                   cost_per_side: float) -> dict:
    """买入持有对照：在 start 收盘买入、end 收盘卖出（两端各收一次成本）。"""
    close = np.asarray(close, dtype=float)
    window = close[start:end + 1]
    if len(window) < 2:
        return {}
    gross = float(window[-1] / window[0] - 1.0)
    net = gross - 2.0 * cost_per_side
    ret = np.diff(window) / window[:-1]
    vol = float(np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else None
    sharpe = None
    if len(ret) > 1 and np.std(ret, ddof=1) > 0:
        sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(TRADING_DAYS))
    years = len(window) / TRADING_DAYS
    annual = float((1.0 + net) ** (1.0 / years) - 1.0) if years > 0 and net > -1 else None
    running_max = np.maximum.accumulate(window)
    max_drawdown = float((window / running_max - 1.0).min())
    return {
        "cum_return": net,
        "gross_return": gross,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "days": len(window),
    }


def cost_scan(close: np.ndarray, positions: np.ndarray, base_cost: float,
              multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
              start: int = 0) -> list[dict]:
    """成本敏感性：把单边成本乘以倍数重算累计收益。"""
    out = []
    for m in multipliers:
        net = strategy_returns(close, positions, base_cost * m)[start:]
        stats = equity_stats(net)
        out.append({
            "multiplier": float(m),
            "cost_per_side": float(base_cost * m),
            "cum_return": stats["cum_return"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })
    return out


def alignment_mask(*arrays: np.ndarray) -> np.ndarray:
    """去掉任一数组里出现 NaN 的位置。"""
    mask = None
    for arr in arrays:
        ok = ~pd.isna(np.asarray(arr, dtype=float))
        mask = ok if mask is None else (mask & ok)
    return mask if mask is not None else np.array([], dtype=bool)
