"""指标实现。

分三类：
  预测精度   rmse / mae / direction_accuracy
  交易绩效   cumulative_return / annual_return / sharpe / max_drawdown / win_rate
  基准       buy_and_hold

约定：所有收益类指标一律用**小数**（0.081 表示 8.1%），不混用百分数。
这条约定来自知识库 `评价指标口径与量纲误用` 卡——原文多处把百分数与小数混用，
导致同一张表里数字差 100 倍。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = _aligned(actual, predicted)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = _aligned(actual, predicted)
    return float(np.mean(np.abs(p - a)))


def direction_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """方向命中率。只比较符号，全零视为未命中。

    参照物是 0.5 而非 1.0：预测方向本身有对有错，拿 1.0 当满分是误读。
    """
    a, p = _aligned(actual, predicted)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.sign(a) == np.sign(p)))


def cumulative_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def annual_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    total = np.prod(1.0 + r)
    if total <= 0:
        return -1.0
    return float(total ** (periods_per_year / r.size) - 1.0)


def annual_volatility(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: np.ndarray, risk_free: float = 0.0,
           periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化夏普。risk_free 为年化无风险利率，默认 0（与知识库案例口径一致）。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - risk_free / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """最大回撤，返回负数（如 -0.12 表示回撤 12%）。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def win_rate(returns: np.ndarray) -> float:
    """盈利期数占比。注意：这是"胜率"的一种口径，不等同于交易胜率。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0))


def profit_loss_ratio(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    wins, losses = r[r > 0], r[r < 0]
    if losses.size == 0 or wins.size == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def price_returns(prices: pd.Series) -> pd.Series:
    """价格序列 → 简单收益率序列（首值为 NaN，调用方自行 dropna）。"""
    s = pd.Series(prices, dtype=float)
    return s.pct_change()


def summarize_returns(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """把一组收益汇总成报告用的交易绩效块。"""
    return {
        "cum_return": cumulative_return(returns),
        "annual_return": annual_return(returns, periods_per_year),
        "annual_volatility": annual_volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "profit_loss_ratio": profit_loss_ratio(returns),
        "n_periods": int(np.isfinite(np.asarray(returns, dtype=float)).sum()),
    }


def _aligned(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    """对齐并剔除任一侧非有限值的样本，避免 NaN 污染指标。"""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"长度不一致: actual={a.shape} predicted={p.shape}")
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]
