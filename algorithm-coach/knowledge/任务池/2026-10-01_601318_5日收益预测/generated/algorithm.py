"""601318（中国平安）未来 5 日收益预测 —— 最终方案：趋势跟踪分类器 → 期望收益。

round1 的教训：把线性回归的正则调大只会把预测进一步压向训练段的负漂移，
样本外方向准确率更差（0.439），样本内外差距反而越过 15 个百分点上限。

最终方案按 `模型假设与适用边界失效`（有缺陷）的修复规则「假设不成立就换族」，
改用 `均线与趋势跟踪策略`（已验证）的模型化路线：价格对均线偏离度 + 量价特征
训练一个方向分类器，再把「上涨概率」按训练段的条件均值映射回期望收益尺度，
使预测值与 harness 的收益标签同量纲。阈值仍由 harness 用训练段预测中位数定。
正则强度 C 由训练段内部的时序 CV 选，不接触样本外数据。

本轮（round3）相对 round2 只做了工程清理：用自写的 TimeSeriesSplit 选 C 循环
替换 GridSearchCV，跳过早段单一类别的折，去掉 sklearn 的 FitFailedWarning。
报告数字与 round2 完全一致（16/20）。

依据卡片：
- `均线与趋势跟踪策略`（已验证）：close_to_ema / 均线偏离度 + 分类模型 → 看涨概率。
- `活牛期货_逻辑回归趋势跟踪策略`（已验证）：LR 在该路线上是有效基线（AUC 0.887）。
- `预测阈值转持仓信号`（已验证）：阈值只能由训练段定、A股 1/0 两态。
- `特征选择方法`（已验证）：TimeSeriesSplit 内层调参，禁止用测试集选特征/参数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 5

FEATURES = [
    "r1", "r5", "r10", "r20", "r60",
    "px_ma5", "px_ma20", "px_ma60",
    "vol20", "vol60",
    "rsi14", "amp", "vol_ratio",
]

CS = (0.01, 0.1, 1.0, 10.0)


class TrendModel:
    """方向分类器 + 训练段条件均值；predict 阶段只读不拟合。"""

    def __init__(self, medians: pd.Series, pipeline, mu_up: float, mu_down: float):
        self.medians = medians
        self.pipeline = pipeline
        self.mu_up = float(mu_up)
        self.mu_down = float(mu_down)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    close = df["close"].astype(float)
    log_close = np.log(close)

    df["r1"] = log_close.diff(1)
    df["r5"] = log_close.diff(5)
    df["r10"] = log_close.diff(10)
    df["r20"] = log_close.diff(20)
    df["r60"] = log_close.diff(60)

    ret1 = df["r1"]
    df["vol20"] = ret1.rolling(20).std()
    df["vol60"] = ret1.rolling(60).std()

    df["px_ma5"] = log_close - np.log(close.rolling(5).mean())
    df["px_ma20"] = log_close - np.log(close.rolling(20).mean())
    df["px_ma60"] = log_close - np.log(close.rolling(60).mean())

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / down.replace(0, np.nan))
    df["rsi14"] = (rsi - 50) / 50

    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    df["amp"] = (high - low) / close

    vma = df["volume"].astype(float).rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = np.log((df["volume"].astype(float) / vma).replace(0, np.nan))

    df["label"] = close.shift(-HORIZON) / close - 1.0
    return df


def _select_C(x: pd.DataFrame, up: pd.Series) -> float:
    """在训练段内部用时序 CV 选正则强度 C。

    早期折可能只有单一类别（全是上涨或全是下跌），这种折直接跳过；
    不使用随机 K 折，也不接触样本外数据。
    """
    cv = TimeSeriesSplit(n_splits=5)
    best_C, best_score = CS[0], -np.inf
    for C in CS:
        scores = []
        for tr_idx, va_idx in cv.split(x):
            y_tr = up.iloc[tr_idx]
            y_va = up.iloc[va_idx]
            if y_tr.nunique() < 2 or y_va.nunique() < 2:
                continue
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
            pipe.fit(x.iloc[tr_idx], y_tr)
            proba = pipe.predict_proba(x.iloc[va_idx])[:, 1]
            scores.append(float(roc_auc_score(y_va, proba)))
        if scores and float(np.mean(scores)) > best_score:
            best_C, best_score = C, float(np.mean(scores))
    return best_C


def fit(train_df: pd.DataFrame) -> TrendModel:
    x = train_df[FEATURES].astype(float)
    medians = x.median()
    x = x.fillna(medians)
    y = train_df["label"].astype(float)

    up = (y > 0).astype(int)
    best_C = _select_C(x, up)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_C, class_weight="balanced", max_iter=2000))
    pipeline.fit(x, up)

    mu_up = float(y[y > 0].mean()) if (y > 0).any() else 0.0
    mu_down = float(y[y <= 0].mean()) if (y <= 0).any() else 0.0
    return TrendModel(medians=medians, pipeline=pipeline,
                      mu_up=mu_up, mu_down=mu_down)


def predict(model: TrendModel, test_df: pd.DataFrame) -> np.ndarray:
    x = test_df[FEATURES].astype(float).fillna(model.medians)
    p = model.pipeline.predict_proba(x)[:, 1]
    return np.asarray(model.mu_down + p * (model.mu_up - model.mu_down), dtype=float)
