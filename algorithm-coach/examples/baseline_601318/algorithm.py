"""601318（中国平安）5 日收益预测——按 harness 契约整理的基线算法。

特征与模型取自 pi-main/solution/gupiao/predict.py（14 个只用过去信息的
技术特征 + 标准化 Ridge），按 harness 契约重写为三个函数：

- build_features: 行数不变、date 列不动，新增特征列与 label 列
  （label = close.shift(-5)/close - 1，与 manifest 声明一致）
- fit: 只用传入的训练段；中位数填充与标准化都只在训练段上拟合
- predict: 只吃特征列，长度与 test_df 行数一致

这份样例同时充当 harness 的标定样本：一个「做法规矩但不作弊」的算法，
应当在三类检查上全过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 5          # 预测未来 5 个交易日
RIDGE_ALPHA = 10.0

# 全部为 t 时刻及之前可得的信息（滚动窗口/差分），不含全样本统计量
FEATURES = [
    "r1", "r5", "r10", "r20", "r60",
    "vol20", "vol60", "vol_ratio",
    "px_ma5", "px_ma20", "px_ma60",
    "rsi14", "amp",
]


class RidgeModel:
    """把训练段拟合出的中位数与标准化器一起打包，predict 阶段只读不拟合。"""

    def __init__(self, medians: pd.Series, pipeline):
        self.medians = medians
        self.pipeline = pipeline


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()                      # 行数不变：预热期只留 NaN，不删行
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

    vma = df["volume"].astype(float).rolling(20).mean().replace(0, np.nan)
    df["vol_ratio"] = np.log((df["volume"].astype(float) / vma).replace(0, np.nan))

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

    df["label"] = close.shift(-HORIZON) / close - 1.0
    return df


def fit(train_df: pd.DataFrame) -> RidgeModel:
    x = train_df[FEATURES].astype(float)
    medians = x.median()                       # 只在训练段上统计
    x = x.fillna(medians)
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    pipeline.fit(x, train_df["label"].astype(float))
    return RidgeModel(medians=medians, pipeline=pipeline)


def predict(model: RidgeModel, test_df: pd.DataFrame) -> np.ndarray:
    x = test_df[FEATURES].astype(float).fillna(model.medians)
    return np.asarray(model.pipeline.predict(x), dtype=float)
