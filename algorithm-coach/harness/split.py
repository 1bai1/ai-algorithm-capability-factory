"""训练段 / 样本外段的切分（由 harness 控制，不由算法控制）。

切分在 harness 手里，意味着「随机切分」「切分前重采样」「把测试集混进训练集」
这几类错误在结构上不可能发生——不是靠检查事后抓，而是根本没机会犯。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    """一次切分的产物。

    - train_df: 训练段（含标签列），供 fit 使用
    - test_features: 样本外段特征（已剥掉标签列），供 predict 使用
    - test_truth: 样本外段标签（真值），只在 harness 手里，用于评分
    - test_positions: 样本外行在原始数据中的位置（0 基），用于对齐价格序列
    """

    train_df: pd.DataFrame
    test_features: pd.DataFrame
    test_truth: pd.Series
    test_positions: np.ndarray

    @property
    def train_dates(self) -> pd.Series | None:
        """时序任务才有 date 列；分类任务返回 None。"""
        return self.train_df["date"] if "date" in self.train_df.columns else None

    @property
    def test_dates(self) -> pd.Series | None:
        return self.test_features["date"] if "date" in self.test_features.columns else None


def split_features(frame: pd.DataFrame, dates: pd.Series, cutoff: pd.Timestamp,
                   label_col: str = "label", min_train: int = 50,
                   min_test: int = 20) -> tuple[Split | None, str | None]:
    """按截止日切分特征表。返回 (Split, 错误说明)。"""
    if len(frame) != len(dates):
        return None, f"特征表 {len(frame)} 行与日期列 {len(dates)} 行不一致"
    if label_col not in frame.columns:
        return None, f"特征表缺少标签列 {label_col}"
    if "date" not in frame.columns:
        return None, "特征表缺少 date 列（切分需要它）"

    frame_dates = pd.to_datetime(frame["date"], errors="coerce")
    if frame_dates.isna().any():
        return None, "特征表的 date 列存在无法解析的值"

    test_mask = (frame_dates >= cutoff).to_numpy()
    label = pd.to_numeric(frame[label_col], errors="coerce")
    feature_cols = [c for c in frame.columns if c not in (label_col, "date")]
    nan_rows = frame[feature_cols].isna().any(axis=1).to_numpy()
    train_mask = (~test_mask) & label.notna().to_numpy() & ~nan_rows

    train_df = frame.loc[train_mask].reset_index(drop=True)
    test_positions = np.where(test_mask)[0]
    test_features = frame.loc[test_mask].drop(columns=[label_col]).reset_index(drop=True)
    test_truth = label.loc[test_mask].reset_index(drop=True)

    if len(train_df) < min_train:
        return None, f"训练段可用行数 {len(train_df)} 少于下限 {min_train}"
    if len(test_features) < min_test:
        return None, f"样本外行数 {len(test_features)} 少于下限 {min_test}"
    return Split(train_df=train_df, test_features=test_features,
                 test_truth=test_truth, test_positions=test_positions), None


def split_classification(df: pd.DataFrame, label_column: str = "label",
                         test_size: float = 0.2, seed: int = 42,
                         min_train: int = 50, min_test: int = 20
                         ) -> tuple[Split | None, str | None]:
    """分类任务的分层随机切分。

    与 split_features（时序）的区别：这里不按时间切，而是**分层随机**切分，
    保证训练集与测试集的类别比例一致——这是文本分类的通行纪律
    （见复用池《文本分类基准数据集与划分纪律》）。

    划分方式与随机种子会原样写进报告 meta：评测口径卡要求「划分方式」必须披露，
    否则不同方法的数字不可比。
    """
    if label_column not in df.columns:
        return None, f"数据缺少标签列 {label_column}"
    if len(df) < min_train + min_test:
        return None, f"样本量 {len(df)} 不足以切分（下限 {min_train + min_test}）"

    from sklearn.model_selection import train_test_split

    y = df[label_column].astype(str)
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None      # 过小的类无法分层

    idx_train, idx_test = train_test_split(
        df.index.to_numpy(), test_size=test_size, random_state=seed, stratify=stratify)

    train_df = df.loc[idx_train].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)
    test_positions = np.asarray(idx_test)
    test_features = test_df.drop(columns=[label_column]).reset_index(drop=True)
    test_truth = test_df[label_column].astype(str).reset_index(drop=True)

    if len(train_df) < min_train:
        return None, f"训练集 {len(train_df)} 行少于下限 {min_train}"
    if len(test_features) < min_test:
        return None, f"测试集 {len(test_features)} 行少于下限 {min_test}"

    missing = sorted(set(y) - set(test_truth.unique()))
    if missing:
        return None, f"测试集缺少类别 {missing}，请增大样本量或取消分层"

    return Split(train_df=train_df, test_features=test_features,
                 test_truth=test_truth, test_positions=test_positions), None
