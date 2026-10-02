"""训练集 / 测试集的切分（由 harness 控制，不由算法控制）。

切分在 harness 手里，意味着「随机切分」「切分前重采样」「把测试集混进训练集」
这几类错误在结构上不可能发生——不是靠检查事后抓，而是根本没机会犯。

分类任务用**分层随机**切分，保证训练集与测试集的类别比例一致——这是文本分类的
通行纪律（见复用池《文本分类基准数据集与划分纪律》）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    """一次切分的产物。

    - train_df: 训练集（含标签列），供 fit 使用
    - test_features: 测试集特征（已剥掉标签列），供 predict 使用
    - test_truth: 测试集标签（真值），只在 harness 手里，用于评分
    - test_positions: 测试行在原始数据中的位置（0 基），用于对齐行号
    """

    train_df: pd.DataFrame
    test_features: pd.DataFrame
    test_truth: pd.Series
    test_positions: np.ndarray


def split_classification(df: pd.DataFrame, label_column: str = "label",
                         test_size: float = 0.2, seed: int = 42,
                         min_train: int = 50, min_test: int = 20
                         ) -> tuple[Split | None, str | None]:
    """分类任务的分层随机切分。

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
