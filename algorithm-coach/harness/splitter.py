"""时间切分。

这一层单独成模块，是因为它承担着整套验证里最关键的一条设计：
**切分由 harness 掌控，生成的代码碰不到。**

只要 fit 只拿得到训练段、predict 只拿得到测试段，"标准化用全量数据拟合"
这类泄漏就不可能发生——它不需要被"检查出来"，而是根本没有发生的路径。
这比事后扫描代码去猜可靠得多，也是本设计选方案 B 的理由。

与之配套的还有 `checks/leakage.py` 里的截断一致性检查，用来补上切分管不住的
那一类泄漏（build_features 内部用了未来信息）。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_TRAIN_RATIO = 0.8


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    ratio: float
    boundary_index: object  # 测试段的第一个索引，用于报错与定位

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def as_dict(self) -> dict:
        return {
            "method": "time",
            "train_ratio": self.ratio,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "boundary_index": str(self.boundary_index),
            "train_range": [str(self.train.index[0]), str(self.train.index[-1])],
            "test_range": [str(self.test.index[0]), str(self.test.index[-1])],
        }


def time_split(df: pd.DataFrame, train_ratio: float = DEFAULT_TRAIN_RATIO,
               min_test_rows: int = 30) -> Split:
    """按时间顺序切分，前 ratio 为训练段，其余为测试段。

    **不做 shuffle，不做随机。** 时序任务用随机切分会把未来信息混进训练集，
    这是知识库 `时序任务使用随机切分与随机K折` 卡记录的典型踩坑。

    调用方须保证 df 已按时间升序。若索引不是单调递增，会直接报错而不是默默纠正——
    索引乱序往往意味着上游数据处理有问题，掩盖它只会让错误更深。
    """
    if df.empty:
        raise ValueError("输入为空表，无法切分")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "索引不是单调递增：时序切分要求数据已按时间升序。"
            "请先排序——不要用随机切分，也不要让 harness 替你掩盖顺序问题。"
        )
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio 必须在 (0,1) 内，实际 {train_ratio}")

    n_train = int(len(df) * train_ratio)
    n_test = len(df) - n_train
    if n_test < min_test_rows:
        raise ValueError(
            f"测试段只有 {n_test} 行，少于下限 {min_test_rows}。"
            "样本太少时样本外指标没有意义——要么加长数据区间，要么调大 train_ratio 的分母。"
        )

    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    return Split(train=train, test=test, ratio=train_ratio, boundary_index=test.index[0])
