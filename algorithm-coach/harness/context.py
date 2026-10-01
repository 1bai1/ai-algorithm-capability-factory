"""检查上下文：一次验证运行中，各步骤产生的中间结果。

检查项通过它读取需要的东西，而不必自己去跑流程——流程由 run.py 统一编排，
这样每个检查只负责"判断"，不负责"准备"。字段是逐步填上的：跑到哪一步，
哪些字段就有值；尚未填上的字段对应的检查会被记为 skip。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .splitter import Split


@dataclass
class CheckContext:
    task_dir: str = ""
    solution_path: str = ""
    profile: dict = field(default_factory=dict)

    # 逐步填充
    module: Any = None                # 已导入的 solution 模块
    interface: dict | None = None     # contract.check_interface 的结果
    df: pd.DataFrame | None = None    # load_data() 的原始输出
    split: Split | None = None
    features: pd.DataFrame | None = None
    truncation: dict | None = None    # 截断一致性检查的原始观测
    fitted: Any = None
    predictions: np.ndarray | None = None
    test_target: np.ndarray | None = None
    metrics: dict | None = None
    trading: dict | None = None
    baseline: dict | None = None

    # 由 run.py 维护的运行信息
    started_at: str = ""
    duration_sec: float = 0.0
    errors: list[dict] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        """这些字段是否都已填充。供检查项的 requires 声明使用。"""
        return all(getattr(self, n, None) is not None for n in names)
