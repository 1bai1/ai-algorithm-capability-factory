"""泄漏检查：截断一致性。

方案 B 把切分权收归 harness，于是"随机切分""fit 看到测试段""切分前重采样"这几类
泄漏在结构上不可能发生。**但有一类管不住**：

    build_features(df) 内部用了未来信息

例如整表标准化（用了全样本均值方差）、`df['x'] = df['close'].shift(-1)`、
或者任何跨越当前时刻的滚动统计。这类问题在 fit/predict 的分工之外。

**判定方法：截断一致性。**

    F_full  = build_features(df)        # 全量
    F_trunc = build_features(df[:t])    # 只喂前 t 行
    前 t 行的特征必须逐值相同

道理很直接：第 t 行时刻不该知道 t 之后的事，那么把 t 之后的数据抽掉，
第 t 行往前的特征就不该有任何变化。变了，就说明用了未来信息。

不需要静态分析、不需要读懂代码，跑一遍就知道。而且取多个截断点，
可以定位到**从哪一行开始**不一致——这对修复很有用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import CheckContext
from .registry import CheckResult, check

# 默认截断点（占全序列的比例）。取多个点是为了能定位最早出问题的位置。
DEFAULT_TRUNCATION_POINTS = (0.3, 0.5, 0.7)

# 逐值比较的容差。浮点运算同一输入应给同一输出，留一点余量防实现差异。
ATOL = 1e-9


def truncation_consistency(build_features, df: pd.DataFrame,
                           points=DEFAULT_TRUNCATION_POINTS,
                           ignore_columns: tuple[str, ...] = ()) -> dict:
    """对每个截断点比较前 t 行特征是否与全量一致，返回原始观测。

    ignore_columns 用来排除**本就应当前瞻**的列——目标列按定义就是"未来 h 期收益"，
    它在末尾 h 行天然为 NaN，拿它做截断比较会永远误报。排除目标列之后，
    剩下的特征列才是有意义的比较对象。

    不抛异常——把观测结果交回给检查项决定如何判级，便于测试时单独调用。
    """
    full = build_features(df)
    if not isinstance(full, pd.DataFrame):
        return {"error": f"build_features 未返回 DataFrame，实际 {type(full).__name__}"}

    columns = [c for c in full.columns if c not in set(ignore_columns)]
    if not columns:
        return {"error": "排除目标列后没有可比对的列，无法判断是否使用未来信息"}

    ignored_present = [c for c in ignore_columns if c in full.columns]

    observations = []
    for ratio in points:
        t = int(len(df) * ratio)
        if t < 2:
            continue
        try:
            trunc = build_features(df.iloc[:t])
        except Exception as exc:  # noqa: BLE001 — 截断后跑不起来本身就是个信号
            observations.append({
                "cut_index": t,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        observations.append(_compare(full, trunc, t, columns))

    return {
        "n_rows": len(df),
        "compared_columns": len(columns),
        "ignored_columns": ignored_present,
        "points": observations,
    }


def _compare(full: pd.DataFrame, trunc: pd.DataFrame, cut_index: int,
             columns: list[str]) -> dict:
    """比较 full 前 cut_index 行与 trunc 是否逐值一致。"""
    if len(trunc) != cut_index:
        return {
            "cut_index": cut_index,
            "mismatch_count": -1,
            "detail": f"截断调用返回了 {len(trunc)} 行，与输入行数 {cut_index} 不符"
                      "（build_features 必须保持行数）",
        }

    common_cols = [c for c in columns if c in trunc.columns]
    if not common_cols:
        return {
            "cut_index": cut_index,
            "mismatch_count": -1,
            "detail": "全量与截断两次调用的输出列名没有交集",
        }

    left = full.iloc[:cut_index][common_cols]
    right = trunc[common_cols]

    diff_mask = _diff_mask(left, right)
    n_mismatch = int(diff_mask.to_numpy().sum())
    out = {"cut_index": cut_index, "mismatch_count": n_mismatch}

    if n_mismatch:
        # 定位最早出问题的行与列
        rows_hit = np.where(diff_mask.any(axis=1).to_numpy())[0]
        cols_hit = [c for c in common_cols if bool(diff_mask[c].any())]
        out["first_row_offset"] = int(rows_hit[0])
        out["columns"] = cols_hit[:10]
        out["detail"] = (
            f"截断到第 {cut_index} 行后，前 {cut_index} 行中有 {n_mismatch} 个值发生变化；"
            f"最早出现在第 {out['first_row_offset']} 行，涉及列 "
            f"{'、'.join(cols_hit[:5])}"
            + ("…" if len(cols_hit) > 5 else "")
        )
    return out


def _diff_mask(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """逐值比较，NaN 与 NaN 视为相同。"""
    eq = left.eq(right)
    # 两侧同为 NaN 的位置，eq 在 pandas 里为 False，需要单独补上
    both_nan = left.isna() & right.isna()
    return ~(eq | both_nan | _close(left, right))


def _close(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """数值列按容差比较，非数值列一律视为不接近。"""
    out = pd.DataFrame(False, index=left.index, columns=left.columns)
    for col in left.columns:
        a, b = left[col], right[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            out[col] = np.isclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float), atol=ATOL, equal_nan=True
            )
    return out


@check("lookahead", "error",
       "截断一致性：build_features 内是否混入了未来信息", requires=("truncation",))
def check_lookahead(ctx: CheckContext) -> CheckResult:
    obs = ctx.truncation or {}
    if "error" in obs:
        return CheckResult(result="fail", detail=obs["error"])

    points = obs.get("points") or []
    if not points:
        return CheckResult(result="skip", detail="数据太短，未取得有效截断点")

    bad = [p for p in points if p.get("mismatch_count", 0) != 0]
    if not bad:
        return CheckResult(
            result="pass",
            detail=None,
            evidence={"points_checked": len(points)},
        )

    worst = max(bad, key=lambda p: abs(p.get("mismatch_count", 0)))
    return CheckResult(
        result="fail",
        detail=(
            "build_features 使用了未来信息——"
            + worst.get("detail", f"截断点 {worst['cut_index']} 出现不一致")
            + "。第 t 行时刻不该知道 t 之后的事，抽掉 t 之后的数据后前 t 行特征不应改变。"
            "常见成因：整表标准化（用了全样本均值方差）、跨时刻滚动统计、`shift(-1)` 类前视。"
            "修法：把依赖全样本的变换移进 fit()（它只收到训练段），"
            "或改成只用历史窗口的滚动计算。"
        ),
        evidence={"points": points},
        location=None,
    )
