"""运行层检查：能不能跑通、跑了多久、两次结果是否一致。

对应任务书第 5 条里的「运行稳定性」。
"""

from __future__ import annotations

import numpy as np

from ..context import CheckContext
from .registry import CheckResult, check


@check("runs", "error", "四个契约函数能否依次跑通，产出非空预测")
def check_runs(ctx: CheckContext) -> CheckResult:
    if ctx.predictions is None:
        reason = "流程未产生预测结果"
        if ctx.errors:
            first = ctx.errors[0]
            reason = f"流程在 {first.get('stage', '?')} 阶段中断：" \
                     f"{first.get('error_type', '')}: {first.get('detail', '')}"
        return CheckResult(result="fail", detail=reason)
    if ctx.predictions.size == 0:
        return CheckResult(result="fail", detail="predict 返回了空数组")
    return CheckResult(result="pass", detail=None)


@check("timeout", "error", "整次验证是否在配置的时限内完成",
       requires=("duration_sec",))
def check_timeout(ctx: CheckContext) -> CheckResult:
    limit = ctx.profile.get("timeout_sec")
    if not limit:
        return CheckResult(result="skip", detail="配置未设 timeout_sec，未做时限检查")
    if ctx.duration_sec > limit:
        return CheckResult(
            result="fail",
            detail=f"耗时 {ctx.duration_sec:.1f} 秒，超过时限 {limit} 秒",
            evidence={"duration_sec": ctx.duration_sec, "limit_sec": limit},
        )
    return CheckResult(result="pass", detail=None)


@check("prediction_valid", "error", "预测值是否有效：无 NaN/Inf、非常数、与测试段等长",
       requires=("predictions", "split"))
def check_prediction_valid(ctx: CheckContext) -> CheckResult:
    pred = np.asarray(ctx.predictions, dtype=float).ravel()
    n_test = ctx.split.n_test

    if pred.shape[0] != n_test:
        return CheckResult(
            result="fail",
            detail=f"预测长度 {pred.shape[0]} 与测试段行数 {n_test} 不一致",
            evidence={"n_predictions": int(pred.shape[0]), "n_test": n_test},
        )

    n_bad = int((~np.isfinite(pred)).sum())
    if n_bad:
        return CheckResult(
            result="fail",
            detail=f"预测值含 {n_bad} 个 NaN/Inf。深度模型在序列前 seq_length 个位置"
                   "常产生无效值，须屏蔽后再评估，不能直接送进指标计算",
            evidence={"n_invalid": n_bad},
        )

    # 判"常数"用极差而不是标准差：常数数组的 std 会因浮点累加误差得到一个
    # 极小非零值（曾因此漏报恒常数预测），而 max-min 在真常数时精确为 0。
    spread = float(np.nanmax(pred) - np.nanmin(pred))
    scale = max(abs(float(np.nanmean(pred))), 1e-12)
    if spread <= 1e-9 * scale:
        return CheckResult(
            result="fail",
            detail=f"预测值恒为常数 {float(pred[0])!r}（极差 {spread:.3e}），"
                   "模型没有输出有效信号。常见成因：模型退化成只输出训练集均值、"
                   "或特征全被标准化成同一值",
            evidence={"constant_value": float(pred[0]), "spread": spread},
        )

    return CheckResult(result="pass", detail=None)
