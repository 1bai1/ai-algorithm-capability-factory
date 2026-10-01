"""接口规范检查。

对应任务书第 5 条明确列出的「接口规范检查」。

这一项和别处不同：它不是"发现问题"，而是"决定后面还能不能跑"。
契约不满足时，后面所有检查都无从进行。
"""

from __future__ import annotations

from ..context import CheckContext
from .registry import CheckResult, check


@check("interface", "error", "四个必需函数齐备、参数个数符合契约", requires=("interface",))
def check_interface(ctx: CheckContext) -> CheckResult:
    info = ctx.interface or {}
    missing = info.get("missing") or []
    bad_arity = info.get("bad_arity") or []

    if not missing and not bad_arity:
        return CheckResult(result="pass", detail=None)

    parts: list[str] = []
    if missing:
        parts.append("缺少函数：" + "、".join(missing))
    if bad_arity:
        parts.append(
            "参数个数不符："
            + "；".join(
                f"{b['name']} 期望 {b['expected_args']} 个位置参数，实际为 {b['found_signature']}"
                for b in bad_arity
            )
        )

    return CheckResult(
        result="fail",
        detail=(
            "接口契约不满足——"
            + "。".join(parts)
            + "。契约要求（见 harness/README.md）："
            "load_data() / build_features(df) / fit(train_df) / predict(fitted, test_df)"
            "。切分由 harness 掌控，因此这四个函数的职责边界是强制的，不是建议。"
        ),
        evidence={"missing": missing, "bad_arity": bad_arity},
    )
