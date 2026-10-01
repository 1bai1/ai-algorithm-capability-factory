"""检查项注册表。

新增一个检查 = 写一个函数 + 挂一个装饰器，不用改 run.py。
任务书进阶要求里的「可扩展」就落在这里。

约定：每个检查函数签名统一为

    def check_xxx(ctx: CheckContext) -> CheckResult

检查作者只负责"判断"，不负责"记录"——异常会由 run.py 统一兜成 error 级结果，
所以单个检查写错不会让整次验证崩掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

LEVELS = ("error", "warn")
RESULTS = ("pass", "fail", "skip")

_registry: dict[str, "Registered"] = {}


@dataclass
class CheckResult:
    result: str                      # pass | fail | skip
    detail: str | None = None        # 失败原因，要写成"哪里、为什么、怎么改"
    location: dict | None = None     # {"file": "solution.py", "line": 42}
    evidence: dict | None = None     # 支撑判断的具体数字，便于人核查


@dataclass
class Registered:
    name: str
    level: str
    description: str
    fn: Callable[[Any], CheckResult]
    requires: tuple[str, ...] = field(default_factory=tuple)


def check(name: str, level: str, description: str, requires: tuple[str, ...] = ()):
    """把一个函数注册成检查项。

    requires 声明依赖上下文里的哪些字段；缺失时该检查记为 skip 而非 fail——
    上游步骤没跑通（比如 load_data 就失败了），下游检查无从判断，不该算它失败。
    """
    if level not in LEVELS:
        raise ValueError(f"level 必须是 {LEVELS} 之一，实际 {level!r}")

    def deco(fn: Callable[[Any], CheckResult]) -> Callable[[Any], CheckResult]:
        if name in _registry:
            raise ValueError(f"检查项重名: {name}")
        _registry[name] = Registered(
            name=name, level=level, description=description, fn=fn, requires=requires
        )
        return fn

    return deco


def all_checks() -> dict[str, Registered]:
    return dict(_registry)


def selected(names: list[str] | None = None) -> list[Registered]:
    """按配置挑检查项。names 为 None 时返回全部，保持注册顺序。"""
    if names is None:
        return list(_registry.values())
    missing = [n for n in names if n not in _registry]
    if missing:
        raise ValueError(f"配置里引用了不存在的检查项: {missing}。已注册的有 {sorted(_registry)}")
    return [_registry[n] for n in names]


def load_builtin_checks() -> None:
    """导入各检查模块，触发装饰器注册。

    只有三个模块——这是有意的。方案 B（切分由 harness 掌控）使若干原计划的
    检查项在结构上不再成立：

      * 随机切分：solution 拿不到切分权，不存在
      * fit 侧的全量泄漏：fit 只收到训练段，不存在
      * 切分前重采样：契约要求 build_features 保持行数，越界即报错

    这些不是"检查通过"，而是"没有发生的路径"，因此记在 README 的
    「结构性保证」一节，不伪装成检查项。真正需要检查的是切分管不到的那一类——
    build_features 内部使用未来信息，见 leakage.py。
    """
    from . import interface, leakage, runtime  # noqa: F401
