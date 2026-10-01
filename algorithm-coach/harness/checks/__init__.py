"""检查项包。

导入本包即完成内置检查的注册。
"""

from .registry import (  # noqa: F401
    CheckResult,
    all_checks,
    check,
    load_builtin_checks,
    selected,
)

load_builtin_checks()
