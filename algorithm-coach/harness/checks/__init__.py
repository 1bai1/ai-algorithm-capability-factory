"""四类检查模块——按类别分文件，每类自带检查清单、判据与依据知识卡。

| 模块 | 文件 | 管什么 |
|---|---|---|
| 接口规范 | `interface.py` | **闸门**：提交物、manifest、导入、签名、冒烟、交付物可独立运行 |
| 功能正确性 | `correctness.py` | 行数守恒、标签列有效性、拟合范围一致性（反泄漏）、预测输出有效性、多数类基线 |
| 指标表现 | `performance.py` | 精度账（对标多数类与参考基线）、类别分布与划分披露、训练/测试差距 |
| 运行稳定性 | `stability.py` | 时间预算、确定性、小样本不崩、坏数据不崩 |

## 加一类检查怎么加

在 `checks/` 下新建一个文件，声明下面这份契约，再把模块加进本文件的
`_MODULES` 一行即可——**harness 内核（`validate.py` / `report.py` / `cli.py`）
不认识任何具体模块**，只按这些约定调度。

    模块契约
    --------
    MODULE      str    模块标识：也是 --modules 的取值，须与检查 id 的前缀一致
    TITLE       str    报告里的中文标题
    ORDER       int    调度顺序，小的先跑（闸门必须最小）
    GATE        bool   这一关不过，后面的模块全部跳过
    NEEDS_CHAIN bool   是否要主流程产物（features / split / chain）
    CHECKS      list   [(检查 id, 中文名, 依据知识卡名或 None), ...]
    run(v)             -> list[CheckResult]      跑这一类检查
    skipped_all(why)   -> list[CheckResult]      整类跳过时用
"""
from __future__ import annotations

from . import correctness, interface, performance, stability

# 注册表：加模块就在这里加一行
_MODULES = (interface, correctness, performance, stability)

MODULES = {m.MODULE: m for m in _MODULES}
ALL_MODULES = tuple(m.MODULE for m in sorted(_MODULES, key=lambda m: m.ORDER))
MODULE_TITLES = {m.MODULE: m.TITLE for m in sorted(_MODULES, key=lambda m: m.ORDER)}
