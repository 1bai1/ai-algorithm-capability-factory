"""统一验证机制（harness）。

对生成的量化算法做三类检查，对应任务书的三个维度：

- 接口规范   checks/interface.py    提交物、导入、函数签名、冒烟运行
- 功能正确性 checks/correctness.py  行数守恒、标签口径、截断一致性、预测有效性、朴素基线
- 运行稳定性 checks/stability.py    超时、确定性、小样本、坏数据

「指标表现」（精度账 / 回测账 / 摩擦敏感性）是独立模块，暂不在此范围内。

用法::

    python -m harness validate <算法目录> --data <日线csv> [--cutoff 2024-01-01]

算法目录的契约见 harness/contract.py 顶部文档。
"""
__version__ = "0.1.0"
