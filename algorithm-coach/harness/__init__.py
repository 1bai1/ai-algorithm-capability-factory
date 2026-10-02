"""统一验证机制（harness）。

对生成的文本分类算法做四类检查，对应任务书的四个维度：

- 接口规范   checks/interface.py     提交物、导入、函数签名、交付层、冒烟运行
- 功能正确性 checks/correctness.py   行数守恒、预测有效性、反泄漏、标签对齐
- 指标表现   checks/performance.py   精度账（对标参考基线）、类别分布与划分披露、训练/测试差距
- 运行稳定性 checks/stability.py     超时、确定性、小样本、坏数据

指标数值由 metrics.py 计算（纯函数，便于单测），判定在 checks/performance.py。

用法::

    python -m harness validate <算法目录> --data <文本csv>

算法目录的契约见 harness/contract.py 顶部文档。
"""
__version__ = "0.1.0"
