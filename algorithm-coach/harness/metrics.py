"""指标计算：准确率、宏 F1、逐类分数、混淆矩阵、类别分布——全部是纯函数。

纯函数的意义：判定逻辑（checks/performance.py）与数值计算分开，数值可以单独测，
判据可以单独改。

口径约定（写进报告，供横向比较）：
- **准确率 / 宏 F1**：宏 F1 对每类先算 F1 再取算术平均，类别不均衡时比准确率更能
  反映真实水平；两类指标都给，避免用单一数字掩盖类别间差异。
- **逐类分数**：precision / recall / F1 / support 四列，support 用来判断小类上的
  指标是否可信（样本太少的类，F1 波动大）。
- **混淆矩阵**：行是真实类别、列是预测类别，用来看错误集中在哪几对类别之间。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------- 分类账（文本分类任务）
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """准确率。类别不平衡时会骗人，必须与宏 F1、多数类基线一起看。"""
    t = np.asarray(y_true).astype(str)
    p = np.asarray(y_pred).astype(str)
    return float(np.mean(t == p))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """宏 F1：各类 F1 的算术平均，不按样本数加权，故对小类敏感。"""
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(y_true).astype(str),
                          np.asarray(y_pred).astype(str),
                          average="macro", zero_division=0))


def per_class_scores(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """逐类的 precision / recall / f1 / support，供混淆矩阵与报告用。"""
    from sklearn.metrics import precision_recall_fscore_support
    t = np.asarray(y_true).astype(str)
    p = np.asarray(y_pred).astype(str)
    labels = sorted(set(t) | set(p))
    pr, rc, f1, sup = precision_recall_fscore_support(
        t, p, labels=labels, zero_division=0)
    return [{"label": lab, "precision": float(a), "recall": float(b),
             "f1": float(c), "support": int(d)}
            for lab, a, b, c, d in zip(labels, pr, rc, f1, sup)]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray
                     ) -> tuple[list[str], list[list[int]]]:
    """混淆矩阵（行=真实，列=预测）。"""
    from sklearn.metrics import confusion_matrix
    t = np.asarray(y_true).astype(str)
    p = np.asarray(y_pred).astype(str)
    labels = sorted(set(t) | set(p))
    matrix = confusion_matrix(t, p, labels=labels)
    return labels, [[int(v) for v in row] for row in matrix]


def class_distribution(values) -> dict[str, float]:
    """类别占比（用于披露训练/测试的分布是否一致）。"""
    s = pd.Series(np.asarray(values).astype(str))
    if not len(s):
        return {}
    share = s.value_counts(normalize=True)
    return {str(k): float(v) for k, v in share.items()}


def majority_share(values) -> float:
    """多数类占比 = 「全猜多数类」基线的准确率。"""
    dist = class_distribution(values)
    return max(dist.values()) if dist else float("nan")
