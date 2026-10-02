"""指标表现检查：跑出来的成绩单好不好。

两类产出：

1. **指标数值**（写进报告的「指标表现」表）：准确率、宏 F1、逐类分数、
   混淆矩阵、多数类基线、参考基线对照、训练/测试差距、类别分布披露；
2. **判定**（过 / 不过）：宏 F1 必须超过多数类基线、不得明显落后于
   harness 自带的参考基线、训练/测试差距要披露且不得过大。

对照纪律：任何"比基线好"的结论都要有**调过的、可比的**基线作陪衬——参考基线是
harness 自带的 TF-IDF + 逻辑回归，同一划分、同一文本列。拿默认值当陪衬不算数。
依据《基线未调优导致虚假提升》。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import metrics
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "performance"
TITLE = "指标表现"
ORDER = 2
GATE = False
NEEDS_CHAIN = True     # 要主流程产物（真跑一遍才有指标）

CHECKS: list[tuple[str, str, str | None]] = [
    ("performance.prediction_quality", "精度账（对标基线）", "基线未调优导致虚假提升"),
    ("performance.class_balance", "类别分布与划分披露", "文本分类评测口径与可复现性"),
    ("performance.overfit_gap", "训练/测试差距", "文本分类评估与交叉验证"),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}

REF_MARGIN = 0.05        # 宏 F1 落后参考基线（TF-IDF+线性）超过这个数就判负
GAP_LIMIT = 0.15         # 训练/测试准确率差距上限
BALANCE_LIMIT = 0.20     # 训练/测试某类占比差异上限


def _result(cid: str, passed: bool, detail: str = "") -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult(id=cid, module=MODULE, name=name, passed=passed,
                       detail=detail, kb_card=card)


def _skip(cid: str, reason: str) -> CheckResult:
    name, card = CHECK_INFO[cid]
    return CheckResult.skipped(cid, MODULE, name, reason, kb_card=card)


def skipped_all(reason: str) -> list[CheckResult]:
    return [_skip(cid, reason) for cid, _, _ in CHECKS]


def run(v: "Validator") -> list[CheckResult]:
    if v.split is None or v.chain is None or not v.chain.ok:
        return [_skip(cid, "主流程未产生测试集预测，跳过") for cid, _, _ in CHECKS]

    label_col = v.manifest.label.column
    text_col = v.manifest.text_column
    truth = v.split.test_truth.to_numpy().astype(str)
    preds = np.asarray(v.chain.value).astype(str)

    acc = metrics.accuracy(truth, preds)
    f1 = metrics.macro_f1(truth, preds)
    per_class = metrics.per_class_scores(truth, preds)
    labels, matrix = metrics.confusion_counts(truth, preds)

    # 基线一：多数类（把训练集的多数类一路猜到底）
    train_labels = v.split.train_df[label_col].astype(str)
    majority_class = train_labels.value_counts().idxmax()
    base_preds = np.full(len(truth), majority_class)
    base_acc = metrics.accuracy(truth, base_preds)
    base_f1 = metrics.macro_f1(truth, base_preds)

    # 基线二：harness 自带的 TF-IDF + 逻辑回归（同一划分、同一文本列）
    ref = _reference_baseline(v, text_col, label_col)

    # 训练/测试差距
    train_acc = None
    if v.insample is not None and v.insample.ok:
        train_acc = metrics.accuracy(
            v.split.train_df[label_col].astype(str),
            np.asarray(v.insample.value).astype(str))

    dist_train = metrics.class_distribution(train_labels)
    dist_test = metrics.class_distribution(truth)
    keys = sorted(set(dist_train) | set(dist_test))
    max_shift = max((abs(dist_train.get(k, 0.0) - dist_test.get(k, 0.0)) for k in keys),
                    default=0.0)

    v.metrics_payload = {
        "setup": {
            "task": "classification",
            "split": "分层随机（stratified random）",
            "test_size": round(len(truth) / max(1, len(v.raw)), 4),
            "seed": v.seed,
            "classes": labels,
            "majority_class": majority_class,
        },
        "prediction": {
            "rows": int(len(truth)),
            "accuracy": acc,
            "macro_f1": f1,
            "per_class": per_class,
            "confusion_labels": labels,
            "confusion": matrix,
        },
        "baselines": {
            "majority": {"class": majority_class, "accuracy": base_acc, "macro_f1": base_f1},
            "tfidf_logreg": ref,
        },
        "distribution": {
            "train": dist_train, "test": dist_test, "max_shift": max_shift,
        },
        "overfit": {
            "train_accuracy": train_acc,
            "test_accuracy": acc,
            "gap": None if train_acc is None else train_acc - acc,
        },
    }

    return [
        _score_check(v.metrics_payload),
        _balance_check(v.metrics_payload),
        _gap_check(v.metrics_payload),
    ]


def _reference_baseline(v: "Validator", text_col: str, label_col: str) -> dict:
    """harness 自带的参考基线：TF-IDF + 逻辑回归，同一划分、同一文本列。

    依据《基线未调优导致虚假提升》：任何"超过基线"的结论都要有调过的、可比的基线作对照，
    不允许拿默认值充当陪衬。
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        if text_col not in v.split.train_df.columns:
            return {"error": f"训练集没有文本列 {text_col}"}
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, C=4.0))
        pipe.fit(v.split.train_df[text_col].astype(str),
                 v.split.train_df[label_col].astype(str))
        ref_pred = pipe.predict(v.split.test_features[text_col].astype(str))
        truth = v.split.test_truth.to_numpy().astype(str)
        return {
            "model": "TF-IDF(1,2) + LogisticRegression(C=4)",
            "accuracy": metrics.accuracy(truth, ref_pred),
            "macro_f1": metrics.macro_f1(truth, ref_pred),
        }
    except Exception as exc:                      # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _score_check(payload: dict) -> CheckResult:
    cid = "performance.prediction_quality"
    p, b = payload["prediction"], payload["baselines"]
    ref = b["tfidf_logreg"]
    detail = (f"准确率 {p['accuracy']:.3f}、宏 F1 {p['macro_f1']:.3f}；"
              f"多数类基线 准确率 {b['majority']['accuracy']:.3f} / 宏 F1 {b['majority']['macro_f1']:.3f}；"
              + (f"参考基线 {ref.get('model')} 准确率 {ref['accuracy']:.3f} / 宏 F1 {ref['macro_f1']:.3f}"
                 if ref.get("accuracy") is not None else f"参考基线不可用（{ref.get('error')}）"))
    problems = []
    if p["macro_f1"] <= b["majority"]["macro_f1"]:
        problems.append("宏 F1 未超过多数类基线")
    if ref.get("macro_f1") is not None and p["macro_f1"] < ref["macro_f1"] - REF_MARGIN:
        problems.append(f"宏 F1 比 TF-IDF+线性参考基线低 {100 * (ref['macro_f1'] - p['macro_f1']):.1f} 个百分点")
    if problems:
        return _result(cid, False, "；".join(problems) + "。" + detail)
    return _result(cid, True, detail)


def _balance_check(payload: dict) -> CheckResult:
    cid = "performance.class_balance"
    d = payload["distribution"]
    detail = (f"划分：分层随机（seed={payload['setup']['seed']}）；"
              f"训练 {'、'.join(f'{k} {v:.2f}' for k, v in sorted(d['train'].items()))} ｜ "
              f"测试 {'、'.join(f'{k} {v:.2f}' for k, v in sorted(d['test'].items()))}；"
              f"最大占比差异 {100 * d['max_shift']:.1f} 个百分点")
    if d["max_shift"] > BALANCE_LIMIT:
        return _result(cid, False, "训练/测试类别分布差异过大（漂移风险）。" + detail)
    return _result(cid, True, detail)


def _gap_check(payload: dict) -> CheckResult:
    cid = "performance.overfit_gap"
    o = payload["overfit"]
    if o["gap"] is None:
        return _skip(cid, "取不到训练集预测，无法比较")
    detail = (f"准确率 训练集 {o['train_accuracy']:.3f} / 测试集 {o['test_accuracy']:.3f}，"
              f"差 {100 * o['gap']:+.1f} 个百分点（上限 {100 * GAP_LIMIT:.0f}）")
    if o["gap"] > GAP_LIMIT:
        return _result(cid, False, "训练/测试差距过大，过拟合风险高。" + detail)
    return _result(cid, True, detail)
