"""验证报告：结构化 JSON（给修复循环用）+ Markdown 摘要（给人看）。

每条检查记录都带：检查项、结果、说明、出错位置（若有）、出自哪张知识卡（若有）。
JSON 是权威产物——修复循环读它决定改哪里；Markdown 只是它的可读渲染。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

def module_titles() -> dict[str, str]:
    """模块中文标题——来自检查模块自己的声明。

    在函数内 import：checks/* 要 import 本模块的 CheckResult，模块级互相 import
    会成环。调用时两边都已加载完毕，安全。
    """
    from .checks import MODULE_TITLES
    return MODULE_TITLES


def _pct(x: float | None, digits: int = 2, signed: bool = True) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and x == float("inf"):
        return "∞"
    sign = "+" if signed else ""
    return f"{100 * x:{sign}.{digits}f}%"


def _num(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and x == float("inf"):
        return "∞"
    return f"{x:.{digits}f}"

STATUS_PASS = "通过"
STATUS_FAIL = "未通过"
STATUS_SKIP = "跳过"


@dataclass
class CheckResult:
    """单条检查结果。passed=None 表示跳过。"""

    id: str
    module: str
    name: str
    passed: bool | None
    detail: str = ""
    location: dict | None = None
    kb_card: str | None = None
    elapsed: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status_text
        return d

    @property
    def status_text(self) -> str:
        if self.passed is None:
            return STATUS_SKIP
        return STATUS_PASS if self.passed else STATUS_FAIL

    @staticmethod
    def skipped(id: str, module: str, name: str, reason: str,
                kb_card: str | None = None) -> "CheckResult":
        return CheckResult(id=id, module=module, name=name, passed=None,
                           detail=reason, kb_card=kb_card)


@dataclass
class Report:
    meta: dict = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)      # 指标表现的全部数值

    # ---------------------------------------------------------------- 统计
    def by_module(self, module: str) -> list[CheckResult]:
        return [c for c in self.checks if c.module == module]

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.passed is False]

    @property
    def passed_all(self) -> bool:
        return not self.failed

    # ---------------------------------------------------------------- 输出
    def to_dict(self) -> dict:
        modules: dict[str, dict] = {}
        for module, title in module_titles().items():
            items = [c.to_dict() for c in self.by_module(module)]
            if not items:
                continue
            modules[module] = {
                "title": title,
                "passed": all(c["passed"] is not False for c in items),
                "checks": items,
            }
        return {
            "meta": self.meta,
            "modules": modules,
            "metrics": self.metrics,
            "conclusion": {
                "passed": self.passed_all,
                "failed": [c.id for c in self.failed],
                "skipped": [c.id for c in self.checks if c.passed is None],
            },
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def write_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        lines: list[str] = ["# 算法验证报告", ""]

        m = self.meta
        lines += [
            f"- 生成时间：{m.get('timestamp', '-')}",
            f"- 任务类型：{m.get('task', 'classification')}",
            f"- 算法目录：`{m.get('algo_dir', '-')}`",
            f"- 数据文件：`{m.get('data_path', '-')}`",
        ]
        classes = m.get("classes") or []
        lines.append(
            f"- 数据集：{m.get('data_rows', '-')} 行 ｜ {len(classes)} 类 "
            f"（多数类占比 {m.get('majority_share') or 0:.3f}）"
            f"　划分：{m.get('split', '-')}（seed={m.get('seed', '-')}）"
            f"　训练 {m.get('train_rows', '-')} / 测试 {m.get('test_rows', '-')} 行")
        manifest = m.get("manifest")
        if manifest:
            lines.append(f"- manifest：`{json.dumps(manifest, ensure_ascii=False)}`")
        counters = m.get("counters", {})
        lines.append(
            f"- 结论：**{'通过' if self.passed_all else '未通过'}**"
            f"（通过 {counters.get('passed', 0)} / 未通过 {counters.get('failed', 0)}"
            f" / 跳过 {counters.get('skipped', 0)}）")
        lines.append("")

        for module, title in module_titles().items():
            items = self.by_module(module)
            if not items:
                continue
            lines += [f"## {title}", ""]
            if module == "performance" and self.metrics:
                lines += self._metrics_markdown()
            lines += ["| 检查项 | 结果 | 说明 |", "|---|---|---|"]
            for c in items:
                icon = {STATUS_PASS: "✓", STATUS_FAIL: "✗", STATUS_SKIP: "—"}[c.status_text]
                detail = (c.detail or "").replace("\n", " ").replace("|", "\\|")
                if c.elapsed is not None:
                    detail = f"{detail}（{c.elapsed:.1f}s）" if detail else f"{c.elapsed:.1f}s"
                lines.append(f"| {c.name} | {icon} {c.status_text} | {detail} |")
            lines.append("")

        if self.failed:
            lines += ["## 未通过项明细", ""]
            for i, c in enumerate(self.failed, 1):
                lines.append(f"### {i}. {c.name}（`{c.id}`）")
                lines.append("")
                lines.append(f"- 说明：{c.detail or '（无）'}")
                if c.location:
                    lines.append(f"- 位置：`{c.location.get('file')}:{c.location.get('line')}`")
                if c.kb_card:
                    lines.append(f"- 依据知识卡：{c.kb_card}")
                lines.append("")
        else:
            lines += ["## 未通过项明细", "", "无。", ""]

        return "\n".join(lines)

    # ------------------------------------------------------- 指标表现明细
    def _metrics_markdown(self) -> list[str]:
        m = self.metrics
        return self._cls_metrics_markdown(m)

    # ------------------------------------------------------- 指标表现明细
    def _cls_metrics_markdown(self, m: dict) -> list[str]:
        s, p, b, d, o = (m["setup"], m["prediction"], m["baselines"],
                         m["distribution"], m["overfit"])
        ref = b.get("tfidf_logreg", {})
        ref_acc = _num(ref.get("accuracy"), 3) if ref.get("accuracy") is not None else "—"
        ref_f1 = _num(ref.get("macro_f1"), 3) if ref.get("macro_f1") is not None else "—"
        ref_name = ref.get("model") or f"不可用（{ref.get('error', '-')}）"

        lines = [
            f"> 口径：{s['split']}，测试占比 {s['test_size']:.0%}，seed={s['seed']}，"
            f"{len(s['classes'])} 类；多数类 = {s['majority_class']}",
            "",
            "### 精度账",
            "",
            "| 指标 | 算法 | 多数类基线 | 参考基线 |",
            "|---|---|---|---|",
            f"| 准确率 | {p['accuracy']:.3f} | {b['majority']['accuracy']:.3f} | {ref_acc} |",
            f"| 宏 F1 | {p['macro_f1']:.3f} | {b['majority']['macro_f1']:.3f} | {ref_f1} |",
            f"| 测试集行数 | {p['rows']} | — | — |",
            "",
            f"> 参考基线 = {ref_name}（harness 自带，与算法同划分、同文本列）",
            "",
            "### 逐类表现（算法）",
            "",
            "| 类别 | precision | recall | F1 | 样本数 |",
            "|---|---|---|---|---|",
        ]
        for row in p["per_class"]:
            lines.append(f"| {row['label']} | {row['precision']:.3f} | {row['recall']:.3f} "
                         f"| {row['f1']:.3f} | {row['support']} |")
        lines += ["", "### 混淆矩阵（行＝真实，列＝预测）", ""]
        labels = p["confusion_labels"]
        lines.append("| 真实\预测 | " + " | ".join(labels) + " |")
        lines.append("|" + "---|" * (len(labels) + 1))
        for lab, row in zip(labels, p["confusion"]):
            lines.append(f"| **{lab}** | " + " | ".join(str(x) for x in row) + " |")
        lines += ["", "### 类别分布（训练 / 测试）", "", "| 类别 | 训练占比 | 测试占比 | 差异 |",
                  "|---|---|---|---|"]
        for k in sorted(set(d["train"]) | set(d["test"])):
            a, c = d["train"].get(k, 0.0), d["test"].get(k, 0.0)
            lines.append(f"| {k} | {a:.3f} | {c:.3f} | {abs(a - c):.3f} |")
        lines += ["", f"最大占比差异 {100 * d['max_shift']:.1f} 个百分点", "",
                  "### 训练/测试差距", ""]
        if o["gap"] is None:
            lines.append("（取不到训练集预测）")
        else:
            lines += ["| 口径 | 准确率 |", "|---|---|",
                      f"| 训练集 | {o['train_accuracy']:.3f} |",
                      f"| 测试集 | {o['test_accuracy']:.3f} |",
                      f"| 差距 | {100 * o['gap']:+.1f} 个百分点 |"]
        lines.append("")
        return lines
