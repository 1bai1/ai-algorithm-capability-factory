"""验证报告：结构化 JSON（给修复循环用）+ Markdown 摘要（给人看）。

每条检查记录都带：检查项、结果、说明、出错位置（若有）、出自哪张知识卡（若有）。
JSON 是权威产物——修复循环读它决定改哪里；Markdown 只是它的可读渲染。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODULE_TITLES = {
    "interface": "接口规范",
    "correctness": "功能正确性",
    "performance": "指标表现",
    "stability": "运行稳定性",
}


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
        for module, title in MODULE_TITLES.items():
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
            f"- 任务类型：{m.get('task', 'time_series')}",
            f"- 算法目录：`{m.get('algo_dir', '-')}`",
            f"- 数据文件：`{m.get('data_path', '-')}`",
        ]
        if m.get("task") == "classification":
            classes = m.get("classes") or []
            lines.append(
                f"- 数据集：{m.get('data_rows', '-')} 行 ｜ {len(classes)} 类 "
                f"（多数类占比 {m.get('majority_share') or 0:.3f}）"
                f"　划分：{m.get('split', '-')}（seed={m.get('seed', '-')}）"
                f"　训练 {m.get('train_rows', '-')} / 测试 {m.get('test_rows', '-')} 行")
        else:
            lines += [
                f"- 数据区间：{m.get('data_start', '-')} ~ {m.get('data_end', '-')}"
                f"（{m.get('data_rows', '-')} 行）",
                f"- 截止日：{m.get('cutoff', '-')}"
                f"　训练段 {m.get('train_start', '-')} ~ {m.get('train_end', '-')}"
                f"　样本外 {m.get('test_start', '-')} ~ {m.get('test_end', '-')}"
                f"（{m.get('test_rows', '-')} 行）",
            ]
        manifest = m.get("manifest")
        if manifest:
            lines.append(f"- manifest：`{json.dumps(manifest, ensure_ascii=False)}`")
        counters = m.get("counters", {})
        lines.append(
            f"- 结论：**{'通过' if self.passed_all else '未通过'}**"
            f"（通过 {counters.get('passed', 0)} / 未通过 {counters.get('failed', 0)}"
            f" / 跳过 {counters.get('skipped', 0)}）")
        lines.append("")

        for module, title in MODULE_TITLES.items():
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
        if m.get("setup", {}).get("task") == "classification":
            return self._cls_metrics_markdown(m)
        setup = m.get("setup", {})
        lines = [
            f"> 口径：{setup.get('rule', '-')}；horizon={setup.get('horizon', '-')}；"
            f"单边成本 {100 * setup.get('cost_per_side', 0):.3f}%；"
            f"阈值（训练段预测中位数）{_num(setup.get('threshold'), 6)}",
            "",
        ]

        p = m.get("prediction")
        if p:
            lines += [
                "### 精度账",
                "",
                "| 指标 | 数值 | 对照 |",
                "|---|---|---|",
                f"| RMSE | {100 * p['rmse']:.2f}% | 零预测基线 {100 * p['rmse_zero_baseline']:.2f}% |",
                f"| 方向准确率 | {p['direction_accuracy']:.3f} | "
                f"全猜涨 {p['baseline_up']:.3f} / 明日=今日 {p['baseline_persistence']:.3f} |",
                f"| IC（Spearman 秩相关） | {_num(p['ic'])} | — |",
                f"| 样本外行数 | {p['rows']} | — |",
                "",
            ]

        t, b, bm = m.get("trading"), m.get("benchmark", {}).get("buy_hold", {}), m.get("benchmark", {})
        if t:
            lines += [
                "### 交易账（样本外，策略 vs 买入持有）",
                "",
                "| 指标 | 策略 | 买入持有 |",
                "|---|---|---|",
                f"| 累计收益率 | {_pct(t['cum_return'])} | {_pct(b.get('cum_return'))} |",
                f"| 年化收益率 | {_pct(t['annual_return'])} | {_pct(b.get('annual_return'))} |",
                f"| 年化波动率（辅助） | {_pct(t['annual_vol'], signed=False)} | "
                f"{_pct(b.get('annual_vol'), signed=False)} |",
                f"| 夏普比率 | {_num(t['sharpe'], 2)} | {_num(b.get('sharpe'), 2)} |",
                f"| 最大回撤 | {_pct(t['max_drawdown'])} | {_pct(b.get('max_drawdown'))} |",
                f"| 胜率（按笔） | {_num(t['win_rate'])} | — |",
                f"| 盈亏比（按笔） | {_num(t['profit_loss_ratio'], 2)} | — |",
                f"| 年化双边换手 | {_num(t.get('turnover_annual'), 0)} | — |",
                f"| 交易笔数（辅助） | {t['trades']} | 1 |",
                f"| 超额收益 | {_pct(bm.get('excess_return'))} | — |",
                f"| 夏普差 | {_num(bm.get('sharpe_diff'), 2)} | — |",
                "",
            ]

        r = m.get("robustness")
        if r:
            scan = r.get("cost_scan", [])
            if scan:
                lines += ["### 成本敏感度", "", "| 成本倍数 | 单边费率 | 累计收益 | 夏普 | 最大回撤 |",
                          "|---|---|---|---|---|"]
                for s in scan:
                    lines.append(f"| ×{s['multiplier']:g} | {100 * s['cost_per_side']:.3f}% | "
                                 f"{_pct(s['cum_return'])} | {_num(s['sharpe'], 2)} | "
                                 f"{_pct(s['max_drawdown'])} |")
                lines.append("")
            ins, outs, gap = r.get("in_sample", {}), r.get("out_of_sample", {}), r.get("gap", {})
            if ins or outs:
                lines += ["### 样本内外差距", "", "| 口径 | 方向准确率 | RMSE |", "|---|---|---|",
                          f"| 样本内（训练段） | {_num(ins.get('direction_accuracy'))} | "
                          f"{_pct(ins.get('rmse'), 2, signed=False)} |",
                          f"| 样本外 | {_num(outs.get('direction_accuracy'))} | "
                          f"{_pct(outs.get('rmse'), 2, signed=False)} |",
                          f"| 差距 | {_num(gap.get('direction_accuracy'))} | "
                          f"{_pct(gap.get('rmse'), 2)} |", ""]
        return lines

    # --------------------------------------------- 指标表现明细（分类任务）
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
