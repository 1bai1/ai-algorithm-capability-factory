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
    "stability": "运行稳定性",
}

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
            f"- 算法目录：`{m.get('algo_dir', '-')}`",
            f"- 数据文件：`{m.get('data_path', '-')}`",
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
            lines += [f"## {title}", "", "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
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
