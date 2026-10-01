"""报告生成：report.json（给 agent 读）+ report.md（给人看）。

两份报告来自同一份内存里的结果，不各算一遍——否则数字对不上时无从判断谁错。
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = "1"

STATUS_TEXT = {
    "passed": "通过",
    "failed": "未通过",
    "error": "执行失败",
}

LEVEL_TEXT = {"error": "错误", "warn": "警告"}
RESULT_TEXT = {"pass": "通过", "fail": "未通过", "skip": "跳过"}


def build_payload(ctx, check_records: list[dict], status: str) -> dict:
    """组装 report.json 的内容。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "task": Path(ctx.task_dir).name,
        "solution": Path(ctx.solution_path).name if ctx.solution_path else None,
        "status": status,
        "started_at": ctx.started_at,
        "duration_sec": round(ctx.duration_sec, 2),
        "interface": ctx.interface or {},
        "split": ctx.split.as_dict() if ctx.split else None,
        "checks": check_records,
        "metrics": _clean(ctx.metrics),
        "trading": _clean(ctx.trading),
        "baseline": _clean(ctx.baseline),
        "errors": ctx.errors,
    }


def write_reports(ctx, payload: dict, md: str, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path


def render_markdown(ctx, payload: dict) -> str:
    lines: list[str] = []
    task = payload["task"]
    lines.append(f"# 验证报告：{task}")
    lines.append("")

    n_err = sum(1 for c in payload["checks"] if c["level"] == "error" and c["result"] == "fail")
    n_warn = sum(1 for c in payload["checks"] if c["level"] == "warn" and c["result"] == "fail")
    summary = f"**结论：{STATUS_TEXT[payload['status']]}**"
    if n_err or n_warn:
        summary += f"（{n_err} 项错误、{n_warn} 项警告）"
    lines.append(f"{summary}　耗时 {payload['duration_sec']} 秒")
    lines.append("")

    if payload["errors"]:
        lines.append("## 执行中断")
        for e in payload["errors"]:
            lines.append(f"- `{e.get('stage')}` 阶段：{e.get('error_type')} — {e.get('detail')}")
        lines.append("")

    failed = [c for c in payload["checks"] if c["result"] == "fail"]
    if failed:
        lines.append("## 失败项")
        for c in failed:
            lines.append("")
            lines.append(f"### [{LEVEL_TEXT[c['level']]}] {c['name']} — {c['description']}")
            lines.append(f"- 详情：{c['detail'] or '（未给出）'}")
            if c.get("location"):
                loc = c["location"]
                lines.append(f"- 位置：`{loc.get('file')}:{loc.get('line')}`")
        lines.append("")

    skipped = [c for c in payload["checks"] if c["result"] == "skip"]
    if skipped:
        lines.append("## 跳过项")
        for c in skipped:
            lines.append(f"- `{c['name']}`：{c['detail'] or '前置步骤未完成'}")
        lines.append("")

    if payload.get("metrics"):
        lines.append("## 预测指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for k, v in payload["metrics"].items():
            lines.append(f"| {k} | {_fmt(v)} |")
        lines.append("")

    if payload.get("trading"):
        lines.append("## 交易表现")
        lines.append("")
        if payload["trading"].get("period_note"):
            lines.append(f"> {payload['trading']['period_note']}")
            lines.append("")
        lines.append("| 指标 | 策略 | 买入持有 |")
        lines.append("|---|---|---|")
        base = payload.get("baseline") or {}
        for k, v in payload["trading"].items():
            if k == "period_note":
                continue
            bv = base.get(k)
            lines.append(f"| {k} | {_fmt(v)} | {_fmt(bv) if bv is not None else '—'} |")
        lines.append("")

    if payload.get("cost_sensitivity"):
        lines.append("## 摩擦敏感性")
        lines.append("")
        lines.append(f"> {payload['cost_sensitivity'].get('note', '')}")
        lines.append("")
        lines.append("| 单边成本 | 换手次数 | 策略累计收益上界 |")
        lines.append("|---|---|---|")
        for row in payload["cost_sensitivity"]["rows"]:
            lines.append(
                f"| {row['cost_rate']:.4f} | {row['trades']} | {row['capped_return']:.4f} |"
            )
        lines.append("")

    if payload.get("split"):
        s = payload["split"]
        lines.append("## 数据切分")
        lines.append("")
        lines.append(f"- 方式：{s['method']}（训练占比 {s['train_ratio']}）")
        lines.append(f"- 训练段：{s['n_train']} 行　{s['train_range'][0]} ~ {s['train_range'][1]}")
        lines.append(f"- 测试段：{s['n_test']} 行　{s['test_range'][0]} ~ {s['test_range'][1]}")
        lines.append("")

    passed = [c for c in payload["checks"] if c["result"] == "pass"]
    if passed:
        lines.append("## 通过的检查")
        lines.append("")
        for c in passed:
            lines.append(f"- `{c['name']}`（{LEVEL_TEXT[c['level']]}）— {c['description']}")
        lines.append("")

    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.4f}"
    return str(v)


def _clean(d) -> dict | None:
    """去掉 NaN（JSON 里不是合法字面量，会写成 null 引起误读）。"""
    if not d:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and v != v:
            out[k] = None
        else:
            out[k] = v
    return out
