#!/usr/bin/env python3
"""从复用池卡片的 frontmatter 生成 knowledge/知识库索引.md。

索引是派生物，不是事实源：复用池增删卡片或改 status 后，重跑本脚本即可。
不要手工编辑生成的索引文件。

用法：
    python scripts/gen_knowledge_index.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

CATEGORIES = [
    "01_数据获取与处理",
    "02_高级特征工程",
    "03_建模方法",
    "04_决策与应用",
    "05_评估与稳健性",
    "06_失败经验",
    "07_验证证据",
]

STATUS_MEANING = {
    "已验证": ("原文给出真实运行结果", "可直接复用"),
    "待验证": ("只有方法，原文无运行结果", "需自行验证后再用"),
    "有缺陷": ("原文自曝或实测存在明显错误", "仅作反例参考，不得直接复用"),
}

EDGE_RE = re.compile(
    r"^-\s*(上游依赖|下游用途|实证证据|常见误用|并列/替代)[：:]\s*\[\[([^\]|]+)\|([^\]]+)\]\]"
)

DESC_LIMIT = 62


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def frontmatter_field(text: str, key: str) -> str:
    m = re.search(r"^" + key + r":\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


def section(text: str, title: str) -> str:
    m = re.search(r"^##\s*" + title + r"\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def first_sentence(body: str, limit: int = DESC_LIMIT) -> str:
    flat = re.sub(r"\s+", "", body)
    parts = re.split(r"[。；]", flat)
    sent = parts[0] if parts and parts[0] else ""
    if not sent:
        return ""
    if len(sent) > limit:
        return sent[: limit - 1] + "…"
    return sent + "。"


def collect(cards_dir: str):
    cards = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(cards_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fn in sorted(os.listdir(cat_dir)):
            if not fn.endswith(".md") or fn == "README.md":
                continue
            path = os.path.join(cat_dir, fn)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            name = frontmatter_field(text, "name") or fn[:-3]
            cards.append(
                {
                    "cat": cat,
                    "name": name,
                    "file": fn,
                    "status": frontmatter_field(text, "status") or "未标注",
                    # 能力卡有「能力说明」；失败模式卡按 schema 用「错误模式」，
                    # 依次回退，别让描述留空。
                    "desc": first_sentence(
                        section(text, "能力说明") or section(text, "错误模式")),
                    "path": f"复用池/{cat}/{fn[:-3]}",
                    "stem": fn[:-3],
                    "edges": [
                        (m.group(1), m.group(2))
                        for m in (EDGE_RE.match(l.strip()) for l in section(text, "相关能力").split("\n"))
                        if m
                    ],
                }
            )
    return cards


def build(cards) -> str:
    distilled_dir = os.path.join(repo_root(), "knowledge", "提炼池")
    distilled_count = (sum(len(fs) for _, _, fs in os.walk(distilled_dir))
                       if os.path.isdir(distilled_dir) else 0)
    by_cat = {cat: [c for c in cards if c["cat"] == cat] for cat in CATEGORIES}
    status_counts = Counter(c["status"] for c in cards)
    edge_counts = Counter(e[0] for c in cards for e in c["edges"])
    total_edges = sum(edge_counts.values())

    out = []
    out.append("---")
    out.append("type: index")
    out.append("generated: true")
    out.append("generator: scripts/gen_knowledge_index.py")
    out.append("---")
    out.append("")
    out.append("# 行业算法能力知识库索引")
    out.append("")
    out.append(
        "> 本文件由 `scripts/gen_knowledge_index.py` 从卡片 frontmatter 自动生成，**请勿手工编辑**。"
        "复用池增删卡片或改动 status 后重跑该脚本即可。"
    )
    out.append("")
    out.append("## 四池结构")
    out.append("")
    out.append("四层递进，每一层是上一层的加工产物：")
    out.append("")
    out.append("| 层 | 位置 | 内容 | 何时读 |")
    out.append("|---|---|---|---|")
    out.append(f"| 复用池 | `复用池/` | {len(cards)} 张能力卡，按能力组织的可执行知识 | **默认工作层**，任务开始就读 |")
    out.append(f"| 提炼池 | `提炼池/<来源类目>/` | {distilled_count} 篇单篇提炼，目录与原始池同构 | 卡片信息不足时下钻 |")
    out.append("| 原始池 | `原始池/<来源类目>/` | **外部来源**的原始材料（付费材料不进仓库） | 需逐字核对原文时下钻 |")
    out.append("| 任务池 | `任务池/` | 单次任务的方案、代码、结果、报告 | 任务开始时建档、结束时写回 |")
    out.append("")
    out.append("**禁止整体读取任何一个池的目录。** 任何一层整读都会撑爆上下文窗口——")
    out.append("这是硬约束，不因为库还小就放宽。检索必须走下面的分层路径。")
    out.append("")
    out.append("## 检索路径")
    out.append("")
    out.append("1. 读本索引，按类别清单定位 1-3 张候选能力卡。")
    out.append("2. 读候选卡全文；卡末的「相关能力」给出上下游、实证证据与已知误用，顺着边扩展。")
    out.append("3. 卡片信息不足时，按卡内 `sources` 字段下钻到提炼池对应文件。")
    out.append("4. 仍需核对原文时，按提炼池的 `source` 字段下钻到原始池。")
    out.append("")
    out.append("## 状态说明")
    out.append("")
    out.append("| status | 含义 | 使用建议 |")
    out.append("|---|---|---|")
    for st, (meaning, advice) in STATUS_MEANING.items():
        out.append(f"| {st} | {meaning} | {advice} |")
    dist = " / ".join(f"{st} {status_counts.get(st, 0)}" for st in STATUS_MEANING)
    out.append("")
    out.append(f"当前分布（共 {len(cards)} 张）：{dist}。")
    out.append("")
    out.append(
        f"**选用任何能力前必须先看 `status` 与卡片内的「不适用条件」"
        f"——库里 {status_counts.get('有缺陷', 0)} 张卡记录的是已验证的错误做法。**"
    )
    out.append("")
    out.append("## 卡片间连接")
    out.append("")
    out.append(f"{len(cards)} 张卡之间共 {total_edges} 条语义边，按类型分布：")
    out.append("")
    out.append("| 边类型 | 条数 |")
    out.append("|---|---|")
    for etype, n in edge_counts.most_common():
        out.append(f"| {etype} | {n} |")
    out.append("")
    out.append("类别间连接密度（行 = 边的起点，列 = 终点）：")
    out.append("")
    header = "| 起点 \\ 终点 | " + " | ".join(c[:2] for c in CATEGORIES) + " |"
    out.append(header)
    out.append("|---" * (len(CATEGORIES) + 1) + "|")
    stem_to_cat = {c["stem"]: c["cat"] for c in cards}

    def _target_cat(raw: str) -> str | None:
        """边目标是裸文件名（可带 |显示名），按 stem 反查所属类目。"""
        tgt = raw.split("|")[0].strip().rsplit("/", 1)[-1]
        if tgt.endswith(".md"):
            tgt = tgt[:-3]
        return stem_to_cat.get(tgt)

    for src in CATEGORIES:
        row = [src[:2]]
        for dst in CATEGORIES:
            n = sum(1 for c in by_cat[src] for _, target in c["edges"]
                    if _target_cat(target) == dst)
            row.append(str(n) if n else "")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("---")
    out.append("")
    out.append("## 能力卡清单")
    out.append("")
    for cat in CATEGORIES:
        items = by_cat[cat]
        if not items:
            continue
        out.append(f"### {cat}（{len(items)} 张）")
        out.append("")
        for c in items:
            desc = f" — {c['desc']}" if c["desc"] else ""
            # 用裸文件名做链接目标：带路径的 wikilink 会被 Obsidian 按相对当前文件
            # 所在目录解析，拼出不存在的位置。
            out.append(f"- [[{c['stem']}|{c['name']}]] `{c['status']}`{desc}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    root = repo_root()
    cards_dir = os.path.join(root, "knowledge", "复用池")
    if not os.path.isdir(cards_dir):
        print(f"找不到复用池目录: {cards_dir}", file=sys.stderr)
        return 1

    cards = collect(cards_dir)
    if not cards:
        print("复用池里没有卡片，未生成索引", file=sys.stderr)
        return 1

    out_path = os.path.join(root, "knowledge", "知识库索引.md")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build(cards))

    print(f"已生成 {out_path}")
    print(f"卡片 {len(cards)} 张，边 {sum(len(c['edges']) for c in cards)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
