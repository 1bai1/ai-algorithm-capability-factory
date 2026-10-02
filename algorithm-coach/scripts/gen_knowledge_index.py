#!/usr/bin/env python3
"""从复用池的两个结构本体生成派生产物。

权威划分（见 `knowledge/知识图谱schema.md`）：

    复用池/nodes.csv   结构本体：id,category,status,sources
    复用池/edges.csv   结构本体：from_id,to_id,type,reason
    复用池/<类目>/<id>.md   内容本体：H1 + 八个能力章节 + 验证状态 + 来源

本脚本只读这两个 CSV（外加读卡片取标题与首句做展示），生成：

    knowledge/知识库索引.md   人读的检索入口
    knowledge/复用池/graph.graphml   给 Gephi 等图工具双击打开

**产物一律不许手工编辑**——改了复用池就重跑本脚本。
"""
from __future__ import annotations

import csv
import os
import re
import sys
from xml.sax.saxutils import escape

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

EDGE_TYPES = ["依赖", "实证证据", "常见误用", "并列替代"]

EDGE_TYPE_MEANING = {
    "依赖": "from 在流水线上先于 to（上游 → 下游）",
    "实证证据": "from 是 07 证据卡，to 是被它验证的能力",
    "常见误用": "from 是被使用的能力，to 是已知的坑（通常是 06 失败卡）",
    "并列替代": "对称关系，同一问题的另一条路线；只存一条，查询时双向看",
}

DESC_LIMIT = 62


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_sentence(body: str, limit: int = DESC_LIMIT) -> str:
    flat = re.sub(r"\s+", "", body)
    parts = re.split(r"[。；]", flat)
    sent = parts[0] if parts and parts[0] else ""
    if not sent:
        return ""
    if len(sent) > limit:
        return sent[: limit - 1] + "…"
    return sent + "。"


def section(text: str, title: str) -> str:
    m = re.search(r"^##\s*" + title + r"\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load(pool_dir: str) -> tuple[list[dict], list[dict]]:
    nodes = read_csv(os.path.join(pool_dir, "nodes.csv"))
    edges = read_csv(os.path.join(pool_dir, "edges.csv"))
    for n in nodes:
        path = os.path.join(pool_dir, n["category"], n["id"] + ".md")
        text = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        m = re.match(r"^#\s*(.+?)\s*$", text, re.M)
        n["title"] = m.group(1) if m else n["id"]
        n["desc"] = first_sentence(section(text, "能力说明")
                                   or section(text, "错误模式"))
    return nodes, edges


def build_index(nodes: list[dict], edges: list[dict]) -> str:
    by_cat = {c: [n for n in nodes if n["category"] == c] for c in CATEGORIES}
    status_counts = {s: sum(1 for n in nodes if n["status"] == s)
                     for s in STATUS_MEANING}
    edge_counts = {t: sum(1 for e in edges if e["type"] == t) for t in EDGE_TYPES}
    stem_to_cat = {n["id"]: n["category"] for n in nodes}

    out: list[str] = []
    out.append("---")
    out.append("type: index")
    out.append("generated: true")
    out.append("generator: scripts/gen_knowledge_index.py")
    out.append("---")
    out.append("")
    out.append("# 行业算法能力知识库索引")
    out.append("")
    out.append("> 本文件由 `scripts/gen_knowledge_index.py` 从 `复用池/nodes.csv` 与")
    out.append("> `复用池/edges.csv` 自动生成，**请勿手工编辑**。改了复用池就重跑该脚本。")
    out.append("")
    out.append("## 四池结构")
    out.append("")
    out.append("四层递进，每一层是上一层的加工产物：")
    out.append("")
    out.append("| 层 | 位置 | 内容 | 何时读 |")
    out.append("|---|---|---|---|")
    out.append(f"| 复用池 | `复用池/` | {len(nodes)} 张能力卡 + 结构本体"
               "（`nodes.csv` / `edges.csv`） | **默认工作层**，任务开始就读 |")
    out.append("| 提炼池 | `提炼池/<来源类目>/` | 单篇提炼，目录与原始池同构 | 卡片信息不足时下钻 |")
    out.append("| 原始池 | `原始池/<来源类目>/` | **外部来源**的原始材料（付费材料不进仓库） | 需逐字核对原文时下钻 |")
    out.append("| 任务池 | `任务池/` | 单次任务的方案、代码、结果、报告 | 任务开始时建档、结束时写回 |")
    out.append("")
    out.append("**禁止整体读取任何一个池的目录。** 任何一层整读都会撑爆上下文窗口——")
    out.append("这是硬约束，不因为库还小就放宽。检索必须走下面的分层路径。")
    out.append("")
    out.append("## 检索路径")
    out.append("")
    out.append("1. **结构在 CSV 里**：读 `复用池/nodes.csv` 与 `复用池/edges.csv`——这就是整张图，"
               "两个文件一次读完，不必开卡片。")
    out.append("2. **在图上遍历与排序**：按 `status`、边型、度数、有没有被 `实证证据` 背过书"
               "收窄到 1–3 个候选 `id`。")
    out.append("3. **只对选中的卡读正文**：打开 `复用池/<类目>/<id>.md`，看「不适用条件」"
               "「关键参数」这些散文细节。")
    out.append("4. 卡片信息不足时，按 `nodes.csv` 的 `sources` 列下钻到提炼池对应文件；"
               "仍需核对原文时再下钻原始池。")
    out.append("")
    out.append("## 状态说明")
    out.append("")
    out.append("| status | 含义 | 使用建议 |")
    out.append("|---|---|---|")
    for s, (meaning, advice) in STATUS_MEANING.items():
        out.append(f"| {s} | {meaning} | {advice} |")
    out.append("")
    out.append(f"当前分布（共 {len(nodes)} 张）："
               + " / ".join(f"{s} {status_counts[s]}" for s in STATUS_MEANING) + "。")
    out.append("")
    out.append("**选用任何能力前必须先看 `status` 与卡片正文里的「不适用条件」——"
               "库里记录着已验证的错误做法。**")
    out.append("")
    out.append("## 边型与方向")
    out.append("")
    out.append("| 边型 | 方向约定 |")
    out.append("|---|---|")
    for t in EDGE_TYPES:
        out.append(f"| `{t}` | {EDGE_TYPE_MEANING[t]} |")
    out.append("")
    out.append(f"{len(nodes)} 张卡之间共 {len(edges)} 条边，按类型分布：")
    out.append("")
    out.append("| 边型 | 条数 |")
    out.append("|---|---|")
    for t in EDGE_TYPES:
        out.append(f"| {t} | {edge_counts[t]} |")
    out.append("")
    out.append("类别间连接密度（行 = 边的起点，列 = 终点；"
               "`并列替代` 只存一条，矩阵不对称是正常的）：")
    out.append("")
    out.append("| 起点 \\ 终点 | " + " | ".join(c[:2] for c in CATEGORIES) + " |")
    out.append("|---" * (len(CATEGORIES) + 1) + "|")
    for src in CATEGORIES:
        row = [src[:2]]
        for dst in CATEGORIES:
            n = sum(1 for e in edges
                    if stem_to_cat.get(e["from_id"]) == src
                    and stem_to_cat.get(e["to_id"]) == dst)
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
        for n in sorted(items, key=lambda x: x["id"]):
            desc = f" — {n['desc']}" if n["desc"] else ""
            # 用裸文件名（= id）做链接目标：带路径的 wikilink 会被 Obsidian
            # 按相对当前文件所在目录解析，拼出不存在的位置。
            out.append(f"- [[{n['id']}|{n['title']}]] `{n['status']}`{desc}")
        out.append("")
    return "\n".join(out)


def build_graphml(nodes: list[dict], edges: list[dict]) -> str:
    """手写 GraphML——不引第三方依赖，字段完全由我们掌控。

    节点带 label(=标题)/category/status，边带 type/reason；
    在 Gephi 里可按 `type` 上色、按边型过滤，这是 Obsidian 图谱视图做不到的。
    """
    out: list[str] = ['<?xml version="1.0" encoding="utf-8"?>',
                      '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">']
    for i, (name, for_) in enumerate([("label", "node"), ("category", "node"),
                                      ("status", "node"),
                                      ("type", "edge"), ("reason", "edge")]):
        out.append(f'  <key id="d{i}" for="{for_}" attr.name="{name}"'
                   ' attr.type="string"/>')
    out.append('  <graph edgedefault="directed">')
    for n in sorted(nodes, key=lambda x: x["id"]):
        out.append(f'    <node id="{escape(n["id"])}">')
        # label 填标题而不是 id：Gephi 用它当节点显示名，id 是下划线连接的文件名，不好读
        for i, val in enumerate([n["title"], n["category"], n["status"]]):
            out.append(f'      <data key="d{i}">{escape(val)}</data>')
        out.append('    </node>')
    for e in sorted(edges, key=lambda x: (x["from_id"], x["to_id"], x["type"])):
        out.append(f'    <edge source="{escape(e["from_id"])}"'
                   f' target="{escape(e["to_id"])}">')
        out.append(f'      <data key="d3">{escape(e["type"])}</data>')
        out.append(f'      <data key="d4">{escape(e["reason"])}</data>')
        out.append('    </edge>')
    out.append('  </graph>')
    out.append('</graphml>')
    return "\n".join(out) + "\n"


def main() -> int:
    root = repo_root()
    pool_dir = os.path.join(root, "knowledge", "复用池")
    if not os.path.isfile(os.path.join(pool_dir, "nodes.csv")):
        print(f"找不到结构本体: {pool_dir}/nodes.csv", file=sys.stderr)
        return 1

    nodes, edges = load(pool_dir)
    if not nodes:
        print("复用池里没有卡片，未生成索引", file=sys.stderr)
        return 1

    index_path = os.path.join(root, "knowledge", "知识库索引.md")
    with open(index_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_index(nodes, edges))
    graphml_path = os.path.join(pool_dir, "graph.graphml")
    with open(graphml_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_graphml(nodes, edges))

    print(f"已生成 {index_path}")
    print(f"已生成 {graphml_path}")
    print(f"卡片 {len(nodes)} 张，边 {len(edges)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
