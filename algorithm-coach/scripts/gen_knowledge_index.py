#!/usr/bin/env python3
"""从复用池的两个结构本体生成派生产物。

权威划分（见 `knowledge/知识图谱schema.md`）：

    复用池/nodes.csv   结构本体：id,category,status,sources
    复用池/edges.csv   结构本体：from_id,to_id,type,reason
    复用池/<类目>/<id>.md   内容本体：H1 + 八个能力章节 + 验证状态 + 来源

本脚本只读这两个 CSV（外加读卡片取标题与首句做展示），生成：

    knowledge/README.md   这个目录的导航 + 能力卡清单（人读）

只放"怎么找东西"和"库里有什么"；规则本身在 AGENTS.md 与 schema.md，这里不抄第二份。
可视化不走这里——要图就跑 `scripts/render_graph.py`（读同一对 CSV，出可交互 HTML）。

**产物一律不许手工编辑**——改了复用池就重跑本脚本。
"""
from __future__ import annotations

import csv
import os
import re
import sys

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


def build_readme(nodes: list[dict], edges: list[dict]) -> str:
    """生成 `knowledge/README.md`：这个目录的导航 + 能力卡清单。

    刻意**只放两样东西**：怎么找到东西（四池表 + 规则指针）、库里有什么（卡片清单）。
    规则本身（检索路径、status 含义、边型方向、不变量）各自在 AGENTS.md 与
    schema.md 里，这里不抄第二份——同一件事写三遍就是漂移的源头。

    也没有规模统计（卡数、边数、密度矩阵）：那是"报表"，需要时现算即可，
    放在目录首页反而像过期倒计时。
    """
    by_cat = {c: [n for n in nodes if n["category"] == c] for c in CATEGORIES}

    out: list[str] = []
    out.append("---")
    out.append("type: index")
    out.append("generated: true")
    out.append("generator: scripts/gen_knowledge_index.py")
    out.append("---")
    out.append("")
    out.append("# 行业算法能力知识库")
    out.append("")
    out.append(f"四层递进，每一层是上一层的加工产物。**当前场景：文本分类｜{len(nodes)} 张卡 / {len(edges)} 条边。**")
    out.append("")
    out.append("| 层 | 位置 | 内容 | 何时读 |")
    out.append("|---|---|---|---|")
    out.append(f"| 复用池 | `复用池/` | 能力卡 + 结构本体（`nodes.csv` / `edges.csv`） | **默认工作层**，任务开始就读 |")
    out.append("| 提炼池 | `提炼池/<来源类目>/` | 单篇提炼，目录与原始池同构 | 卡片信息不足时下钻 |")
    out.append("| 原始池 | `原始池/<来源类目>/` | **外部来源**的原始材料（付费材料不进仓库） | 需逐字核对原文时下钻 |")
    out.append("| 任务池 | `任务池/` | 单次任务的方案、代码、结果、报告 | 任务开始时建档、结束时写回 |")
    out.append("")
    out.append("**禁止整体读取任何一个池的目录**——任何一层整读都会撑爆上下文窗口。")
    out.append("检索怎么走、能力怎么选，见 `../AGENTS.md`；卡片与边的定义、不变量，见")
    out.append("`知识图谱schema.md`。**本文件由 `../scripts/gen_knowledge_index.py` 生成，")
    out.append("请勿手工编辑**——改了复用池就重跑该脚本。")
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
    out.append("## 看图")
    out.append("")
    out.append("```bash")
    out.append("python scripts/render_graph.py --serve   # 出可交互图谱并打印访问地址")
    out.append("```")
    out.append("")
    return chr(10).join(out)

def main() -> int:
    root = repo_root()
    pool_dir = os.path.join(root, "knowledge", "复用池")
    if not os.path.isfile(os.path.join(pool_dir, "nodes.csv")):
        print(f"找不到结构本体: {pool_dir}/nodes.csv", file=sys.stderr)
        return 1

    nodes, edges = load(pool_dir)
    if not nodes:
        print("复用池里没有卡片，未生成 README", file=sys.stderr)
        return 1

    readme_path = os.path.join(root, "knowledge", "README.md")
    with open(readme_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_readme(nodes, edges))

    print(f"已生成 {readme_path}")
    print(f"卡片 {len(nodes)} 张，边 {len(edges)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
