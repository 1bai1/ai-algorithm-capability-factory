#!/usr/bin/env python3
"""按 knowledge/知识图谱schema.md 校验复用池——分三层查。

权威划分（改动前必读）：

    复用池/nodes.csv   结构本体：id,category,status,sources
    复用池/edges.csv   结构本体：from_id,to_id,type,reason
    复用池/<类目>/<id>.md   内容本体：H1 + 八个能力章节 + 验证状态 + 来源

三层各管一段，互不越界：

    内容层（读 MD）  章节齐不齐、正文空不空、有没有残留结构字段
    结构层（读 CSV） 枚举合法、id 唯一、外键存在、reason 非空单行、禁止互指对
    对齐层           表里的行必须有文件、目录里的文件必须有行——分家之后最易烂的地方

检查项分两级：错误级违反即退出码 1；警告级只提示不阻断。

用法：
    python scripts/validate_knowledge.py            # 校验并输出摘要
    python scripts/validate_knowledge.py -v         # 同时列出全部警告明细
    python scripts/validate_knowledge.py --root <dir>   # 校验别的根（测试夹具用）
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Schema 常量（与 知识图谱schema.md 保持一致；改动需同步三处：
# 本文件、schema 文档、knowledge/README.md）
# --------------------------------------------------------------------------

CATEGORIES = [
    "01_数据获取与处理",
    "02_高级特征工程",
    "03_建模方法",
    "04_决策与应用",
    "05_评估与稳健性",
    "06_失败经验",
    "07_验证证据",
]

FAILURE_CATEGORY = "06_失败经验"
EVIDENCE_CATEGORY = "07_验证证据"

STATUS_VALUES = {"已验证", "待验证", "有缺陷"}

EDGE_TYPES = ["依赖", "实证证据", "常见误用", "并列替代"]

# 内容层：卡片正文该有的章节。`相关能力` 不在其列——它已随结构搬进 edges.csv，
# 正文里再出现就是没迁移干净。
COMMON_SECTIONS = ["验证状态", "来源"]

CAPABILITY_SECTIONS = [
    "能力说明", "输入契约", "输出契约", "调用方式",
    "关键参数", "依赖", "适用条件", "不适用条件",
]

FAILURE_SECTIONS = ["错误模式", "触发场景", "如何识别", "后果", "修复规则"]

NODE_COLUMNS = ["id", "category", "status", "sources"]
EDGE_COLUMNS = ["from_id", "to_id", "type", "reason"]

SOURCES_PREFIX = "提炼池/"

# 节点度的实测区间，超出只警告：枢纽卡改动影响面大，提示一下
DEGREE_WARN = 14


# --------------------------------------------------------------------------
# 报告器
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.errors: list[tuple[str, int, str]] = []
        self.warnings: list[tuple[str, int, str]] = []

    def error(self, path: str, line: int, msg: str) -> None:
        self.errors.append((path, line, msg))

    def warn(self, path: str, line: int, msg: str) -> None:
        self.warnings.append((path, line, msg))

    def dump(self, verbose: bool) -> None:
        if self.errors:
            print(f"\n错误 {len(self.errors)} 项：")
            for path, line, msg in self.errors:
                loc = f"{path}:{line}" if line else path
                print(f"  [E] {loc}\n      {msg}")
        if self.warnings:
            shown = self.warnings if verbose else self.warnings[:5]
            print(f"\n警告 {len(self.warnings)} 项（不阻断）：")
            for path, line, msg in shown:
                loc = f"{path}:{line}" if line else path
                print(f"  [W] {loc}\n      {msg}")
            if not verbose and len(self.warnings) > len(shown):
                print(f"  ... 另有 {len(self.warnings) - len(shown)} 项，用 -v 查看")


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def find_root(argv: list[str]) -> str:
    """返回 algorithm-coach 根目录。--root <dir> 可指定其他根，便于对测试夹具校验。"""
    for i, a in enumerate(argv):
        if a == "--root" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def section_count(text: str, title: str) -> int:
    return len(re.findall(r"^##\s*" + re.escape(title) + r"\s*$", text, re.M))


def section_body(text: str, title: str) -> str:
    m = re.search(r"^##\s*" + re.escape(title) + r"\s*$(.*?)(?=^##\s|\Z)", text,
                  re.M | re.S)
    return m.group(1) if m else ""


def read_table(path: str, columns: list[str], rep: Report,
               label: str) -> list[dict]:
    """读一张 CSV，带行号返回；表头不符或读不出时返回空表并记错误。

    `label` 是报错时显示的名字（不复用 relpath——夹具可能建在别的盘符上，
    跨盘符求相对路径会抛异常）。
    """
    rel = label
    if not os.path.isfile(path):
        rep.error(rel, 0, f"缺少结构本体文件：{rel}")
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        rep.error(rel, 1, "空文件，连表头都没有")
        return []
    header = [h.strip() for h in rows[0]]
    if header != columns:
        rep.error(rel, 1, f"表头应为 {','.join(columns)}，实际 {','.join(header)}")
        return []
    out = []
    for i, row in enumerate(rows[1:], start=2):
        if not row or all(not c.strip() for c in row):
            continue                     # 空行跳过，不算错
        if len(row) != len(columns):
            rep.error(rel, i, f"列数应为 {len(columns)}，实际 {len(row)}")
            continue
        rec = dict(zip(columns, [c.strip() for c in row]))
        rec["_line"] = i
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# 内容层：卡片正文
# --------------------------------------------------------------------------

def collect_cards(pool_dir: str, rep: Report) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    for cat in CATEGORIES:
        d = os.path.join(pool_dir, cat)
        if not os.path.isdir(d):
            continue          # 空库或新场景尚未建该类目，跳过而非报错
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md") or fn == "README.md":
                continue
            stem = fn[:-3]
            path = os.path.join(d, fn)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            cards[stem] = {"stem": stem, "rel": f"{cat}/{fn}", "category": cat,
                           "path": path, "text": text}
    return cards


def validate_content(cards: dict[str, dict], rep: Report) -> None:
    for stem, c in cards.items():
        rel, text = c["rel"], c["text"]

        if text.startswith("---"):
            rep.error(rel, 1, "正文里残留 frontmatter——结构字段属于 nodes.csv，"
                              "迁移时应当删除")
        if re.search(r"^##\s*相关能力\s*$", text, re.M):
            rep.error(rel, 0, "正文里残留「相关能力」小节——边属于 edges.csv，"
                              "迁移时应当删除")

        # `#` 后面必须跟空白，否则 `## 能力说明` 这种二级标题也会被当成 H1
        if not re.match(r"^#\s+\S", text, re.M):
            rep.error(rel, 1, "缺少一级标题（卡片标题）")

        for title in COMMON_SECTIONS:
            n = section_count(text, title)
            if n != 1:
                rep.error(rel, 0, f"章节 `## {title}` 应恰好出现 1 次，实际 {n} 次")

        extra = FAILURE_SECTIONS if c["category"] == FAILURE_CATEGORY \
            else CAPABILITY_SECTIONS
        kind = "失败模式卡" if c["category"] == FAILURE_CATEGORY else "能力卡"
        for title in extra:
            if section_count(text, title) != 1:
                rep.error(rel, 0, f"{kind}缺少章节 `## {title}`")

        first = "错误模式" if c["category"] == FAILURE_CATEGORY else "能力说明"
        body = section_body(text, first)
        if len(re.sub(r"\s+", "", body)) < 20:
            rep.error(rel, 0, f"`## {first}` 正文太短（不足 20 字），像是空壳")


# --------------------------------------------------------------------------
# 结构层：两张 CSV
# --------------------------------------------------------------------------

def validate_nodes(nodes: list[dict], rep: Report, refine_dir: str) -> None:
    rel = "复用池/nodes.csv"
    seen: dict[str, int] = {}
    for n in nodes:
        nid, line = n["id"], n["_line"]
        if not nid:
            rep.error(rel, line, "id 不能为空")
            continue
        if nid in seen:
            rep.error(rel, line, f"id 重复：{nid!r} 已出现在第 {seen[nid]} 行")
        seen[nid] = line
        if "/" in nid or nid.endswith(".md"):
            rep.error(rel, line, f"id 应是文件名去掉 .md 后的裸名：{nid!r}")
        if not re.match(r"^\S+$", nid):
            rep.error(rel, line, f"id 不能含空格：{nid!r}")
        if n["category"] not in CATEGORIES:
            rep.error(rel, line, f"category 非法：{n['category']!r}")
        if n["status"] not in STATUS_VALUES:
            rep.error(rel, line,
                      f"status 非法：{n['status']!r}，应为 {'/'.join(sorted(STATUS_VALUES))} 之一")
        items = [s.strip() for s in n["sources"].split(";") if s.strip()]
        if not items:
            rep.error(rel, line, "sources 不能为空")
        for item in items:
            if not item.startswith(SOURCES_PREFIX):
                rep.error(rel, line, f"sources 项未以 {SOURCES_PREFIX!r} 开头：{item!r}")
                continue
            target = os.path.join(os.path.dirname(refine_dir),
                                  item.replace("/", os.sep))
            if not os.path.isfile(target):
                rep.error(rel, line, f"sources 指向的文件不存在：{item}")


def validate_edges(edges: list[dict], nodes: list[dict], rep: Report) -> Counter:
    rel = "复用池/edges.csv"
    by_id = {n["id"]: n for n in nodes}
    type_counter: Counter = Counter()
    seen: dict[tuple[str, str, str], int] = {}
    pairs: dict[tuple[str, str, str], int] = {}
    degree: Counter = Counter()

    for e in edges:
        line = e["_line"]
        f, t, ty, reason = e["from_id"], e["to_id"], e["type"], e["reason"]
        if ty not in EDGE_TYPES:
            rep.error(rel, line, f"type 非法：{ty!r}，应为 {'/'.join(EDGE_TYPES)} 之一")
            continue
        type_counter[ty] += 1
        if f not in by_id:
            rep.error(rel, line, f"from_id 不存在于 nodes.csv：{f!r}")
        if t not in by_id:
            rep.error(rel, line, f"to_id 不存在于 nodes.csv：{t!r}")
        if f == t:
            rep.error(rel, line, f"不允许自指：{f!r}")
        if not reason:
            rep.error(rel, line, "reason 不能为空——判断依据是这条边可被反驳的抓手")
        if "\n" in reason or "\r" in reason:
            rep.error(rel, line, "reason 必须单行（多行会毁掉 diff）")

        key = (f, t, ty)
        if key in seen:
            rep.error(rel, line, f"重复的边：{f} → {t} [{ty}]，已出现在第 {seen[key]} 行")
        seen[key] = line

        # 禁止互指对：同一对节点之间同类型只允许存一条
        if (t, f, ty) in pairs:
            rep.error(rel, line,
                      f"互指对：{f} → {t} [{ty}] 与第 {pairs[(t, f, ty)]} 行的反向边表达同一事实，"
                      "只应保留一条")
        pairs[key] = line

        if ty == "实证证据" and f in by_id and by_id[f]["category"] != EVIDENCE_CATEGORY:
            rep.error(rel, line,
                      f"实证证据的 from 必须是 {EVIDENCE_CATEGORY} 的卡，"
                      f"而 {f!r} 属于 {by_id[f]['category']}")
        if ty == "并列替代" and f > t:
            rep.error(rel, line,
                      f"并列替代按 id 字典序只存一条，{f!r} > {t!r}，应把两者对调")

        degree[f] += 1
        degree[t] += 1

    for n in nodes:
        if degree[n["id"]] == 0:
            rep.error(rel, n["_line"],
                      f"孤儿节点：{n['id']!r} 在 edges.csv 里一条边都没有")
        elif degree[n["id"]] >= DEGREE_WARN:
            rep.warn(rel, n["_line"],
                     f"枢纽节点：{n['id']!r} 度 {degree[n['id']]}，改动它影响面较大")
    return type_counter


# --------------------------------------------------------------------------
# 对齐层：表 ↔ 文件
# --------------------------------------------------------------------------

def validate_alignment(cards: dict[str, dict], nodes: list[dict],
                       rep: Report) -> None:
    """两张表与磁盘上的卡片文件必须一一对应——分家之后最容易烂的地方。"""
    node_ids = {n["id"]: n for n in nodes}
    for n in nodes:
        stem, cat = n["id"], n["category"]
        found = cards.get(stem)
        if found is None:
            rep.error("复用池/nodes.csv", n["_line"],
                      f"表里有 {stem!r}，但文件不存在：复用池/{cat}/{stem}.md")
        elif found["category"] != cat:
            rep.error("复用池/nodes.csv", n["_line"],
                      f"{stem!r} 表里属 {cat}，文件却在 {found['category']}/ 下")
    for stem, c in cards.items():
        if stem not in node_ids:
            rep.error(c["rel"], 0,
                      f"目录里有这张卡（{stem}），但 nodes.csv 里没有对应行")


def check_stem_uniqueness(cards: dict[str, dict], knowledge_dir: str,
                          rep: Report) -> None:
    """id 同时也是文件名，索引里的 [[id|标题]] 按名字匹配；工作区别处若有同名
    .md，链接会指向歧义。"""
    if not os.path.isdir(knowledge_dir):
        return
    for root, dirs, files in os.walk(knowledge_dir):
        if ".obsidian" in root:
            continue
        for fn in files:
            if not fn.endswith(".md"):
                continue
            stem = fn[:-3]
            if stem not in cards:
                continue
            dup = os.path.relpath(os.path.join(root, fn), knowledge_dir)
            owner = f"复用池/{cards[stem]['category']}/{fn}"
            if dup.replace(os.sep, "/") != owner:
                rep.error(cards[stem]["rel"], 0,
                          f"卡片标识在整个工作区内不唯一：{stem} 还存在于 {dup}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    root = find_root(sys.argv)
    pool_dir = os.path.join(root, "knowledge", "复用池")
    refine_dir = os.path.join(root, "knowledge", "提炼池")

    if not os.path.isdir(pool_dir):
        print(f"找不到复用池目录: {pool_dir}", file=sys.stderr)
        return 1

    rep = Report()
    cards = collect_cards(pool_dir, rep)
    nodes = read_table(os.path.join(pool_dir, "nodes.csv"), NODE_COLUMNS, rep,
                       "复用池/nodes.csv")
    edges = read_table(os.path.join(pool_dir, "edges.csv"), EDGE_COLUMNS, rep,
                       "复用池/edges.csv")

    validate_content(cards, rep)
    validate_nodes(nodes, rep, refine_dir)
    type_counter = validate_edges(edges, nodes, rep)
    validate_alignment(cards, nodes, rep)
    check_stem_uniqueness(cards, os.path.join(root, "knowledge"), rep)

    # 素材孤儿：提炼池里有文件，却没有任何卡的 sources 引用它
    used = {s.strip() for n in nodes for s in n["sources"].split(";") if s.strip()}
    if os.path.isdir(refine_dir):
        for cur, _, files in os.walk(refine_dir):
            for fn in files:
                if not fn.endswith(".md"):
                    continue
                rel = os.path.relpath(os.path.join(cur, fn), refine_dir)
                if f"提炼池/{rel}".replace(os.sep, "/") not in used:
                    rep.warn(f"提炼池/{rel}", 0, "没有被任何卡片的 sources 引用")

    n_nodes = len(cards)
    print(f"节点 {n_nodes}   边 {sum(type_counter.values())}")
    print("状态分布  " + "  ".join(
        f"{k} {sum(1 for n in nodes if n['status'] == k)}"
        for k in ("已验证", "待验证", "有缺陷")))
    print("边型分布  " + "  ".join(f"{k} {v}" for k, v in type_counter.most_common()))

    rep.dump(verbose)

    if rep.errors:
        print(f"\n校验失败：{len(rep.errors)} 项错误")
        return 1
    print(f"\n校验通过：内容层/结构层/对齐层全部符合 schema"
          + (f"（{len(rep.warnings)} 项警告）" if rep.warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
