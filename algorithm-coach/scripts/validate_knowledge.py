#!/usr/bin/env python3
"""按 knowledge/知识图谱schema.md 校验复用池的能力知识图谱。

检查项分两级：
  错误级 —— 违反即退出码 1，会破坏图谱的可解析性或引用完整性
  警告级 —— 只提示，不阻断（当前为边显示名不一致与出度偏离）

用法：
    python scripts/validate_knowledge.py            # 校验并输出摘要
    python scripts/validate_knowledge.py -v         # 同时列出全部警告明细
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Schema 常量（与 知识图谱schema.md 保持一致，改动需同步三处：
# 本文件、schema 文档、knowledge/README.md 的边说明）
# --------------------------------------------------------------------------

CATEGORIES = [
    "01_数据获取与处理",
    "02_高级特征工程",
    "03_预测模型",
    "04_交易策略",
    "05_回测系统与风险评估",
    "06_失败经验",
    "07_验证证据",
]

FAILURE_CATEGORY = "06_失败经验"

STATUS_VALUES = {"已验证", "待验证", "有缺陷"}

REQUIRED_FIELDS = ["id", "name", "category", "status", "sources"]

COMMON_SECTIONS = ["验证状态", "来源", "相关能力"]

CAPABILITY_SECTIONS = [
    "能力说明", "输入契约", "输出契约", "调用方式",
    "关键参数", "依赖", "适用条件", "不适用条件",
]

FAILURE_SECTIONS = ["错误模式", "触发场景", "如何识别", "后果", "修复规则"]

EDGE_TYPES = ["上游依赖", "下游用途", "实证证据", "常见误用", "并列/替代"]

# 边的目标必须是裸文件名：不含 `/`。
# Obsidian 把带路径的 wikilink 按「相对当前文件所在目录」解析，而卡片本身就在
# 复用池/<类>/ 下，链接路径再以复用池/<类>/ 开头就会拼出不存在的位置，
# 导致链接解析失败、点击时在错位目录新建文件。裸文件名不含斜杠，不存在该歧义。
EDGE_RE = re.compile(
    r"^-\s*(?P<type>" + "|".join(EDGE_TYPES) + r")[：:]\s*"
    r"\[\[(?P<target>[^\]|/]+)\|(?P<display>[^\]]+)\]\]\s*"
    r"——\s*(?P<reason>.+)$"
)

EDGE_FORMAT_HINT = "`- <边型>：[[<卡片文件名>|<显示名>]] —— <判断依据>`"

ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")

SOURCES_PREFIX = "提炼池/"

# 出度实测区间，用作警告阈值
OUT_DEGREE_RANGE = (3, 8)


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


def parse_frontmatter(text: str) -> tuple[str, int]:
    """返回 frontmatter 原文与其结束行号；无 frontmatter 时返回空串。"""
    if not text.startswith("---"):
        return "", 0
    end = text.find("\n---", 3)
    if end == -1:
        return "", 0
    fm = text[4:end]
    return fm, fm.count("\n") + 2


def fm_field(fm: str, key: str) -> str | None:
    m = re.search(r"^" + key + r":\s*(.+)$", fm, re.M)
    return m.group(1).strip() if m else None


def fm_list(fm: str, key: str) -> list[str]:
    """读取 YAML 块序列字段（形如 `sources:` 后跟缩进的 `- 值`）。"""
    m = re.search(r"^" + key + r":\s*$\n((?:[ \t]+-[ \t]*.+\n?)+)", fm, re.M)
    if not m:
        return []
    return [ln.strip()[1:].strip() for ln in m.group(1).splitlines() if ln.strip().startswith("-")]


def section_count(text: str, title: str) -> int:
    return len(re.findall(r"^##\s*" + re.escape(title) + r"\s*$", text, re.M))


def section_body(text: str, title: str) -> str:
    m = re.search(r"^##\s*" + re.escape(title) + r"\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return text[:idx].count("\n") + 1 if idx >= 0 else 0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def collect_cards(cards_dir: str, rep: Report) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    for cat in CATEGORIES:
        d = os.path.join(cards_dir, cat)
        if not os.path.isdir(d):
            rep.error(os.path.relpath(d, cards_dir), 0, "类目目录不存在")
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md") or fn == "README.md":
                continue
            rel = f"{cat}/{fn}"
            key = f"{cat}/{fn[:-3]}"
            path = os.path.join(d, fn)
            text = open(path, encoding="utf-8").read()
            fm, fm_end = parse_frontmatter(text)
            if not fm:
                rep.error(rel, 1, "缺少 YAML frontmatter")
                continue
            cards[key] = {
                "key": key, "rel": rel, "stem": fn[:-3],
                "text": text, "fm": fm, "fm_end": fm_end,
                "category": cat, "path": path,
            }
    return cards


def validate_fields(cards, rep: Report) -> None:
    for key, c in cards.items():
        rel, fm = c["rel"], c["fm"]
        for field in REQUIRED_FIELDS:
            if field == "sources":
                if not fm_list(fm, "sources"):
                    rep.error(rel, 1, "frontmatter 缺少非空的 sources")
            elif fm_field(fm, field) is None:
                rep.error(rel, 1, f"frontmatter 缺少必需字段 {field}")

        status = fm_field(fm, "status")
        if status is not None and status not in STATUS_VALUES:
            rep.error(rel, line_of(c["text"], f"status: {status}"),
                      f"status 取值非法: {status!r}，应为 {'/'.join(sorted(STATUS_VALUES))} 之一")

        cat = fm_field(fm, "category")
        if cat is not None and cat != c["category"]:
            rep.error(rel, line_of(c["text"], f"category: {cat}"),
                      f"category 与所在目录不符: {cat!r} != {c['category']!r}")

        cid = fm_field(fm, "id")
        if cid is not None and not ID_RE.match(cid):
            rep.error(rel, line_of(c["text"], f"id: {cid}"),
                      f"id 不符合字符集（小写字母数字与 ._-）: {cid!r}")

        for item in fm_list(fm, "sources"):
            if not item.startswith(SOURCES_PREFIX):
                rep.error(rel, line_of(c["text"], item),
                          f"sources 项未以 {SOURCES_PREFIX!r} 开头: {item!r}")


def check_uniqueness(cards, rep: Report) -> None:
    seen_id = defaultdict(list)
    seen_name = defaultdict(list)
    for key, c in cards.items():
        cid = fm_field(c["fm"], "id")
        nm = fm_field(c["fm"], "name")
        if cid:
            seen_id[cid].append(key)
        if nm:
            seen_name[nm].append(key)
    for cid, keys in seen_id.items():
        if len(keys) > 1:
            for key in keys:
                rep.error(cards[key]["rel"], line_of(cards[key]["text"], f"id: {cid}"),
                          f"id 重复: {cid!r} 同时出现于 {[cards[k]['rel'] for k in keys]}")
    for nm, keys in seen_name.items():
        if len(keys) > 1:
            for key in keys:
                rep.error(cards[key]["rel"], line_of(cards[key]["text"], f"name: {nm}"),
                          f"name 重复: {nm!r} 同时出现于 {[cards[k]['rel'] for k in keys]}")


def validate_sources(cards, refine_dir: str, rep: Report) -> None:
    if not os.path.isdir(refine_dir):
        return
    for key, c in cards.items():
        rel = c["rel"]
        for item in fm_list(c["fm"], "sources"):
            target = os.path.join(os.path.dirname(refine_dir), item.replace("/", os.sep))
            if not os.path.isfile(target):
                rep.error(rel, line_of(c["text"], item), f"sources 指向的文件不存在: {item}")


def validate_sections(cards, rep: Report) -> None:
    for key, c in cards.items():
        text, cat, rel = c["text"], c["category"], c["rel"]
        for title in COMMON_SECTIONS:
            n = section_count(text, title)
            if n != 1:
                rep.error(rel, 0, f"章节 `## {title}` 应恰好出现 1 次，实际 {n} 次")
        extra = FAILURE_SECTIONS if cat == FAILURE_CATEGORY else CAPABILITY_SECTIONS
        for title in extra:
            if section_count(text, title) != 1:
                rep.error(rel, 0, f"{'失败模式卡' if cat == FAILURE_CATEGORY else '能力卡'}"
                                  f"缺少章节 `## {title}`")


def validate_edges(cards, rep: Report) -> Counter:
    by_stem = {c["stem"]: c for c in cards.values()}
    type_counter = Counter()
    for key, c in cards.items():
        rel = c["rel"]
        body = section_body(c["text"], "相关能力")
        n_edges = 0
        for raw in body.split("\n"):
            line = raw.strip()
            if not line.startswith("-"):
                continue
            m = EDGE_RE.match(line)
            if not m:
                hint = ("边目标不能带路径，请写裸文件名" if "[[" in line and "/" in line.split("[[")[-1].split("|")[0]
                        else "边格式不合规")
                rep.error(rel, line_of(c["text"], line),
                          f"{hint}（应为 {EDGE_FORMAT_HINT}）: {line[:60]}")
                continue
            n_edges += 1
            type_counter[m.group("type")] += 1

            target = m.group("target")
            if target == c["stem"]:
                rep.error(rel, line_of(c["text"], line), "不允许自链接")
                continue
            tgt = by_stem.get(target)
            if tgt is None:
                rep.error(rel, line_of(c["text"], line),
                          f"边的目标卡片不存在: {target}")
                continue

            tname = fm_field(tgt["fm"], "name")
            if tname and m.group("display") != tname:
                rep.warn(rel, line_of(c["text"], line),
                         f"边的显示名与目标 name 不一致: {m.group('display')!r} != {tname!r}")

        if n_edges == 0:
            rep.error(rel, 0, "每张卡至少要有 1 条边")
        lo, hi = OUT_DEGREE_RANGE
        if n_edges and not (lo <= n_edges <= hi):
            rep.warn(rel, 0, f"出度 {n_edges} 超出实测区间 {lo}–{hi}")
    return type_counter


def validate_stem_uniqueness(cards, knowledge_dir: str, rep: Report) -> None:
    """边的目标是裸文件名，因此卡片文件名必须在整个 vault 内唯一，否则会指向歧义。"""
    if not os.path.isdir(knowledge_dir):
        return
    stems = {c["stem"]: key for key, c in cards.items()}
    for root, dirs, files in os.walk(knowledge_dir):
        if ".obsidian" in root:
            continue
        for fn in files:
            if not fn.endswith(".md"):
                continue
            stem = fn[:-3]
            if stem not in stems:
                continue
            owner = stems[stem]
            dup_path = os.path.relpath(os.path.join(root, fn), knowledge_dir)
            if dup_path.replace(os.sep, "/") != f"复用池/{owner}.md":
                rep.error(cards[owner]["rel"], 0,
                          f"卡片文件名在整个 vault 内不唯一，裸文件名链接会产生歧义: "
                          f"{stem} 同时存在于 {dup_path}")


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    root = find_root(sys.argv)
    cards_dir = os.path.join(root, "knowledge", "复用池")
    refine_dir = os.path.join(root, "knowledge", "提炼池")

    if not os.path.isdir(cards_dir):
        print(f"找不到复用池目录: {cards_dir}", file=sys.stderr)
        return 1

    rep = Report()
    cards = collect_cards(cards_dir, rep)
    validate_fields(cards, rep)
    check_uniqueness(cards, rep)
    validate_sources(cards, refine_dir, rep)
    validate_sections(cards, rep)
    validate_stem_uniqueness(cards, os.path.join(root, "knowledge"), rep)
    type_counter = validate_edges(cards, rep)

    n_nodes = len(cards)
    n_edges = sum(type_counter.values())
    print(f"节点 {n_nodes}   边 {n_edges}")
    print(f"状态分布  " + "  ".join(
        f"{k} {sum(1 for c in cards.values() if fm_field(c['fm'], 'status') == k)}"
        for k in ("已验证", "待验证", "有缺陷")))
    print(f"边型分布  " + "  ".join(f"{k} {v}" for k, v in type_counter.most_common()))

    rep.dump(verbose)

    if rep.errors:
        print(f"\n校验失败：{len(rep.errors)} 项错误")
        return 1
    print(f"\n校验通过：{n_nodes} 个节点、{n_edges} 条边全部符合 schema"
          + (f"（{len(rep.warnings)} 项警告）" if rep.warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
