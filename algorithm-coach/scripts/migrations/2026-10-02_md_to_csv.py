#!/usr/bin/env python3
"""一次性迁移：复用池从「结构内嵌在 MD 里」切到「CSV 管结构」。

从 24 张卡片的 frontmatter 与「相关能力」小节抽出节点与边，按 schema 的
「边型方向」做规范化，写出两个结构本体文件：

    复用池/nodes.csv    id,category,status,sources
    复用池/edges.csv    from_id,to_id,type,reason

规范化做三件事：

1. **翻转**：`上游依赖`（to 是上游）与一条照抄文档示例的 `实证证据`
   要把 from/to 对调，统一成「from = 上游」「from = 证据卡」。
2. **合并**：同一个事实在两张卡上各写一次（镜像对/互指/对称边），合成一行；
   两条判断依据不同时都保留，用 `；` 连接（不丢信息，便于人工取舍）。
3. **改枚举**：`并列/替代` → `并列替代`（值里去掉斜杠）。

用法::

    python scripts/migrations/2026-10-02_md_to_csv.py --dry-run   # 只看报告
    python scripts/migrations/2026-10-02_md_to_csv.py             # 真写文件

**迁移完成后本脚本不再使用**——卡片已无 frontmatter 与「相关能力」，
再跑它只会报错。保留在仓库里是为了记录这次迁移的做法。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

CATEGORIES = [
    "01_数据获取与处理", "02_高级特征工程", "03_建模方法",
    "04_决策与应用", "05_评估与稳健性", "06_失败经验", "07_验证证据",
]

EDGE_TYPES_FINAL = ["依赖", "实证证据", "常见误用", "并列替代"]

# 原写法 → 规范写法
EDGE_RENAME = {
    "上游依赖": "依赖",
    "下游用途": "依赖",
    "实证证据": "实证证据",
    "常见误用": "常见误用",
    "并列/替代": "并列替代",
}

EDGE_RE = re.compile(
    r"^-\s*(上游依赖|下游用途|实证证据|常见误用|并列/替代)[：:]\s*"
    r"\[\[([^\]|]+)\|([^\]]+)\]\]"
    r"(?:\s*——\s*(.*))?$"
)

FM_LIST_RE = re.compile(r"^(\w+):\s*(.*)$")


def merge_reasons(reasons: list[str]) -> str:
    """合并同一事实的多条判断依据。

    三种情况：完全包含 → 留更全的；措辞相近（相似度 ≥ 0.45，实测分界：
    0.06–0.38 是不同角度、0.46 以上是同一句的改写）→ 留更全的；
    **真正互补的用 `；` 拼接**——宁可啰嗦，不丢信息。
    """
    import difflib

    out: list[str] = []
    for r in reasons:
        for i, kept in enumerate(out):
            if r in kept:
                break
            if kept in r or difflib.SequenceMatcher(None, kept, r).ratio() >= 0.45:
                out[i] = kept if len(kept) >= len(r) else r
                break
        else:
            out.append(r)
    return "；".join(p.rstrip("。；;，, ") for p in out)


def slim_card(text: str) -> str:
    """把一张卡的 MD 从「文案 + 结构」瘦成「只有文案」。

    去掉 frontmatter（字段已进 nodes.csv）和「相关能力」小节（边已进
    edges.csv）；其余逐字保留——H1 标题、八个能力章节、验证状态、来源。
    """
    # 去 frontmatter
    m = re.match(r"^---\n.*?\n---\n", text, re.S)
    if m:
        text = text[m.end():]
    # 去「相关能力」小节（截到下一个同级或更高级标题，或文末）
    text = re.sub(r"^##\s*相关能力\s*$.*?(?=^##\s|\Z)", "", text,
                  flags=re.M | re.S)
    return text.lstrip("\n").rstrip("\n") + "\n"


def repo_root() -> str:
    """本脚本在 <项目根>/scripts/migrations/ 下，退回三级即项目根。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def read_card(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    fm = {}
    body = text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        raw = m.group(1)
        body = text[m.end():]
        list_key = None
        for line in raw.split("\n"):
            lm = FM_LIST_RE.match(line)
            if lm:
                key, val = lm.group(1), lm.group(2).strip()
                if val:
                    fm[key] = val
                    list_key = None
                else:
                    fm[key] = []
                    list_key = key
            elif line.startswith("  - ") and list_key:
                fm[list_key].append(line[4:].strip())
            elif line.strip() == "":
                continue
    return {"fm": fm, "body": body}


def section(text: str, title: str) -> str:
    m = re.search(r"^##\s*" + title + r"\s*$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def walk_cards(cards_dir: str) -> list[dict]:
    cards = []
    for cat in CATEGORIES:
        d = os.path.join(cards_dir, cat)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            path = os.path.join(d, fn)
            c = read_card(path)
            stem = fn[:-3]
            cards.append({
                "stem": stem,
                "path": path,
                "cat": cat,
                "name": c["fm"].get("name", stem),
                "status": c["fm"].get("status", ""),
                "sources": c["fm"].get("sources", []),
                "id": c["fm"].get("id", ""),
                "edges": parse_edges(section(c["body"], "相关能力")),
            })
    return cards


def parse_edges(block: str) -> list[tuple[str, str, str]]:
    out = []
    for line in block.split("\n"):
        m = EDGE_RE.match(line.strip())
        if not m:
            continue
        etype, target, _alias, reason = m.groups()
        out.append((etype, target.split("|")[0].strip().rsplit("/", 1)[-1],
                    (reason or "").strip()))
    return out


def canonical(card: dict, etype: str, target: str, reason: str,
              unknown: list[str]) -> tuple[str, str, str] | None:
    """返回规范的 (from_id, to_id, type)，方向按 schema 规范摆正。"""
    src_cat = card["cat"]
    tgt_cat = CAT_OF.get(target)
    if tgt_cat is None:
        unknown.append(f"{card['stem']} → {target}（目标卡不存在）")
        return None
    a, b = card["stem"], target
    final_type = EDGE_RENAME[etype]

    if etype == "上游依赖":          # to 是上游 → 翻成 from=上游
        return (b, a, final_type)
    if etype == "下游用途":          # to 是下游 → 方向已对
        return (a, b, final_type)
    if etype == "实证证据":          # 规范 from=07 证据卡
        if src_cat == "07_验证证据":
            return (a, b, final_type)
        if tgt_cat == "07_验证证据":
            return (b, a, final_type)
        unknown.append(f"{card['stem']} → {target}（两端都不是 07，方向存疑）")
        return (a, b, final_type)
    if etype == "常见误用":          # 规范 from=能力卡, to=失败卡(06)
        if src_cat == "06_失败经验" and tgt_cat != "06_失败经验":
            return (b, a, final_type)
        if tgt_cat == "06_失败经验":
            return (a, b, final_type)
        unknown.append(f"{card['stem']} → {target}"
                       "（两端都不含 06：坑的载体不是失败卡，按原方向保留）")
        return (a, b, final_type)
    if etype == "并列/替代":         # 对称 → 按 id 字典序存一条
        return (a, b, final_type) if a <= b else (b, a, final_type)
    raise AssertionError(etype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印报告，不写文件")
    ap.add_argument("--slim-cards", action="store_true",
                    help="同时把 24 张卡的 MD 瘦成纯内容")
    args = ap.parse_args()

    global CAT_OF
    root = repo_root()
    cards_dir = os.path.join(root, "knowledge", "复用池")
    cards = walk_cards(cards_dir)
    CAT_OF = {c["stem"]: c["cat"] for c in cards}

    unknown: list[str] = []
    merged: dict[tuple[str, str, str], list[str]] = {}
    origin: dict[tuple[str, str, str], int] = collections.Counter()
    for c in cards:
        for etype, target, reason in c["edges"]:
            key = canonical(c, etype, target, reason, unknown)
            if key is None:
                continue
            merged.setdefault(key, [])
            if reason and reason not in merged[key]:
                merged[key].append(reason)
            origin[key] += 1

    # ---- 报告 ----
    raw_edges = sum(len(c["edges"]) for c in cards)
    print(f"卡片 {len(cards)} 张；原始边 {raw_edges} 条 → 规范化后 {len(merged)} 条")
    by_type = collections.Counter(k[2] for k in merged)
    print("按边型：", dict(by_type))

    dupes = {k: v for k, v in origin.items() if v > 1}
    print(f"\n合并掉 {sum(v - 1 for v in dupes.values())} 条重复表达（{len(dupes)} 组）：")
    for (a, b, t), n in sorted(dupes.items()):
        print(f"  [{t}] {a} → {b}   （{n} 条写法合成 1 条）")
        for r in merged[(a, b, t)]:
            print(f"      依据: {r[:70]}…" if len(r) > 70 else f"      依据: {r}")

    both = {k: v for k, v in merged.items() if len(v) > 1}
    print(f"\n依据有两条、需要人工取舍的：{len(both)} 组")

    if unknown:
        print(f"\n⚠ 需要人工确认的 {len(unknown)} 条：")
        for u in unknown:
            print("  ", u)

    if args.dry_run:
        print("\n（dry-run，未写文件）")
        return 0

    # ---- 写文件 ----
    nodes_path = os.path.join(cards_dir, "nodes.csv")
    edges_path = os.path.join(cards_dir, "edges.csv")
    with open(nodes_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "category", "status", "sources"])
        for c in sorted(cards, key=lambda x: x["stem"]):
            w.writerow([c["stem"], c["cat"], c["status"], ";".join(c["sources"])])
    with open(edges_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["from_id", "to_id", "type", "reason"])
        for (a, b, t) in sorted(merged):
            w.writerow([a, b, t, merge_reasons(merged[(a, b, t)])])
    print(f"\n已写 {nodes_path}")
    print(f"已写 {edges_path}")

    if args.slim_cards:
        changed = 0
        for c in cards:
            old = open(c["path"], encoding="utf-8").read()
            new = slim_card(old)
            if new != old:
                open(c["path"], "w", encoding="utf-8", newline="\n").write(new)
                changed += 1
        print(f"已瘦身 {changed} 张卡片（去掉 frontmatter 与「相关能力」）")
    return 0


CAT_OF: dict[str, str] = {}

if __name__ == "__main__":
    raise SystemExit(main())
