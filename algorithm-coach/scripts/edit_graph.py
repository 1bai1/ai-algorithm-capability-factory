#!/usr/bin/env python3
"""改复用池的结构本体——加/删节点与边、改状态、给节点改名。

结构本体是两张 CSV（`复用池/nodes.csv`、`复用池/edges.csv`），手改容易
改坏（列错位、枚举写错、互指对、方向反了）。本脚本把常用改动做成命令，
**写之前先校验这次改动本身合不合法**，写之后再提示跑一次全量校验。

用法::

    python scripts/edit_graph.py check
    python scripts/edit_graph.py add-edge 卡A 卡B 依赖 "B 在流水线上先于 A"
    python scripts/edit_graph.py del-edge 卡A 卡B 依赖
    python scripts/edit_graph.py add-node 新卡 03_建模方法 待验证 提炼池/示例/来源.md
    python scripts/edit_graph.py del-node 新卡
    python scripts/edit_graph.py set-status 卡A 已验证
    python scripts/edit_graph.py rename 卡A 卡A_v2

方向约定（写错的会在写盘前被挡下或自动摆正）：

    依赖      from 在流水线上先于 to（上游 → 下游），方向自由但同类型不许互指
    实证证据  from 必须是 07_验证证据 的卡
    常见误用  from 是被使用的能力，to 是坑（通常是 06 失败卡）
    并列替代  对称——脚本会自动按 id 字典序摆正
"""

from __future__ import annotations

import csv
import os
import shutil
import sys

import validate_knowledge as vk

POOL = os.path.join("knowledge", "复用池")


def repo_root(argv: list[str] | None = None) -> str:
    """默认是脚本所在的算法工厂根；`--root <dir>` 可指向别处，便于测试。"""
    for i, a in enumerate(argv or []):
        if a == "--root" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Graph:
    def __init__(self, root: str) -> None:
        self.root = root
        self.pool = os.path.join(root, POOL)
        self.nodes_path = os.path.join(self.pool, "nodes.csv")
        self.edges_path = os.path.join(self.pool, "edges.csv")
        self.nodes = self._read(self.nodes_path, vk.NODE_COLUMNS)
        self.edges = self._read(self.edges_path, vk.EDGE_COLUMNS)

    @staticmethod
    def _read(path: str, columns: list[str]) -> list[dict]:
        with open(path, encoding="utf-8", newline="") as fh:
            return [dict(zip(columns, row)) for row in list(csv.reader(fh))[1:]
                    if any(c.strip() for c in row)]

    def _write(self, path: str, columns: list[str], rows: list[dict],
               sort_key) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
            w.writeheader()
            for r in sorted(rows, key=sort_key):
                w.writerow({c: r[c] for c in columns})
        os.replace(tmp, path)

    def save(self) -> None:
        self._write(self.nodes_path, vk.NODE_COLUMNS, self.nodes,
                    lambda n: n["id"])
        self._write(self.edges_path, vk.EDGE_COLUMNS, self.edges,
                    lambda e: (e["from_id"], e["to_id"], e["type"]))

    # ---- 查询 ----
    def node(self, nid: str) -> dict | None:
        return next((n for n in self.nodes if n["id"] == nid), None)

    def card_path(self, nid: str) -> str | None:
        n = self.node(nid)
        return os.path.join(self.pool, n["category"], nid + ".md") if n else None


# --------------------------------------------------------------------------
# 各命令
# --------------------------------------------------------------------------

def cmd_add_edge(g: Graph, from_id: str, to_id: str, etype: str,
                 reason: str) -> int:
    if etype not in vk.EDGE_TYPES:
        return fail(f"边型非法：{etype!r}，应为 {'/'.join(vk.EDGE_TYPES)} 之一")
    if from_id == to_id:
        return fail("不允许自指")
    if not g.node(from_id):
        return fail(f"节点不存在：{from_id}")
    if not g.node(to_id):
        return fail(f"节点不存在：{to_id}")
    if not reason.strip():
        return fail("reason 不能为空——判断依据是这条边可被反驳的抓手")
    if "\n" in reason or "\r" in reason:
        return fail("reason 必须单行")

    if etype == "并列替代" and from_id > to_id:
        from_id, to_id = to_id, from_id
        print(f"  并列替代是对称关系，已按 id 字典序摆正为 {from_id} → {to_id}")
    if etype == "实证证据" and g.node(from_id)["category"] != vk.EVIDENCE_CATEGORY:
        return fail(f"实证证据的 from 必须是 {vk.EVIDENCE_CATEGORY} 的卡，"
                    f"而 {from_id!r} 属于 {g.node(from_id)['category']}")

    for e in g.edges:
        if (e["from_id"], e["to_id"], e["type"]) == (from_id, to_id, etype):
            return fail(f"这条边已存在：{from_id} → {to_id} [{etype}]")
        if (e["from_id"], e["to_id"], e["type"]) == (to_id, from_id, etype):
            return fail(f"已有反向边 {to_id} → {from_id} [{etype}]，"
                        "互指对表达同一事实，应先删掉那条或改这条的方向")

    g.edges.append({"from_id": from_id, "to_id": to_id, "type": etype,
                    "reason": reason.strip()})
    g.save()
    print(f"已加边：{from_id} → {to_id} [{etype}]")
    return 0


def cmd_del_edge(g: Graph, from_id: str, to_id: str, etype: str) -> int:
    keep = [e for e in g.edges
            if (e["from_id"], e["to_id"], e["type"]) != (from_id, to_id, etype)]
    if len(keep) == len(g.edges):
        return fail(f"没有这条边：{from_id} → {to_id} [{etype}]")
    g.edges = keep
    g.save()
    print(f"已删边：{from_id} → {to_id} [{etype}]")
    lonely = [n["id"] for n in g.nodes
              if not any(e["from_id"] == n["id"] or e["to_id"] == n["id"]
                         for e in g.edges)]
    if lonely:
        print(f"  ⚠ 现在这些节点没有边了：{lonely}（校验会报孤儿节点）")
    return 0


def cmd_add_node(g: Graph, nid: str, category: str, status: str,
                 sources: list[str]) -> int:
    if "/" in nid or nid.endswith(".md") or not nid.strip():
        return fail("id 应是文件名去掉 .md 后的裸名，且不含空格与斜杠")
    if g.node(nid):
        return fail(f"节点已存在：{nid}")
    if category not in vk.CATEGORIES:
        return fail(f"category 非法：{category!r}")
    if status not in vk.STATUS_VALUES:
        return fail(f"status 非法：{status!r}")
    if not sources:
        return fail("至少要有一个 sources（提炼池里的溯源文件）")
    for s in sources:
        if not s.startswith(vk.SOURCES_PREFIX):
            return fail(f"sources 项未以 {vk.SOURCES_PREFIX!r} 开头：{s!r}")
        if not os.path.isfile(os.path.join(g.root, "knowledge",
                                           s.replace("/", os.sep))):
            return fail(f"sources 指向的文件不存在：{s}")

    g.nodes.append({"id": nid, "category": category, "status": status,
                    "sources": ";".join(sources)})
    g.save()
    print(f"已加节点：{nid}（{category} / {status}）")

    path = os.path.join(g.pool, category, nid + ".md")
    if os.path.isfile(path):
        print(f"  卡片文件已存在，未改动：{path}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sections = (vk.FAILURE_SECTIONS if category == vk.FAILURE_CATEGORY
                    else vk.CAPABILITY_SECTIONS)
        body = [f"# {nid}", ""]
        for s in sections:
            body += [f"## {s}", "", ]
        for s in vk.COMMON_SECTIONS:
            body += [f"## {s}", ""]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(body).rstrip() + "\n")
        print(f"  已生成卡片骨架，请补齐正文：{path}")
        print("  ⚠ 骨架的章节是空的，校验会报『正文太短』——这是提醒你还没写内容")
    return 0


def cmd_del_node(g: Graph, nid: str) -> int:
    if not g.node(nid):
        return fail(f"节点不存在：{nid}")
    touched = [e for e in g.edges if nid in (e["from_id"], e["to_id"])]
    if touched:
        return fail(f"{nid} 还有 {len(touched)} 条边牵着，先删边再删节点\n"
                    + "\n".join(f"    {e['from_id']} → {e['to_id']} [{e['type']}]"
                                for e in touched[:5]))
    path = g.card_path(nid)          # 必须在删行之前取——删完就查不到了
    g.nodes = [n for n in g.nodes if n["id"] != nid]
    g.save()
    # 卡片文件一并删掉：留着它就变成"有文件没行"的孤儿，校验会报错。
    # 删错也不怕——仓库有版本控制，git 历史里找得回来。
    if path and os.path.isfile(path):
        os.remove(path)
        print(f"已删节点：{nid}（卡片文件也删了：{path}）")
        print("  找回来的办法：git checkout <那次提交> -- <该文件>")
    else:
        print(f"已删节点：{nid}（本来就没有卡片文件）")
    return 0


def cmd_set_status(g: Graph, nid: str, status: str) -> int:
    if status not in vk.STATUS_VALUES:
        return fail(f"status 非法：{status!r}，应为 {'/'.join(sorted(vk.STATUS_VALUES))} 之一")
    n = g.node(nid)
    if not n:
        return fail(f"节点不存在：{nid}")
    old, n["status"] = n["status"], status
    g.save()
    print(f"已改状态：{nid}  {old} → {status}")
    return 0


def cmd_rename(g: Graph, old: str, new: str) -> int:
    n = g.node(old)
    if not n:
        return fail(f"节点不存在：{old}")
    if g.node(new):
        return fail(f"新 id 已被占用：{new}")
    if "/" in new or new.endswith(".md") or not new.strip():
        return fail("新 id 应是文件名去掉 .md 后的裸名")
    src = g.card_path(old)
    dst = os.path.join(g.pool, n["category"], new + ".md")
    if not os.path.isfile(src):
        return fail(f"卡片文件不存在：{src}")

    n["id"] = new
    n_edges = 0
    for e in g.edges:
        if e["from_id"] == old:
            e["from_id"] = new
            n_edges += 1
        if e["to_id"] == old:
            e["to_id"] = new
            n_edges += 1
    g.save()
    shutil.move(src, dst)
    print(f"已改名：{old} → {new}")
    print(f"  卡片文件：{os.path.basename(src)} → {os.path.basename(dst)}")
    print(f"  跟着改的边：{n_edges} 处引用")
    return 0


def cmd_check(g: Graph) -> int:
    return subprocess_check(g.root)


def subprocess_check(root: str) -> int:
    import subprocess

    proc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "validate_knowledge.py"), "--root", root],
        text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode


def fail(msg: str) -> int:
    print(f"✗ {msg}", file=sys.stderr)
    return 1


USAGE = __doc__.split("用法::")[1].strip()


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[1], argv[2:]
    if "--root" in rest:                     # --root 是给工具自己用的，不参与命令参数
        i = rest.index("--root")
        rest = rest[:i] + rest[i + 2:]
    g = Graph(repo_root(argv))
    n_before = (len(g.nodes), len(g.edges))

    try:
        if cmd == "check":
            return cmd_check(g)
        if cmd == "add-edge":
            return cmd_add_edge(g, rest[0], rest[1], rest[2], rest[3])
        if cmd == "del-edge":
            return cmd_del_edge(g, rest[0], rest[1], rest[2])
        if cmd == "add-node":
            return cmd_add_node(g, rest[0], rest[1], rest[2], rest[3:])
        if cmd == "del-node":
            return cmd_del_node(g, rest[0])
        if cmd == "set-status":
            return cmd_set_status(g, rest[0], rest[1])
        if cmd == "rename":
            return cmd_rename(g, rest[0], rest[1])
    except IndexError:
        print(f"✗ 参数不够\n\n{USAGE}", file=sys.stderr)
        return 2

    print(f"✗ 未知命令：{cmd}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    code = main(sys.argv)
    if code == 0 and len(sys.argv) > 1 and sys.argv[1] != "check":
        print("\n提示：结构改完记得跑一次全量校验")
        print("      python scripts/validate_knowledge.py")
        print("      python scripts/gen_knowledge_index.py   # 刷新索引")
        print("      python scripts/render_graph.py          # 想看图谱时再跑")
    raise SystemExit(code)
