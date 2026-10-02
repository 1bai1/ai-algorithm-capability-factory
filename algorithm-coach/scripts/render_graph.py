#!/usr/bin/env python3
"""把复用池的结构画成可交互的知识图谱（HTML）。

读 `复用池/nodes.csv` 与 `复用池/edges.csv`——**改了库重跑本脚本，图跟着变**，
脚本里没有任何写死的节点数、类目或边。

布局：同心圆。中心是《知识库索引》（它给每张卡发一条链接，是天然的枢纽），
其余卡片按**度数降序**往外摊——连接最多的靠中心，边缘的在外圈。

    第 k 圈容量 = BASE + k * GROWTH   （外圈周长更长，能放更多）
    第 k 圈半径 = R0   + k * STEP

装满了就自动往外开新圈，所以库变大不用改代码：

    24 张 → 6+8+10 = 24，正好三圈
    25 张 → 自动开第四圈（容量 12），只放 1 张
   100 张 → 八圈（6+8+10+12+14+16+18+20 = 104 个位）

用法::

    python scripts/render_graph.py                     # 生成 复用池/graph.html
    python scripts/render_graph.py --serve             # 生成并起本地服务，输出访问地址
    python scripts/render_graph.py --open              # 生成后直接用浏览器打开
    python scripts/render_graph.py --root TF-IDF词项加权  # 换中心：以某张卡为圆心
    python scripts/render_graph.py --out /tmp/g.html   # 换个输出位置

产物是**自包含**的单个 HTML（vis.js 内嵌，离线可开、能拖能缩放、悬停看详情），
已加进 `.gitignore`——它是随时可再生的看板，不进仓库。
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

# ---- 布局参数（改这里就能调形状，不涉及具体数据）-------------------------

BASE_CAPACITY = 6      # 内圈容量
CAPACITY_GROWTH = 2    # 每往外一圈多放几个
R0 = 340.0             # 内圈半径
R_STEP = 210.0         # 每往外一圈半径加多少

INDEX_NODE = "知识库索引"   # 合成枢纽的 id（它不在 nodes.csv 里）

# ---- 配色 ----------------------------------------------------------------

CAT_COLOR = {
    "01_数据获取与处理": "#4f9dd9",
    "02_高级特征工程": "#e08c1a",
    "03_建模方法": "#2f9e6f",
    "04_决策与应用": "#a86fbf",
    "05_评估与稳健性": "#d9534f",
    "06_失败经验": "#8c6a52",
    "07_验证证据": "#d8c34a",
}

# 出现新类目时的备用调色板（按名字取模，保证同类目同色、跨类目可区分）
FALLBACK_COLORS = ["#7f8c8d", "#16a085", "#c0392b", "#8e44ad", "#2c3e50",
                   "#d35400", "#27ae60", "#2980b9"]

EDGE_COLOR = {
    "依赖": "#c4c4c4",
    "并列替代": "#7fb3d5",
    "常见误用": "#d9534f",
    "实证证据": "#2f9e6f",
}


def cat_color(cat: str) -> str:
    if cat in CAT_COLOR:
        return CAT_COLOR[cat]
    return FALLBACK_COLORS[sum(map(ord, cat)) % len(FALLBACK_COLORS)]


def vis_color(background: str) -> dict:
    """包成 vis 的四态颜色。

    必须显式给 highlight/hover——vis 的默认高亮是一层灰蓝，会把节点染成灰的
    （按用户要求：只有中心节点可以是灰的，其余节点选中后保持本色，只把边框压深）。
    """
    return {
        "background": background,
        "border": background,
        "highlight": {"background": background, "border": "#1f1f1f"},
        "hover": {"background": background, "border": "#1f1f1f"},
    }


# ---- 读结构本体 ----------------------------------------------------------

def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load(pool_dir: str) -> tuple[list[dict], list[dict], dict[str, int]]:
    nodes = read_csv(os.path.join(pool_dir, "nodes.csv"))
    edges = read_csv(os.path.join(pool_dir, "edges.csv"))
    degree: collections.Counter = collections.Counter()
    for e in edges:
        degree[e["from_id"]] += 1
        degree[e["to_id"]] += 1
    for n in nodes:
        n["degree"] = degree.get(n["id"], 0)
    return nodes, edges, degree


# ---- 布局：同心圆，满了往外开新圈 ----------------------------------------

def ring_capacity(k: int) -> int:
    """第 k 圈（从 0 数）能放几个。外圈周长更长，所以容量更大。"""
    return BASE_CAPACITY + k * CAPACITY_GROWTH


def ring_radius(k: int) -> float:
    return R0 + k * R_STEP


def layout(nodes: list[dict], root: str
           ) -> tuple[dict[str, tuple[float, float]], int]:
    """按度数降序把卡片铺进同心圈；装满了自动往外开新圈。

    返回（坐标表, 实际用到的圈数）。
    """
    import math

    pos: dict[str, tuple[float, float]] = {root: (0.0, 0.0)}
    ranked = sorted(nodes, key=lambda n: (-n["degree"], n["id"]))
    i = 0
    k = 0
    while i < len(ranked):
        cap = ring_capacity(k)
        r = ring_radius(k)
        ring = ranked[i:i + cap]
        i += cap
        for j, n in enumerate(ring):
            # 每圈错开半格，避免里外节点连成一条直线
            ang = (2 * math.pi * (j + 0.5 * (k % 2)) / len(ring)
                   - math.pi / 2)
            pos[n["id"]] = (r * math.cos(ang), r * math.sin(ang))
        k += 1
    return pos, k


# ---- 出图 ----------------------------------------------------------------

def build_html(nodes: list[dict], edges: list[dict], pos: dict,
               root: str, root_is_index: bool, title: str, node_size: float,
               index_size: float) -> str:
    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222",
                  cdn_resources="in_line",      # vis.js 内嵌，离线可开
                  select_menu=True, filter_menu=False)   # 只留按结点搜索那个下拉
    net.set_options("""
{"physics":{"enabled":false},
 "interaction":{"hover":true,"tooltipDelay":120,"navigationButtons":true},
 "edges":{"smooth":{"type":"continuous","roundness":0.08}},
 "nodes":{"font":{"size":14,"face":"Microsoft YaHei","strokeWidth":4,
                  "strokeColor":"#ffffff"}}}
""")

    # 中心：合成枢纽（默认《知识库索引》），位置钉死在原点
    net.add_node(root, label=root, x=0, y=0, physics=False, shape="dot",
                 size=index_size,
                 color=vis_color("#cfcfcf"),
                 title=(f"{root} —— 每张卡都从它引出一条链接" if root_is_index
                        else f"{root}（本图的圆心）"), status="索引")

    for n in nodes:
        x, y = pos[n["id"]]
        # 提示用纯文本：vis 这个版本不渲染 HTML 标签
        net.add_node(
            n["id"], label=n["id"], x=x, y=y, physics=False, shape="dot",
            size=node_size, color=vis_color(cat_color(n["category"])),
            title=(f"{n['id']}\n类目：{n['category']}\n状态：{n['status']}"
                   f"\n度数：{n['degree']}"
                   f"\n正文：复用池/{n['category']}/{n['id']}.md"),
            # 注意：**一个字都不能提 group**——pyvis 的 add_node 里只要 options 带
            # group，就把 color 参数整个丢掉（源码里 `if "group" in options` 那一支），
            # 节点会退回 vis 默认浅蓝、选中还会套灰蓝高亮。传 group=None 同样触发。
            # 按类目筛选改用 category 属性（下拉菜单读节点属性，不读 group）。
            category=n["category"], status=n["status"])
        if root_is_index:
            # 放射线只在中心是索引时才画——它本来就给每张卡发一条链接
            net.add_edge(root, n["id"], color="#e0e0e0", width=0.7,
                         title="索引收录")

    for e in edges:
        net.add_edge(
            e["from_id"], e["to_id"],
            color=EDGE_COLOR.get(e["type"], "#bbbbbb"),
            dashes=(e["type"] == "并列替代"),
            width=2.2 if e["type"] in ("实证证据", "常见误用") else 1.1,
            title=f"[{e['type']}] {e['reason'][:150]}",
            label=e["type"] if e["type"] in ("实证证据", "常见误用") else "")

    html = net.generate_html(notebook=False)
    # 先 fit 定比例，再把中心节点移到画面正中
    hook = ("<script>"
            "function centerHub(){network.fit({animation:false});"
            "var s=network.getScale();"
            f"network.moveTo({{position:{{x:0,y:0}},scale:s*0.95,animation:false}});}}"
            "network.once('afterDrawing',centerHub);"
            "window.addEventListener('resize',centerHub);"
            "</script></body>")
    return html.replace("</body>", hook), title


def serve(directory: str, port: int) -> None:
    """在本机起一个只读的静态服务，把图挂出去。

    **这是给用户看图的正式方式**：用户不跑命令、不看代码，agent 生成后把
    `http://127.0.0.1:<port>/graph.html` 交给他即可。只绑 127.0.0.1，不对外。
    """
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=directory)
    for candidate in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", candidate),
                                                    handler)
            break
        except OSError:
            continue
    else:
        print(f"端口 {port}–{port + 19} 都被占用，换一个 --serve-port", file=sys.stderr)
        return
    print(f"访问地址 http://127.0.0.1:{candidate}/graph.html")
    print("（本地服务，只绑 127.0.0.1；Ctrl+C 或结束进程即停）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把复用池画成可交互知识图谱")
    ap.add_argument("--out", default=None,
                    help="输出路径，默认 复用池/graph.html")
    ap.add_argument("--root", default=INDEX_NODE,
                    help=f"中心节点，默认 {INDEX_NODE}；也可指定某张卡的 id")
    ap.add_argument("--node-size", type=float, default=18,
                    help="卡片节点大小（默认 18；中心节点另由 --index-size 控制）")
    ap.add_argument("--index-size", type=float, default=34,
                    help="中心节点大小（默认 34）")
    ap.add_argument("--open", action="store_true", help="生成后打开浏览器")
    ap.add_argument("--serve", action="store_true",
                    help="生成后起本地服务并输出访问地址（给用户看图的正式方式）")
    ap.add_argument("--serve-port", type=int, default=8765,
                    help="本地服务起始端口，默认 8765（被占用就往后试）")
    args = ap.parse_args(argv)

    root = repo_root()
    pool_dir = os.path.join(root, "knowledge", "复用池")
    if not os.path.isfile(os.path.join(pool_dir, "nodes.csv")):
        print(f"找不到结构本体: {pool_dir}/nodes.csv", file=sys.stderr)
        return 1

    nodes, edges, degree = load(pool_dir)
    root_is_index = args.root == INDEX_NODE
    if not root_is_index:
        if not any(n["id"] == args.root for n in nodes):
            print(f"中心节点不存在：{args.root}", file=sys.stderr)
            return 1
        nodes = [n for n in nodes if n["id"] != args.root]   # 中心不放两遍

    pos, n_rings = layout(nodes, args.root)
    title = (f"复用池知识图谱：{len(nodes) + 1} 个节点 / {len(edges)} 条边"
             f"｜中心 {args.root}")
    html, _ = build_html(nodes, edges, pos, args.root, root_is_index, title,
                         args.node_size, args.index_size)

    out = args.out or os.path.join(pool_dir, "graph.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)

    print(f"已生成 {out}")
    print(f"节点 {len(nodes) + 1}（含中心）／结构边 {len(edges)}"
          + (f" ＋ 索引收录 {len(nodes)}" if root_is_index else ""))
    print(f"同心圈 {n_rings} 圈，容量 "
          + "、".join(str(ring_capacity(k)) for k in range(n_rings))
          + "，半径 "
          + "、".join(str(int(ring_radius(k))) for k in range(n_rings)))
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))
    if args.serve:
        serve(os.path.dirname(os.path.abspath(out)), args.serve_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
