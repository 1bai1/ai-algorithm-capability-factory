"""把网页 / 论文抓成本地 Markdown 素材，落进原始池。

原始池的规矩是「保留原貌」，所以这里只做 HTML → Markdown 的机械转换
（保留标题层级、列表、代码块），不做摘要、不改写。

用法::

    python scripts/fetch_article.py <url> --out <输出路径.md> \
        [--title 标题] [--source 出处] [--proxy http://127.0.0.1:7890] [--raw]

- 默认走 `--proxy`（本机常用 127.0.0.1:7890）；不带代理时直连。
- `--raw` 用于本身就是 Markdown / 纯文本的来源（GitHub raw、arXiv 导出接口），
  原样保存不做转换。
- 文件头会自动写入溯源 frontmatter（source / title / url / collected）。

注意：站点可能反爬（CSDN 返回 521、知乎 403），那种情况只能人工提供正文。
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript",
             "svg", "form", "iframe", "aside", "button"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "ul", "ol", "li",
              "h1", "h2", "h3", "h4", "h5", "h6", "pre", "table", "tr",
              "blockquote", "figcaption", "dd", "dt"}


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    lines: list[str] = []

    def emit(text: str = "") -> None:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            lines.append(text)
        elif lines and lines[-1] != "":
            lines.append("")

    def walk(node: Tag | NavigableString) -> None:
        if isinstance(node, NavigableString):
            emit(str(node))
            return
        name = getattr(node, "name", None)
        if name in SKIP_TAGS:
            return
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            emit("#" * level + " " + node.get_text(" ", strip=True))
            emit()
            return
        if name == "pre":
            code = node.get_text("\n", strip=False).strip("\n")
            text = "\n".join("    " + ln for ln in code.splitlines())
            emit("```")
            lines.append(text)
            emit("```")
            emit()
            return
        if name == "code" and node.parent and node.parent.name != "pre":
            emit("`" + node.get_text(" ", strip=True) + "`")
            return
        if name == "li":
            marker = "-"
            parent = node.parent
            if parent and parent.name == "ol":
                pos = len([s for s in parent.find_all("li", recursive=False)])
                marker = f"{pos}."
            emit(f"{marker} " + node.get_text(" ", strip=True))
            return
        if name in ("table",):
            for tr in node.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if cells:
                    emit("| " + " | ".join(cells) + " |")
            emit()
            return
        if name in ("br", "hr"):
            emit()
            return
        for child in node.children:
            walk(child)
        if name in BLOCK_TAGS:
            emit()

    walk(root)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch(url: str, proxy: str | None) -> str:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    resp = requests.get(url, headers={"User-Agent": UA}, proxies=proxies, timeout=45)
    resp.raise_for_status()
    if resp.encoding is None or resp.encoding.lower() in ("iso-8859-1", "ascii"):
        resp.encoding = resp.apparent_encoding
    return resp.text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--proxy", default="http://127.0.0.1:7890")
    ap.add_argument("--raw", action="store_true",
                    help="来源本身是 Markdown/纯文本，原样保存")
    args = ap.parse_args()

    try:
        html = fetch(args.url, args.proxy or None)
    except Exception as exc:                     # noqa: BLE001
        print(f"抓取失败: {exc}", file=sys.stderr)
        return 2

    body = html if args.raw else html_to_markdown(html)
    if len(body) < 500:
        print(f"内容过短（{len(body)} 字符），可能是反爬页面，已放弃", file=sys.stderr)
        return 3

    title = args.title
    if not title and not args.raw:
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.get_text(strip=True) if soup.title else "")[:120]

    cited = args.source or args.url.split("/")[2]
    today = datetime.date.today().isoformat()
    header = (f"---\nsource: {cited}\ntitle: {title}\n"
              f"url: {args.url}\ncollected: {today}\n---\n\n")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + body + "\n")
    print(f"OK  {args.out}  {len(body)} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
