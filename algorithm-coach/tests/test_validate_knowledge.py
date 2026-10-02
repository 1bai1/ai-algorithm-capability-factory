#!/usr/bin/env python3
"""validate_knowledge.py 的自测：构造含各类违规的夹具，验证每类都能被抓出来。

夹具建在系统临时目录，不触碰真实知识库。校验器本身也必须被验证过——
否则它只是个"永远通过"的摆设。

夹具里同时验证一件事：**干净夹具必须退出码 0**。少了这条，下面所有
"违规能抓出来"的断言都可能只是因为校验器对什么都没报错。

用法：
    python tests/test_validate_knowledge.py
退出码 0 表示干净夹具通过、且以下各违规类全部命中。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR = os.path.join(ROOT, "scripts", "validate_knowledge.py")

NODE_COLUMNS = "id,category,status,sources\n"
EDGE_COLUMNS = "from_id,to_id,type,reason\n"

SOURCE = "提炼池/示例/来源.md"

CAPABILITY_SECTIONS = ["能力说明", "输入契约", "输出契约", "调用方式",
                       "关键参数", "依赖", "适用条件", "不适用条件"]
FAILURE_SECTIONS = ["错误模式", "触发场景", "如何识别", "后果", "修复规则"]


def card(cid: str, sections: list[str], *, frontmatter: str = "",
         extra_section: str = "", body_len: int = 40,
         with_h1: bool = True) -> str:
    out = []
    if frontmatter:
        out.append(frontmatter)
    if with_h1:
        out.append(f"# {cid}\n")
    for s in sections:
        filler = "内容" * body_len if s in ("能力说明", "错误模式") else "内容"
        out.append(f"## {s}\n{filler}\n")
    for s in ("验证状态", "来源"):
        out.append(f"## {s}\n说明\n")
    if extra_section:
        out.append(extra_section)
    return "\n".join(out) + "\n"


def build_fixture() -> dict[str, str]:
    """生成干净夹具，返回 {相对路径: 内容}。各用例在它基础上单点破坏。"""
    files: dict[str, str] = {}
    files["knowledge/提炼池/示例/来源.md"] = "---\ntitle: 示例\n---\n\n正文\n"
    files["knowledge/复用池/nodes.csv"] = NODE_COLUMNS + (
        f"卡A,03_建模方法,已验证,{SOURCE}\n"
        f"卡B,02_高级特征工程,待验证,{SOURCE}\n"
        f"失败卡,06_失败经验,有缺陷,{SOURCE}\n"
        f"证据卡,07_验证证据,已验证,{SOURCE}\n"
    )
    files["knowledge/复用池/edges.csv"] = EDGE_COLUMNS + (
        "卡B,卡A,依赖,表示的输入先于模型\n"
        "证据卡,卡A,实证证据,该卡在本数据集上跑到 0.885\n"
        "卡A,失败卡,常见误用,该失败模式由本卡环节引入\n"
        "卡A,卡B,并列替代,同一问题的两条路线\n"
    )
    files["knowledge/复用池/03_建模方法/卡A.md"] = card("卡A", CAPABILITY_SECTIONS)
    files["knowledge/复用池/02_高级特征工程/卡B.md"] = card("卡B", CAPABILITY_SECTIONS)
    files["knowledge/复用池/06_失败经验/失败卡.md"] = card("失败卡", FAILURE_SECTIONS)
    files["knowledge/复用池/07_验证证据/证据卡.md"] = card("证据卡", CAPABILITY_SECTIONS)
    return files


def write_fixture(base: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = os.path.join(base, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)


def run_validator(base: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, VALIDATOR, "--root", base, "-v"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main() -> int:
    base = tempfile.mkdtemp(prefix="kg_fixture_")
    print(f"夹具: {base}")
    failures = 0
    total = 0

    # 0) 干净夹具必须通过——下面所有断言的基线
    write_fixture(base, build_fixture())
    total += 1
    code, out = run_validator(base)
    if code == 0:
        print("  PASS  干净夹具通过（退出码 0）")
    else:
        failures += 1
        print(f"  FAIL  干净夹具竟未通过：\n{out}")

    def mutate(name: str, fn) -> None:
        nonlocal failures, total
        total += 1
        files = build_fixture()
        fn(files)
        shutil.rmtree(os.path.join(base, "knowledge"), ignore_errors=True)
        write_fixture(base, files)
        code, _ = run_validator(base)
        if code != 0:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name} 未被检出")

    # ---- 内容层 ----
    mutate("能力卡缺少章节", lambda f: f.update({
        "knowledge/复用池/03_建模方法/卡A.md": card(
            "卡A", [s for s in CAPABILITY_SECTIONS if s != "关键参数"])}))

    mutate("失败模式卡缺少章节", lambda f: f.update({
        "knowledge/复用池/06_失败经验/失败卡.md": card(
            "失败卡", [s for s in FAILURE_SECTIONS if s != "修复规则"])}))

    mutate("正文太短（空壳）", lambda f: f.update({
        "knowledge/复用池/03_建模方法/卡A.md": card(
            "卡A", CAPABILITY_SECTIONS, body_len=1)}))

    mutate("正文残留 frontmatter", lambda f: f.update({
        "knowledge/复用池/03_建模方法/卡A.md": card(
            "卡A", CAPABILITY_SECTIONS, frontmatter="---\nid: leftover\n---\n")}))

    mutate("正文残留「相关能力」", lambda f: f.update({
        "knowledge/复用池/03_建模方法/卡A.md": card(
            "卡A", CAPABILITY_SECTIONS,
            extra_section="## 相关能力\n- 上游依赖：[[卡B]] —— 说明\n")}))

    mutate("缺少一级标题", lambda f: f.update({
        "knowledge/复用池/03_建模方法/卡A.md": card(
            "卡A", CAPABILITY_SECTIONS, with_h1=False)}))

    # ---- 结构层：nodes.csv ----
    mutate("缺少 nodes.csv", lambda f: f.pop("knowledge/复用池/nodes.csv"))

    mutate("nodes.csv 表头不符", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            NODE_COLUMNS.replace("sources", "src")
            + f"卡A,03_建模方法,已验证,{SOURCE}\n"}))

    mutate("id 重复", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"] + f"卡A,03_建模方法,已验证,{SOURCE}\n"}))

    mutate("id 带 .md 后缀", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace("卡A,", "卡A.md,", 1)}))

    mutate("category 非法", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace(
                "卡A,03_建模方法", "卡A,09_不存在的类目")}))

    mutate("status 非法", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace(
                "卡A,03_建模方法,已验证", "卡A,03_建模方法,大概行吧")}))

    mutate("sources 前缀错", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace(SOURCE, "别的池/来源.md", 1)}))

    mutate("sources 文件不存在", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace(
                SOURCE, "提炼池/示例/不存在.md", 1)}))

    mutate("sources 为空", lambda f: f.update({
        "knowledge/复用池/nodes.csv":
            f["knowledge/复用池/nodes.csv"].replace(
                f"卡A,03_建模方法,已验证,{SOURCE}", "卡A,03_建模方法,已验证,")}))

    # ---- 结构层：edges.csv ----
    mutate("边的 type 非法", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace(",依赖,", ",有点像,", 1)}))

    mutate("from_id 不存在", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace("卡B,卡A,依赖", "幽灵卡,卡A,依赖")}))

    mutate("to_id 不存在", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace("卡B,卡A,依赖", "卡B,幽灵卡,依赖")}))

    mutate("自指边", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace("卡B,卡A,依赖", "卡A,卡A,依赖")}))

    mutate("reason 为空", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace(
                "卡B,卡A,依赖,表示的输入先于模型", "卡B,卡A,依赖,")}))

    mutate("重复的边", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"] + "卡B,卡A,依赖,又写了一遍\n"}))

    mutate("互指对", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"] + "卡A,卡B,依赖,反过来说一次\n"}))

    mutate("实证证据的 from 不是 07", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace(
                "证据卡,卡A,实证证据", "卡A,卡B,实证证据")}))

    mutate("并列替代未按字典序", lambda f: f.update({
        "knowledge/复用池/edges.csv":
            f["knowledge/复用池/edges.csv"].replace(
                "卡A,卡B,并列替代", "卡B,卡A,并列替代")}))

    def _orphan(files: dict[str, str]) -> None:
        files["knowledge/复用池/nodes.csv"] += f"孤卡,03_建模方法,待验证,{SOURCE}\n"
        files["knowledge/复用池/03_建模方法/孤卡.md"] = card("孤卡", CAPABILITY_SECTIONS)

    mutate("孤儿节点", _orphan)

    # ---- 对齐层 ----
    mutate("表里有行、文件不存在", lambda f: f.pop("knowledge/复用池/03_建模方法/卡A.md"))

    mutate("目录里有文件、表里没行", lambda f: f.update({
        "knowledge/复用池/03_建模方法/野卡.md": card("野卡", CAPABILITY_SECTIONS)}))

    def _category_mismatch(files: dict[str, str]) -> None:
        moved = files.pop("knowledge/复用池/03_建模方法/卡A.md")
        files["knowledge/复用池/05_评估与稳健性/卡A.md"] = moved

    mutate("表里类目与文件所在目录不符", _category_mismatch)

    shutil.rmtree(base, ignore_errors=True)
    print()
    if failures:
        print(f"FAIL {total - failures}/{total} 类通过，{failures} 类未检出")
        return 1
    print(f"PASS {total}/{total} 类全部按预期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
