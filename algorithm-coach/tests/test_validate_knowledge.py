#!/usr/bin/env python3
"""validate_knowledge.py 的自测：构造含各类违规的夹具，验证每类都能被抓出来。

夹具建在系统临时目录，不触碰真实知识库。校验器本身也必须被验证过——
否则它只是个"永远通过"的摆设。

用法：
    python tests/test_validate_knowledge.py
退出码 0 表示 15 类违规全部命中且退出码正确。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR = os.path.join(ROOT, "scripts", "validate_knowledge.py")

ALL_CATEGORIES = [
    "01_数据获取与处理", "02_高级特征工程", "03_建模方法",
    "04_决策与应用", "05_评估与稳健性", "06_失败经验", "07_验证证据",
]

CAPABILITY_SECTIONS = ["能力说明", "输入契约", "输出契约", "调用方式",
                       "关键参数", "依赖", "适用条件", "不适用条件"]


def make_card(cid="model.demo.v1", name="示例能力卡", category="03_预测模型",
              status="已验证", sources=("提炼池/示例/来源.md",),
              omit_section=None, edges=("- 上游依赖：[[正常卡|正常卡]] —— 示例依据",)):
    """生成一张卡片；各违规用例在它基础上做单点破坏。"""
    secs = [s for s in CAPABILITY_SECTIONS if s != omit_section]
    body = [f"## {s}\n内容" for s in secs]
    body.append("## 验证状态\n原文未给出运行结果")
    body.append("## 来源\n- 示例来源")
    body.append("## 相关能力\n" + "\n".join(edges))
    src = "\n".join(f"  - {s}" for s in sources)
    return (f"---\nid: {cid}\nname: {name}\ncategory: {category}\n"
            f"status: {status}\nsources:\n{src}\n---\n\n"
            f"# {name}\n\n" + "\n\n".join(body) + "\n")


def build_fixture(base: str) -> dict[str, str]:
    """建夹具，返回 {用例名: 期望在校验输出中出现的子串}。"""
    pool = os.path.join(base, "knowledge", "复用池")
    for cat in ALL_CATEGORIES:
        os.makedirs(os.path.join(pool, cat), exist_ok=True)
    ref_dir = os.path.join(base, "knowledge", "提炼池", "示例")
    os.makedirs(ref_dir, exist_ok=True)
    open(os.path.join(ref_dir, "来源.md"), "w", encoding="utf-8").write("占位来源\n")
    # 与卡片同名的提炼池文件：用来验证「卡片文件名全库唯一」这条不变量
    open(os.path.join(ref_dir, "正常卡.md"), "w", encoding="utf-8").write("占位同名文件\n")

    c1 = os.path.join(pool, "01_数据获取与处理")
    c6 = os.path.join(pool, "06_失败经验")

    def put(d, fn, text):
        open(os.path.join(d, fn), "w", encoding="utf-8").write(text)

    # 合规卡：既是边的目标，本身也不应报错
    put(c1, "正常卡.md", make_card(cid="data.normal.v1", name="正常卡",
                                   category="01_数据获取与处理"))

    cases = {
        "缺少必需字段 status":
            "缺少必需字段 status",
        "status 取值非法":
            "status 取值非法",
        "category 与所在目录不符":
            "category 与所在目录不符",
        "id 不符合字符集":
            "id 不符合字符集",
        "sources 未以 提炼池/ 开头":
            "sources 项未以",
        "sources 指向的文件不存在":
            "sources 指向的文件不存在",
        "缺少章节":
            "能力卡缺少章节 `## 调用方式`",
        "失败模式卡缺少章节":
            "失败模式卡缺少章节 `## 错误模式`",
        "边格式不合规":
            "边格式不合规",
        "边目标带路径":
            "边目标不能带路径",
        "边的目标卡片不存在":
            "边的目标卡片不存在",
        "不允许自链接":
            "不允许自链接",
        "每张卡至少要有 1 条边":
            "每张卡至少要有 1 条边",
        "id 重复":
            "id 重复",
        "name 重复":
            "name 重复",
        "卡片文件名全库不唯一":
            "卡片文件名在整个工作区内不唯一",
    }

    put(c1, "缺字段.md", make_card(category="01_数据获取与处理").replace("status: 已验证\n", ""))
    put(c1, "status非法.md", make_card(category="01_数据获取与处理", status="大概可以"))
    put(c1, "类目不符.md", make_card(category="03_预测模型"))
    put(c1, "id非法.md", make_card(cid="Model.Bad_ID", category="01_数据获取与处理"))
    put(c1, "来源前缀错.md", make_card(category="01_数据获取与处理",
                                    sources=("原始池/某文件.md",)))
    put(c1, "来源不存在.md", make_card(category="01_数据获取与处理",
                                    sources=("提炼池/根本不存在的文件.md",)))
    put(c1, "缺章节.md", make_card(category="01_数据获取与处理", omit_section="调用方式"))
    put(c6, "失败卡缺章.md", make_card(category="06_失败经验"))
    put(c1, "边格式错.md", make_card(category="01_数据获取与处理",
                                  edges=("- 上游依赖：[[正常卡|正常卡]]",)))
    put(c1, "边目标带路径.md", make_card(category="01_数据获取与处理",
                                    edges=("- 上游依赖：[[复用池/01_数据获取与处理/正常卡|正常卡]] —— 依据",)))
    put(c1, "边目标不存在.md", make_card(category="01_数据获取与处理",
                                     edges=("- 上游依赖：[[幽灵卡|幽灵卡]] —— 依据",)))
    put(c1, "自链接.md", make_card(name="自链接", category="01_数据获取与处理",
                                 edges=("- 上游依赖：[[自链接|自链接]] —— 依据",)))
    put(c1, "零边.md", make_card(category="01_数据获取与处理", edges=()))
    put(c1, "重复id.md", make_card(cid="data.normal.v1", name="另一个名字",
                                 category="01_数据获取与处理"))
    put(c1, "重复name.md", make_card(cid="data.other.v1", name="正常卡",
                                   category="01_数据获取与处理"))
    return cases


def main() -> int:
    base = tempfile.mkdtemp(prefix="kg_fixture_")
    try:
        cases = build_fixture(base)
        print(f"夹具: {base}")
        print(f"构造违规用例 {len(cases)} 个\n")

        r = subprocess.run(
            [sys.executable, VALIDATOR, "--root", base, "-v"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")

        misses = []
        for label, expect in cases.items():
            hit = expect in out
            print(f"  {'PASS' if hit else 'MISS'}  {label}")
            if not hit:
                misses.append(label)

        print()
        if r.returncode != 1:
            print(f"FAIL 退出码应为 1，实际 {r.returncode}")
            return 1
        print(f"退出码 {r.returncode} 正确（检出错误时非零退出）")

        if misses:
            print(f"\nFAIL 漏报 {len(misses)} 类：{misses}")
            return 1
        print(f"\nPASS {len(cases)}/{len(cases)} 类违规全部检出")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
