"""索引生成器的自检：文档里的数字必须来自实际结构，派生物不许漂移。

为什么要盯这个：索引是自动生成的、明文写着「请勿手工编辑」，于是**没人读它**。历史上两个 bug 就这么活了很久——

- `92 张卡` 是写死的字面量（金融时代正好 92 张，所以看着一直正常）；
- 类别间连接密度矩阵要求边目标写成 `复用池/<类目>/…` 这种带路径的形式，
  而项目约定早已改成裸文件名，于是那张表恒为空。

现在的规矩是：结构在 `复用池/nodes.csv` 与 `edges.csv`，索引是**从这两个文件生成的
派生物**。所以这里盯两件事：数字来自结构、**在库的索引与现场重算一致**（改了卡却忘了
重跑生成器，会在这里红）。

运行::

    python -m unittest tests.test_knowledge_index -v
"""
from __future__ import annotations

import csv
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "knowledge" / "复用池"

_spec = importlib.util.spec_from_file_location(
    "gen_knowledge_index", ROOT / "scripts" / "gen_knowledge_index.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestKnowledgeIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.edges = gen.load(str(POOL))
        cls.text = gen.build_index(cls.nodes, cls.edges)
        cls.lines = cls.text.splitlines()

    def test_card_count_comes_from_structure(self):
        """卡片数必须由 nodes.csv 算出，不能写死。"""
        self.assertIn(f"（共 {len(self.nodes)} 张）", self.text)
        self.assertNotIn("92 张卡", self.text)

    def test_density_matrix_is_filled(self):
        """类别间连接密度矩阵必须有数，且总数等于边数。"""
        start = next((i for i, l in enumerate(self.lines)
                      if "类别间连接密度" in l), None)
        self.assertIsNotNone(start, "索引里找不到密度矩阵")
        rows = [l for l in self.lines[start:start + 14]
                if l.startswith("| ") and set(l) - set("|- 0123456789") == set()]
        self.assertTrue(rows, "矩阵表格没渲染出来")
        total = 0
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")][1:]
            total += sum(int(c) for c in cells if c.isdigit())
        self.assertEqual(total, len(self.edges),
                         "矩阵里的边数总和与 edges.csv 对不上——起点/终点类目解析有问题")

    def test_every_card_listed(self):
        """每张卡都要出现在清单里（链接目标用 id，显示名用正文 H1）。"""
        for n in self.nodes:
            self.assertIn(f"[[{n['id']}|{n['title']}]]", self.text)

    def test_edge_types_and_directions_documented(self):
        """四种边型与各自的方向约定都要写进索引。"""
        for t in gen.EDGE_TYPES:
            self.assertIn(f"`{t}`", self.text)
            self.assertIn(gen.EDGE_TYPE_MEANING[t], self.text)

    def test_no_hardcoded_pool_sizes(self):
        """池的体量不要写死数字（会随库变化而过期）。"""
        self.assertNotIn("万 token", self.text)
        self.assertNotIn("194 篇", self.text)

    def test_committed_index_is_fresh(self):
        """在库的索引必须与现场重算一致——改了结构忘了重跑生成器，这里会红。"""
        index_path = ROOT / "knowledge" / "知识库索引.md"
        self.assertTrue(index_path.is_file(), "索引文件不存在，跑一次生成器")
        self.assertEqual(
            index_path.read_text(encoding="utf-8").replace("\r\n", "\n"),
            self.text,
            "知识库索引.md 与现场重算不一致——改了复用池就重跑 gen_knowledge_index.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
