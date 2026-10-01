"""索引生成器自检：文档里的数字必须来自实际卡片，图不能是空的。

为什么要盯这个：索引是自动生成的、明文写着「请勿手工编辑」，于是**没人读它**。
两个 bug 就这么活了很久——

- `92 张卡` 是写死的字面量（金融时代正好 92 张，所以看着一直正常）；
- 类别间连接密度矩阵要求边目标写成 `复用池/<类目>/…` 这种带路径的形式，
  而项目约定早已改成裸文件名，于是那张表恒为空。

测试比人可靠：只要生成器再写死一个数、或者类目解析再坏掉，这里就会红。

运行::

    python -m unittest tests.test_knowledge_index -v
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "gen_knowledge_index", ROOT / "scripts" / "gen_knowledge_index.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestKnowledgeIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = gen.collect(str(ROOT / "knowledge" / "复用池"))
        cls.text = gen.build(cls.cards)
        cls.lines = cls.text.splitlines()

    def test_card_count_comes_from_cards(self):
        """卡片数必须由实际卡片算出，不能写死。"""
        self.assertIn(f"{len(self.cards)} 张卡之间共", self.text)
        self.assertNotIn("92 张卡", self.text)

    def test_distilled_count_comes_from_repo(self):
        """提炼池篇数必须来自实际目录（与原始池同构）。"""
        base = ROOT / "knowledge" / "提炼池"
        n = sum(len(fs) for _, _, fs in os.walk(base)) if base.is_dir() else 0
        self.assertIn(f"{n} 篇单篇提炼", self.text)

    def test_density_matrix_is_filled(self):
        """类别间连接密度矩阵必须有数，且总数等于边数。"""
        start = next((i for i, l in enumerate(self.lines) if "类别间连接密度" in l), None)
        self.assertIsNotNone(start, "索引里找不到密度矩阵")
        rows = [l for l in self.lines[start:start + 14]
                if l.startswith("| ") and set(l) - set("|- 0123456789") == set()]
        self.assertTrue(rows, "矩阵表格没渲染出来")
        # 上面的过滤已经把表头（含中文）与分隔行（首字符后无空格）排除，rows 就是 7 行数据
        total = 0
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")][1:]   # 去掉行首类目
            total += sum(int(c) for c in cells if c.isdigit())
        self.assertEqual(
            total, sum(len(c["edges"]) for c in self.cards),
            "矩阵里的边数总和与卡片上的边数对不上——目标类目解析有问题")

    def test_no_hardcoded_pool_sizes(self):
        """pool 体量不要写死数字（会随库变化而过期）。"""
        self.assertNotIn("万 token", self.text)
        self.assertNotIn("194 篇", self.text)

    def test_every_card_listed(self):
        """每张卡都要出现在清单里。"""
        for card in self.cards:
            self.assertIn(f"[[{card['stem']}|{card['name']}]]", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
