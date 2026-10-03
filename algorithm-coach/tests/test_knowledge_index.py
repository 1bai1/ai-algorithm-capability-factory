"""`knowledge/README.md` 的自检：数字来自结构、不抄规则、派生物不许漂移。

它是生成物、明文写着「请勿手工编辑」，于是**没人读它**。历史上两个 bug 就这么活了很久——
`92 张卡` 是写死的字面量（金融时代正好 92 张，看着一直正常）、类别间密度矩阵因为
解析口径变了而恒为空。现在文件换了形态（索引 → 目录 README），同样的风险还在，所以盯三件事：

1. **数字来自结构**：卡数/边数由 `nodes.csv` 与 `edges.csv` 算出，不是字面量；
2. **不抄规则**：检索路径、status 含义、边型方向分别住在 AGENTS.md 与 schema.md 里，
   README 只做导航与清单——同一件事写两遍就是漂移的源头；
3. **在库的那份与现场重算一致**：改了卡却忘了重跑生成器，会在这里红。

运行::

    python -m unittest tests.test_knowledge_index -v
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "knowledge" / "复用池"

_spec = importlib.util.spec_from_file_location(
    "gen_knowledge_index", ROOT / "scripts" / "gen_knowledge_index.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestKnowledgeReadme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.edges = gen.load(str(POOL))
        cls.text = gen.build_readme(cls.nodes, cls.edges)

    def test_counts_come_from_structure(self):
        """卡数与边数必须由两张 CSV 算出，不能写死。"""
        self.assertIn(f"{len(self.nodes)} 张卡 / {len(self.edges)} 条边", self.text)
        self.assertNotIn("92 张卡", self.text)

    def test_every_card_listed(self):
        """每张卡都要出现在清单里（链接目标用 id，显示名用正文 H1）。"""
        for n in self.nodes:
            self.assertIn(f"[[{n['id']}|{n['title']}]]", self.text)

    def test_does_not_duplicate_rules(self):
        """不抄规则：检索路径、status 含义表、边型方向表都不该出现在这里。

        这三样分别住在 AGENTS.md（检索与选用）与 schema.md（结构与边型）里。
        抄进 README 就多一份要同步的东西——历史上那些过期数字就是这么来的。
        """
        for banned in ("## 检索路径", "## 状态说明", "## 边型与方向", "类别间连接密度"):
            self.assertNotIn(banned, self.text, f"README 里不该出现「{banned}」，它在别的文件里")

    def test_points_to_where_rules_live(self):
        """导航要指对地方：规则在 AGENTS.md 与 schema.md，看图有命令。"""
        self.assertIn("AGENTS.md", self.text)
        self.assertIn("知识图谱schema.md", self.text)
        self.assertIn("render_graph.py", self.text)

    def test_no_hardcoded_pool_sizes(self):
        """池的体量不要写死数字（会随库变化而过期）。"""
        self.assertNotIn("万 token", self.text)
        self.assertNotIn("194 篇", self.text)

    def test_committed_readme_is_fresh(self):
        """在库的 README 必须与现场重算一致——改了结构忘了重跑生成器，这里会红。"""
        path = ROOT / "knowledge" / "README.md"
        self.assertTrue(path.is_file(), "knowledge/README.md 不存在，跑一次生成器")
        self.assertEqual(
            path.read_text(encoding="utf-8").replace("\r\n", "\n"),
            self.text,
            "knowledge/README.md 与现场重算不一致——改了复用池就重跑 gen_knowledge_index.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
