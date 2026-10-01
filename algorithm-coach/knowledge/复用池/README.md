# 复用池

能力卡按**能力**组织（不按来源组织）：同一能力来自多份材料的，合并成一张卡。

## 分类目录

`01_数据获取与处理` / `02_高级特征工程` / `03_建模方法` / `04_决策与应用` /
`05_评估与稳健性` / `06_失败经验` / `07_验证证据`

**当前为空。** 目录是按场景定义的——换场景时同步改三处：
`scripts/validate_knowledge.py`、`scripts/gen_knowledge_index.py` 的 `CATEGORIES`
常量，以及 `knowledge/知识图谱schema.md` 的 `category` 取值一节。

## 卡片规范

frontmatter 字段、章节结构、`status` 取值、边格式与全部不变量，一律以
`knowledge/知识图谱schema.md` 为准。最容易出错的几条：

- 文件名不含空格，多词用 `_` 连接；**文件名须在工作区内唯一**——边按裸文件名匹配
- 卡片必须含 `## 相关能力` 一节，边写成
  `- <边型>：[[<卡片文件名>|<显示名>]] —— <判断依据>`
- 目标是不含 `/` 的裸文件名（原因见 `knowledge/README.md`）

## 改完必须做的两件事

1. 跑 `python scripts/validate_knowledge.py`，必须 0 错误；
2. 跑 `python scripts/gen_knowledge_index.py` 重新生成 `knowledge/知识库索引.md`。

本池的写入只由**知识管理员**（被委派时）执行，主 agent 不得直接改动。
