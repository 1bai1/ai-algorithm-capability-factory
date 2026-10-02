# algorithm-coach —— 算法能力工厂的业务层

**项目整体说明**（背景与目标、系统架构、知识图谱 schema、Agent 工作流、环境配置、
示例与任务、生成代码示例、验证报告、挑战与扩展方向）在仓库根的
[`README.md`](../README.md) —— **先看那一份**，本文件只做目录导览。

## 目录

| 路径 | 内容 |
|---|---|
| `AGENTS.md` | 规则层：角色与写权限、知识库结构、检索协议、算法契约、验收命令 |
| `knowledge/` | 四池知识库（原始池 → 提炼池 → 复用池 → 任务池）+ 索引 + 图谱 schema |
| `harness/` | 统一验证机制：接口规范 / 功能正确性 / 指标表现 / 运行稳定性 四模块 |
| `examples/text_cls_demo/` | 契约化的示例算法（`generated/` 里含可直接运行的 `run.py`） |
| `scripts/` | 知识库工具：校验器、索引生成器、素材抓取 |
| `tests/` | 自测：38 个用例 + 校验器自测（16 类违规） |
| `.pi/` | 技能、知识管理员 agent 定义、subagent 扩展 |
| `run/` | 本机启动命令备忘（未入库） |

## 与 Pi 的关系

`../pi-main` 提供 Agent 循环、LLM 接入、会话管理与内置工具——它是**框架**；
本目录是**业务层**：知识库、验证机制与规则。两者都在仓库里，clone 下来即可运行。

## 常用命令

```bash
cd algorithm-coach

# 验证一个算法（四模块 19 项检查，产出 report.json / report.md）
python -m harness validate examples/text_cls_demo/generated \
    --data examples/text_cls_demo/data/agnews_sample.csv

# 直接跑生成出来的算法（用户视角，不需要 harness）
cd examples/text_cls_demo/generated
python run.py --data ../data/agnews_sample.csv --out 预测.csv

# 知识库校验与索引
python scripts/validate_knowledge.py
python scripts/gen_knowledge_index.py

# 自测
python -m unittest tests.test_harness tests.test_harness_classification tests.test_knowledge_index
python tests/test_validate_knowledge.py
```

## 启动 agent

见仓库根 README 的「环境配置和运行方法」一节（含完整的 PowerShell 启动命令）。
