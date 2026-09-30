---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-3-01）基于OpenAI和LangChain的上市公司估值系统：质性分析(1)年度报告参考.md
column: 金融大模型专栏系列-1
title: （26-3-01）基于OpenAI和LangChain的上市公司估值系统：质性分析(1)年度报告参考
distilled: 2026-09-30
relevance: 低
---

# （26-3-01）基于OpenAI和LangChain的上市公司估值系统：质性分析(1)年度报告参考

## 筛选结论
不适用 —— 内容是美股 10-K/10-Q 年报的目录结构与文本分节解析（配合 LangChain + OpenAI 做质性分析），属基本面文本处理链路，与面向 A 股个股的量价特征、预测建模、策略回测无直接关系。

## 核心内容
- 年报没有统一标准结构，同一公司不同年份也可能不同；可行做法是利用多数文档开头的目录来推断全篇结构。
- 原文用三个静态清单把报告结构固化为可匹配的参考：`list_10k_items`、`default_10k_sections`、`list_10q_items`。
- `list_10k_items` 为 21 个 10-K 常见章节标题字符串，覆盖 business、risk factors、properties、legal proceedings、MD&A、financial statements、controls and procedures、executive compensation、exhibits 等。
- `default_10k_sections` 是以序号 1~21 为键的字典，每项含 item 编号（item 1、1a、1b、2 直到 15）与 title 关键词列表，并为同一章节登记多种写法变体（如 item 6 对应 "reserved" 或 "selected financial data"，item 10 对应两种董事/高管表述，item 15 对应两种 exhibits 表述）。
- `list_10q_items` 为 11 个 10-Q 目录关键词，用于先判断报告是否带目录、再定位目录位置。
- 匹配方式为小写关键词包含匹配：关键词全部小写，并使用截断词干（如 'propert'）以同时命中 properties 与 property 等不同形态。

## 可复用要点
- 无。对量价预测算法生成 Agent 无可直接使用的特征、模型或策略要素；唯一可类比的是"用关键词清单切分长文档结构"这一通用工程思路，与金融特征工程无关。

## 关键实现
- 三个模块级常量：`list_10k_items`（list[str]）、`default_10k_sections`（dict[int, {item: str, title: list[str]}]）、`list_10q_items`（list[str]）。
- 项目定位是 LangChain 调用 OpenAI 模型抽取并理解报告文本，本篇只提供分节所需的参考清单，未出现模型调用代码。

## 数据与假设
- 输入为美股上市公司的 10-K / 10-Q 报告文本；本篇仅涉及美国 SEC 披露体系，不涉及 A 股年报、招股书或公告格式。
- 假设报告为英文，且章节标题可被小写关键词包含匹配命中。

## 局限与风险
- 纯关键词包含匹配较脆弱：未登记的标题变体、带页码的目录行、跨行断词都会漏匹配；原文未给出任何解析成功率或验证结果。
- 清单靠人工维护，随监管披露规则调整（item 编号或命名变化）会失效。
- 属 LLM 质性分析的前置准备，本篇未涉及任何模型调用、估值计算或效果评估，价值无法独立验证。
- 与 A 股场景的格式差异大，直接迁移需要重建整份清单。
