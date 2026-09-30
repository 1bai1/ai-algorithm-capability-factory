---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-4-04）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(4)LangChain处理.md
column: 金融大模型专栏系列-1
title: （26-4-04）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(4)LangChain处理
distilled: 2026-09-30
relevance: 不适用
---

# （26-4-04）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(4)LangChain处理

## 筛选结论
不适用 —— 本篇是大模型对 10-K 财报文本的章节摘要工程，与量化预测算法的数据/特征/模型/策略链路无直接关系。

## 核心内容
不适用 —— 演示用 LangChain 的 refine 链 + gpt-3.5-turbo 对 Alphabet 10-K 各章节（business、risk、property、legal、other）逐节摘要，并统计压缩率、成本与耗时，与价格预测无关。

## 可复用要点
无（如日后确需处理财报文本，可参考其做法：先按 ITEM 章节结构化，长文本改用具更长上下文的 16k 模型，并记录压缩率与调用成本）。

## 关键实现
无量化相关实现。文中函数为 summarize_section(section_text, model, chain_type, verbose) 与 sections_summary(parsed_doc)，chain_type="refine"，模型 gpt-3.5-turbo / gpt-3.5-turbo-16k。

## 数据与假设
Alphabet 10-K 原文（business 25020 字符、risk 82337 字符等），英文；假设按 ITEM 章节切分后逐节摘要可代表全文要点。

## 局限与风险
摘要质量未经人工校验，可能丢失估值所需细节；成本与耗时（合计约 0.0934 美元、67.3 秒）随模型版本与报价变化；该摘要结果与量化选股/预测无衔接，不能作为因子有效性的证据。
