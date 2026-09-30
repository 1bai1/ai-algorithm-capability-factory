---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-4-02）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(2)Langchain和OpenAI处理.md
column: 金融大模型专栏系列-1
title: （26-4-02）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(2)Langchain和OpenAI处理
distilled: 2026-09-30
relevance: 低
---

# （26-4-02）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(2)Langchain和OpenAI处理

## 筛选结论
不适用 —— 本篇只是 LLM 估值系统中的文本加载与分块工具代码，不涉及行情、特征、预测、策略或回测，与 A 股量化预测算法生成无关。

## 核心内容
1. 编写 summarizer.py，用 LangChain + OpenAI 做非结构化文本与文档的加载、分块与摘要预览。
2. 自定义 `UnstructuredStringLoader` 继承 `UnstructuredBaseLoader`，承载原始字符串并附带 source 元数据。
3. 两个分块函数 `split_text_in_chunks` / `split_doc_in_chunks` 均基于 RecursiveCharacterTextSplitter，参数固定 chunk_size=20000、chunk_overlap=100。

## 可复用要点
无。

## 关键实现
`UnstructuredStringLoader(UnstructuredBaseLoader)`：`_get_elements()` 调 `unstructured.partition.text.partition_text`，`_get_metadata()` 返回 `{"source": source}`；`split_text_in_chunks(text, chunk_size=20000)`、`split_doc_in_chunks(doc, chunk_size=20000)` 内部构造 `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=100)`。依赖 langchain、unstructured。原文提到的 doc_summary 未给出实现。

## 数据与假设
输入为未结构化文本或文档（原文未给具体语料）；假设长文先分块再交给大模型处理。

## 局限与风险
- 代码不完整：构造签名转写为 `def init(...)`（疑为 `__init__` 丢失），doc_summary 无实现。
- chunk_size=20000 远超常见切分实践，原文未说明与模型上下文的匹配依据。
- 不含估值模型、行情数据、因子或回测内容。
