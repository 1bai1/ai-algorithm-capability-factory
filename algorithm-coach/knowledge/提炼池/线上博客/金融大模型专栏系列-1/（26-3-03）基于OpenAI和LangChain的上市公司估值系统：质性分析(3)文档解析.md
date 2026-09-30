---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-3-03）基于OpenAI和LangChain的上市公司估值系统：质性分析(3)文档解析.md
column: 金融大模型专栏系列-1
title: （26-3-03）基于OpenAI和LangChain的上市公司估值系统：质性分析(3)文档解析
distilled: 2026-09-30
relevance: 低
---

# （26-3-03）基于OpenAI和LangChain的上市公司估值系统：质性分析(3)文档解析

## 筛选结论
部分保留 —— 内容是把 SEC 年报 HTML 解析成结构化文本并入库，服务于大模型基本面估值；与 A 股价格预测链路不直接相关，仅"数据解析入库"的工程范式可借鉴。

## 核心内容
- parse_document(doc)：按表单类型分支（10-K / 10-K/A / 10-Q / 8-K）选择对应的章节定义，不支持的类型直接返回。
- 解析流程：BeautifulSoup 解析 HTML → 定位目录确定章节位置 → 按目录链接或默认章节标题抽取内容 → 整理为字典 → 写入 MongoDB 的 parsed_documents 集合。
- 输入文档来自 documents 集合，字段含 _id（原文 URL）、form_type、filing_date、cik、html；先经 company_from_cik 映射公司信息，映射不到则早退。
- find_auditor(doc)：把正文抽为纯文本并压缩换行/空格，循环搜索 's/' 签名锚点，命中 'auditor since' 后用正则抽取审计机构与年份。
- 示例以 Google 10-K（CIK 1652044，GOOGL，Nasdaq）验证，输出公司元信息。

## 可复用要点
- 文档结构化范式：类型分支 → 目录定位 → 按节抽取 → 存库，可迁移到 A 股年报/公告的分章节解析。
- 无效输入前置校验（未知表单类型、公司信息缺失即在入口返回），避免脏数据入库。
- 审计机构抽取用"签名锚点 + 正则"，比全文关键字匹配更少误命中。
- 原始 HTML 与解析结果分集合存放（documents / parsed_documents），便于重跑解析。

## 关键实现
- 函数：parse_document(doc)、find_auditor(doc)；配置常量 list_10k_items / default_10k_sections（10-Q、8-K 同构）。
- 依赖：BeautifulSoup(features="html.parser")、re、unidecode、MongoDB。
- 抽取后按字典组织 sections，写回 parsed_documents 集合；示例从 documents 集合按 _id 取文档后调用。

## 数据与假设
- 数据源为 SEC EDGAR 披露原文（示例 goog-20221231.htm），以 URL 为主键；表单类型限于 10-K/10-K/A/10-Q/8-K。
- 假设：年报按固定章节组织，目录链接或默认章节标题可作为定位依据。

## 局限与风险
- 代码块在章节抽取与审计正则后半段均被截断，无法直接复用。
- 仅适配美股披露体系（CIK、10-K 章节），A 股无对应结构，迁移需重写表单定义与数据源。
- 未涉及表格/脚注处理、解析质量评估与文本长度截断问题；与价格、成交量等量化数据无关联。
