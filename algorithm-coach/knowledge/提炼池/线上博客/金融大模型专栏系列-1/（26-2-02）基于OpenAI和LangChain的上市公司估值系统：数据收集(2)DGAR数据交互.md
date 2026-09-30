---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-2-02）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)DGAR数据交互.md
column: 金融大模型专栏系列-1
title: （26-2-02）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)DGAR数据交互
distilled: 2026-09-30
relevance: 低
---

# （26-2-02）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)DGAR数据交互

## 筛选结论
不适用 —— 内容是美国 SEC EDGAR 财报/公告数据的抓取与 MongoDB 入库，服务于 LLM 上市公司估值系统；标的是美股基本面披露文件，与本项目 A 股价格/因子量化预测无直接关系，仅数据管道设计思路可参考。

## 核心内容
- EDGAR 是 SEC 的公开披露数据库，收录 10-K、10-Q、8-K 等文件；edgar_utils.py 封装了与它交互并落库 MongoDB 的整套方法。
- 能力覆盖六类：CIK 与股票代码映射的下载与缓存、按 CIK 取公司信息、按 ticker 反查 CIK、下载指定公司全部 submissions、按表单类型与年份范围下载文档、下载财务数据。
- 还包含工程实用方法：从提交索引页解析文档 URL、CIK 补零到 10 位、字符串大小转字节、按起始日期抓取全市场最新提交（增量更新）。
- 存储设计分集合：cik_ticker 存映射，submissions 与 documents 存提交与文档，均通过 upsert 写入。
- 请求侧设置了浏览器 User-Agent 与 Accept-Encoding，属于对外部数据源的基本适配。

## 可复用要点
- 数据管道模式可迁移：标识映射（ticker↔CIK，对应 A 股的证券代码↔内部 ID）→ 元数据与文档分层落库 → 按类型 + 年份范围批量下载 → 按开始日期做增量拉取。
- 对外的数据源请求应固定 UA 与编码头，批量下载按公司/标的维度组织。
- 在 A 股量化预测场景下没有价格、成交量或因子层面的直接可用内容。

## 关键实现
- edgar_utils.py 中的函数：make_edgar_request(url)、download_cik_ticker_map()、get_df_cik_ticker_map()、company_from_cik(cik)、cik_from_ticker(ticker)、download_all_cik_submissions(cik)、download_submissions_documents(cik, forms_to_download=("10-Q","10-K","8-K"), years=5)、download_financial_data(cik)、get_filing_from_index(url)、add_trailing_to_cik(cik)、get_size_in_bytes(size_string)、get_latest_filings(form_type, start_date)。
- 映射数据源为 `https://www.sec.gov/files/company_tickers_exchange.json`，落地为 MongoDB 集合 cik_ticker / submissions / documents。
- 依赖 requests、pandas 与项目自研 mongodb 模块（upsert_document、get_collection_documents）。

## 数据与假设
数据源为 SEC EDGAR，标的是美股公司（示例 GOOGL）；默认抓取近 5 年的 10-Q/10-K/8-K；前提是 EDGAR 接口可访问、MongoDB 可用且 CIK 映射有效。

## 局限与风险
- 与 A 股无关，迁移需要替换整个数据源与标识体系，成本高。
- 原文未涉及限流、重试、异常处理与数据校验，SEC 对访问频率有要求，工程上不完整。
- 未给出字段说明、数据质量检查与使用效果评估，无法判断入库数据是否可直接用于分析。
- 属于估值系统的数据收集模块，脱离该系统的取用与分析部分难以独立验证。
