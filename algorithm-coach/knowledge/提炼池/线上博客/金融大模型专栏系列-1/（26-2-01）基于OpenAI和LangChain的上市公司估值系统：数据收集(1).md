---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-2-01）基于OpenAI和LangChain的上市公司估值系统：数据收集(1).md
column: 金融大模型专栏系列-1
title: （26-2-01）基于OpenAI和LangChain的上市公司估值系统：数据收集(1)
distilled: 2026-09-30
relevance: 中
---

# （26-2-01）基于OpenAI和LangChain的上市公司估值系统：数据收集(1)

## 筛选结论
部分保留 —— 本篇是 LLM 估值系统的数据层设计，本身不产出预测信号，但其中的多源财务数据获取、分层存储与主数据（公司/行业/地区）标准化口径，对本项目的数据获取与股票画像字段设计有参考价值。

## 核心内容
- 数据分两类存储：MongoDB 存年报原文与解析结果，PostgreSQL 存 Damodaran 与 Yahoo Finance 的结构化财务/市场数据。
- MongoDB 六个集合：cik_ticker（CIK↔Ticker 映射）、submissions（公司级申报清单）、documents（每份 SEC 申报的原始 HTML）、financial_data（公司财务全历史）、parsed_documents（按 SEC 条目切分的解析文本）、items_summary（申报要点摘要）。
- postgresql.py 提供取数接口：get_df_from_table(tablename, where, most_recent) 直接返回 DataFrame，get_generic_info(ticker) 返回公司名/国家/行业/地区。
- 用 area_to_repr_country、country_to_region、industry_translation 三张字典统一国家、地区与行业口径。
- 数据库凭据从 credentials.cfg 读取，连接串支持本地 mongodb://localhost:27017 与 Atlas 两种模式。

## 可复用要点
- 数据分层思路：原始文档层、解析文本层、结构化财务事实层、公司主数据层分开存放，便于特征工程与溯源。
- 主数据标准化：用映射字典统一"国家/地区/行业"口径，可直接借鉴为 A 股个股画像（profile）的字段规范。
- 取数接口形态：get_df_from_table(table, where, most_recent) 统一返回 DataFrame，便于上层特征工程直接消费。
- 数据源清单可复用：SEC EDGAR（CIK/Ticker 映射、申报清单、原始报备文本、财务历史）、Damodaran（财务比率、市场数据、宏观经济指标）、Yahoo Finance。
- 与价格预测直接相关的因子/指标：无（本篇止于数据收集）。

## 关键实现
- mongodb.py：get_mongodb_client()，依赖 pymongo.MongoClient 与 configparser，DB_NAME='company_eval'，凭据取 credentials.cfg 的 [mongo_db] username/password。
- postgresql.py：get_connection()、get_df_from_table(tablename, where=";", most_recent=False)、get_generic_info(ticker)；涉及表 yahoo_equity_tickers、tickers_additional_info；含 area_to_repr_country / country_to_region / industry_translation 三个映射字典。
- 原文代码块被大量省略（映射字典只保留两行示例），无法直接运行。

## 数据与假设
- 标的：美股上市公司，主键体系为 SEC 的 CIK + 交易所 Ticker 双标识。
- 数据内容：SEC 申报文件（原始 HTML 与按条目切分的文本）、公司财务历史、Damodaran 财务比率/市场数据/宏观指标、Yahoo Finance 基础信息。
- 假设：申报文本可按 SEC 条目稳定切分并归纳为结构化要点。

## 局限与风险
- 全部为美股/EDGAR 语境，字段、标识体系与接口不能直接用于 A 股，需替换为 A 股数据源（如交易所/巨潮/行情商）与本地代码体系。
- 原文代码缺失严重，只有架构说明，无法复用实现。
- 无任何建模、估值结果或验证内容；本篇属 LLM 检索式估值系统的数据层，不产生可回测信号。
- 数据入库的更新频率、增量同步与质量校验机制未说明。
