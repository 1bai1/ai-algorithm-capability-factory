---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-5-02）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(2)数据处理.md
column: 金融大模型专栏系列-1
title: （26-5-02）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(2)数据处理
distilled: 2026-09-30
relevance: 低
---

# （26-5-02）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(2)数据处理

## 筛选结论
低 —— 内容是美股财报（XBRL 口径）的定量数据抽取与合并，服务于基本面估值，与 A 股个股量价预测的特征/模型/策略环节基本不重叠。

## 核心内容
- 本篇是估值系统"定量分析"子模块的数据处理篇，核心是定义一组函数把财报原始数据整理成可分析的结构。
- merge_subsets_yearly：把多个子集（dates/values 结构）并入总集，按日期去重，同日期数值累加；支持 must_include 指定必须保留的日期集合，且要求传入元组。
- merge_subsets_most_recent：比较总集与各子集的日期，取最新日期并把该日期上各子集的数值求和，用于合并最新一期财务数据。
- extract_shares：从财报提取股本，兼顾普通股股数（EntityCommonStockSharesOutstanding，dei 口径，取时点值）与加权平均股数（WeightedAverageNumberOfSharesOutstandingBasic），并处理单位不一致。
- extract_income_statement：抽取收入（Revenues、含/不含税客户合同收入、SalesRevenueNet）并计算 TTM 与年度值，同时抽取研发费用（含剔除收购在研成本的版本）。
- extract_balance_sheet_current_assets：抽取现金及受限现金、各类存货、预付费用与其他流动资产、应收账款/票据/贷款、各类证券。
- extract_balance_sheet_noncurrent_assets：抽取权益法投资与其他金融资产、不动产厂房设备、投资性房地产、递延税项与未确认税收优惠。
- 整体设计是把"度量口径清单 → 取值 → 按年/最新值合并"抽象成可复用函数，并对财报单位错误等异常做兜底。

## 可复用要点
- 基本无。可借鉴的仅是"多来源时间序列按日期去重合并 + 最新值合并 + 缺失兜底为 {date:None, value:0}"的合并函数设计思路，与行情多源对齐问题形式相近，但字段体系完全不同。

## 关键实现
- 主要函数：merge_subsets_yearly(superset, subsets, must_include=None)、merge_subsets_most_recent(superset, subsets)、extract_shares(doc, quarter_of_annual_report, years_diff)、extract_income_statement(doc)、extract_balance_sheet_current_assets / extract_balance_sheet_noncurrent_assets(doc, quarter_of_annual_report, years_diff)。
- 依赖内部工具：build_financial_df、get_values_from_measures、get_most_recent_value_from_df、get_last_annual_report_date_and_fy，均未在本文给出实现。
- 关键参数：unit="shares"、"dei" 税源标签、instant=True 表示时点值（资产负债表项）、get_ttm 控制是否取滚动十二个月。

## 数据与假设
- 数据源为美股上市公司定期财报的 XBRL 条目，按 us-gaap / dei 度量名索引；标的是美股，非 A 股。
- 假设财报数据能以统一度量名映射，缺失值可安全置 0。

## 局限与风险
- 大量依赖前文未展示的辅助函数，本篇代码无法独立运行。
- 缺失值兜底为 0 有污染风险：区分"真的为 0"与"未披露"没有任何处理。
- 度量名清单是手工枚举的兼容写法，覆盖不全时会静默漏取。
- 面向基本面估值，与价格/收益预测所需的时序建模、回测、评价体系无关。
