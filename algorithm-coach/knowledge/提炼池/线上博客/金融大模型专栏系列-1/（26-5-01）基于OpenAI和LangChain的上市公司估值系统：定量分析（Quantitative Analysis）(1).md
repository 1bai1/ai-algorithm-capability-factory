---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-5-01）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(1).md
column: 金融大模型专栏系列-1
title: （26-5-01）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(1)
distilled: 2026-09-30
relevance: 中
---

# （26-5-01）基于OpenAI和LangChain的上市公司估值系统：定量分析（Quantitative Analysis）(1)

## 筛选结论
部分保留 —— 提供从 XBRL 财务数据构建 DataFrame、计算 TTM、提取年度/季度值的成体系数据接口，对财务基本面数据接入有参考价值；但估值模型本体（FCFF/股利）不属于行情预测场景。

## 核心内容
- 估值模型按达摩达伦原则建立，围绕自由现金流（FCFF）与股利构造4种场景：季度收益+历史增长率（及归一化版）、季度收益+季度增长率（及归一化版）。
- 每个场景额外计算衰退情形，取 FCFF、衰退FCFF、股利、衰退股利的中位数，再按衰退概率加权出两个期望值，最终估值取偏保守的低值。
- 财务数据从文档 facts 树提取：doc["facts"][tax][measure]["units"][unit]，tax 默认 "us-gaap"，单位默认 USD。
- build_financial_df 把提取结果转为 DataFrame，并把 val 规范为数值、start/end 规范为日期。
- get_ttm_from_df 用 end-start 天数区分期间类型（>100 天视为年度），取最后一个年度值后拼接其后季度值（减前对应季度值）计算 TTM。
- 一系列函数按财年 frame 字段（如 CY2022Q3）对齐年度与瞬时（资产负债表）数据：get_yearly_values_from_df 按年报发布季度锚点提年度值，并用 years_diff 做年份校正。

## 可复用要点
- TTM 计算法：以期间天数区分年/季报，年度值 + 最新季度 - 上年同期季度。
- 数据字段规范：us-gaap taxonomy 与 frame 字段（CYxxxxQxI）是接入美股财报的标准接口，可用于构造财务因子。
- 区分 instant（资产负债表）与非 instant（利润表/现金流）数据的一致性检查思路。
- "取衰退与正常情形期望的较低者"的保守估值思想，可用于风险调整后的目标价生成。

## 关键实现
- 函数：build_financial_df(doc, measure, unit="USD", tax="us-gaap")、get_ttm_from_df(df)、get_most_recent_value_from_df(df)、get_last_annual_report_date_and_fy(df)、get_quarter_of_annual_report(df, last_annual_report_date, last_annual_report_fy)、get_yearly_values_from_df(df, instant=False, quarter_of_annual_report=None, years_diff=0)、get_values_from_measures(doc, measures, get_ttm=True, get_most_recent=True, get_yearly=True, ...)。
- 依赖 pandas；输入为公司财务文档 JSON（含 facts 树），输出为 DataFrame 或 {"dates": [...], "values": [...]} 字典。

## 数据与假设
- 数据源：美股上市公司财报 XBRL 结构化数据；频率为季度+年度；原文未给具体公司与时间范围。

## 局限与风险
- 代码多处被截断，get_values_from_measures 等核心函数的完整逻辑与容错未见。
- 估值依赖"衰退概率"这一外部输入，原文未说明其来源与估计方法。
- 面向基本面估值而非价格/收益预测，与A股行情预测仅共享数据层经验。
- 保守取值会系统性偏低，原文未回测其对投资收益的实际影响。
