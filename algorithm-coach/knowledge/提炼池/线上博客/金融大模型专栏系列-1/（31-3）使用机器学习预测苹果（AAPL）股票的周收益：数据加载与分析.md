---
source: 原始池/线上博客/金融大模型专栏系列-1/（31-3）使用机器学习预测苹果（AAPL）股票的周收益：数据加载与分析.md
column: 金融大模型专栏系列-1
title: （31-3）使用机器学习预测苹果（AAPL）股票的周收益：数据加载与分析
distilled: 2026-09-30
relevance: 高
---

# （31-3）使用机器学习预测苹果（AAPL）股票的周收益：数据加载与分析

## 筛选结论
保留 —— 给出从 yfinance 获取周频行情、计算周收益率、描述统计与分布可视化的完整范式，与本项目的数据获取、特征工程和标签构造环节直接相关。

## 核心内容
- 用 yf.download("AAPL", start='1990-01-01', end='2023-07-28', interval='1wk') 下载周频数据，返回 Open/High/Low/Close/Adj Close/Volume 六列，共 1752 行。
- 周收益率以复权收盘价构造：Adj Close.pct_change()*100 保留两位小数，首行无前值为 NaN。
- 描述统计：Adj Close 跨度 0.10~195.83，均值23.44、标准差42.62，说明价格量纲差异大，直接入模需标准化；成交量均值约1.78e9。
- 波动率随制度下降：1990s 单周最大涨24.86%/跌-25.17%；2000-10-25 单周暴跌50.66%；2020年最大单周跌幅-17.53%。
- 箱形图显示多数周收益集中在 -11.84%~+12.85%，最优周+39.74%，最差-50.66%；直方图集中在0%附近。

## 可复用要点
- 数据接口：yfinance download 的 ticker/start/end/interval='1wk' 参数组合与返回列结构可直接复用。
- 标签构造：以 Adj Close 的 pct_change 计算周收益，是周频预测标签的常用做法。
- 可视化组件职责划分（蜡烛图/累积收益/收益曲线/箱线+直方图）可作为回测与数据报告模板。
- 波动率时变提示：训练/验证切分必须按时间顺序，并考虑波动率区制变化。

## 关键实现
- plot_returns(df, name, period)：plotly go.Scatter 画周收益折线，带 rangeslider。
- histogram_boxplot(df, feat)：make_subplots 并排绘制箱形图与直方图。
- plot_candlestick(df, name, period)、plot_cumulative_returns(df, name, period)：原文只给调用与说明，未给函数体。
- 依赖 yfinance、pandas、numpy、plotly（graph_objects/make_subplots）。

## 数据与假设
- 数据源：Yahoo Finance；标的 AAPL；频率周频；范围 1990-01-01 至 2023-07-28；1752 条。
- 假设：Adj Close 已复权，pct_change 可代表周收益。

## 局限与风险
- 只做描述性分析，未建模、未预测、未回测，可预测性未验证。
- 未处理停牌/节假日导致的缺周与 1990 年代低价区间的复权误差。
- 单一标的、单一市场，波动率下降等结论外推到A股不成立。
- 部分图表数值依赖图片展示，脚本未给全量可复现代码。
