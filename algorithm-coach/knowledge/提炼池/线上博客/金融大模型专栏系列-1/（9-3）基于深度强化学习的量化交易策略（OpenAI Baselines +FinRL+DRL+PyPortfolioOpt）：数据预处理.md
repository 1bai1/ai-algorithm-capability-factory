---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-3）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：数据预处理.md
column: 金融大模型专栏系列-1
title: （9-3）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：数据预处理
distilled: 2026-09-30
relevance: 高
---

# （9-3）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：数据预处理

## 筛选结论
保留 —— 展示了 FinRL 风格的特征工程做法（技术指标 + 湍流指数 + 协方差矩阵状态），特征清单与状态构造方式可直接为本项目的特征工程环节提供候选集与实现路径。

## 核心内容
- 预处理分两步：先补技术指标与风险指标，再统一做标准化与缺失值处理。
- 技术指标示例为 MACD 与 RSI 两个趋势跟踪指标；处理后的实际列还包括 boll_ub / boll_lb（布林带上下轨）、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma，覆盖趋势、超买超卖、波动通道与均线族。
- 湍流指数（turbulence index）用于度量极端资产价格波动，目的是在 2007-2008 式危机中控制风险；本例中该开关被关闭（use_turbulence=False），仅作概念介绍。
- 协方差矩阵作为状态：以 lookback=252 个交易日（约一年）为回看窗口，滚动计算多资产收益率的协方差矩阵，让模型感知资产间联动与整体风险。
- 状态构造后数据规模从 (97524, 17) 变为 (90468, 19)，行数减少源于回看窗口的消耗，列数增加 2 列对应 cov_list 与 return_list。
- 工程流程：数据按 ['date','tic'] 排序、以日期因子化值作索引，再用 FeatureEngineer.preprocess_data 一步完成指标添加、标准化与缺失值处理。

## 可复用要点
- 特征清单可直接作为本项目特征工程候选：macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma，叠加原始 OHLCV、adjcp、volume，以及 day（星期几）这类日历特征。
- 状态构造：把 N×N 协方差矩阵连同收益率序列一起并入状态，回看窗口取 252（一年交易日），可迁移为多标的联动特征。
- 风险控制：湍流指数可作为极端市场状态信号，用于策略降仓/停手开关。
- 工程规范：指标计算、标准化与缺失填补统一走一个预处理类，避免手工处理口径不一致。

## 关键实现
- 类 FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)，入口方法 `df = fe.preprocess_data(df)`。
- 协方差矩阵构造：`df.sort_values(['date','tic'], ignore_index=True)`，`df.index = df.date.factorize()[0]`；窗口 `lookback=252`，截取 `df.loc[i-lookback:i, :]`，用 `pivot_table(index='date', columns='tic', values='close').pct_change().dropna()` 得收益率，再算协方差，回填 cov_list / return_list。
- 依赖 FinRL、pandas、numpy；输出数据字段含 date、open、high、low、close、adjcp、volume、tic、day 及各技术指标。

## 数据与假设
多只美股日线（示例含 AAPL、AMGN、AXP、BA、CAT），起始日期 2008-12-31；日频；样本 97524 行 × 17 列 → 90468 行 × 19 列；假设一年窗口的协方差足以表征资产间联动与整体风险。

## 局限与风险
- 协方差矩阵直接进状态会显著抬高状态维度，样本效率与训练稳定性风险原文未讨论。
- 标准化方式未说明（preprocess_data 内部实现未给出），若使用全样本统计量会引入前视偏差。
- 协方差窗口为滚动历史窗口，但收益率序列与协方差是否对齐到同一决策时点未说明，存在时序错位风险。
- 湍流指数被关闭，原文只介绍概念，没有效果验证。
- 缺少特征有效性检验（重要性、IC、增量收益），只是完成工程拼装。
