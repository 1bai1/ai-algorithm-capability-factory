---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-5-01）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（1）.md
column: 金融大模型专栏系列-1
title: （29-5-01）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（1）
distilled: 2026-09-30
relevance: 高
---

# （29-5-01）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（1）

## 筛选结论
保留 —— ARIMA 原理与平稳性/季节性检验流程，属时间序列建模环节，可作 Agent 选型时的统计基线方法。

## 核心内容
- ARIMA(p,d,q)：AR 用过去值的加权线性组合，I 用 d 阶差分去趋势/季节使序列平稳，MA 用过去预测误差加权修正。
- ARMA 面向平稳序列，ARIMA 通过差分扩展到非平稳序列。
- 平稳性判据：均值、方差与自相关随时间恒定；任一随时间变化即非平稳。
- ADF 检验流程：逐列输出 ADF 统计量、p 值与临界值，p ≤ 0.05 判为平稳。
- 示例中前四列数据的 ADF 统计量 -0.198、p 值 0.939，判为非平稳，需差分。
- 季节性分解把序列拆成趋势、季节、残差三部分，周期设为 365（对应日频年度周期）。

## 可复用要点
- 建模前先做 ADF 单位根检验，用 p ≤ 0.05 决定差分阶数 d。
- 季节性分解参数 period=365 适用于日频数据的一年周期设定。
- 误差评价准备用 MSE（已导入 sklearn 的 mean_squared_error）。
- 工具链：statsmodels 的 adfuller、seasonal_decompose、plot_acf、plot_pacf。

## 关键实现
依赖 statsmodels.tsa.stattools(adfuller)、seasonal_decompose、plot_acf/plot_pacf、sklearn.metrics.mean_squared_error。核心循环：for i in df.columns[:4]: ts.adfuller(df[i])，按 result[1] ≤ 0.05 判定平稳；seasonal_decompose(df[j], period=365) 取 .trend/.seasonal/.resid 并绘三联子图。

## 数据与假设
df 前四列的日频时间序列，属"通过回测、ARIMA 和 GRU 预测股票价格"项目的一节；本篇未交代具体标的、字段含义与时间区间；假设日频数据存在年度季节周期。

## 局限与风险
只有检验与分解，未给出 p/d/q 定阶、拟合参数、预测结果与精度，ARIMA 效果无法评估；ADF 输出示例未说明对应哪一列；对股票价格做 365 天季节分解的假设较强（价格通常无稳定年度季节性）；未讨论差分带来的信息损失与多重检验问题。
