---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-5-03）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（3）.md
column: 金融大模型专栏系列-1
title: （29-5-03）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（3）
distilled: 2026-09-30
relevance: 高
---

# （29-5-03）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（3）

## 筛选结论
保留 —— 时间序列定阶（ACF/PACF）、按年份切分、带外生变量的 ARIMAX 建模与 RMSE 评估，是价格预测基线模型的完整可复用流程。

## 核心内容
- 用 PACF 图定阶：分别画原始序列、一阶差分、二阶差分的 PACF（lags=5），在相关值降到阈值以下的截止点确定 p 与 q。
- 原文给出的选型理由：ARIMA 比 ARMA 更好，因为能处理非平稳数据、可用差分去掉趋势与季节性、可包含季节成分。
- 数据按年份切分而非随机切分：2020 年之前为训练集（3867 行），2020 年及以后为测试集（770 行），理由是时间序列的顺序性、避免随机切分造成空隙。
- 用 ARIMA(1,1,1) 加外生变量（Open/High/Low）预测 Close，即多变量 ARIMAX；摘要显示外生变量系数均显著：Open -0.5615、High 0.7750、Low 0.7870（p≈0.000），ar.L1=-0.0658、ma.L1=-0.9998、sigma2=0.0234。
- 残差诊断：Ljung-Box Q=0.19（p=0.67，无显著自相关），Jarque-Bera=9485.55（p=0.00，非正态），异方差检验显著，偏度 0.10、峰度 10.67（尖峰厚尾）。
- 测试集逐步外推：对每个时点调用 model_fit.forecast(exog=test[exogenous_features].iloc[i]) 生成 Forecast 列。
- 多变量 ARIMAX 在测试集上的 RMSE = 0.7477（在价格水平上计算）。

## 可复用要点
- 定阶流程：对原序列与一/二阶差分画 PACF(lags=5)，按截止点取 p、q；参数示例为 (1,1,1)。
- 切分规则：train = df[df.index.year < 2020]、test = df[df.index.year >= 2020]，严格按时间先后，并核对训练/测试行数（3867/770）。
- 外生变量用法：exog=['Open','High','Low']，预测时逐时点传入当期外生变量。
- 残差诊断清单可直接作为模型验收项：Ljung-Box（自相关）、Jarque-Bera（正态性）、异方差检验、偏度与峰度。
- 评价指标：在价格序列上算 RMSE，便于直接解释预测误差幅度。

## 关键实现
- 使用 statsmodels 的 ARIMA（摘要输出为 SARIMAX Results）；训练集内预测用 train['Predictions'] = model_fit.predict()。
- 测试集预测用列表推导逐点 forecast：forecast = [model_fit.forecast(exog=test[exogenous_features].iloc[i]).values[0] for i in range(len(test))]。
- 定阶绘图为三行子图，基于 statsmodels.graphics.tsaplots.plot_pacf，并对原始/一阶差分/二阶差分分别作图。
- RMSE 计算：np.sqrt(mean_squared_error(test['Close'], test['Forecast'])).
- 数据列为 Date 索引（带 -04:00/-05:00 时区偏移）+ Open/High/Low/Close，样本自 2004-08-23 开始。

## 数据与假设
单只股票的日频 OHLC 数据（原文未给出标的名称），共约 4637 个样本；训练期 2004 年至 2019 年，测试期 2020-01 至 2023-01；endog 为 Close，exog 为 Open/High/Low；假设差分后序列平稳，且预测时点的外生变量已知（预测中使用了当期开高低价）。

## 局限与风险
逐步预测使用了当期 Open/High/Low，这在实盘中属于不可得的未来信息，RMSE 0.7477 存在乐观偏差（原文未讨论）；在价格水平而非收益率上评估，掩盖了相对误差；残差非正态、厚尾且异方差，ARIMA 的高斯假设不成立，理应改用 GARCH 类或稳健方法；模型参数只用训练期估计一次，未做滚动重估，长期外推能力未验证；"ARIMA 优于 ARMA"为教科书式论断，本项目未做对照实验。
