---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-6-01）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（1）.md
column: 金融大模型专栏系列-1
title: （29-6-01）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（1）
distilled: 2026-09-30
relevance: 高
---

# （29-6-01）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（1）

## 筛选结论
保留 —— 典型的日频股票多特征 + GRU 回归建模流程，特征清单、标准化方法与滑动窗口样本构造函数都可直接迁移到 A 股预测建模。

## 核心内容
- 序列数据建模选用 RNN 家族：标准 RNN 存在梯度消失问题，LSTM 与 GRU 用门控机制解决；GRU 参数更少、更快，本项目选用 GRU。
- 选定 13 个预测变量：Open、High、Low、Close、weekly_mean、quarterly_mean、annual_mean、annual_weekly_mean、annual_quarterly_mean、weekly_trend、open_close_ratio、high_close_ratio、low_close_ratio；显式剔除 Volume 与 Target。
- 数据标准化给出四种方法（归一化、标准化、Min-Max 缩放、鲁棒缩放），本文采用 MinMaxScaler 把所有特征压到 0–1。
- 用滑动窗口把时序 DataFrame 转成 (样本数, 时间步, 特征数) 的 X 与对应标签 y，窗口默认 13。
- 标签取窗口后一行的第 0 列（Open），即预测下一日开盘价。
- 工具链：Keras Sequential + GRU 层 + 早停回调 + RMSE 指标 + Adam 优化器。
- 示例数据规模 4637 行 × 13 列，时间跨度 2004-08-23 至 2023-01-23。

## 可复用要点
- 特征工程清单：周/季/年三级均值及交叉均值（annual_weekly_mean、annual_quarterly_mean）、周趋势标记 weekly_trend、三类 OHLC 比率（open_close / high_close / low_close），可直接作为价格类因子模板。
- 标准化：MinMaxScaler 对所有特征列统一 fit/transform。
- 样本构造参数：window_size=13；y 取窗口后一行指定列，便于把预测目标换成任意列（收盘价/收益率）。
- 训练配置：EarlyStopping 防过拟合、RootMeanSquaredError 监控、Adam 优化器。
- 数据接口字段：Date、Open、High、Low、Close、Volume、Target 及上述衍生列。

## 关键实现
- 函数 `df_to_X_y2(df, window_size=13)`：`df.to_numpy()` 后按 `df_as_np[i:i+window_size]` 切窗口入 X，取 `df_as_np[i+window_size][0]` 作标签入 y，返回两个 np.array。
- 模型：`Sequential` + `InputLayer` + `GRU` + `Dense`；回调 `EarlyStopping`；指标 `RootMeanSquaredError`；优化器 `Adam`。
- 依赖：tensorflow.keras、sklearn.metrics.mean_squared_error、sklearn.preprocessing.MinMaxScaler。
- 标准化写法：`scaler = MinMaxScaler(); scaler.fit(NN_df[NN_df.columns]); NN_df[NN_df.columns] = scaler.transform(...)`。

## 数据与假设
- 数据：2004-08-23 至 2023-01-23 的日频行情，4637 行；示例价格从约 2.5 涨到约 92（同一价格口径的连续序列）。
- 频率：日频；预测目标为下一日开盘价。
- 假设：13 个特征均为历史可得的滞后/衍生量，不引入未来信息。

## 局限与风险
- 本篇未展示训练/测试损失与预测精度，模型效果无法判断。
- 归一化在全量数据上 fit 后再用于建模，若与训练/测试切分配合不当会造成信息泄露（本篇尚未出现切分步骤）。
- 周/季/年均值类特征的统计窗口口径未说明，若按全样本统计则含未来信息。
- 未涉及交易成本、涨跌停与可交易性约束，"预测价格"到"可执行策略"之间仍有距离。
- 图片未收录。
