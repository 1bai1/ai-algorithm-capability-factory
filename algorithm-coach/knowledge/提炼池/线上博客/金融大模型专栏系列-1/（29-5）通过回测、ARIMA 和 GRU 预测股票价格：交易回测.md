---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-5）通过回测、ARIMA 和 GRU 预测股票价格：交易回测.md
column: 金融大模型专栏系列-1
title: （29-5）通过回测、ARIMA 和 GRU 预测股票价格：交易回测
distilled: 2026-09-30
relevance: 高
---

# （29-5）通过回测、ARIMA 和 GRU 预测股票价格：交易回测

## 筛选结论
保留 —— 提供可直接复用的滚动前向回测函数、概率阈值化信号生成、多尺度滚动均值比率特征与精确率对比，是 Agent 回测与特征工程环节的核心素材。

## 核心内容
- `backtest` 采用滚动前向切分：从第 1000 行开始，每次在前 i 行上训练、在其后 step 行上预测，随后 i 递增 step（step=50）直至覆盖全样本；每轮只用历史数据训练，避免前视。
- 信号不取 0.5 默认阈值，而是取 `predict_proba[:,1]` 后以 0.6 二值化；原文建议在 0.6~1 之间试不同阈值看能否改善。
- 新增特征分两类：多尺度滚动均值及其比值（7 日周均值、90 日季均值、365 日年均值；各均值/Close 比率、年均值/周均值、年均值/季均值），以及日内价格比率（Open/Close、High/Close、Low/Close）；另加 Target 的周趋势（shift(1) 后取 7 日均值）。
- 空值处理：先逐列统计缺失比例（周均值 0.13%、季均值 1.92%、年均值及其派生列 7.85%），再统一 fillna(0)；原文自承这不是最优策略。缺失源自滚动窗口在序列早期必然不足。
- 特征列从 5 列扩展到 15 列。加入新特征后精确率由 67.83% 提升到 85.28%。
- 分类报告（support 3637）：类 0 精确率 0.80 / 召回 0.85，类 1 精确率 0.85 / 召回 0.80，准确率 0.82，macro 与 weighted F1 均 0.82，两类表现均衡。
- 回测输出为逐段拼接的 DataFrame，含 Target 与 Predictions 两列、日期索引，可直接用于分段指标计算或错误分布作图（countplot）。

## 可复用要点
- 回测框架模板（可直接移植）：`backtest(data, model, predictors, start=1000, step=50)`——扩窗训练 + 等长测试段滚动，返回拼接后的预测序列；比一次性 train_test_split 更贴近实盘使用方式。
- 阈值化信号：`preds = model.predict_proba(test[predictors])[:, 1]`，再按业务阈值（原文 0.6）转 0/1；阈值应作为可调参数而非固定 0.5。
- 特征模板：多尺度滚动均值相对当前收盘价的比率（7/90/365 日）+ 均值间比率（年/周、年/季）+ 日内比率（开/收、高/收、低/收）+ 目标周趋势。比率型特征对量纲不敏感，跨标的与跨时段迁移性优于原始价格。
- 特征增益归因：固定同一模型与切分方式，只替换 predictors 集合，才能把指标变化归于特征工程（本例 67.83% → 85.28%）。
- 缺失值体检先行：逐列打印空值占比再决定填充方式；滚动窗口越长，早期空值比例越高（365 日窗口对应约 7.85%）。

## 关键实现
- `def backtest(data, model, predictors, start=1000, step=50)`：`train = data.iloc[0:i].copy()`、`test = data.iloc[i:(i+step)].copy()`；`model.fit(train[predictors], train["Target"])`；预测取正类概率并按 0.6 阈值二值化后 `pd.Series(preds, index=test.index)`；每轮 `pd.concat({"Target": ..., "Predictions": ...}, axis=1)`，最后 `pd.concat(predictions)` 返回。
- 滚动特征：`df.rolling(7).mean()`、`rolling(90).mean()`、`rolling(365).mean()`；`df.shift(1).rolling(7).mean()["Target"]` 作周趋势；比率列如 `df["open_close_ratio"] = df["Open"] / df["Close"]`。
- predictors 完整清单（14 个）：Open、High、Low、Close、Volume、weekly_mean、quarterly_mean、annual_mean、annual_weekly_mean、annual_quarterly_mean、weekly_trend、open_close_ratio、high_close_ratio、low_close_ratio。
- 评价与诊断：`precision_score(...)`、`classification_report(...)`，以及错误预测的计数图与 `time.time()` 计时。

## 数据与假设
- 数据为 GOOG 日频行情（Date 索引，列含 Open/High/Low/Close/Volume），样本区间原文未给出；回测后有效样本 3637 行。
- Target 为二分类 0/1（涨跌方向，前序章节定义），属分类任务而非收益率回归；模型需支持 predict_proba。
- 滚动均值自数据集首行起算，故前 365 行相关特征缺失并被填 0；训练起点设为 1000 行使测试段特征基本有效。
- 精确率的语义是"预测为正且实际也为正"，即重点衡量买入信号的可靠性。

## 局限与风险
- `fillna(0)` 使早期样本的多尺度均值比率变为 0 或虚假持平信号，模型可能学到"早期即异常"的伪模式；原文也承认不是最优做法。
- 完全没有交易成本、滑点、涨跌停与 T+1 约束；精确率提升不等于策略净值提升，二者的落差未做验证。
- 阈值 0.6 是手工设定，无敏感性分析；提高阈值以牺牲召回换精确率，需按策略目标权衡。
- 特征一次性在全样本上计算后再回测：滚动窗口本身为后视，尚不构成泄露，但若后续加入需要跨样本拟合的变换（标准化、编码、填充统计量），必须移入滚动循环内部，否则会引入未来信息。
- 单标的、单次回测，缺跨标的与跨时段稳健性验证。
- 67.83% → 85.28% 的对比未说明是否使用同一模型超参与随机种子，存在不可比风险。
