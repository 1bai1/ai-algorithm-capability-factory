---
source: 原始池/线上博客/金融大模型专栏系列-1/（31-2）使用机器学习预测苹果（AAPL）股票的周收益：工具函数.md
column: 金融大模型专栏系列-1
title: （31-2）使用机器学习预测苹果（AAPL）股票的周收益：工具函数
distilled: 2026-09-30
relevance: 高
---

# （31-2）使用机器学习预测苹果（AAPL）股票的周收益：工具函数

## 筛选结论
保留 —— 一套可直接复用的数据拆分、可视化与分类模型评估工具函数集，对应本项目的 EDA、回测报告与评价指标环节。

## 核心内容
- 本篇是"预测 AAPL 周收益"项目的工具函数库，覆盖数据拆分、可视化、模型评估三类能力。
- X_y_split：按目标列名把 DataFrame 拆成 X/y，并打印形状、样本数、特征数及前 10 行便于核对。
- 可视化函数族：特征相关性热图（用掩码去掉对称重复）、K 线图、累计收益曲线、周收益曲线、箱线图+直方图并排、双特征散点图（附 OLS 趋势线）、类别分布环形饼图。
- plot_model_performance：输入 model_name、y_test、y_pred，绘制带 AUC 标注的 ROC 曲线（含随机猜测虚线）与混淆矩阵热图。
- 评估口径面向二分类，即把周收益预测转化为方向分类（0/1）。

## 可复用要点
- 特征-目标拆分函数可直接用于任何监督式个股预测任务，含形状自检输出。
- 评价指标组合：ROC/AUC + 混淆矩阵；混淆矩阵前对概率输出做 round() 取硬标签。
- EDA/回测报告的可视化清单：收益曲线、累计收益曲线（以 Adj Close 为基准）、K 线、分布与离群值、特征相关性、类别分布。
- 累计收益以调整后收盘价（Adj Close）为口径，避免除权除息干扰。
- 相关性热图做对称掩码，散点图配 OLS 趋势线以定性判断线性关系。
- 时间周期以参数 period 传入，仅用于图表副标题，便于按周/月/年复用同一函数。

## 关键实现
- 函数签名：X_y_split(df, target_variable) -> (X, y)；plot_correlation(df)；plot_candlestick(df, name, period)；plot_cumulative_returns(df, name, period)；plot_returns(df, name, period)；histogram_boxplot(df, feat)；correlation_scatterplot(...)；pie_plot(df, feat)；plot_model_performance(model_name, y_test, y_pred)。
- 依赖：plotly（graph_objects、express、make_subplots）、sklearn.metrics（roc_curve、auc、confusion_matrix）、seaborn 主题、pandas。
- 关键参数：饼图 hole=0.75 做环形图、切片 pull=0.05；箱线图/直方图用 1×2 子图、horizontal_spacing=0.2；ROC 图 fill='tozeroy'、AUC 以 %0.4f 标注；图表高度/宽度 450、paper_bgcolor='#F6F5F5'。
- 原文多数函数只给出签名与片段，函数体被截断，需自行补全。

## 数据与假设
- 输入为 AAPL 周频行情数据框，含 Adj Close 及目标列（周收益方向标签）。
- 假设特征已在上游构造完成，本层只做拆分、展示与评估。

## 局限与风险
- 函数体大量缺失（如 correlation_scatterplot 无代码），不能照抄即用。
- plot_model_performance 直接把 y_pred 传入 roc_curve，若上游给的是硬标签则 ROC 退化为单点，须确认输入为概率或得分。
- 混淆矩阵用 round() 按 0.5 硬阈值切分，未考虑类别不平衡与阈值寻优。
- 全部为可视化与评估层，无特征构造、模型选择与统计显著性检验。
- 未涉及交易成本、滑点，收益曲线不能直接等价于策略净值。
