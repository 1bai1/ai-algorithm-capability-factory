---
source: 原始池/线上博客/AI金融实战系列-2（持续更新）/（1-2-5）金融时间序列分析：ARCH和GARCH模型.md
column: AI金融实战系列-2（持续更新）
title: （1-2-5）金融时间序列分析：ARCH和GARCH模型
distilled: 2026-09-30
relevance: 高
---

# （1-2-5）金融时间序列分析：ARCH和GARCH模型

## 筛选结论
保留 —— 波动率预测与 VaR 的完整可运行范式，可直接作为 A 股个股的风险侧模型与风险指标来源。

## 核心内容
1. ARCH（Engle, 1982）把条件方差建模为过去误差项平方的函数，突破"误差方差恒定"的假设；GARCH（Bollerslev, 1986）引入条件方差自身的滞后项，以更少参数刻画长期记忆性。
2. GARCH(1,1) 实际最常用，3 个参数即可拟合多数金融序列的波动率特征，可作默认基线。
3. 应用面：波动率预测、VaR 计算、资产配置与对冲（高波动期降仓）、修正期权定价的波动率恒定假设、市场有效性检验。
4. 流程：对数收益率（×100）→ 8:2 时序切分 → 多阶 ARCH/GARCH 以 AIC 选阶 → forecast 取条件方差 → 开方得波动率 → 以测试期收益绝对值为真实波动率算 MSE/RMSE/MAE；本案例 GARCH(1,1) 的 RMSE 低于 ARCH(1)。
5. VaR 取 95% 置信水平 z=1.645 乘预测波动率，并标记超限的极端负收益。

## 可复用要点
- 基线配置：arch_model(returns, vol='GARCH', p=1, q=1, mean='Zero')，均值方程置零、专注波动率。
- 收益率预处理：np.log(close).diff() * 100 后 dropna。
- 选阶：在 ARCH(1)/ARCH(2) 与 GARCH(1,1)/(1,2)/(2,1) 间按 AIC 筛选。
- 评价：MSE/RMSE/MAE 三指标；可输出未来 30 个交易日（freq='B'）的波动率路径。
- 风险指标：VaR_95 = 1.645 × σ_pred，统计超限次数可作覆盖性检验雏形；波动率序列可作门控特征供后续模型或仓位规则使用。

## 关键实现
- 依赖 pandas、numpy、matplotlib、arch、sklearn.metrics。
- arch_model(...).fit(disp='off') → .aic / .summary()；.forecast(horizon=N).variance.values[-1, :] 取条件方差，np.sqrt 转波动率。
- 自定义 evaluate_volatility(actual, predicted, model_name) 返回 (mse, rmse, mae)；日期以 format='%Y%m%d' 解析。

## 数据与假设
贵州茅台日线，2020-01-03 至 2023-09-01，890 条有效记录（训练 712 条，8:2 切分）；假设残差正态、均值方程为零、按时间顺序切分。

## 局限与风险
原文自述：假设波动率对称、无法捕捉杠杆效应，对极端事件预测有限，应改用 EGARCH/TGARCH。此外：单一标的且样本仅约 890 日，未做滚动重估与多次样本外验证；以收益绝对值作波动率代理本身含噪，两模型 RMSE 接近时结论不稳健；收益率乘 100 会改变似然尺度，跨数据集比较 AIC 需口径一致。
