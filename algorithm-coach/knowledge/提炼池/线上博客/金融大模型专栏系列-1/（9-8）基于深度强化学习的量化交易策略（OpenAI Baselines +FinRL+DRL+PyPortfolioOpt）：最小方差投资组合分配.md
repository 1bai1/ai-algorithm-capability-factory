---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-8）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：最小方差投资组合分配.md
column: 金融大模型专栏系列-1
title: （9-8）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：最小方差投资组合分配
distilled: 2026-09-30
relevance: 中
---

# （9-8）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：最小方差投资组合分配

## 筛选结论
部分保留 —— 组合层的最小方差权重求解与"策略 vs 基准"累计收益对比逻辑可用，但目标是多资产组合配置而非 A 股个股预测，且组合是逐日按协方差重算的静态最优解。

## 核心内容
- 用 PyPortfolioOpt 的 EfficientFrontier 做最小方差组合：以当日收益率协方差矩阵为输入，权重上下限设为 (0, 0.1)，调用 min_volatility() 求原始权重、clean_weights() 取清洗后权重。
- 组合推进按日循环：取出当日与次日数据，用当日权重和当前总资产算出每只股票的买入金额，除以当日 close 得到持股数，再乘次日 close 得到下一期账户价值，初始资金 100 万。
- 协方差来源是数据集中预置的 return_list 字段（逐日保存的收益率序列），`df_temp.return_list[0].cov()` 直接取协方差矩阵。
- 评价方式是三条累计收益曲线对比：A2C 模型组合、最小方差组合、道琼斯指数基准，均用 `(日收益+1).cumprod()-1` 换算为累计收益。
- 可视化用 Plotly 的 go.Scatter 分别绘制三条曲线并叠加在同一 Figure 中比较。
- 本系列结论指向 DRL、最小方差与 DJIA 的横向比较，本节的产出是最小方差这条基线。

## 可复用要点
- 组合优化接口：`EfficientFrontier(None, Sigma, weight_bounds=(0, 0.1)).min_volatility()` + `clean_weights()`，可直接用于 A 股多标的的权重生成；0.1 上限强制分散。
- 账户价值递推写法可直接复用：`shares = 资金 × 权重 / 当日 close`，`账户价值 = Σ shares × 次日 close`，是最小化假设下的日频调仓回测循环。
- 评价口径：所有策略统一换算成累计收益曲线（cumprod-1）再与指数基准同图比较，避免只看绝对收益。
- 数据字段模板：date、open、high、low、close、adjcp、volume、tic、day + 8 个技术指标（macd、boll_ub/lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma）。

## 关键实现
- 依赖：`pip install PyPortfolioOpt`、`pip install plotly`；模块 `from pypfopt.efficient_frontier import EfficientFrontier`、`from pypfopt import risk_models`。
- 关键变量：initial_capital = 1000000，portfolio 为 1×unique_trade_date 的结果表；循环 `for i in range(len(unique_trade_date)-1)`。
- 协方差：`Sigma = df_temp.return_list[0].cov()`；权重清洗后按股票代码顺序对齐到当日 close 列。
- 累计收益：`(portfolio.account_value.pct_change()+1).cumprod()-1`、`(df_daily_return.daily_return+1).cumprod()-1`、`(baseline_returns+1).cumprod()-1`。
- 绘图：trace0/trace1/trace2 分别对应 A2C、DJIA、Min-Variance，另有 DDPG、Adaptive-DDPG、Min-Variance 的注释代码可启用扩展对比。

## 数据与假设
- 标的：美股（样例行为 AAPL、AMGN、AXP、BA、CAT 等），日频；样例日期从 2008-12-31 起，结果区间显示 2020-07，说明使用 FinRL 默认数据集。
- 假设：每日可按收盘价无成本调仓；协方差矩阵已由外部预先算好并存在 return_list 字段中；权重上限 0.1 保证分散。

## 局限与风险
- 完全未考虑交易成本、滑点、印花税与最小交易单位，每日调仓的换手成本会显著侵蚀最小方差组合的收益。
- 用当日及历史数据算协方差、却按次日价格结算，属于典型的静态优化+一期持有近似，未做滚动样本外检验，也未说明 return_list 的窗口长度（若含全样本信息则有前视嫌疑）。
- 只有累计收益一条对比线，缺波动率、夏普、最大回撤等风险指标，无法判断"最小方差"目标是否真的降低了风险。
- 与 A 股个股预测 Agent 的契合度有限：多资产组合配置、美股数据、日频无摩擦假设均需改造。
- 原文多处代码被注释掉（DDPG、Adaptive-DDPG），对比不完整。
