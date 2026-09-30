---
id: evidence.dow30.finrl_drl
name: 道琼斯成分股 × FinRL 深度强化学习
category: 07_验证证据
status: 已验证
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-1）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：背景介绍+项目目标+模块架构.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-2）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：准备环境+下载数据.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-3）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：数据预处理.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-4）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：构建交易环境.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-5）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：深度强化学习算法模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-6）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：交易.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-7）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：回测交易策略.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-8）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：最小方差投资组合分配.md
---

# 道琼斯成分股 × FinRL 深度强化学习

## 能力说明
本卡记录 FinRL（Stable-Baselines3 + gym 组合环境）在道指成分股上的训练与样本外交易数字：A2C 交易段期末 1,363,804 美元（+36.4%）、夏普 1.8078，以及 Pyfolio 对策略与道指基准的逐项对比。它是判断"DRL 组合交易是否真的能跑赢指数"的关键证据。

## 输入契约
- 标的：FinRL 预置道指成分股清单 config_tickers.DOW_30_TICKER；Yahoo Finance 日频，2008-01-01 至 2021-10-31，下载结果 101,615 行 × 9 列（多标的长表；含 date/open/high/low/close/adjcp/volume/tic/day）。
- 特征工程：8 个技术指标 macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma；湍流指数开关关闭（use_turbulence=False）。
- 状态构造：以 lookback=252 个交易日滚动计算多资产收益率协方差矩阵，展平后与技术指标拼接；数据规模 (97524,17) → (90468,19)。
- 切分：训练 `data_split(df,'2009-01-01','2020-07-01')`；交易/回测 `data_split(df,'2020-07-01','2021-10-31')`，trade 形状 (9436, 19)。

## 输出契约
- 环境输出：逐日资产价值（date + daily_return）与逐日持仓权重（date × 各 ticker），落盘 df_daily_return.csv / df_actions.csv。
- 训练日志：每个 episode 的 begin_total_asset / end_total_asset / total_reward / Sharpe 等。
- 回测：Pyfolio `timeseries.perf_stats` 全指标 + `create_full_tear_sheet`；最差 5 段回撤与压力事件。

## 调用方式
```
env = StockPortfolioEnv(df=trade, **env_kwargs); env_train, _ = e_train_gym.get_sb_env()
agent = DRLAgent(env=env_train); model = agent.get_model("a2c"|"ddpg"|"sac"|"td3"|"ppo", model_kwargs=PARAMS)
trained = agent.train_model(model=model, tb_log_name='a2c', total_timesteps=50000)
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)
pyfolio.timeseries.perf_stats(returns=DRL_strat, factor_returns=DRL_strat, turnover_denom="AGB")
backtest_stats(get_baseline(ticker="^DJI", start=..., end=...), value_col_name='close')
```

## 关键参数
- 环境：动作 = 各股票权重经 softmax 归一化（和为 1）；hmax、initial_amount、transaction_cost_pct、reward_scaling、state_space 等 8 项参数；重置时动作记忆初始化为等权 [1/stock_dim]。
- 训练超参：A2C {n_steps 5, ent_coef 0.005, learning_rate 0.0002, 50000 步}；PPO {n_steps 2048, ent_coef 0.005, learning_rate 0.0001, batch_size 128, 80000 步}；DDPG {batch_size 128, buffer_size 50000, learning_rate 0.001, 50000 步}；SAC {batch_size 128, buffer_size 100000, learning_rate 0.0003, learning_starts 100, ent_coef auto_0.1}；TD3 {batch_size 100, buffer_size 1000000, learning_rate 0.001, 30000 步}。
- 初始资金 1,000,000；交易段 2020-07-01 至 2021-10-31。
- 最小方差基线（9-8）：PyPortfolioOpt EfficientFrontier，权重上下限 (0, 0.1)，min_volatility()，每日按收盘价无成本调仓，初始 100 万——原文未给出该基线的绩效数值。

## 依赖
FinRL（DRLAgent、StockPortfolioEnv、YahooFinanceProcessor、config_tickers）、Stable-Baselines3、gym、pandas/numpy、pyfolio、TensorBoard、PyPortfolioOpt、plotly。

## 适用条件
- 多标的（20–30 只）日频组合交易、样本 10 年以上、允许按权重连续调仓的场景；想复现"DRL 组合 vs 指数"对比时。
- 需要研究 RL 在组合权重分配（而非单标的择时）上表现时，本卡的训练日志与 Pyfolio 报表可作对照。

## 不适用条件
- 不能把训练日志里的"100 万 → 400 多万"当作策略业绩：那是训练 episode 内的资产变化，不是样本外结果。
- 唯一样本外结果（A2C 交易段）在年化收益、累计收益、夏普上均略低于道指基准，不能声称跑赢指数。
- 协方差展平状态与 state_space=28 的口径不一致（原文自己的输出），状态维度的正确性未经校验。
- 未给出 transaction_cost_pct 的具体取值，无法判断回测是否扣费。

## 验证状态
原文给出真实运行结果：
- 数据：下载 101,615 行 × 9 列；预处理后 (97524,17) → (90468,19)；trade (9436, 19)。
- 训练日志（周期末资产 / 夏普）：DDPG begin 1,000,000 → end 4,369,306.145455855（Sharpe 0.8034072979350758）与 end 4,365,995.854896107（Sharpe 0.8200827579868865）；SAC end 4,774,375.224598323（0.8157447898211176）、4,851,457.312329918（0.817397961885012）、4,851,717.33279626（0.8174262460980435）、4,851,205.14751689（0.8173829155723342）；TD3 end 4,609,152.895393911（0.8172592399889653）、4,798,090.361426867（0.835226336478133）。
- A2C 样本外交易（2020-07-01~2021-10-31）：begin_total_asset 1,000,000 → end_total_asset 1,363,803.996631671，Sharpe 1.8078156710226434；首日各票等权 0.035714，之后典型权重 0.0267~0.0725。
- Pyfolio（DRL 策略）：Annual return 0.261142（26.114%）、Cumulative returns 0.363804（36.38%）、Annual volatility 13.33%、Sharpe 1.81、Calmar 3.32、Stability 0.91、Max drawdown -7.871%、Omega 1.35、Sortino 2.74、Skew -0.18、Kurtosis 1.13、Tail ratio 1.07、Daily VaR -1.584%、Alpha 0.00、Beta 0.94。
- 基准 ^DJI（336×8）：Annual return 0.279047、Cumulative returns 0.388402、Annual volatility 0.139129、Sharpe 1.84456、Calmar 3.124551、Stability 0.918675、Max drawdown -0.089308、Omega 1.35896、Sortino 2.734872、Tail ratio 1.052781、Daily VaR -0.01651。
- 最差回撤段：7.87%（2020-09-02 峰值 / 2020-10-28 谷值 / 2020-11-09 恢复，49 天）、5.17%（2021-08-16 起，未恢复）、4.06%（40 天）、3.52%（11 天）、3.40%（13 天）；压力事件 New Normal 均值 0.10%、区间 -3.32%~3.32%。

### 可信度评估
- 样本量：训练约 11 年、样本外约 1 年 4 个月（336 个交易日），对 DRL 而言样本外太短。
- 泄漏：训练与交易段无重叠；协方差状态用 lookback=252 滚动构造，方向正确；但状态维度口径不自洽，且环境参数（交易成本/缩放）未完整披露。
- 基准对比：有 ^DJI 同区间对照——这是一个诚实且有价值的对比，结论是策略未跑赢指数（累计 36.38% vs 38.84%，夏普 1.81 vs 1.84），仅在最大回撤上更小（-7.87% vs -8.93%）。
- 结论稳定性：中等。训练日志的高资产值不能作为业绩；样本外一年的结果在统计上不显著，且不同算法训练表现（DDPG/SAC/TD3）差异不足以区分。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（9-1）背景介绍+项目目标+模块架构.md —— MDP 建模、动作空间、奖励 r=v'−v、六大模块与绩效口径。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-2）准备环境+下载数据.md —— Yahoo 数据、DOW_30_TICKER、2008-01-01~2021-10-31、101,615 行×9 列。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-3）数据预处理.md —— 8 个技术指标、lookback=252 协方差状态、(97524,17)→(90468,19)。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-4）构建交易环境.md —— StockPortfolioEnv、softmax 归一化、8 项环境参数、get_sb_env。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-5）深度强化学习算法模型.md —— A2C/PPO/DDPG/SAC/TD3 的超参与各次训练日志的期末资产与夏普。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-6）交易.md —— 交易段 (9436,19)、end_total_asset 1,363,803.996631671、Sharpe 1.8078156710226434、首日等权 0.035714。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-7）回测交易策略.md —— Pyfolio 策略与 ^DJI 基准的全部指标、最差 5 段回撤与压力事件。
- 提炼池/线上博客/金融大模型专栏系列-1/（9-8）最小方差投资组合分配.md —— PyPortfolioOpt 最小方差基线的构建方式（权重上限 0.1、每日无成本调仓、初始 100 万）；无绩效数字。

## 相关能力

- 上游依赖：[[复用池/03_预测模型/深度强化学习交易模型（A2C-PPO-DDPG-SAC-TD3）|深度强化学习交易模型（A2C-PPO-DDPG-SAC-TD3）]] —— 本卡是该能力五算法（A2C/PPO/DDPG/SAC/TD3）训练日志与 A2C 样本外交易的实测。
- 上游依赖：[[复用池/04_交易策略/强化学习组合持仓权重策略|强化学习组合持仓权重策略]] —— 动作是 softmax 归一化的组合权重、逐日输出持仓权重，是该策略路线在本池里最新的实证。
- 下游用途：[[复用池/05_回测系统与风险评估/交易绩效指标与回撤分析|交易绩效指标与回撤分析]] —— 策略与 ^DJI 的收益序列由 Pyfolio perf_stats 全套指标与最差 5 段回撤报告，本卡的绩效数字出自该能力。
- 常见误用：[[复用池/06_失败经验/回测无买入持有基准与风险指标|回测无买入持有基准与风险指标]] —— 有 ^DJI 同区间对照却显示 Alpha 0.00、Beta 0.94，收益基本来自跟随大盘，原文未作解释。
- 常见误用：[[复用池/06_失败经验/回测无交易成本与滑点|回测无交易成本与滑点]] —— 权重逐日变动却未给出 transaction_cost_pct 取值，也无换手约束说明，无法判断是否扣费。
- 并列/替代：[[复用池/07_验证证据/A股15只_FinRL的A2C策略|A股15只_FinRL的A2C策略]] —— 同一 FinRL 框架在 A 股 15 只上的对照实证（5 个月亏损 4.75%），两卡合看才能判断该框架的适用边界。
