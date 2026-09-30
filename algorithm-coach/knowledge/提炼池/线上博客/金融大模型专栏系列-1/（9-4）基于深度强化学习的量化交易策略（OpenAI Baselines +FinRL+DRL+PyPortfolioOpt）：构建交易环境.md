---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-4）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：构建交易环境.md
column: 金融大模型专栏系列-1
title: （9-4）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：构建交易环境
distilled: 2026-09-30
relevance: 高
---

# （9-4）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：构建交易环境

## 筛选结论
保留 —— 把组合交易显式建模为 MDP 并用 OpenAI Gym 实现 StockPortfolioEnv，状态构造、动作归一化、交易成本与绩效记录接口可直接迁移到本项目的策略与回测环节。

## 核心内容
1. 问题建模：自动股票交易具随机性与交互性，建为马尔可夫决策过程（MDP）；代理观测股价变化、执行动作、计算奖励，通过与环境交互学出随时间最大化奖励的策略。
2. 环境基于 OpenAI Gym，按时间驱动模拟真实市场，使用真实行情数据。
3. 数据拆分：`data_split(df, '2009-01-01','2020-07-01')` 得训练集；交易/回测集区间在原文中被注释掉未执行。
4. 状态空间：每个时点取该日协方差矩阵 cov_list 展平后与技术指标列表 tech_indicator_list 的指标值拼接；技术指标含 macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma。
5. 动作：代理输出各股票权重，经 softmax_normalization 归一化以保证权重合法（和为 1）。
6. 环境参数 8 项：hmax（单次最大交易股数）、initial_amount（初始资金）、transaction_cost_pct（交易成本百分比）、state_space、stock_dim、tech_indicator_list、action_space（=股票数）、reward_scaling（奖励缩放，利于训练）。
7. 关键接口：step 推进一步并在末端输出总资产、夏普比率等信息并保存组合表现图表；reset 重置环境（asset_memory 置为 [initial_amount]、day=0、动作记忆初始化为等权 [1/stock_dim]）；save_asset_memory 输出逐日 date+daily_return 表，save_action_memory 输出逐日持仓表。
8. 与 Stable-Baselines3 对接：get_sb_env() 用 DummyVecEnv 把环境包装成向量化环境供 SB3 算法训练（向量化可并行加速）。

## 可复用要点
- MDP 三要素落地口径（可直接作为策略生成模板）：状态 = 协方差矩阵 + 技术指标；动作 = 各标的权重（softmax 归一化）；奖励经 reward_scaling 缩放。
- 现实约束必须显式参数化，回测才可信：初始资金、单次交易上限、交易成本百分比三项都在环境初始化时传入。
- 逐日记录组合日收益、动作/持仓与账户价值，是绩效归因与净值曲线的基础（对应 save_asset_memory / save_action_memory）。
- 终止条件用 `self.day >= len(self.df.index.unique()) - 1`，即按交易日推进到数据末端。
- 绩效口径：终止时输出总资产与夏普比率；初始持仓用等权 1/stock_dim 作基准。
- 随机性控制：`_seed()` 用 `seeding.np_random(seed)` 固定环境随机性，保证可复现。

## 关键实现
类 `StockPortfolioEnv(gym.Env)`，属性 df、stock_dim、hmax、initial_amount、transaction_cost_pct、reward_scaling、state_space。方法：__init__、step(actions)、reset()、render(mode='human')、softmax_normalization(actions)、save_asset_memory()、save_action_memory()、_seed(seed=None)、get_sb_env()。
- softmax：`np.exp(actions) / np.sum(np.exp(actions))`
- get_sb_env：`e = DummyVecEnv([lambda: self]); obs = e.reset(); return e, obs`
- 依赖 gym、numpy、pandas、stable_baselines3.common.vec_env.DummyVecEnv。
- 原文示例输出 Stock Dimension: 28, State Space: 28 —— 即把 state_space 直接设为股票数，与 reset() 实际拼接出的状态维度口径不一致，原文未解释。

## 数据与假设
输入 df 为多标的日频面板，来自前述 Yahoo 道指成分股数据加技术指标：date/open/high/low/close/adjcp/volume/tic/day 以及 macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma、cov_list、return_list。假设市场可按日离散推进、交易成本按成交额固定比例计提、协方差矩阵与技术指标足以刻画状态。

## 局限与风险
- 原文把交易集的 data_split 行注释掉，只演示了训练集；本篇无训练、无回测、无绩效数字，环境是否可用未验证；初始资金、成本比例、reward_scaling 的具体取值均未给出。
- step 中的奖励公式未展示，奖励定义不透明；交易成本仅按固定百分比计，未建模冲击成本、滑点与流动性限制，hmax 只约束单次股数。
- state_space 与实际状态维度口径不一致，照抄可能引发维度错误；softmax 只保证权重归一，未约束单票集中度。
- 状态包含 28×28 协方差矩阵的高维展开，易过拟合，原文未做降维与敏感性分析。
