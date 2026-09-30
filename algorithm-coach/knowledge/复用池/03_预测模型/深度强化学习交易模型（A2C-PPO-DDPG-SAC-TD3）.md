---
id: model.drl.trading
name: 深度强化学习交易模型（A2C/PPO/DDPG/SAC/TD3）
category: 03_预测模型
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-1）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：背景介绍+项目目标+模块架构.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-2）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：准备环境+下载数据.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-3）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：数据预处理.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-4）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：构建交易环境.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-5）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：深度强化学习算法模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-6）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：交易.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-7）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：回测交易策略.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-8）基于深度强化学习的量化交易策略（OpenAI Baselines +FinRL+DRL+PyPortfolioOpt）：最小方差投资组合分配.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-1）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-2）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-3）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（27-1）基于深度强化学习（DRL）的比特币交易系统：背景介绍+系统介绍.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（27-3）基于深度强化学习（DRL）的比特币交易系统：比特币交易数据集.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（27-4）基于深度强化学习（DRL）的比特币交易系统：测试验证.md
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/(6-1-02)高频交易与量化交易：机器学习在高频交易中的应用.md
---

# 深度强化学习交易模型（A2C/PPO/DDPG/SAC/TD3）

## 能力说明
把交易决策建模为马尔可夫决策过程（MDP）：代理观察市场状态，输出仓位/交易动作，以组合价值变化为奖励，用 DRL 算法（A2C/PPO/DDPG/SAC/TD3/DQN）学出端到端的交易或组合权重策略。模型侧产出的是"每交易日的各标的权重或买卖股数"，不是价格预测值。原文在美股组合上有正收益与完整 pyfolio 报告，但 A 股实例亏损且存在配置错误，故本卡按"记录踩坑"处理。

## 输入契约
- 多标的长表面板（date + tic 纵向堆叠），字段：date、open、high、low、close、adjcp、volume、tic、day。
- 技术指标列（FinRL 预处理固定集，8 个）：macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma。
- 可选状态增强：以 lookback=252 个交易日滚动计算的协方差矩阵 + 收益率序列（cov_list / return_list），状态维度从 (97524, 17) 变为 (90468, 19)。
- 数据规模示例：
  - 美股道指 30 只：2008-01-01 至 2021-10-31，'1D'，101,615 行 × 9 列。
  - A 股 15 只沪市股票：训练 2015-01-01 至 2019-08-01，交易 2019-08-01 至 2020-01-03；清洗后加指标 (18270, 17)，训练集 (16695, 17)。
  - BTC 小时级：2020-08-17 04:00 起 27812 条，特征 ema_13/25/32/100/200、vol_close=(high-low)/close 及其 3/6/12 周期 EMA、hour、day。
- 状态空间公式（多标的）：`state_space = stock_dim × (指标数 + 2) + 1`（每只股票贡献"指标数 + 现金/持仓 2 维"，整体再加 1 个时间步）；示例 15 × 10 + 1 = 151。
- 数据划分：训练/交易段必须按时间切分，交易段单独留出（示例 `data_split(df, '2020-07-01','2021-10-31')`，trade 表 9436×19）。

## 输出契约
- 每交易日的各标的权重矩阵（df_actions，行=日期、列=ticker，典型取值 0.0267~0.0725，首日等权 1/stock_dim）或买卖股数矩阵（action.csv，0 表示不动、1000 表示买卖 1000 股）。
- 每日收益序列（df_daily_return）。
- 绩效口径：期初/期末总资产（begin_total_asset / end_total_asset）、Sharpe、total_reward、total_cost、total_trades、time_elapsed；回测侧用 Pyfolio `perf_stats` 输出年化/累计收益、年化波动率、Sharpe、Calmar、Stability、最大回撤、Omega、Sortino、偏度、峰度、尾部比率、日 VaR、Alpha、Beta，并与基准（^DJI / 沪深300 399300）同区间对比。

## 调用方式
```python
# FinRL + Stable-Baselines3（9-5 / 8-4-x）
agent = DRLAgent(env=env_train)
model = agent.get_model("a2c"|"ppo"|"ddpg"|"sac"|"td3", model_kwargs=PARAMS)
trained = agent.train_model(model=model, tb_log_name='a2c', total_timesteps=50000)
trained.save('/path/trained_a2c.zip')

# 交易段预测（输出每日收益 + 每日权重）
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)

# 环境（9-4 / 8-4-2）
env = StockPortfolioEnv(df=trade, **env_kwargs)   # 或 StockTradingEnv
env_train, _ = env.get_sb_env()                   # DummyVecEnv 包装，适配 SB3

# 绩效（9-7）
pyfolio.timeseries.perf_stats(returns, factor_returns=returns, positions=None,
                              transactions=None, turnover_denom="AGB")
```
- 动作归一化（组合环境）：`np.exp(actions) / np.sum(np.exp(actions))`（softmax 保证权重和为 1）。
- 奖励设计：组合权重型环境用 r(s,a,s') = v' − v（动作前后组合价值变化量）；单标的买卖环境用"账户余额与持仓收益变化"。
- 自定义 Gym 环境（27-x）：类名 CryptoTradingEnv，动作买/卖/持有，状态 = 市场数据 + 账户状态；用 `from stable_baselines3 import PPO` + DummyVecEnv 训练。

## 关键参数
| 算法 | 原文超参 | 训练步数 |
|---|---|---|
| A2C（9-5） | n_steps=5、ent_coef=0.005、learning_rate=0.0002 | 50,000 |
| PPO（9-5） | n_steps=2048、ent_coef=0.005、learning_rate=0.0001、batch_size=128 | 80,000 |
| DDPG（9-5） | batch_size=128、buffer_size=50,000、learning_rate=0.001 | 50,000 |
| SAC（9-5） | batch_size=128、buffer_size=100,000、learning_rate=0.0003、learning_starts=100、ent_coef="auto_0.1" | 原文未给 |
| TD3（9-5） | batch_size=100、buffer_size=1,000,000、learning_rate=0.001 | 30,000 |
| A2C（8-4-3） | n_steps=5、ent_coef=0.01、learning_rate=0.0007 | 50,000 |
| DDPG（8-4-2） | batch_size=256、buffer_size=50,000、learning_rate=0.0005、NormalActionNoise(mu=0, sigma=0.1) | 10,000 |

环境参数（A 股实例，8-4-2/8-4-3）：hmax=1000（单标的最大持仓）、initial_amount=1,000,000、buy_cost_pct=6.87e-5（约万分之 0.687）、sell_cost_pct=1.0687e-3（约千分之 1.07，卖贵于买，符合含印花税结构）、reward_scaling=1e-4、initial_buy=True/False、hundred_each_trade=True（每笔 100 股）、state_space=stock_dim×(8+2)+1。训练器：`seeding.np_random(seed)` 固定环境随机性；get_sb_env() 向量化提升吞吐。

## 依赖
FinRL（DRLAgent、StockTradingEnv、StockPortfolioEnv、DataProcessor）、Stable-Baselines3 / OpenAI Baselines、gym（+shimmy>=0.2.1）、TensorBoard、Pyfolio、PyPortfolioOpt（对照组）、Plotly、pandas/numpy。

## 适用条件
- 多标的组合权重决策、希望端到端学习而非人工设计规则；允许用较长训练时间与 GPU 做实验。
- 需要"每日给出各标的权重"的输出形态时，权重矩阵便于集中度检查与归因分析（比 0/1 信号更通用）。
- 作为规则策略的对照实验：与最小方差组合、等权、Buy & Hold 同图比较累计收益。
- 单标的择时也可用 PPO + 自定义 Gym 环境（27-x），动作空间为买/卖/持有。

## 不适用条件
- 只有训练期表现就下结论：9-5 报告的期末总资产 436 万~485 万（期初 100 万）都来自训练环境，没有独立测试期，极可能是过拟合环境的产物。
- 样本外无基准对比：9-6 的 Sharpe 1.8078 无任何基准对照，无法判断优劣；9-7 实测 Alpha=0.00、Beta=0.94，收益基本来自跟随大盘，并非超额能力。
- 配置前后不一致的实现：8-4-3 训练用 A2C（trained_a2c）却调用 trained_ddpg 预测，该结果不可信，A 股实测期末资产 952,511（初始 100 万）、total_reward −47,488.68、Sharpe −0.366。
- 把训练集当验证集：27-4 直接 `val_data = train_df`，无样本外数据，无法评估过拟合。
- 无成本/无摩擦假设：9-1/9-6/9-8 均未计交易成本、滑点与换手约束；权重逐日变动时成本会显著侵蚀收益。
- 直接迁移到 A 股：T+1、涨跌停、融券限制、最小交易单位与印花税都会改变动作空间与可行性；原文未覆盖。
- 算法间横向比较：9-5 五种算法的训练步数不统一（30k~80k）、未说明随机种子与重复次数，Sharpe 0.80~0.835 的差异不具统计显著性。
- 状态含 28×28 协方差矩阵展平的高维做法：样本效率与训练稳定性风险未讨论；state_space 与实际拼接维度口径不一致，照抄会报维度错误。

## 验证状态
原文给出真实运行结果，但多处存在过拟合、无基准、配置错误等缺陷，故标为"有缺陷"：
- 训练期（9-5，道指 30 只，期初 100 万）：DDPG 期末总资产约 436.6 万~436.9 万、Sharpe 约 0.80~0.82；SAC 约 485 万、Sharpe 约 0.817；TD3 约 479.8 万、Sharpe 约 0.835；A2C 日志 fps 193~237（训练 50,000 步）、PPO 训练 80,000 步。全部为训练期指标。
- 样本外交易段（9-6，A2C，2020-07-01 至 2021-10-31）：期末总资产 1,363,804（对初始 100 万约 +36.4%），Sharpe = 1.8078；df_actions 权重取值 0.0267~0.0725，首日等权 0.035714。
- 回测细节（9-7，约 336 个交易日，基准 ^DJI）：策略年化 26.11%、累计 36.38%、年化波动 13.33%、Sharpe 1.81、Calmar 3.32、Stability 0.91、最大回撤 −7.87%、Sortino 2.74、日 VaR −1.58%、Alpha 0.00、Beta 0.94；基准同期年化 27.90%、累计 38.84%、波动 13.91%、Sharpe 1.84、Calmar 3.12、最大回撤 −8.93%、日 VaR −1.65%。策略各项均略逊或持平基准，Skew/Kurtosis 为 NaN。
- A 股实例（8-4-3，回测 2019-08-01 至 2019-12-31，初始 100 万）：期末资产 952,511、total_reward −47,488.68、Sharpe −0.366、成交 608 笔、成本 68.68；且训练 A2C 却用 DDPG 预测。
- 8-4-2（A 股 DDPG 训练 10,000 步）：只有训练配置与 Episode 监控字段，无任何收益数字；原文指出 10,000 步对 DDPG 通常远不足收敛。
- 27-x（比特币，小时级 27,812 条，2020-08-17 04:00 起）：验证数据即训练集，PPO 输出每秒/每回合奖励与交易次数，无累计收益、夏普或手续费假设；PPO 超参、奖励函数、网络结构均未给出，测试输出被截断。
- 9-8 最小方差对照（PyPortfolioOpt EfficientFrontier + min_volatility，权重上限 0.1，初始 100 万）：只有三条累计收益曲线的可视化描述，缺波动率、夏普、最大回撤等风险指标，且每日按收盘价无成本调仓。

## 来源
- `（9-1）...背景介绍+项目目标+模块架构.md`：MDP 三要素定义（连续权重动作、组合价值差值奖励）、六大模块与库选型、绩效口径。
- `（9-2）...准备环境+下载数据.md`：数据字段模板（date/tic/day/adjcp）、Yahoo 下载接口与限速、道指 30 只 101,615×9。
- `（9-3）...数据预处理.md`：FeatureEngineer 与 8 个技术指标清单、252 日协方差矩阵状态构造。
- `（9-4）...构建交易环境.md`：StockPortfolioEnv 类接口（step/reset/softmax/save_asset_memory/get_sb_env）、8 项环境参数、state_space 口径不一致问题。
- `（9-5）...深度强化学习算法模型.md`：A2C/PPO/DDPG/SAC/TD3 超参与训练步数、训练期总资产与 Sharpe。
- `（9-6）...交易.md`：样本外 A2C 结果（1,363,804 / Sharpe 1.8078）、输出契约（daily_return + actions）。
- `（9-7）...回测交易策略.md`：Pyfolio 指标清单、回撤表与压力区间、策略与 ^DJI 的全部对比数字（Alpha 0.00）。
- `（9-8）...最小方差投资组合分配.md`：PyPortfolioOpt 最小方差对照组的权重求解与累计收益对比写法。
- `（8-4-x）...制作股票交易策略模型（1/2/3）.md`：A 股 FinRL 数据流水线与 state_space 公式、StockTradingEnv 成本/持仓/整手参数、A2C/DDPG 训练配置与 A 股实测亏损及模型对象不一致的 bug。
- `（27-1）...背景介绍+系统介绍.md`、`（27-3）...比特币交易数据集.md`、`（27-4）...测试验证.md`：PPO + 自定义 Gym 环境（CryptoTradingEnv）路线、小时级加密数据与 EMA/波幅特征、训练集当验证集的问题。
- `(6-1-02)机器学习在高频交易中的应用.md`：RL 在做市价差优化与智能拆单中的用法（DQN/PPO）与风控侧的模型漂移监测思路。
