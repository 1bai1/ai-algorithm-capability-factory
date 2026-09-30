---
id: evidence.ashare15.finrl_a2c
name: A股15只 × FinRL 的 A2C 策略（亏损记录）
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-1）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-2）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-3）股票交易策略实战：制作股票交易策略模型.md
---

# A股15只 × FinRL 的 A2C 策略（亏损记录）

## 能力说明
本卡记录 FinRL 在 A 股 15 只沪市股票上的唯一带数字的样本外结果：5 个月亏损 4.75%、夏普 -0.366。它的主要价值是给出"DRL 交易在 A 股上未必有效"的反面证据，以及一段可复用的环境参数配置（含印花税式的非对称成本）。

## 输入契约
- 标的池：15 只上交所股票：600000.SH、600009.SH、600016.SH、600028.SH、600030.SH、600031.SH、600036.SH、600050.SH、600104.SH、600196.SH、600276.SH、600309.SH、600519.SH、600547.SH、600570.SH（含茅台、招行、中信证券等权重股）。
- 数据源：TuShare，日频，interval=1d；下载区间 2015-01-01 至 2020-01-03；规模：原始下载 (17960, 8) → 清洗后 (18315, 8) → 加技术指标后 (18270, 17) → 训练集 (16695, 17)。
- 切分：训练 2015-01-01 至 2019-08-01；交易/样本外 2019-08-01 至 2020-01-03。
- 技术指标 8 个：macd、boll_ub、boll_lb、rsi_30、cci_30、dx_30、close_30_sma、close_60_sma；状态空间 stock_dim×(指标数+2)+1 = 15×10+1 = 151。

## 输出契约
- 交易环境输出：df_account_value（逐日账户价值）与 df_actions（日期 × 股票的当日买卖股数，0 表示不动、1000 表示成交 1000 股）。
- 监控输出：每 episode 的 begin_total_asset、end_total_asset、total_reward、total_cost、total_trades、Sharpe、time_elapsed。
- 绩效对比：以沪深300（399300）为基准，PyFolio `timeseries.perf_stats` 计算夏普、年化收益、最大回撤等。

## 调用方式
```
p = DataProcessor(data_source='tushare', start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL)
p.download_data(ticker_list=ticker_list); p.clean_data(); p.fillna(); p.add_technical_indicator(config.INDICATORS); p.fillna()
train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
e_train_gym = StockTradingEnv(df=train, **env_kwargs); env_train, _ = e_train_gym.get_sb_env()
agent = DRLAgent(env=env_train); model_a2c = agent.get_model("a2c")
trained_a2c = agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=50000)
df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)  # 原文此处用的是 ddpg
```

## 关键参数
- 环境：hmax=1000（单标的最大持仓）、initial_amount=1,000,000、buy_cost_pct=6.87e-5（约万分之 0.687）、sell_cost_pct=1.0687e-3（约千分之 1.07，卖出成本约为买入的 15.6 倍，符合含印花税结构）、reward_scaling=1e-4、initial_buy=False、hundred_each_trade=True（按一手 100 股成交）。
- A2C 默认超参：{'n_steps': 5, 'ent_coef': 0.01, 'learning_rate': 0.0007}，训练 50,000 步（CPU）。
- DDPG 配置（同项目另一路）：batch_size=256、buffer_size=50000、learning_rate=0.0005、动作噪声 NormalActionNoise(mu=0, sigma=0.1)，训练 10,000 步。
- 环境假设：按日频价格成交、成本按固定比例扣除；未处理复权/停牌/涨跌停/T+1 的显式约束。

## 依赖
FinRL（DataProcessor、StockTradingEnv、DRLAgent、config）、Stable-Baselines3、TuShare（需 token）、pandas/numpy、PyFolio、matplotlib。

## 适用条件
- 想评估"FinRL 默认 A2C 直接套 A 股"的效果时，本卡可作为对照基线（结果是亏损）。
- 其非对称买卖成本设置（买入 6.87e-5 / 卖出 1.0687e-3）与整手成交开关，可作为 A 股环境建模的参考。

## 不适用条件
- 不要直接复用为可交易策略：样本外 5 个月亏损 4.75%。
- 结论不能归因于 A2C 本身：原文训练用 A2C，预测时却加载 trained_ddpg，模型对象不一致，使这次结果无法用于评价任一算法。
- 成本项形同虚设（total_cost 68.68 对 100 万本金），说明回测并未真实模拟换手成本。
- 样本外区间仅 5 个月、覆盖单一市场状态，任何优劣判断都不稳。

## 验证状态
原文给出真实运行结果（A 股 15 只，训练 2015-01-01~2019-08-01，交易 2019-08-01~2020-01-03，初始 100 万）：
- 训练日志（A2C，50,000 步）：fps 251，iterations 100，learning_rate 0.0007，entropy_loss -21.3，explained_variance -0.0322，policy_loss -2.66，reward -0.5146969，value_loss 2.24。
- 样本外交易：Episode 2 / day 103，begin_total_asset 1,000,000.00 → end_total_asset 952,511.32；total_reward -47,488.68；total_cost 68.68；total_trades 608；Sharpe -0.366。
- 决策数据：action.csv 逐日逐票记录买卖股数（0 或 1000 的整数倍）；示例日期 2019-08-01 起，2019-12-31 全为 0。
- 基准对比：代码取 399300 作基准并计算 perf_stats，但收录文本**未给出基准的任何指标数值**。
- 数据规模与状态空间：训练集 (16695, 17)，state_space = 15×(8+2)+1 = 151。

### 可信度评估
- 样本量：训练 4.6 年（16695 行多标的长表）、样本外 5 个月（约 103 个交易日），样本外严重不足。
- 泄漏：训练/交易区间不重叠，切分正确；但技术指标在个股维度独立计算、缺失值直接填充，未说明是否避免跨期污染。
- 基准对比：有基准设置（沪深300）但没输出数字，等于没有对照；策略亏损 4.75% 是否劣于基准无从判断。
- 结论稳定性：差。模型对象不一致 + 成本近乎为零 + 样本外过短，本卡只能作为"踩坑记录"使用，不能作为算法评价。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（8-4-1）股票交易策略实战：制作股票交易策略模型.md —— 15 只标的清单、FinRL 安装与 DataProcessor 流水线、时间切分、8 个技术指标、state_space 公式与数据规模。
- 提炼池/线上博客/金融大模型专栏系列-1/（8-4-2）股票交易策略实战：制作股票交易策略模型.md —— StockTradingEnv 环境参数（hmax、initial_amount、买卖成本、reward_scaling、成交规则）与 DDPG 配置、训练 10000 步。
- 提炼池/线上博客/金融大模型专栏系列-1/（8-4-3）股票交易策略实战：制作股票交易策略模型.md —— A2C 训练日志、样本外结果为 952,511.32 / reward -47,488.68 / 608 笔 / 成本 68.68 / Sharpe -0.366、action.csv 结构、基准 399300 与 perf_stats 调用。

## 相关能力

- 上游依赖：[[深度强化学习交易模型（A2C-PPO-DDPG-SAC-TD3）|深度强化学习交易模型（A2C/PPO/DDPG/SAC/TD3）]] —— 本卡是该能力（A2C 训练 50,000 步 + DDPG 配置）在 A 股 15 只沪市股上的实测，环境参数可参照，5 个月亏损 4.75% 不能反过来评价任一算法。
- 常见误用：[[回测无买入持有基准与风险指标|回测无买入持有基准与风险指标]] —— 代码取了沪深300 作基准却没输出任何基准指标，亏损 4.75% 是优是劣无从判断。
- 常见误用：[[报告自相矛盾与结论与数据不符|报告自相矛盾与结论与数据不符]] —— 训练用 A2C、预测却加载 trained_ddpg，模型对象不一致使这次结果无法归因。
- 常见误用：[[回测无交易成本与滑点|回测无交易成本与滑点]] —— 买 6.87e-5 / 卖 1.0687e-3 的成本项形同虚设（total_cost 68.68 对 100 万本金），换手成本未真实扣除。
- 并列/替代：[[道琼斯成分股_FinRL深度强化学习|道琼斯成分股 × FinRL 深度强化学习]] —— 同一 FinRL 框架在美股道指的对照实证（样本外略逊指数），判断"FinRL 直接套某个市场"是否可行须两卡合看。
