---
id: evidence.aapl.gbdt_ensemble
name: 苹果AAPL × 梯度提升与集成周收益预测
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（31-1）使用机器学习预测苹果（AAPL）股票的周收益：项目介绍+准备环境.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（31-3）使用机器学习预测苹果（AAPL）股票的周收益：数据加载与分析.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（31-4）使用机器学习预测苹果（AAPL）股票的周收益：准备训练和验证数据.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（31-5）使用机器学习预测苹果（AAPL）股票的周收益：特征选择.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（31-6）使用机器学习预测苹果（AAPL）股票的周收益：构建模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/(31-7）使用机器学习预测苹果（AAPL）股票的周收益：精调选定模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/(31-8-01）使用机器学习预测苹果（AAPL）股票的周收益：集成方法（1）定义并训练集成方法.md
  - 提炼池/线上博客/金融大模型专栏系列-1/(31-8-02）使用机器学习预测苹果（AAPL）股票的周收益：集成方法（2）策略1.md
  - 提炼池/线上博客/金融大模型专栏系列-1/(31-8-03）使用机器学习预测苹果（AAPL）股票的周收益：集成方法（3）策略2.md
  - 提炼池/线上博客/金融大模型专栏系列-1/(31-9）使用机器学习预测苹果（AAPL）股票的周收益：结论.md
---

# 苹果AAPL × 梯度提升与集成周收益预测

## 能力说明
本卡记录"周频技术指标 + 梯度提升树族 + 投票/堆叠集成 + 择时两策略"在苹果股票上的完整数字：从单模型 ROC-AUC 0.51–0.58、集成准确率 59.5%，到策略累计收益 908% vs 基准 855%。核心用途是判断"周收益方向预测"这类任务的真实上限在哪里。

## 输入契约
- 标的与数据：AAPL 周频（yfinance，interval='1wk'），1990-01-01 至 2023-07-28，共 1752 行 × 6 列；周收益 = Adj Close.pct_change()×100。
- 切分：训练 ≤2014-01-01，测试 2014-01-01 至 2023-07-24；dropna 后 X_train (1233, 21)、X_test (479, 21)。
- 标签：cat_target = 下周收益>0（0/1）；训练集正类占比 53.7%；另有 cont_target（下周收益）。
- 特征：Open/High/Low/Close/Adj Close/Volume、returns、sma_4/12/20 与比率、rsi 与超买超卖、obv 与 obv_divergence_12_weeks、4week_vol、12week_vol、velocity、acceleration。

## 输出契约
- 特征选择：RFECV(估计器, cv=TimeSeriesSplit(n_splits=5), scoring='roc_auc')，不同估计器给出不同子集；最终建模统一用 9 特征清单：sma_12、sma4_ratio、rsi、obv、4week_vol、12week_vol、acceleration、velocity、obv_divergence_12_weeks。
- 模型输出正类概率 → ROC 曲线/AUC 与混淆矩阵（plot_model_performance）。
- 策略输出：position = Predicted.shift(1)，model_returns = position×returns；
  - 策略 1：预测看涨才持仓（0/1），每周五最后交易时段买入、下周五卖出；
  - 策略 2：始终 100% 在市，预测看跌则做空（0 → -1）。
- 绩效报告：QuantStats `qs.reports.full(model_returns/100, benchmark=returns/100)`。

## 调用方式
```
cv = TimeSeriesSplit(n_splits=5); rfe = RFECV(estimator=GradientBoostingClassifier(random_state=seed), cv=cv, scoring='roc_auc')
clf.fit(X_train, y_train); y_pred = clf.predict_proba(X_test)[:,1]
hard_voting = VotingClassifier(estimators, voting="hard"); soft_voting = VotingClassifier(estimators, voting="soft")
tuned_voting_clf = VotingClassifier(estimators, voting='soft', weights=best_weights)
X_test['position'] = X_test['Predicted'].shift(1); X_test['model_returns'] = X_test['position'] * X_test['returns']
```

## 关键参数
- seed = 123（全部分类器 random_state=seed）。
- 单个模型：AdaBoost / GradientBoosting / HistGradientBoosting(scoring='auc') / CatBoost(verbose=False) / XGBoost(eval_metric='auc', objective='binary:logistic')。
- Optuna 各 1000 次试验：HBGB 搜索 learning_rate 0.001~0.1、max_iter 100~1000、max_leaf_nodes 2~100、max_depth 1~10、min_samples_leaf 1~15、l2_regularization 1e-4~0.1；CatBoost 最优 depth=2 等（见验证状态）。
- 集成：Hard Voting、Soft Voting（VotingClassifier）、Stacking（元学习器 DecisionTreeClassifier）；权重用 Optuna 搜索，优化指标由 ROC-AUC 改为 Accuracy。
- 策略假设：无交易成本、无滑点、无风险利率 0、策略 2 允许无约束做空。

## 依赖
pandas、numpy、yfinance、ta、scikit-learn（TimeSeriesSplit、RFECV、VotingClassifier、StackingClassifier、DecisionTreeClassifier、各种 GBDT）、xgboost、catboost、optuna、quantstats、plotly。

## 适用条件
- 单只美股、周频、30 年历史；想评估"用技术指标预测周涨跌方向"这一任务的现实水平时，本卡提供了可直接对照的数字基准。
- 特征选择用 TimeSeriesSplit 的做法可复用（时序任务不要用随机 K 折）。

## 不适用条件
- 不要用 AUC 0.51–0.58 的模型做方向交易：这几乎等于随机；测试集正类基准线就有 53.7%。
- 不要引用 Optuna 报出的最优分数作为泛化能力：objective 直接在 X_test 上算 AUC（测试集参与调参），分数是乐观估计。
- 策略 908% / 夏普 2.49 不能归功于模型：策略 94% 时间在市，本质接近满仓持有；且未计交易成本。
- 指标口径自相矛盾：原文同时报告"加权软投票准确率最好（59.5%）但 ROC-AUC 更低"，说明所选模型并未在概率排序上更优。

## 验证状态
原文给出真实运行结果（AAPL 周频 1990-2023，测试段 2014-01-01 起；回测报告区间 2014-06-02 至 2023-07-17）：
- 单模型（9 特征，X_test 上的 ROC-AUC）：性能最好的是 Gradient Boosting 0.53，其次 CatBoost 0.51。
- Optuna 调参：HBGB 由 0.51 提升到 0.55（试验日志示例 Trial 0 = 0.4653670302768478、Trial 1 = 0.4735572596032689、Trial 2 = 0.5471253195089463）；CatBoost 由 0.51 提升至 0.58，最优超参 learning_rate=0.03638764157612463、depth=2、l2_leaf_reg=0.05590867142014544、random_strength=0.03636195872651924、bagging_temperature=0.01514447911663602、border_count=18、min_data_in_leaf=55（Trial 0 = 0.4894337041437161）。
- 集成：硬投票 ROC-AUC 0.57（三种集成里最好）；加权软投票最优权重 dict_values([0.8891140385078973, 0.3631724786178146])，测试集准确率 59.5%（原文称是"至今最好"），但 ROC-AUC 反而更低。
- 特征选择（RFECV + TimeSeriesSplit(5)）：GradientBoosting 选 7 个（sma_12、sma4_ratio、rsi、obv、4week_vol、12week_vol、acceleration）；RandomForest 选 4 个（obv、4week_vol、12week_vol、acceleration）；XGBoost 选 3 个（Close、Adj Close、sma_12）；AdaBoost 选 3 个。X_train 1233×21、X_test 479×21。
- 策略 1（仅预测看涨时持仓）：Time in Market 94.0% vs 基准 100%；累计收益 908.02% vs 855.4%；CAGR 19.09% vs 18.61%；Sharpe 2.49 vs 2.29；Prob. Sharpe 99.96% vs 99.91%；Smart Sharpe 2.48 vs 2.28；Sortino 3.91 vs 3.56；Smart Sortino 3.88 vs 3.54；Sortino/√2 2.76 vs 2.52；最大回撤与年化波动率低于基准（原文未给数值）；异常损失比率高于基准。
- 策略 2（始终满仓、看跌做空）：只有文字结论——累计收益与 CAGR 优于买入持有，但夏普低于基准、最大回撤更高、Serenity Index 更高；**未给出具体数字**。
- 数据侧描述统计：Adj Close 0.10~195.83（均值 23.44、标准差 42.62）；1990s 单周最大涨 24.86% / 跌 -25.17%；2000-10-25 单周 -50.66%；2020 年最大单周 -17.53%；多数周收益落在 -11.84%~+12.85%，最优周 +39.74%。

### 可信度评估
- 样本量：训练 1233 周、测试 479 周，样本量充足；但方向标签的正类占比 53.7%，无信息基线就接近 54%。
- 泄漏：明显的超参选择泄漏——Optuna 的 objective 直接以 X_test 的 ROC-AUC/Accuracy 为优化目标（1000 次试验），报出的 0.55/0.58/0.57/59.5% 都是在测试集上择优的结果；特征选择与归一化在 train 上完成，这部分规范。
- 基准对比：策略层面做了 Buy & Hold 对比（这是优点），但收益差异（908% vs 855%）在 9 年、94% 在市的背景下不足以证明选时有效。
- 结论稳定性：弱。核心预测能力接近随机，策略收益主要来自长期持有；本卡按"指标/流程陷阱"标记为有缺陷。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（31-1）项目介绍+准备环境.md、（31-9）结论.md —— 项目定位、两条策略、seed=123 与结论（策略 I 优于策略 II）。
- 提炼池/线上博客/金融大模型专栏系列-1/（31-3）数据加载与分析.md —— 1752 行、1990-01-01~2023-07-28、收益构造与描述统计、极端周涨跌。
- 提炼池/线上博客/金融大模型专栏系列-1/（31-4）准备训练和验证数据.md —— 2014-01-01 切分、cont_target/cat_target、特征清单与相关性检查、53.7% 正类。
- 提炼池/线上博客/金融大模型专栏系列-1/（31-5）特征选择.md —— TimeSeriesSplit(5)、RFECV 四种估计器的特征子集、1233×21 / 479×21 与 9 特征清单。
- 提炼池/线上博客/金融大模型专栏系列-1/（31-6）构建模型.md —— 五个 GBDT、Gradient Boosting AUC 0.53、CatBoost 0.51。
- 提炼池/线上博客/金融大模型专栏系列-1/(31-7）精调选定模型.md —— Optuna 1000 trials、HBGB 0.51→0.55、CatBoost 0.51→0.58 与最优超参、试验日志数值。
- 提炼池/线上博客/金融大模型专栏系列-1/(31-8-01）集成方法（1）.md —— 硬投票 0.57、最优权重、加权软投票 59.5% 准确率。
- 提炼池/线上博客/金融大模型专栏系列-1/(31-8-02）集成方法（2）策略1.md —— 策略 1 全部 QuantStats 数字与时间在市。
- 提炼池/线上博客/金融大模型专栏系列-1/(31-8-03）集成方法（3）策略2.md —— 策略 2 规则与定性结论（无数字）。
