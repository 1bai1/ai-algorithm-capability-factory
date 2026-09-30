---
id: evidence.goog.backtest_arima_gru
name: 谷歌GOOG × 滚动回测与 ARIMA/GRU 价格预测
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-1）通过回测、ARIMA 和 GRU 预测股票价格：项目介绍+准备环境.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-2）通过回测、ARIMA 和 GRU 预测股票价格：EDA.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-3）通过回测、ARIMA 和 GRU 预测股票价格：机器学习模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-4）通过回测、ARIMA 和 GRU 预测股票价格：参数调优.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-5）通过回测、ARIMA 和 GRU 预测股票价格：交易回测.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-5-03）通过回测、ARIMA 和 GRU 预测股票价格：ARIMA模型预测（3）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-6-01）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（1）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-6-02）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（2）.md
---

# 谷歌GOOG × 滚动回测与 ARIMA/GRU 价格预测

## 能力说明
本卡记录同一只美股（GOOG，价格量级 2.5→92，疑为复权后口径）上三条路线的真实数字：XGBoost 方向分类（准确率 0.49–0.55）、ARIMAX(1,1,1) 预测收盘价（RMSE 0.7477）、GRU(64) 预测下一日开盘价（MSE 0.003459）。它同时暴露了两个严重的方法学问题：切分前过采样与同期外生变量。

## 输入契约
- 数据口径在原文里前后不一致：项目介绍给出 GOOG.csv（1257 行 × 14 列），EDA 给出 yfinance 全历史 4639 行 × 7 列，实际建模数据为 4637 行 × 13 列、2004-08-23 至 2023-01-23 日频。
- 特征：Open/High/Low/Close/Volume + weekly_mean、quarterly_mean、annual_mean、annual_weekly_mean、annual_quarterly_mean、weekly_trend、open_close_ratio、high_close_ratio、low_close_ratio（共 15 列）。
- 标签：Target = 当日 Close > 前一日 Close（rolling(2) 比较），特征整体 shift(1)；轻微类别不平衡，用 RandomOverSampler(random_state=0) 过采样。
- 切分：按时间 75%/25% → X_train (3633, 5)、X_test (1211, 5)（基础模型阶段）；GRU 阶段特征数为 13，窗口 13 步。

## 输出契约
- 分类：classification_report（精确率/召回/F1/accuracy）、误分类计数图、Log Loss 与分类误差曲线。
- 回测：`backtest(data, model, predictors, start=1000, step=50)` 返回逐段拼接的 Target/Predictions 两列表（含日期索引），可用于分段指标与错误分布分析。
- ARIMAX：训练集/测试集预测列 + RMSE；摘要含系数、显著性、Ljung-Box/Jarque-Bera 诊断。
- GRU：test_results 表（Test Predictions / Actuals）与测试集 MSE。

## 调用方式
```
# 方向分类（XGBoost）
model = XGBClassifier(max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train, early_stopping_rounds=10,
          eval_set=[(X_train,y_train),(X_test,y_test)], eval_metric=["error","logloss"], verbose=0)
# 滚动回测
preds = model.predict_proba(test[predictors])[:,1]; preds[preds > .6] = 1; preds[preds <= .6] = 0
# ARIMAX（statemodels，SARIMAX 结果）
model_fit = ARIMA(train['Close'], exog=train[['Open','High','Low']], order=(1,1,1)).fit()
forecast = [model_fit.forecast(exog=test[exogenous_features].iloc[i]).values[0] for i in range(len(test))]
# GRU
model.add(InputLayer((X_train.shape[1], X_train.shape[2]))); model.add(GRU(64)); model.add(Dense(8,'relu')); model.add(Dense(1,'linear'))
```

## 关键参数
- XGBoost：max_depth 3 / 7 / 15，learning_rate=0.1，reg_lambda=0.6（调优后训练用 0.8），early_stopping_rounds 10 或 2；GridSearchCV 网格 max_depth∈{18,21,25} × reg_lambda∈{0.1,0.4,0.8}、cv=3，选出 {max_depth 25, reg_lambda 0.1}。
- 回测：start=1000、step=50，概率阈值 0.6（不是 0.5）；新增特征后 predictors 共 14 个。
- ARIMAX：ARIMA(1,1,1) + 外生 Open/High/Low 预测 Close；按年份切分（2020 年前训练 3867 行、2020 起测试 770 行）。
- GRU：window_size=13、13 个特征、MinMaxScaler 到 0-1、MSE 损失 + Adam、EarlyStopping、epoch 上限 100。
- 空值处理：weekly_mean 0.13%、quarterly_mean 1.92%、annual_mean 及其派生列 7.85% 的缺失统一 fillna(0)（原文自认不是最优）。

## 依赖
pandas、numpy、yfinance、scikit-learn（RandomOverSampler 来自 imbalanced-learn、GridSearchCV、classification_report、precision_score、mean_squared_error、MinMaxScaler）、xgboost、statsmodels（ARIMA/SARIMAX）、TensorFlow/Keras（GRU、Dense）、matplotlib/seaborn。

## 适用条件
- 只有在需要"反面教材"或想复现原文数字时使用；其滚动回测框架（start/step 前向切分）与特征工程思路可以借用。
- 单只美股、日频、20 年数据、方向分类或下一日开盘价回归的演示场景。

## 不适用条件
- 不能把 85.28% 的精确率当作交易能力：这只是"预测上涨且真上涨"的比例，在样本上涨占比高的年份可以轻易达到；且无手续费、无滑点、无换手统计。
- 不能把 ARIMAX 的 RMSE 0.7477 当成可用预测：它用当日的 Open/High/Low 预测当日 Close，属同期信息，实盘中不可得。
- 不能相信 GRU 的 MSE 0.003459：目标（下一日开盘价）经 MinMaxScaler 归一化，MSE 在 0~1 尺度上，且未与"上一日收盘价"基线对比。
- 数据集规模三种口径互相矛盾（1257 / 4639 / 4637 行），复用前必须自己核对数据源。

## 验证状态
原文给出真实运行结果（GOOG 日频，2004-08-23~2023-01-23，4637 行）：
- XGBoost 方向分类（过采样后 75/25 切分，3633/1211）：max_depth=3 时 accuracy 0.49（类 0 精确率 0.70 / 召回 0.15；类 1 精确率 0.46 / 召回 0.92）；max_depth=7（reg_lambda=0.6）accuracy 0.50（类 0 0.72/0.16，类 1 0.46/0.92）；max_depth=15 accuracy 0.50；GridSearchCV 调优后训练（max_depth=25、reg_lambda=0.8）accuracy 0.55（类 0 精确率 0.71 / 召回 0.33 / F1 0.45；类 1 精确率 0.49 / 召回 0.83 / F1 0.62）。
- 滚动回测（阈值 0.6，support 3637）：加特征前精确率 67.83%，加特征后 85.28%；分类报告：类 0 精确率 0.80 / 召回 0.85，类 1 精确率 0.85 / 召回 0.80，accuracy 0.82，macro 与 weighted F1 均 0.82。
- ARIMAX(1,1,1)+外生 Open/High/Low：训练 3867 行（2020 年前）、测试 770 行（2020 年起）；Log Likelihood 1770.361、AIC -3528.722、BIC -3491.163、HQIC -3515.385；系数 Open -0.5615、High 0.7750、Low 0.7870、ar.L1 -0.0658、ma.L1 -0.9998、sigma2 0.0234（p 值均 0.000）；Ljung-Box Q 0.19（p 0.67）、Jarque-Bera 9485.55（p 0.00）、异方差检验 8.22（p 0.00）、偏度 0.10、峰度 10.67；测试集 RMSE = 0.7477（在价格水平上计算）。
- GRU：X_train (3468, 13, 13)、X_test (1156, 13, 13)；GRU(64) → Dense(8, relu) → Dense(1, linear)；测试集 MSE = 0.003459。
- 特征规模：从 5 列扩展到 15 列（14 个预测变量）。

### 可信度评估
- 样本量：4637 个交易日，充足；测试段 1211 行 / 1156 个窗口。
- 泄漏：严重。① RandomOverSampler 在训练/测试切分之前执行（原文代码顺序如此），过采样会把未来的重复样本混入训练集；② ARIMAX 使用当期 Open/High/Low 作外生变量预测当期 Close，是同期信息；③ 调参网格在过采样数据上做，最终训练用原始切分（原文自述口径不同）。
- 基准对比：无。XGBoost 三期实验的准确率都贴着"全预测上涨"的水平（类 1 召回 0.83–0.92），说明模型几乎没有分类能力，但原文未设无信息基线。
- 结论稳定性：差。方向分类准确率 0.49–0.55 明确说明无预测力；回测精确率的提升来自阈值 0.6 与特征组合的单次选择。本卡按"泄漏与指标陷阱"标记为有缺陷。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（29-1）项目介绍+准备环境.md —— GOOG.csv 1257 行×14 列口径与项目模块划分。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-2）EDA.md —— yfinance 全历史 4639 行×7 列与可视化手法。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-3）机器学习模型.md —— 目标构造与 shift(1)、RandomOverSampler、3633/1211 切分、三次 XGBoost 的分类报告数字。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-4）参数调优.md —— GridSearchCV 网格与最优参数、调优后 accuracy 0.55 的分类报告。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-5）交易回测.md —— backtest 函数（start=1000、step=50、阈值 0.6）、精确率 67.83% → 85.28%、support 3637 的分类报告、空值率与 15 列特征。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-5-03）ARIMA模型预测（3）.md —— 2020 年切分（3867/770）、ARIMAX 摘要全部数值、外生系数、残差诊断与 RMSE 0.7477。
- 提炼池/线上博客/金融大模型专栏系列-1/（29-6-01）深度学习模型预测（1）.md、（29-6-02）深度学习模型预测（2）.md —— 13 特征/窗口 13、GRU 结构与 (3468,13,13)/(1156,13,13) 切分、测试集 MSE 0.003459。

## 相关能力

- 上游依赖：[[复用池/03_预测模型/自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）|自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）]] —— 本卡是该能力（ARIMAX(1,1,1) + 外生变量、RMSE 0.7477）的实测，也暴露了用同期外生变量导致 RMSE 偏低的陷阱。
- 上游依赖：[[复用池/03_预测模型/GRU时序预测|GRU时序预测]] —— GRU(64)+Dense 预测下一日开盘价的实测（测试集 MSE 0.003459，在归一化尺度上）。
- 上游依赖：[[复用池/05_回测系统与风险评估/滚动回测引擎|滚动回测引擎]] —— backtest(start=1000, step=50) 的扩窗前向切分是该能力可照搬的骨架。
- 常见误用：[[复用池/06_失败经验/切分前重采样泄漏|切分前重采样泄漏]] —— RandomOverSampler 在训练/测试切分之前执行，过采样样本把测试信息带进训练集。
- 常见误用：[[复用池/06_失败经验/特征与标签时序对齐前视|特征与标签时序对齐前视]] —— ARIMAX 用当期 Open/High/Low 预测当期 Close，实盘中该信息不可得。
- 常见误用：[[复用池/06_失败经验/缺朴素基线导致预测力误判|缺朴素基线导致预测力误判]] —— XGBoost 准确率 0.49–0.55 贴着"全预测上涨"水平，GRU 的 MSE 也未与"上一日收盘价"对照。
