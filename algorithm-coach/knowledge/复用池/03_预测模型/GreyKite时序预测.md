---
id: model.greykite
name: GreyKite（Silverkite）时序预测
category: 03_预测模型
status: 已验证
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4 -1 ）比特币价格预测系统：GreyKite介绍.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-3）比特币价格预测系统：创建预测.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-4）比特币价格预测系统：交叉验证.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-5）比特币价格预测系统：后测试（Backtest）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-6）比特币价格预测系统：预测.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-7）比特币价格预测系统：模型诊断.md
---

# GreyKite（Silverkite）时序预测

## 能力说明
LinkedIn 开源的时间序列预测库，核心算法为 Silverkite，支持交互式单序列预测与自动化批量预测，内置特征工程、模型解释性与可视化。本项目用它对比特币加权价格做单变量基线预测：声明"时间列 + 值列 + 频率"三要素即可启动，一次调用同时产出未来预测值、原始序列、历史预测性能（grid_search 交叉验证 + backtest 留存测试）。

## 输入契约
- 单变量 DataFrame，至少两列：时间列与数值列。原文 time_col="Timestamp"、value_col="Weighted_Price"（加权价，而非 Close）。
- 频率字符串 freq 必须与数据实际粒度一致，否则预测窗口含义会错。原文正文称 'D'（每日）而代码写 'W'（每周），前后矛盾，采用前必须自行核对。
- 数据为比特币历史价格（Bitstamp 分钟数据聚合后的日/周频序列）；诊断示例训练样本 222 条、特征 96 个。
- 无需手工做缺失/异常处理：pipeline 自带 ZscoreOutlierTransformer 与 NullTransformer(impute_algorithm='interpolate', axis=0)，处理顺序为 列选择(ts/y) → z-score 异常值替换 → 插值补缺。

## 输出契约
- `result.forecast`：预测对象，`forecast.plot()` 直接出图，`forecast.df` 为预测结果表（先 `.head().round(2)` 核对数值量级）；字段含 forecast、forecast_lower、forecast_upper、err_std（点预测 + 上下界 + 残差标准差）。
- `result.timeseries`：原始序列与预测区间，`result.timeseries.plot()` 可视化；`make_future_dataframe(periods=4, include_history=False)` 生成未来时间索引。
- `result.grid_search`：交叉验证汇总（`summarize_grid_search_results`），列含 rank、mean_test、split_test、mean_train、split_train、mean_fit_time、mean_score_time、params。
- `result.backtest`：留存测试集评估，`backtest.plot()` 出预测/实际对比图；指标含 CORR、R²、MSE、RMSE、MAE、MedAE、MAPE、MedAPE。
- `result.model[-1].summary()`：管道最后一个 estimator 的参数估计、显著性与置信区间。
- `forecast.plot_components()`：成分分解图（趋势/事件假期/季节性），用于上线前判断模型是否适配数据。

## 调用方式
```python
from greykite.framework.templates.autogen.forecast_config import (ForecastConfig, MetadataParam,
                                                                 ModelTemplateEnum)
from greykite.framework.forecast.forecaster import Forecaster

metadata = MetadataParam(time_col="Timestamp", value_col="Weighted_Price", freq="W")
result = Forecaster().run_forecast_config(
    df=df_price_include,
    config=ForecastConfig(model_template=ModelTemplateEnum.SILVERKITE.name,
                          forecast_horizon=30, coverage=0.95,
                          metadata_param=metadata))

fig = result.timeseries.plot(); fig.show()
grid_search = result.grid_search      # 交叉验证
backtest = result.backtest            # 留存测试
forecast = result.forecast            # 未来预测
forecast.plot().show(); forecast.df.head().round(2)
result.model[-1].summary(); result.timeseries.make_future_dataframe(periods=4, include_history=False)
```
- CV 汇总写法：`summarize_grid_search_results(grid_search=..., decimals=2, cv_report_metrics=None, column_order=[...])`，随后把 params 转 str、设为索引并 transpose()，一行一个配置便于横向比较。

## 关键参数
| 参数 | 含义 | 原文取值 |
|---|---|---|
| model_template | 模型模板 | ModelTemplateEnum.SILVERKITE |
| forecast_horizon | 外推步数 | 30（另有周频模板默认 default_horizon=12） |
| coverage | 预测区间 | 0.95 |
| freq | 频率 | 原文 'D' 与 'W' 自相矛盾，需自行核对 |
| 交叉验证 | 折数 | 3 折（日志 "Fitting 3 folds for each of 1 candidates, totalling 3 fits"） |
| 不确定度 | 区间估计 | uncertainty_method='simple_conditional_residuals'、quantile_estimation_method='normal_fit'、quantiles=[0.025, 0.975]；有效样本 <5 时退化为 std_quantiles 并把分位改用 0.98（sample_size_thresh=5） |
| 异常值/缺失处理 | pipeline | ZscoreOutlierTransformer + NullTransformer(impute_algorithm='interpolate', axis=0) |

## 依赖
greykite（`pip install greykite`，原文未给版本号；Forecaster、ForecastConfig、MetadataParam、ModelTemplateEnum、Silverkite）、pandas；绘图依赖库自带接口。

## 适用条件
- 需要快速建立单变量时序基线：只声明时间列/值列/频率即可启动，适合作为 ARIMA、GBDT、LSTM 之外的一条候选路线。
- 大规模多序列的自动化批量预测（库的核心定位），需要自带 CV、backtest 与预测区间的场景。
- 需要开箱即用的成分分解与可解释性（趋势/季节性/事件项分解、模型摘要）时。
- 需要一个省事的"训练/测试指标成对输出"框架做模型体检。

## 不适用条件
- 追求预测精度：本例 CV 的 mean_test_MAPE=60.13（折间 28.96~116.93，差异极大），backtest 测试集 CORR=−0.213、R²=−1.358，R² 为负意味着不如直接预测历史均值，CORR 由正转负意味着方向性预测不可用。
- 只用训练/交叉验证指标选模型：必须看 backtest 的留存测试集结果，本例训练集 CORR 0.594/R² 0.290 与测试集全面崩坏形成典型过拟合/分布漂移对照。
- 需要非负价格区间：预测区间下界出现负值（−2914~−3112），对价格这类非负量不合理，说明残差正态假设与区间估计不适配，不能直接当风险边界。
- 单一指标判优劣：测试集 RMSE 比训练集扩大近 9 倍，但 MAPE 反而更低（66.15 → 54.91），因为测试期币价量级更高，MAPE 与 RMSE 必须结合价格区间解读。
- 需要交易层面的可用性：无方向准确率、无成本模型、无交易信号与回测收益；比特币单一标的的结果迁移到 A 股需重新验证（涨跌停、休市、T+1 会改变误差结构）。
- 频率口径未核对就照抄：原文 freq 自相矛盾，预测 30 步的含义随频率而变。
- 库版本未锁定：原文未给版本号，接口可能已变化，接入前需查证 API 与数据格式。

## 验证状态
原文给出真实运行结果（比特币价格预测，数字完整但结论是模型样本外失效）：
- 交叉验证（result.grid_search）：mean_test_MAPE = 60.13，三折分别为 116.93 / 34.5 / 28.96；mean_train_MAPE = 73.62（高于 mean_test，说明整体精度不足而非过拟合）；平均拟合时间 12.56s、平均评分时间 0.86s。
- 后测试（result.backtest，训练/测试成对）：训练集 CORR 0.594、R² 0.290、RMSE 2868、MAE 2218、MAPE 66.15；测试集 CORR −0.213、R² −1.358、RMSE 25532、MAE 19314、MAPE 54.91。
- 模型诊断（周长/周频示例，start_year=2017，训练 222 条、96 个特征）：pipeline 为 ColumnSelector(ts)/(y) → ZscoreOutlierTransformer → NullTransformer(interpolate, axis=0)；不确定度用 simple_conditional_residuals + 分位 0.025/0.975，样本 <5 时退化；预测输出含上下界与残差标准差；预测区间下界为负（−2914 ~ −3112）。
- 预测环节（11-4-6）：只演示 result.forecast 取对象、绘图与 `.df.head().round(2)` 核对，无精度数字。
- 库介绍（11-4-1）："快速、准确、直观"属官方宣传口径，无实测对比与基准数据；本篇未说明 GreyKite 与项目后续 LSTM 如何组合或对比。

## 来源
- `（11-4-3）...创建预测.md`：MetadataParam/ForecastConfig/Forecaster 的最小可运行配置、三要素声明、3 折交叉验证日志、结果三段式（forecast + 原始序列 + backtest）。
- `（11-4-4）...交叉验证.md`：grid_search 汇总字段与列顺序、MAPE 口径与折间方差、训练/测试 MAPE 对比判读、模型选择兼顾精度与算力成本。
- `（11-4-5）...后测试（Backtest）.md`：backtest 流程、完整指标清单（CORR/R²/MSE/RMSE/MAE/MedAE/MAPE/MedAPE）、train/test 成对汇报规范与全部实测数字。
- `（11-4-6）...预测.md`：result.forecast / forecast.df 的访问约定与输出核对顺序。
- `（11-4-7）...模型诊断.md`：plot_components、model[-1].summary()、pipeline 与不确定度参数、make_future_dataframe 用法、负区间问题。
- `（11-4 -1 ）...GreyKite介绍.md`：库定位（Silverkite 核心算法、批量预测、解释性与可视化）、安装命令、无版本锁定的风险。

## 相关能力

- 上游依赖：[[行情降采样与频率统一|行情降采样与频率统一]] —— 分钟数据聚合为日/周频序列，且 freq 字符串必须与数据实际粒度一致
- 并列/替代：[[自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）|自回归族单变量时序预测（AR/MA/ARMA/ARIMA/SARIMA）]] —— 同为单变量时序基线候选，按精度、算力成本与可解释性对比选择
- 实证证据：[[比特币_GreyKite时序预测|比特币 × GreyKite/Silverkite 时序预测]] —— CV MAPE 60.13、backtest 测试集 CORR −0.213/R²=−1.358 的样本外失败记录
- 常见误用：[[过拟合与结果不稳定|过拟合与结果不稳定]] —— 训练集 CORR 0.594/R² 0.290 与测试集全面崩坏的典型对照，必须看 backtest 留存测试
- 常见误用：[[模型假设与适用边界失效|模型假设与适用边界失效]] —— 残差正态假设下预测区间下界为负，不能直接当非负价格的边界
