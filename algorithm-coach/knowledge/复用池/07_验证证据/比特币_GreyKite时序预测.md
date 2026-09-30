---
id: evidence.btc.greykite
name: 比特币 × GreyKite/Silverkite 时序预测
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4 -1 ）比特币价格预测系统：GreyKite介绍.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-2 ）比特币价格预测系统：数据预处理.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-3）比特币价格预测系统：创建预测.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-4）比特币价格预测系统：交叉验证.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-5）比特币价格预测系统：后测试（Backtest）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-6）比特币价格预测系统：预测.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-7）比特币价格预测系统：模型诊断.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-9）比特币价格预测系统：模型性能可视化.md
---

# 比特币 × GreyKite/Silverkite 时序预测

## 能力说明
本卡记录 LinkedIn GreyKite（Silverkite 模板）在比特币价格上的完整评估数字，是本池中"样本外彻底失败"最典型的一条证据：交叉验证 MAPE 60%、backtest 测试段 CORR 转负、R² = -1.358。

## 输入契约
- 数据：Bitstamp 比特币 1 分钟历史（2012-01-01 至 2021-03-31）→ 按日取 Weighted_Price 日均 → 过滤 2017 年及以后 → 以 Timestamp 为索引。
- 输入 DataFrame 只需两列：Timestamp 与 Weighted_Price；`MetadataParam(time_col="Timestamp", value_col="Weighted_Price", freq=...)`。
- 频率口径矛盾：正文文字称按日 'D'，代码实际写 `freq="W"`（每周）；模型摘要显示训练样本 222 条、96 个特征，与周频口径吻合。
- 预测期 50 天（数据预处理中设定的预测天数），实际 ForecastConfig 的 forecast_horizon=30。

## 输出契约
- `Forecaster().run_forecast_config(df, config)` 返回三部分：forecast（未来预测）、timeseries（历史序列）、grid_search（交叉验证）与 backtest（留存测试集评估）。
- 指标表（defaultdict 汇总，行=指标，列=train/test）：CORR、R2、MSE、RMSE、MAE、MedAE、MAPE、MedAPE。
- 另有 `result.model[-1].summary()`（pipeline 末端 estimator）与 `plot_components()`。

## 调用方式
```
metadata = MetadataParam(time_col="Timestamp", value_col="Weighted_Price", freq="W")
result = Forecaster().run_forecast_config(df=df_price_include,
          config=ForecastConfig(model_template=ModelTemplateEnum.SILVERKITE.name,
                                forecast_horizon=30, coverage=0.95, metadata_param=metadata))
result.timeseries.plot(); result.backtest.plot(); result.grid_search
summarize_grid_search_results(grid_search=..., decimals=2,
    column_order=["rank","mean_test","split_test","mean_train","split_train","mean_fit_time","mean_score_time","params"])
```

## 关键参数
- model_template = SILVERKITE；forecast_horizon = 30；coverage = 0.95；交叉验证 3 折、1 组候选参数。
- 不确定度：uncertainty_method='simple_conditional_residuals'，quantiles=[0.025, 0.975]，quantile_estimation_method='normal_fit'，sample_size_thresh=5（有效样本 <5 时退化为 std_quantiles，分位改用 0.98）。
- pipeline：ColumnSelector(ts)/(y) → ZscoreOutlierTransformer → NullTransformer(impute_algorithm='interpolate', axis=0)。
- 未来预测 `make_future_dataframe(periods=4, include_history=False)`。

## 依赖
greykite（Forecaster、ForecastConfig、MetadataParam、ModelTemplateEnum、summarize_grid_search_results）、pandas、numpy、matplotlib。

## 适用条件
- 只适合当"如何正确做时序交叉验证与后测试"的示范模板：按折报告指标、同时看 train/test、用 backtest 验证外推能力。
- 可参考其 pipeline 的异常值替换（z-score）与插值补缺做法。

## 不适用条件
- 不要用它预测加密货币价格：本卡的后测试结果说明该配置在留存测试集上连均值预测都不如（R² 为负），相关性为负。
- 频率口径必须先统一（'D' 与 'W' 的矛盾会直接改变样本量、horizon 与全部指标）。
- 不要只看 mean_test 排名：折间 MAPE 从 28.96 到 116.93，均值 60.13 掩盖了模型对时间段高度敏感。
- 没有与随机游走/朴素预测的对照，任何"预测可用"的推断都不成立。

## 验证状态
原文给出真实运行结果（比特币，Bitstamp 聚合日/周序列，2017 年起，模型摘要 222 条样本 × 96 特征）：
- 交叉验证（3 折，MAPE）：rank_test_MAPE 1，mean_test_MAPE = 60.13，split_test_MAPE = (116.93, 34.5, 28.96)，mean_train_MAPE = 73.62，split_train_MAPE = (23.55, 101.05, 96.27)，mean_fit_time = 12.56，mean_score_time = 0.86。
- 后测试（训练集 / 测试集）：CORR 0.594112 / -0.213172；R2 0.289701 / -1.358174；MSE 8227240.311111 / 651879964.932465；RMSE 2868.316634 / 25531.940093；MAE 2218.435046 / 19313.942228；MedAE 2001.118287 / 11810.323572；MAPE 66.153838 / 54.906148；MedAPE 24.928294 / 55.241113。
- 预测示例（周频点，2021-04-04）：forecast 13684.712459，forecast_lower -2914.324065，forecast_upper 30283.748984，err_std 8469.051807。
- 原文明确点出：测试集相关性变负、R² 深负，说明模型在样本外连均值预测都不如，是典型的过拟合/分布漂移。

### 可信度评估
- 样本量：222 条（周频 2017 起）对 96 个特征严重不足，参数/特征比失衡是过拟合的直接原因。
- 泄漏：train/test 按时间切分，无前视；但特征量大而样本少，本质上是"用 96 个特征拟合 222 个点"。
- 基准对比：无朴素基准；不过 R² 为负这一事实本身就等价于"不如取均值"。
- 结论稳定性：高（负面结论稳定）。三折 CV 与 backtest 两套评估一致指向失败；本卡按"踩坑/失败证据"标记为有缺陷。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4 -1 ）比特币价格预测系统：GreyKite介绍.md —— 库定位与安装。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-2 ）比特币价格预测系统：数据预处理.md —— Bitstamp 1 分钟数据区间、日频聚合与 2017 年过滤、预测天数 50。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-3）比特币价格预测系统：创建预测.md —— MetadataParam/ForecastConfig 配置（SILVERKITE、horizon 30、coverage 0.95、3 折）。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-4）比特币价格预测系统：交叉验证.md —— 交叉验证全部数值（mean_test 60.13、三折 116.93/34.5/28.96、mean_train 73.62、时间 12.56/0.86）。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-5）比特币价格预测系统：后测试（Backtest）.md —— 训练/测试八个指标的全部数值。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-6）比特币价格预测系统：预测.md、（11-4-9）模型性能可视化.md —— 预测对象取用与可视化流程。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-7）比特币价格预测系统：模型诊断.md —— 222 条样本/96 特征、不确定度配置与 2021-04-04 的预测区间数值。

## 相关能力

- 上游依赖：[[复用池/03_预测模型/GreyKite时序预测|GreyKite时序预测]] —— 本卡是该能力（Silverkite 模板 + ForecastConfig）的实测，也是本池"样本外彻底失败"最典型的一条。
- 上游依赖：[[复用池/01_数据获取与处理/行情降采样与频率统一|行情降采样与频率统一]] —— 1 分钟数据聚合到日/周，且正文 'D' 与代码 freq='W' 的口径矛盾直接改变样本量（222 条）与全部指标。
- 上游依赖：[[复用池/05_回测系统与风险评估/时序切分与滚动验证|时序切分与滚动验证]] —— 本卡可复用的部分正是它示范的"3 折交叉验证按折报告 + backtest 留存测试"协议。
- 常见误用：[[复用池/06_失败经验/过拟合与结果不稳定|过拟合与结果不稳定]] —— 222 条样本 × 96 个特征，折间 MAPE 从 28.96 到 116.93，均值 60.13 掩盖了模型对时间段的极端敏感。
- 常见误用：[[复用池/06_失败经验/评价指标口径与量纲误用|评价指标口径与量纲误用]] —— 测试集 RMSE 比训练集扩大近 9 倍而 MAPE 反而更低（66.15→54.91），量级变化下指标互相矛盾。
- 常见误用：[[复用池/06_失败经验/预测输出有效性未校验|预测输出有效性未校验]] —— 预测区间下界为负（forecast_lower -2914.32）仍直接出图，未做后置校验。
