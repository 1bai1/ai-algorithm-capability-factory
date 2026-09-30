---
id: evidence.zinc.arma_monthly
name: IMF锌价 × ARMA 月度预测
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（4-2-02）常用的时间序列分析方法（2）自回归移动平均模型（ARMA）.md
---

# IMF锌价 × ARMA 月度预测

## 能力说明
本卡记录大宗商品（IMF 锌价月频）上"对数 + 滚动均值去趋势 → ADF 检验 → ARMA 拟合 → 未来 24 个月外推"的真实拟合摘要与平稳性检验数值，是本池里唯一一条商品月频 ARMA 证据。

## 输入契约
- 数据：zinc.csv（IMF 锌价数据集），字段 Date、Price；样本 1980-12 至 2016-02，月频，共 423 个观测。
- 变换：`tsl = np.log(df)`；`ma = tsl.rolling(window=12).mean()`；以 `tsl - ma`（原文记作 tslma）作为建模序列。
- 建模前做两次平稳性检验（原始 Price 一次、对数去趋势后一次），并画 ACF/PACF（nlags=10）辅助定阶。

## 输出契约
- ADF 结果（Series 形式）：T 统计量、P 值、#Lags Used、#Observations Used 及 1%/5%/10% 临界值。
- ARIMA/ARMA 拟合摘要：系数、标准误、z、P 值、置信区间、AIC/BIC/HQIC、Ljung-Box、Jarque-Bera、偏度/峰度。
- 预测：生成未来 24 个月日期索引 → `predict(start, end, dynamic=False)` → 逆对数变换与加回滚动均值还原为价格。

## 调用方式
```
def Staionarity_Check(ts): adfuller(ts, autolag='AIC') 并画 rolling(52) 均值/标准差
Arima = ARIMA(tslma, order=(2,5,1)).fit()
Arma  = ARIMA(tslma, order=(2, 0, 1)).fit(); Arma_fit.summary()
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,24)]
model.predict(start=..., end=..., dynamic=False)
```

## 关键参数
- 滚动窗口：平稳性检查用 window=52；去趋势用 window=12（原文明说 ARMA 版本 order=(2,0,1)、ARIMA 版本 order=(2,5,1)）。
- ACF/PACF nlags=10，置信带用 ±1.96/√len。
- 预测跨度 24 个月；`dynamic=False`（不使用动态多步递推）。
- 原文未说明未来期滚动均值如何取值（直接影响还原精度）。

## 依赖
statsmodels（adfuller、acf、pacf、ARIMA）、pandas（DateOffset）、numpy、seaborn、matplotlib。

## 适用条件
- 商品/宏观等确有月度周期与平滑趋势的序列；样本数 400 左右、月频、单变量。
- 需要"平稳性检验 + 对数去趋势 + 定阶 + 预测还原"这一整套模板时可直接复用。

## 不适用条件
- 不要照抄本卡的模型阶数：摘要标注的模型与代码拟合的 order 不一致（见验证状态），d=5 的设定在月度价格上也不合理。
- 没有样本外误差（MSE/RMSE/MAE 均未给出），不能判断预测精度。
- 残差诊断不过关（Ljung-Box p=0.00、Jarque-Bera p=0.00），说明残差仍有自相关且厚尾，标准误与置信区间不可信。
- 金融资产价格序列（股票/加密）不要用本配置：商品月频与股票日频的平稳性/季节性结构完全不同。

## 验证状态
原文给出真实运行结果（IMF 锌价，月频 1980-12~2016-02，423 个观测）：
- 原始 Price 的 ADF：T Statistic -3.139601，P-Value 0.023758，#Lags Used 7，#Observations Used 426；临界值 1% -3.445794、5% -2.868349、10% -2.570397。
- 对数去滚动均值后的 ADF：T Statistic -5.898484，P-Value 2.814411e-07，#Lags Used 4，#Observations Used 418；临界值 1% -3.446091、5% -2.868479、10% -2.570466。
- 拟合摘要（原文输出）：No. Observations 423，Log Likelihood 267.664，AIC -527.328，BIC -511.186，HQIC -520.947，sigma2 0.0159；系数 ar.L1 -1.0827（z -25.852，p 0.000）、ar.L2 -0.5279（z -13.607，p 0.000）、ma.L1 -0.9972（z -5.933，p 0.000）；Ljung-Box (L1) Q=25.69（Prob 0.00）、Jarque-Bera 23.63（Prob 0.00）、Heteroskedasticity 1.24（Prob 0.21）、Skew 0.12、Kurtosis 4.14。
- **口径矛盾**：该摘要的 Model 字段标注为 ARIMA(2, 5, 1)，而生成它的代码是 `ARIMA(tslma, order=(2, 0, 1))`（ARMA 版本），两者不一致；ARIMA(2,5,1) 与 ARMA(2,0,1) 分别只有拟合调用，未各自给出独立摘要。
- 预测部分：仅画出未来 24 个月预测曲线，**未给出任何预测值或误差数字**。

### 可信度评估
- 样本量：423 个月（约 35 年），对 ARMA(2,1) 足够。
- 泄漏：模型拟合与评估都在同一段样本内，无样本外评估；逆变换依赖未来期滚动均值，而原文未定义该值，导致预测不可复现。
- 基准对比：无（无随机游走基准）。
- 结论稳定性：差。系数接近 -1（ar.L1 -1.08、ma.L1 -0.997）提示接近单位根，参数解释不可靠；残差自相关显著；加上阶数标注与代码矛盾，本卡只能作为"流程可用、配置不可照抄"的记录，标记为有缺陷。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（4-2-02）常用的时间序列分析方法（2）自回归移动平均模型（ARMA）.md —— 锌价数据规模与区间、Staionarity_Check 函数、两次 ADF 全部数值、ACF/PACF、ARIMA(2,5,1)/ARMA(2,0,1) 调用、拟合摘要全部统计量、未来 24 个月预测流程。

## 相关能力

- 上游依赖：[[自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）|自回归族单变量时序预测（AR/MA/ARMA/ARIMA/SARIMA）]] —— 本卡是该能力在商品月频上的实测（423 个观测、AIC -527.328、完整拟合摘要）。
- 上游依赖：[[价格序列平稳化（对数、差分与ADF检验）|价格序列平稳化（对数、差分与ADF检验）]] —— "对数 + 12 月滚动均值去趋势 + ADF"三步与两次 ADF 数值都属该能力，是本池唯一一条商品月频平稳化样本。
- 常见误用：[[模型假设与适用边界失效|模型假设与适用边界失效]] —— 代码里的 d=5 在月度价格上属过度差分，系数接近 -1 提示接近单位根，参数的统计解释不可靠。
- 常见误用：[[报告自相矛盾与结论与数据不符|报告自相矛盾与结论与数据不符]] —— 摘要 Model 字段标 ARIMA(2,5,1)、生成它的代码是 ARMA(2,0,1)，模型阶数口径自相矛盾。
- 常见误用：[[无独立验证集与样本外评估|无独立验证集与样本外评估]] —— 全样本拟合、无样本外误差，未来 24 个月外推只有曲线，没有任何预测值或误差数字。
