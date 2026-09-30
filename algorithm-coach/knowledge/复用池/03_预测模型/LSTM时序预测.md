---
id: model.lstm
name: LSTM 时序预测
category: 03_预测模型
status: 已验证
sources:
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/（1-2-7）金融时间序列分析：金融时间序列分析中的机器学习方法.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（4-2-01）常用的时间序列分析方法（1）.md
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/（2-2-01）金融风险建模与管理：传统风险建模方法回顾+机器学习在金融风险建模中的应用.md
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/(6-3-01)高频交易与量化交易：量化交易基础+基于LSTM的量化交易程序.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-8）比特币价格预测系统：使用LSTM训练模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-9）比特币价格预测系统：模型性能可视化.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（15-7）基于时间序列预测的比特币自动交易系统：构建深度学习模块.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（30-6）基于NLP用户舆情的交易策略：使用增加嵌入维度的深度学习模型.md
---

# LSTM 时序预测

## 能力说明
用 LSTM（门控循环网络）拟合价格/收益率序列的长期依赖，输出下一期价格（回归）或未来 N 日涨跌概率（分类）。LSTM 用门控机制缓解长序列的梯度消失/爆炸，是本项目深度学习侧最常用的价格预测模型；同一网络结构也可前置嵌入层处理文本，产出情感标签供下游模型当特征。

## 输入契约
- 多特征日频行情：OHLCV + 派生指标。原文配方（1-2-7）：OHLC + volume，派生 MA5、MA20、RSI(14)、Momentum（close − close.shift(5)）、Vol_Rate（成交量 pct_change）。
- 或单变量收盘价（11-4-8：前一日价格作输入、当日价格作目标；15-7：时间步 15 的单特征序列）。
- 样本构造：多特征滑窗 `time_step=60` 预测下一日 close；单变量用 `create_dataset` 滑窗（示例 time_step=15）。
- 形状约定：X 为三维 `(样本数, 时间步, 特征数)`；单变量单步时 (N−1, 1, 1)。
- 预处理：MinMaxScaler 缩放到 [0,1]（LSTM 对尺度敏感），预测后把收盘价列填回同宽数组再 inverse_transform 还原到价格尺度。
- 切分：按时间顺序 8:2 或 7:3，禁止随机打乱（防未来信息泄露）。
- 分类变体（6-3-01）：日频 OHLCV + 复权收盘价，60 天回溯窗口，标签为未来 5 日涨跌二分类。
- 文本变体（30-6）：词表 20000、单条最大长度 50 词（超出截断、不足补零）、100 维可训练嵌入层前置 LSTM。

## 输出契约
- 回归：下一期价格预测值；反归一化后与真实价格对比；指标为 MSE / RMSE（元）/ MAE（元）/ R² / explained_variance。
- 分类：未来 5 日涨跌概率；建议信号门槛为"预测上涨概率 > 55% 且趋势因子（均线交叉）确认"后开仓。
- 多步外推：滚动预测——以最近 60 天为初始输入，预测值写回序列后整体前移一格，连续外推未来 N 天（示例 15 天 / 30 天，freq='B'）。
- 训练过程件：loss / val_loss 曲线、预测与实际并排曲线（原文要求评估前必须 inverse_transform 回原始价格尺度）。

## 调用方式
```python
# 三层多特征版（1-2-7）
model = Sequential([
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32, return_sequences=True), Dropout(0.2),
    LSTM(16), Dropout(0.2), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=32, epochs=100,
          validation_data=(X_test, y_test), callbacks=[
              EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
              ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)])

# 单变量版（15-7）
Sequential([LSTM(10, activation='relu'), Dense(1)])       # MSE + Adam, 100 epoch, batch 32

# 最小版（11-4-8）
model1.add(LSTM(10, activation="sigmoid", return_sequences=True, input_shape=(None, 1)))
model1.add(Dense(1)); compile(loss='mean_squared_error', optimizer='adam')
model1.fit(X_train, y_train, batch_size=5, epochs=20)
```
- 预测与还原：`model.predict(X) → reshape(-1,1) → scaler.inverse_transform(...)`；作图时训练预测从 look_back 位置写入、测试预测从 `len(train)+2*look_back+1` 位置写入同一长度数组，避免曲线错位。

## 关键参数
| 参数 | 含义 | 原文取值 |
|---|---|---|
| time_step / lookback | 滑窗时间步 | 60（1-2-7 茅台）、60（6-3-01）、15（15-6/15-7）、5（2-2-01）、1（11-4-8） |
| 网络结构 | 层数与单元 | 64→32→16 + Dropout 0.2（1-2-7）；LSTM(10)+Dense(1)（15-7/11-4-8）；2 层 LSTM+Dropout（2-2-01） |
| 优化器/损失 | 训练 | Adam + MSE |
| batch_size / epochs | 训练规模 | 32 / 100（1-2-7、15-7）；5 / 20（11-4-8） |
| 早停与调度 | 正则化 | EarlyStopping(patience=15, restore_best_weights) + ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6) |
| 缩放 | 输入输出 | MinMaxScaler(feature_range=(0,1)) |
| 切分 | 训练/测试 | 8:2（1-2-7）、7:3（2-2-01）、顺序不打乱 |
| 外推长度 | 未来预测 | 15 天 / 30 天，freq='B' |
| 文本变体 | 嵌入与文本长度 | vocab 20000、maxlen 50、embedding dim 100、BCE + Adam、10 epochs |

## 依赖
TensorFlow/Keras（Sequential、LSTM、Dense、Dropout、EarlyStopping、ReduceLROnPlateau）、scikit-learn（MinMaxScaler、mean_squared_error、mean_absolute_error、r2_score、explained_variance_score）、pandas/numpy/matplotlib（或 plotly）。

## 适用条件
- 有多列价量/技术指标特征、需要捕捉较长历史依赖的日频价格序列预测；深度基线的首选。
- 需要多步滚动外推（未来 N 天路径）而不是单点预测时。
- 数据量足够（示例：比特币 472 个窗口样本、隆基绿能 2020-01 至 2021-08 日线、茅台未给样本量）且可用 GPU/较长时间训练。
- 文本情绪因子生产（30-6 变体）：用标注标题训练 LSTM+嵌入做情感二分类，再批量推断标签与事件收益对齐，供下游模型使用。
- 需要"信号 + 确认"双门槛时，把 LSTM 概率输出与均线交叉等技术确认结合（6-3-01）。

## 不适用条件
- 单变量、时间步=1 的配置：11-4-8 用前一日价格预测当日价格，在价格水平上几乎无预测力（接近随机游走的复制），且未划分验证/测试集，只有训练 loss。
- 把测试集当 validation_data：1-2-7 与 15-7 都用测试集做早停/调学习率，测试集参与模型选择，评估偏乐观。
- 归一化在全量数据上 fit 后再切分：测试集参与 scaler 参数估计，属轻微泄漏。
- 指标前视：MA20/RSI/Momentum 在全序列上 rolling 后再切窗，窗口边界处含未来信息（1-2-7 自陈）。
- 只看 R² 判优劣：价格序列强自相关会让 R² 虚高（15-7 测试 R² 0.9457），必须核对比朴素基线的改进与交易层面收益。
- 单标的、单数据集、单次留出的结果外推：原文未验证跨股票泛化，也没有涨跌停、T+1、交易成本约束。
- 特征含未来信息：把预测时点不可得的数据入模（例如用当日 OHLC 预测当日收盘）会使结果不可信。
- 期望高精度方向预测：4-2-01 的 LSTM 只报 RMSE，无方向准确率；30-6 的 0.97 验证准确率是文本二分类，不可与价格预测混用。

## 验证状态
原文给出多处真实运行结果：
- 15-7（比特币收盘价单变量，时间步 15，LSTM(10)+Dense(1)，100 epoch、batch 32，472 个窗口样本：训练 336 / 测试 136）：训练集 explained_variance 0.9850 / R² 0.9849；测试集 0.9457 / 0.9457。测试集被当作 validation_data。
- 4-2-01（隆基绿能 601012.SH，2020-01-01 至 2021-08-31 日线，MinMaxScaler、8:2、100 epoch、逐步外推未来 30 天）：训练 RMSE 76.27、测试 RMSE 41.99（测试误差低于训练，原文未解释）。
- 1-2-7（贵州茅台日线，三层 LSTM 64→32→16 + Dropout 0.2，time_step=60，外推 15 天）：原文未给任何结果数字，只给出结构与流程。
- 11-4-8（比特币，LSTM(10, sigmoid)+Dense(1)，491 个可训练参数，batch 5、epochs 20）：训练 loss 从第 1 轮 0.2960 降到第 3–6 轮的 0.0134/0.0123/0.0099/0.0095；无样本外评估、无 RMSE/MAE。
- 2-2-01（贵州茅台日线 CSV）：流程与指标口径（MSE/RMSE/MAE/MAPE、7:3、滑窗 5、2 层 LSTM+Dropout），无结果数字。
- 6-3-01（Oracle 股票日线，LSTMQuantStrategy，lookback=60，预测未来 5 日涨跌，概率阈值 55% + 均线确认）：未展示回测结果，LSTM 层数/轮数/优化器等超参缺失。
- 30-6（文本变体，LSTM + 100 维嵌入，10 epoch）：验证准确率从第 1 轮 0.9449 稳定在 0.97 附近，训练准确率升到 0.9996（明显过拟合，无早停/正则）。
- 11-4-9：只有 loss 曲线与预测/真实叠图，无任何误差指标。

## 来源
- `（1-2-7）金融时间序列分析中的机器学习方法.md`：三层 LSTM 完整结构、特征集（MA5/MA20/RSI14/Momentum/Vol_Rate）、time_step=60、早停与学习率调度、滚动多步外推、前视与验证集泄漏的缺陷。
- `（15-7）...构建深度学习模块.md`：LSTM(10)+Dense(1) 模板、输入形状、逆变换与指标口径、R²/解释方差数字、作图对齐写法。
- `（11-4-8）...使用LSTM训练模型.md`：最小单变量模板与训练 loss 数字、输入形状 (N,1,1)、超参起点与缺陷。
- `（11-4-9）...模型性能可视化.md`：loss 曲线 + 真实/预测叠图的最小评估可视化、keras plot_model 用法。
- `（4-2-01）常用的时间序列分析方法（1）.md`：Tushare 数据接口、滑窗样本构造、隆基绿能 LSTM 的 RMSE 数字。
- `（2-2-01）金融风险建模与管理：传统风险建模方法回顾+机器学习在金融风险建模中的应用.md`：LSTM/GRU+蒙特卡洛算 VaR 的风险侧用法、7:3 切分与滑窗 5。
- `(6-3-01)量化交易基础+基于LSTM的量化交易程序.md`：LSTMQuantStrategy 类结构、因子公式族、概率阈值 + 趋势双确认开仓、回测评价三指标。
- `（30-6）...使用增加嵌入维度的深度学习模型.md`：LSTM+嵌入的文本情感分类变体与因子生产链。

## 相关能力

- 上游依赖：[[复用池/02_高级特征工程/标准化与归一化|标准化与归一化]] —— LSTM 对尺度敏感，输入须缩放、输出须反变换回价格尺度后再评估
- 上游依赖：[[复用池/02_高级特征工程/技术指标与量价特征构造|技术指标与量价特征构造]] —— MA5/MA20/RSI/Momentum/Vol_Rate 等派生指标构成多特征输入
- 并列/替代：[[复用池/03_预测模型/GRU时序预测|GRU时序预测]] —— 更轻的门控 RNN 替代路线，参数更少、训练更快
- 并列/替代：[[复用池/03_预测模型/自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）|自回归族时序预测（AR-MA-ARMA-ARIMA-SARIMA）]] —— 单变量统计基线，任何深度模型应至少不劣于它
- 实证证据：[[复用池/07_验证证据/隆基绿能_LSTM价格预测|隆基绿能_LSTM价格预测]] —— A股个股上 LSTM 收盘价预测的真实 RMSE 样本
- 实证证据：[[复用池/07_验证证据/比特币_机器学习与LSTM价格预测|比特币_机器学习与LSTM价格预测]] —— 比特币 LSTM 训练损失与解释方差/R² 的真实数字
- 常见误用：[[复用池/06_失败经验/测试集参与调参与模型选择|测试集参与调参与模型选择]] —— 把测试集当 validation_data 做早停与学习率调度，评估偏乐观
