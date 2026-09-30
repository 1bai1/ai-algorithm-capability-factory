---
id: model.gru
name: GRU 时序预测
category: 03_预测模型
status: 已验证
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-6-01）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（1）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（29-6-02）通过回测、ARIMA 和 GRU 预测股票价格：深度学习模型预测（2）.md
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/(6-1-03)高频交易与量化交易：高频交易中的预测建模.md
---

# GRU 时序预测

## 能力说明
用 GRU（门控循环单元）做日频多特征序列回归：GRU 以重置门与更新门替代 LSTM 的三门结构，参数更少、计算更快，在长序列上同样缓解梯度消失/爆炸。适用于"需要一个比 LSTM 更轻的门控 RNN 基线"的价格/收益率滚动预测。

## 输入契约
- 13 个特征列：Open、High、Low、Close、weekly_mean、quarterly_mean、annual_mean、annual_weekly_mean、annual_quarterly_mean、weekly_trend、open_close_ratio、high_close_ratio、low_close_ratio；显式剔除 Volume 与 Target。
- 特征必须是历史可得的滞后/衍生量；原文隐含假设不引入未来信息（但周/季/年均值的统计窗口口径未说明，若按全样本统计则含未来信息）。
- 标准化：MinMaxScaler 把所有特征压到 0–1，`scaler.fit(NN_df[NN_df.columns])` 后统一 transform。
- 样本构造：滑动窗口把时序 DataFrame 转成三维张量 `(样本数, 时间步, 特征数)`；窗口 `window_size=13`，标签取窗口后一行的第 0 列（Open），即预测下一日开盘价；y 为单标量。
- 数据规模：4637 行 × 13 列，2004-08-23 至 2023-01-23 日频行情（示例价格从约 2.5 涨到约 92）。
- 切分：按样本顺序 75% 训练 / 25% 测试（训练 3468 / 测试 1156），不使用随机切分。
- 模型输入张量形状：X (4624, 13, 13)。

## 输出契约
- 单一标量回归值（下一日指定列，示例为开盘价；换标签列即可改预测目标）。
- 评估指标：MSE（示例 0.003459，归一化尺度）、RMSE；原文另要求把 `model.predict(X_test).flatten()` 与实际值并排成 DataFrame 画曲线目视核对形态。
- 训练件：EarlyStopping 监控验证集损失；epoch 上限 100。

## 调用方式
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    InputLayer((X_train.shape[1], X_train.shape[2])),   # (13, 13)
    GRU(64),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')])
model.compile(loss='mse', optimizer='adam', metrics=[RootMeanSquaredError()])
model.fit(X_train, y_train, epochs=100, callbacks=[EarlyStopping(...)])

preds = model.predict(X_test).flatten()
pd.DataFrame({'Test Predictions': preds, 'Actuals': y_test})
```
- 样本构造函数 `df_to_X_y2(df, window_size=13)`：`df.to_numpy()` 后按 `df_as_np[i:i+window_size]` 切窗口入 X，取 `df_as_np[i+window_size][0]` 作标签入 y，返回两个 np.array。

## 关键参数
| 参数 | 含义 | 原文取值 |
|---|---|---|
| window_size | 时间步 | 13 |
| 特征数 | 输入维度 | 13（剔除 Volume/Target 后） |
| 网络结构 | 层 | GRU(64) → Dense(8, relu) → Dense(1, linear) |
| 损失/优化器 | 训练 | MSE / Adam |
| 指标 | 监控 | RMSE（RootMeanSquaredError） |
| EarlyStopping | 正则化 | 监控验证集损失（patience 等细节原文被省略） |
| epochs | 上限 | 100 |
| 切分 | 训练/测试 | 75% / 25%，按顺序 |
| 缩放 | 预处理 | MinMaxScaler → [0,1] |

## 依赖
TensorFlow/Keras（Sequential、InputLayer、GRU、Dense、EarlyStopping、RootMeanSquaredError）、scikit-learn（MinMaxScaler、mean_squared_error）、pandas、numpy、matplotlib。

## 适用条件
- 有 10 个以上价量衍生特征、要做单目标连续值滚动预测，且希望控制参数量与训练成本（GRU 参数少于 LSTM，原文明确选用理由）。
- 数据为日频、样本数千行（示例 4637 行）时参数量匹配，不易过拟合到极端。
- 序列存在较强的非线性与长程依赖（原文档化选型：LSTM/GRU 抓时序长期依赖，CNN 提局部形态，Transformer 建注意力）。
- 作为 ARIMA 之外的深度对照：同项目已用 ARIMA 做统计基线，GRU 用于比较非线性模型是否带来增量。

## 不适用条件
- 只看 MSE 数值判优劣：0.003459 是归一化尺度下的 MSE，未给 RMSE、方向准确率，也未与"预测值=上期值"的朴素基准对比，价格近似随机游走时小 MSE 不代表有预测力。
- 需要模型输出直接当交易信号：原文未考虑交易成本、涨跌停与 T+1，预测价格到可执行策略之间仍有距离。
- 一次性留出、无滚动回测或交叉验证的结果：13 个时间步、13 个特征、64 隐状态等超参未做敏感性分析，结果稳健性未知。
- 全量 fit 的归一化：29-6-01 在全量数据上 fit 后再建模，若与训练/测试切分配合不当会造成信息泄露。
- 周/季/年均值类特征的统计窗口口径不明：若按全样本统计则含未来信息，须改为滚动窗口并只使用历史数据。
- 复现信息不完整的早期试验：29-6-02 收录文本中 EarlyStopping 与 compile 部分被省略，复现需自行补齐。

## 验证状态
原文给出真实运行结果：
- 29-6-02（2004-08-23 至 2023-01-23 日频，4624 个窗口样本，输入 (4624, 13, 13)，GRU(64)+Dense(8,relu)+Dense(1,linear)，Adam+MSE，epochs ≤100，75/25 顺序切分训练 3468 / 测试 1156）：测试集 **MSE = 0.003459**；同时用预测值与实际值曲线目视核对。未给 RMSE、方向准确率或与朴素基线的对比。
- 29-6-01（同一项目的特征工程篇）：给出 13 个预测变量清单、MinMaxScaler 与 `df_to_X_y2(window_size=13)` 实现，但本篇未展示训练/测试损失与预测精度，模型效果无法判断；数据为 4637 行 × 13 列，2004-08-23 至 2023-01-23。
- 6-1-03：GRU/LSTM 用于高频非线性序列的选型规则（需剪枝轻量化）以及"预测指标 + 交易可行性"双轨评估口径，属定性论述，无实验数字。

## 来源
- `（29-6-01）...深度学习模型预测（1）.md`：13 个预测变量清单、MinMaxScaler 标准化、滑窗样本构造函数、RNN/LSTM/GRU 的门控原理与选型理由、Keras 工具链。
- `（29-6-02）...深度学习模型预测（2）.md`：三维张量切分、GRU 网络四层结构、训练配置、测试 MSE 0.003459、五条注意事项（规范化、学习率、正则化、超参、验证方式）。
- `(6-1-03)高频交易与量化交易：高频交易中的预测建模.md`：模型分工与延迟/精度权衡、结构化特征优先树模型、序列依赖强再用 LSTM/Transformer 的选型顺序。
