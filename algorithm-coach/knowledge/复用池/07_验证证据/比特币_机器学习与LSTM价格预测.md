---
id: evidence.btc.ml_lstm_price
name: 比特币 × 机器学习与 LSTM 价格预测
category: 07_验证证据
status: 有缺陷
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（15-3）基于时间序列预测的比特币自动交易系统：数据探索分析.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（15-4）基于时间序列预测的比特币自动交易系统：数据转换.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（15-6）基于时间序列预测的比特币自动交易系统：创建机器学习模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（15-7）基于时间序列预测的比特币自动交易系统：构建深度学习模块.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-8）比特币价格预测系统：使用LSTM训练模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（11-4-9）比特币价格预测系统：模型性能可视化.md
---

# 比特币 × 机器学习与 LSTM 价格预测

## 能力说明
本卡记录比特币单变量价格预测的两组真实数字：线性回归/XGBoost/随机森林的横向对比（MAE/RMSE/解释方差/R²）与 LSTM 的训练损失、解释方差/R²。核心价值是揭示"R²≈0.95 的价格水平预测"这一指标陷阱。

## 输入契约
- 数据：BTC-USD 日频，本地 CSV 2587 条（2015-01-01 起，7 列，无缺失）；截取日期 > 2020-09-13 后得 504 条。
- 特征只用 Close 单列；MinMaxScaler(feature_range=(0,1)) 归一化。
- 监督样本：`create_dataset(dataset, time_step=15)`，用过去 15 步预测下一步；训练/测试按时间 70/30 → 352 / 152。
- 另一路（11-4-8）同样是单变量价格序列，直接以"前一时刻价格 → 当前时刻价格"构造样本，输入 reshape 成 (N-1, 1, 1)。

## 输出契约
- 回归指标：MAE、RMSE、explained_variance、R²，分训练集与测试集；预测曲线需 inverse_transform 回美元尺度后再展示。
- 原文的 MAE/RMSE 打印发生在逆变换之前，因此数值落在归一化尺度（0~1），与价格误差不可直接比较。

## 调用方式
```
X_train, y_train = create_dataset(train_data, 15); X_test, y_test = create_dataset(test_data, 15)
models = {"Linear Regression": LinearRegression(),
          "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.3, max_depth=6, n_jobs=4, random_state=0),
          "Random Forest Regressor": RandomForestRegressor(max_depth=1000)}
LSTM_model.add(LSTM(10, input_shape=(None,1), activation="relu")); LSTM_model.add(Dense(1))
LSTM_model.compile(loss="mean_squared_error", optimizer="adam")
history = LSTM_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
```

## 关键参数
- time_step / 窗口 = 15；训练/测试 = 70/30（352/152）；MinMaxScaler 到 [0,1]。
- XGBoost：n_estimators=100、learning_rate=0.3、max_depth=6、random_state=0；随机森林 max_depth=1000。
- LSTM（15-7）：单层 LSTM(10, relu) + Dense(1)，100 epoch、batch 32，直接把测试集当 validation_data。
- LSTM（11-4-8）：LSTM(10, activation='sigmoid', return_sequences=True) + Dense(1)，共 491 个可训练参数（LSTM 480 + Dense 11），batch_size=5、epochs=20，loss=MSE、optimizer=adam。

## 依赖
pandas、numpy、plotly、seaborn、scikit-learn（LinearRegression、MinMaxScaler、metrics、r2_score、explained_variance_score）、xgboost（XGBRegressor）、TensorFlow/Keras（Sequential、LSTM、Dense）。

## 适用条件
- 只适合演示"滑窗 + 机器学习/深度学习"的工程流程，或用于验证"价格水平预测指标会被自相关抬高"这一现象。
- 数据只有单列价格、样本数百条时能跑通。

## 不适用条件
- 不能把 R² 0.94~0.95 理解为预测能力：目标 y 是价格水平本身（归一化后），只要模型近似输出"上一期价格"就能得到高 R²。
- 不能引用 MAE/RMSE 作为价格精度：这两个数在归一化尺度上计算（LR 0.0247 等），既不是美元也不是百分比。
- 15-7 把测试集同时用作 `validation_data`，模型选择基于测试集 → 测试指标偏乐观，不能再当作样本外证据。
- 周/日频口径矛盾（15-6 图注写"周收盘价"，但 504 行与 2020-09-13 至 2022-01 的日频区间一致），引用前必须自行确认频率。

## 验证状态
原文给出真实运行结果（BTC-USD，2020-09-13 之后 504 条，7:3 切分 352/152，窗口 15）：
- 三个回归模型（在归一化尺度上）：Linear Regression MAE 0.02468877316851655 / RMSE 0.03204950005533036；XGBoost Regressor MAE 0.047398661971883777 / RMSE 0.06095256391999026；Random Forest Regressor MAE 0.030420887668338583 / RMSE 0.03881724290350216。
- Linear Regression 的拟合优度：训练集解释方差 0.9884835588523104、测试集 0.9522424065346115；训练集 R² 0.9884835588523104、测试集 R² 0.9522347837752287。
- LSTM（15-7）：X_train (336, 15, 1)、X_test (136, 15, 1)；训练日志 Epoch 1 的 loss 0.2066 / val_loss 0.3584，Epoch 2 0.1695 / 0.2930，Epoch 3 0.1338 / 0.2299；训练集解释方差 0.9849308552789744、测试集 0.9457335146928992；训练集 R² 0.9849171729689591、测试集 R² 0.9456756404595442。
- LSTM（11-4-8）：491 个可训练参数；训练损失从第 1 轮 0.2960 降到第 3–6 轮的 0.0134 / 0.0123 / 0.0099 / 0.0095。
- LSTM 的 MAE/RMSE 打印语句存在，但收录文本中输出为空，**未给出数值**。

### 可信度评估
- 样本量：352 个训练窗口 / 152 个测试窗口，对 491–10000 参数的模型偏小；比特别加密资产的日频段仅 1.4 年。
- 泄漏：滑窗在训练/测试边界可能重叠（原文未做 gap），且 15-7 直接以测试集作验证集；指标口径（归一化尺度）进一步放大乐观程度。
- 基准对比：无。没有"上一期价格"这一朴素基准，无法判断 R² 0.94 是否只是自相关的产物。
- 结论稳定性：差。不同章的 LSTM 配置（relu vs sigmoid、20 epoch vs 100 epoch）与指标口径不一致；本卡按"记录指标陷阱"标记为有缺陷。

## 来源
- 提炼池/线上博客/金融大模型专栏系列-1/（15-3）数据探索分析.md、（15-4）数据转换.md —— 2587 条、7 列、2015-01-01 起、无缺失；2020-01-01 后 761 行。
- 提炼池/线上博客/金融大模型专栏系列-1/（15-6）创建机器学习模型.md —— 504 条、70/30 → 352/152、time_step 15、三个模型的 MAE/RMSE 与解释方差/R²。
- 提炼池/线上博客/金融大模型专栏系列-1/（15-7）构建深度学习模块.md —— LSTM(10, relu)+Dense(1)、100 epoch/batch 32、输入形状 (336,15,1)/(136,15,1)、解释方差与 R²、损失曲线前 3 轮数值。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-8）使用LSTM训练模型.md —— 单变量单步 LSTM 结构（10 单元、sigmoid、return_sequences）、491 参数、batch 5/epoch 20 与训练损失序列。
- 提炼池/线上博客/金融大模型专栏系列-1/（11-4-9）模型性能可视化.md —— 预测需 reshape 成 (样本数,1,1) 并经 inverse_transform 还原，与训练 loss 曲线的判读方法。

## 相关能力

- 上游依赖：[[复用池/03_预测模型/LSTM时序预测|LSTM时序预测]] —— 本卡是该能力两套配置（LSTM(10,relu) 100 epoch 与 LSTM(10,sigmoid) 20 epoch）在 BTC 价格上的实测。
- 上游依赖：[[复用池/02_高级特征工程/标准化与归一化|标准化与归一化]] —— MinMaxScaler 到 [0,1] 的口径决定了 MAE/RMSE 落在归一化域，引用误差数字前必须先确认尺度。
- 并列/替代：[[复用池/07_验证证据/隆基绿能_LSTM价格预测|隆基绿能_LSTM价格预测]] —— 同为"LSTM 直接预测价格、无朴素基线"的另一条实测（A 股个股），两卡可对照该路线的误差量级与指标陷阱。
- 常见误用：[[复用池/06_失败经验/缺朴素基线导致预测力误判|缺朴素基线导致预测力误判]] —— R² 0.95 拟合的是价格水平而非收益率，来自价格强自相关，不是预测力。
- 常见误用：[[复用池/06_失败经验/测试集参与调参与模型选择|测试集参与调参与模型选择]] —— 15-7 直接把测试集当 validation_data，模型选择建立在测试集上，指标偏乐观。
- 常见误用：[[复用池/06_失败经验/全样本统计量泄漏|全样本统计量泄漏]] —— 归一化与滑窗构造在全量序列上完成后再切分，测试段信息进入变换参数。
