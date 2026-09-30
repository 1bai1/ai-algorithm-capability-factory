---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-3）通过回测、ARIMA 和 GRU 预测股票价格：机器学习模型.md
column: 金融大模型专栏系列-1
title: （29-3）通过回测、ARIMA 和 GRU 预测股票价格：机器学习模型
distilled: 2026-09-30
relevance: 高
---

# （29-3）通过回测、ARIMA 和 GRU 预测股票价格：机器学习模型

## 筛选结论
保留 —— 完整演示了"价格预测转监督分类"的全流程（标签构造、防未来函数平移、类别重采样、时序切分、XGBoost 训练调参、分类报告与学习曲线），与本项目模型训练与验证环节一一对应。

## 核心内容
- 标签构造：用 `rolling(2)` 比较相邻两日收盘价，当前值高于前值记 1、否则记 0，把价格预测改造成二分类监督任务。
- 防未来函数：特征整体 `shift(1)`，用前一天及更早的数据预测下一日走势；原文强调不平移就等于用当天数据预测当天目标，结果不可信。
- 类别轻微不平衡，用 `RandomOverSampler(random_state=0)` 对 OHLCV 特征与标签过采样；但重采样发生在切分之前，会打乱时间顺序。
- 划分按时间顺序 75%/25%（`threshold=int(len(X_resampled)*0.75)`），训练集 3633 行、测试集 1211 行。
- 基线模型 XGBClassifier(max_depth=3, learning_rate=0.1)，early_stopping_rounds=10，eval_metric=["error","logloss"]；结果 accuracy 0.49，类别 1 召回 0.92、类别 0 召回仅 0.15，模型几乎把一切都预测成上涨。
- 调参尝试：max_depth=7（reg_lambda=0.6，early_stopping_rounds=2）→ accuracy 0.50；max_depth=15 → accuracy 0.50，均无提升；原文承认出现过拟合并观察到训练/测试误差趋同。
- 评价手段：classification_report、误分类计数图、Log Loss 曲线与分类错误曲线（取 history.evals_result_）。

## 可复用要点
- 标签与特征对齐范式：Target = (Close_t > Close_{t-1})，全部特征 shift(1) 后再入模，是本项目生成监督样本的必备步骤。
- 时序切分必须按时间保序（75/25 或按日期切点），不能随机乱序。
- 评价指标组合：precision/recall/f1 + accuracy，辅以误分类分布图与 error/logloss 学习曲线判断是否过拟合。
- 无 alpha 基线参考：仅用 OHLCV 的 XGBoost 日频涨跌分类约 0.49~0.50 准确率、且呈"高召回低精确"的偏多输出，可作为本项目判断模型是否真有预测力的对照下限。
- 过拟合控制手段：限制 max_depth、加 L2（reg_lambda）、早停（early_stopping_rounds）。

## 关键实现
- 依赖 xgboost（XGBClassifier）、imbalanced-learn（RandomOverSampler）、seaborn、matplotlib、sklearn.metrics.classification_report。
- 关键参数：max_depth ∈ {3, 7, 15}；learning_rate=0.1；reg_lambda=0.6；early_stopping_rounds ∈ {10, 2}；eval_set=[(X_train,y_train),(X_test,y_test)]；eval_metric=["error","logloss"]；verbose=0。
- 特征列固定为 Open、High、Low、Close、Volume 五列，无任何技术指标或外部因子。
- 曲线绘制取 history.evals_result_['validation_0'|'validation_1']['error'|'logloss']。

## 数据与假设
标的未明确，价格量级约 2.5~2.8（疑为低价股或已复权数据），样本自 2004-08-19 起，日频 OHLCV；重采样后训练 3633 行、测试 1211 行；假设历史价量形态可外推至未来。

## 局限与风险
- 先做 RandomOverSampler 再按位置切分训练/测试，合成样本会同时进入训练与测试集，测试集已被污染，0.49/0.50 的准确率不可信。
- 特征只有 OHLCV，无任何因子或技术指标，准确率与随机猜测无异，模型实际没有预测力。
- 原文称"模型出现了过拟合现象，并且训练集和测试集的错误率趋于一致"，二者逻辑矛盾（误差趋同更接近欠拟合）。
- 只评分类指标，未给交易层面的累计收益、最大回撤、夏普，无法判断能否盈利。
- 类别 0 召回仅 0.15~0.21，模型不会识别下跌，直接用于多空决策风险极大。
