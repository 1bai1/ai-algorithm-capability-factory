---
source: 原始池/线上博客/金融大模型专栏系列-1/（29-4）通过回测、ARIMA 和 GRU 预测股票价格：参数调优.md
column: 金融大模型专栏系列-1
title: （29-4）通过回测、ARIMA 和 GRU 预测股票价格：参数调优
distilled: 2026-09-30
relevance: 高
---

# （29-4）通过回测、ARIMA 和 GRU 预测股票价格：参数调优

## 筛选结论
保留 —— XGBoost 网格调参、早停配置以及用 Log Loss / 分类错误率曲线诊断过拟合的标准流程，可直接用于我们模型的训练与验证环节。

## 核心内容
- 因 XGBoost 会对树剪枝、存在过拟合风险，用 GridSearchCV 做超参数调优，网格仅两个维度：max_depth ∈ {18,21,25}、reg_lambda ∈ {0.1,0.4,0.8}，cv=3。
- 搜索结果为 max_depth=25、reg_lambda=0.1，best_estimator_ 是对应的 XGBClassifier。
- 但随后训练实际使用的是 max_depth=25、learning_rate=0.1、reg_lambda=0.8，与网格选出的 reg_lambda=0.1 不一致（原文未解释）。
- 训练配置 early_stopping_rounds=10，eval_set 同时包含训练集与测试集，监控指标为 error 与 logloss，返回 history 供画曲线。
- 测试集（1211 条）分类报告：accuracy 0.55，类别 0 precision 0.71 / recall 0.33 / F1 0.45，类别 1 precision 0.49 / recall 0.83 / F1 0.62，明显偏向预测类别 1。
- 用分类错误样本计数图展示错误类别分布，用 Log Loss 与分类错误率随迭代轮次的变化曲线判断过拟合。
- 原文指出 Log Loss 专为二分类设计，对错误分类的惩罚重于准确率，更适合不平衡分类的评估。

## 可复用要点
- 调参协议：GridSearchCV(model, param_grid, cv=3).fit(X_resampled, y_resampled)，再取 clf.best_params_ / clf.best_estimator_。
- 可借用的超参先验：max_depth 18~25（深树）、reg_lambda 0.1~0.8、learning_rate 0.1、n_estimators 100。
- 早停配置：early_stopping_rounds=10，eval_set=[(X_train,y_train),(X_test,y_test)]，eval_metric=['error','logloss']。
- 过拟合诊断：读取 history.evals_result_['validation_0'/'validation_1']['logloss']（或 ['error']）分别画训练/测试曲线对比。
- 不平衡二分类应以 Log Loss 和类别级 precision/recall 为主，而不是总体 accuracy。
- 模型实例化示例：XGBClassifier(max_depth=25, learning_rate=0.1, reg_lambda=0.8)。

## 关键实现
- 依赖 xgboost.XGBClassifier、sklearn.model_selection.GridSearchCV、sklearn.metrics.classification_report。
- 训练：model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_train,y_train),(X_test,y_test)], eval_metric=["error","logloss"], verbose=0)。
- 画曲线：epochs = len(history.evals_result_['validation_0']['error'])，x_axis = range(0, epochs)，ax.plot 叠加 Train / Test 两条曲线。
- 调参使用重采样后的 X_resampled/y_resampled，最终训练使用 X_train/y_train，说明前序步骤做过数据重采样。

## 数据与假设
项目为"通过回测、ARIMA 和 GRU 预测股票价格"，本篇只覆盖其中的 XGBoost 调参与训练；标签为二分类（0.0/1.0），特征构造见前文；测试集 1211 条样本；调参在重采样数据上进行，最终训练与早停使用原始训练/测试切分。

## 局限与风险
网格搜索选出的 reg_lambda=0.1 与最终训练使用的 0.8 直接矛盾，属实质性前后不一致；测试集被用于早停的 eval_set，会产生乐观偏差；accuracy 0.55、类别 0 召回仅 0.33，模型接近随机并严重偏向一类，原文仍称"稳定性"提升，结论与数据不符；网格只覆盖两个超参数，learning_rate、n_estimators 等未调；没有收益、交易成本或回测层面的验证。
