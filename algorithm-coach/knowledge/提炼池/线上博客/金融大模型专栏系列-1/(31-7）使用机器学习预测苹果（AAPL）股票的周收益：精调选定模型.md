---
source: 原始池/线上博客/金融大模型专栏系列-1/(31-7）使用机器学习预测苹果（AAPL）股票的周收益：精调选定模型.md
column: 金融大模型专栏系列-1
title: (31-7）使用机器学习预测苹果（AAPL）股票的周收益：精调选定模型
distilled: 2026-09-30
relevance: 高
---

# (31-7）使用机器学习预测苹果（AAPL）股票的周收益：精调选定模型

## 筛选结论
保留 —— AAPL 周收益二分类模型的超参数调优章节，Optuna 搜索空间、以 ROC-AUC 为目标的调参流程和调优后的实际分数都可直接迁移到我们的模型生成与训练环节。

## 核心内容
- 用 Optuna 分别对 Histogram-based Gradient Boosting（HBGB）与 CatBoost 做 1000 次试验的超参数搜索，优化方向为最大化测试集 ROC-AUC。
- HBGB 搜索空间：learning_rate 对数均匀 0.001~0.1、max_iter 100~1000、max_leaf_nodes 2~100、max_depth 1~10、min_samples_leaf 1~15、l2_regularization 1e-4~0.1。
- HBGB 调优后 ROC-AUC 由 0.51 提升至 0.55，幅度有限；试验日志显示单次试验分数在约 0.47~0.55 间波动，说明结果对超参/数据切分敏感。
- CatBoost 搜索空间含 learning_rate、depth、l2_leaf_reg、random_strength、bagging_temperature、border_count、min_data_in_leaf；最优组合为很浅的树（depth=2）。
- CatBoost 调优后 ROC-AUC 由 0.51 提升至 0.58，优于 HBGB；原文归因于类别特征自动处理、缺失值自动处理、抗过拟合与并行训练。
- 评估统一用 predict_proba 取类别 1 的概率计算 ROC 曲线与 AUC，并配合绘图函数输出 ROC 与混淆矩阵。

## 可复用要点
- 调参目标与协议：objective(trial) 返回 AUC，optuna.create_study(direction='maximize')，用 best_params 重新实例化模型（random_state=seed）后 fit + predict_proba 评估。
- 可直接借用的搜索空间边界：learning_rate 对数均匀 1e-3~0.1；max_depth 1~10；min_samples_leaf 1~15；L2 正则 1e-4~0.1；CatBoost 的 border_count、min_data_in_leaf 等。
- 随机种子固定（seed）以保证复现。
- 低信噪比金融分类任务上，浅树（depth=2）在调优中胜出，可作为先验起点。
- 评估可视化组件：roc_curve/auc + 混淆矩阵的组合使用。

## 关键实现
- objective_hbgb(trial)：按 trial 建议超参实例化 HistGradientBoostingClassifier，训练后 predict_proba(X_test)[:,1]，用 roc_curve/auc 返回分数。
- objective_catboost(trial)：结构同上，模型为 CatBoostClassifier，依赖 optuna 与 catboost。
- study.optimize(objective, n_trials=1000)；best_params 与 best_value 分别取最优超参与分数。
- 原文代码块多处被截断（CatBoost 的 objective 函数体、训练与绘图代码不完整），plot_model_performance 只给出调用未见定义。

## 数据与假设
标的为苹果（AAPL）股票，任务是预测周收益的二分类标签（0/1）；特征与训练/测试切分见前文各章，本篇仅使用已划分的 X_train/X_test、y_train/y_test；固定随机种子；评估指标为 ROC-AUC。

## 局限与风险
ROC-AUC 仅 0.55~0.58，接近随机水平，模型实用价值有限；以测试集 AUC 作为 Optuna 的优化目标会造成测试集参与调参（原文未意识到或未讨论），且无嵌套验证；只报告 AUC，没有收益、回测或成本评估；HBGB 与 CatBoost 的性能图被标注为同一图号"图5-8"，疑为原文编号错误；类别不平衡时 ROC-AUC 可能高估实际表现。
