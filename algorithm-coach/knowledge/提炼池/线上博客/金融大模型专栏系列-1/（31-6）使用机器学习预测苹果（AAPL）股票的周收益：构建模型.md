---
source: 原始池/线上博客/金融大模型专栏系列-1/（31-6）使用机器学习预测苹果（AAPL）股票的周收益：构建模型.md
column: 金融大模型专栏系列-1
title: （31-6）使用机器学习预测苹果（AAPL）股票的周收益：构建模型
distilled: 2026-09-30
relevance: 高
---

# （31-6）使用机器学习预测苹果（AAPL）股票的周收益：构建模型

## 筛选结论
保留 —— AAPL 周收益方向预测的建模环节，给出梯度提升树分类器候选池、统一 ROC-AUC 评估口径与可照搬的多模型训练/评估循环，可直接对应本项目的模型生成与选型环节。

## 核心内容
1. 选型理由：分类器统一取梯度提升树族，因其对数据比例、分布与异常值更鲁棒。
2. 分类器清单 5 个：AdaBoostClassifier、GradientBoostingClassifier、HistGradientBoostingClassifier、CatBoostClassifier、XGBClassifier；全部设 random_state=seed（seed=123）保证可复现。
3. 评估口径统一：能用评估指标的算法一律用 ROC-AUC—HistGradientBoostingClassifier 设 scoring='auc'，XGBClassifier 设 eval_metric='auc' 与 objective='binary:logistic'；CatBoostClassifier 设 verbose=False。
4. 建模前先收敛特征：X_train/X_test 只保留 selected_features 的 9 列——sma_12、sma4_ratio、rsi、obv、4week_vol、12week_vol、acceleration、velocity、obv_divergence_12_weeks（来自前一章的特征选择结果）。
5. 统一循环：fit → `predict_proba(X_test)[:,1]` 取正类概率 → roc_curve 算 FPR/TPR → auc 算面积 → 自定义函数 plot_model_performance 输出 ROC 曲线与混淆矩阵。
6. 结果很差：最好的是 Gradient Boosting，ROC-AUC 仅 0.53；CatBoost 0.51。原文据此决定对 HistGradientBoosting 与 CatBoost 做进一步调参（理由是此前测试中 HistGradientBoosting 更稳定）。

## 可复用要点
- 多模型横向对比模板：同一份特征集上遍历分类器列表，统一用 ROC-AUC 比较，输出 ROC 曲线+混淆矩阵，是模型选型的低成本基线做法，可直接作为 Agent 的模型筛选骨架。
- 两个工程细节：算 AUC 必须取 `predict_proba(X_test)[:,1]` 的正类概率；随机种子全局统一（seed=123）以保证可复现。
- 重要的负面基准：9 个技术指标 + 周频方向标签的 AUC 只有 0.51-0.53，接近随机水平，说明该因子组合对周度方向几乎无预测力，可作为同类特征集的信息量参照。
- 流程策略：先多模型粗筛，锁定 2 个候选（HistGradientBoosting、CatBoost）再进入调参阶段。

## 关键实现
依赖 scikit-learn（AdaBoostClassifier、GradientBoostingClassifier、HistGradientBoostingClassifier、roc_curve、auc）、catboost、xgboost。核心循环：
```
for (name, clf) in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plot_model_performance(name, y_test, y_pred)
```
`plot_model_performance(name, y_test, y_pred)` 为自定义函数（原文未给实现，功能是绘制 ROC 曲线与混淆矩阵）。所有分类器均在实例化时传入 random_state=seed。

## 数据与假设
AAPL 周频样本；X 为 selected_features 指定的 9 个技术指标列，y 为二分类方向标签（cat_target，定义见前序章节）；训练/测试集沿用前文按时间顺序的切分。本篇未重复给出样本量与日期范围。

## 局限与风险
- 结果几乎无预测力（AUC 0.51-0.53），原文未分析原因，也未回退检查标签定义或特征口径。
- 只做单次留出集评估，无交叉验证，5 个模型间 0.51 与 0.53 的差异不具统计意义；各模型准确率与混淆矩阵数值均未报告。
- 调参决策依据是"此前测试中 HistGradientBoosting 更稳定"的主观印象，稳定性如何度量未说明。
- AUC 与策略盈亏不等价，原文未计交易成本、未做回测；结论不能直接迁移到 A 股或用于实盘。
