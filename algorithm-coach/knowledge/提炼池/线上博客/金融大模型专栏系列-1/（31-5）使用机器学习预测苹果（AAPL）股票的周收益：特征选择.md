---
source: 原始池/线上博客/金融大模型专栏系列-1/（31-5）使用机器学习预测苹果（AAPL）股票的周收益：特征选择.md
column: 金融大模型专栏系列-1
title: （31-5）使用机器学习预测苹果（AAPL）股票的周收益：特征选择
distilled: 2026-09-30
relevance: 高
---

# （31-5）使用机器学习预测苹果（AAPL）股票的周收益：特征选择

## 筛选结论
保留 —— 与本项目最贴近的单篇：以 AAPL 周频数据为标的，用 TimeSeriesSplit + RFECV + ROC-AUC 做特征选择，给出了完整候选因子名与各模型选出的最优特征集，可直接作为 Agent 特征选择模块的实现模板。

## 核心内容
1. 因股票数据有时间顺序，交叉验证必须用 TimeSeriesSplit(n_splits=5)，不能用随机 K 折，以保证训练集与验证集日期顺序一致。
2. 候选特征共 21 个：Open/High/Low/Close/Adj Close/Volume，returns、cont_target、cat_target，sma_4/12/20 及 sma4/12/20_ratio，rsi 与 rsi_overbound/rsi_underbound，obv 与 obv_divergence_12_weeks，4week_vol、12week_vol、velocity、acceleration。
3. 目标定义为 cat_target（方向分类标签），另保留 cont_target（连续收益）与 returns；训练前先 dropna，再 drop 掉 returns 与 cont_target 构造 X。
4. 统一用 RFECV(estimator, cv=cv, scoring='roc_auc') 递归消除特征，最优特征数由交叉验证的 AUC 决定，不同估计器得到不同子集。
5. 梯度提升树选出 7 个特征：sma_12、sma4_ratio、rsi、obv、4week_vol、12week_vol、acceleration。
6. 随机森林选出 4 个：obv、4week_vol、12week_vol、acceleration。
7. XGBoost 选出 3 个：Close、Adj Close、sma_12；AdaBoost 也选出 3 个（原文未列出具体特征名）。
8. 测试阶段汇总了一份 9 特征清单（sma_12、sma4_ratio、rsi、obv、4week_vol、12week_vol、acceleration、velocity、obv_divergence_12_weeks）用于准备模型测试数据。

## 可复用要点
- 时序特征选择范式：TimeSeriesSplit 交叉验证 + RFECV + scoring='roc_auc'，全过程不使用未来数据。
- 因子池模板（可直接照搬）：4/12/20 周均线及其与价格的比值、RSI 及超买/超卖标记位、OBV 及 12 周背离、4 周与 12 周波动率、速度 velocity、加速度 acceleration。
- 多估计器交叉验证思想：被多种模型共同选中的特征（sma_12、4week_vol、12week_vol、acceleration 出现频次最高）可信度更高，可作为稳定因子候选。
- 数据规模参考：训练 1233 周 × 21 特征，测试 479 周 × 21 特征。
- 目标设计：分类标签配 ROC-AUC，同时保留连续收益列便于回归建模对照。
- 特征选择与后续训练解耦：先定 selected_features 列表，再统一切分 X_test/y_test。

## 关键实现
依赖 scikit-learn（TimeSeriesSplit、RFECV、GradientBoostingClassifier、RandomForestClassifier、AdaBoostClassifier）与 xgboost（XGBClassifier），统一 random_state=seed 保证可复现。核心片段：
- cv = TimeSeriesSplit(n_splits=5)
- estimator = GradientBoostingClassifier(random_state=seed); rfe = RFECV(estimator=estimator, cv=cv, scoring='roc_auc'); rfe.fit(X_train, y_train)
- print("Optimal number of features: %d" % rfe.n_features_); print(list(X_train.columns[rfe.support_]))
- train = train.dropna(); X_train, y_train = X_y_split(train.drop(['returns','cont_target'], axis=1), 'cat_target')

## 数据与假设
AAPL 周频数据，训练集自 1990-05-14 起共 1233 个样本 × 21 个属性，测试集 479 个样本 × 21 个属性；特征已做标准化/比率化处理。假设周收益方向（cat_target）可由上述技术指标分类预测，且必须按时间顺序做交叉验证。

## 局限与风险
- 不同估计器选出的最优特征差异极大（XGBoost 选到 Close/Adj Close，树模型选到波动率类），说明结果对估计器高度敏感；原文未做稳定性检验，也未对比最终模型表现。
- 未报告 RFECV 各步的 AUC 数值、最终模型准确率/AUC 或任何回测收益，效果不可验证。
- cat_target 的构造方式与阈值（涨跌分界、收益窗口定义）原文未交代；dropna 造成的样本损失未统计。
- 仅在 AAPL 单只标的上验证，跨股票泛化能力未知。
