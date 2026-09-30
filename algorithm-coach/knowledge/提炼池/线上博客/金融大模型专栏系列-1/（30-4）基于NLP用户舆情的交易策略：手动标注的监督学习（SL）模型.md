---
source: 原始池/线上博客/金融大模型专栏系列-1/（30-4）基于NLP用户舆情的交易策略：手动标注的监督学习（SL）模型.md
column: 金融大模型专栏系列-1
title: （30-4）基于NLP用户舆情的交易策略：手动标注的监督学习（SL）模型
distilled: 2026-09-30
relevance: 高
---

# （30-4）基于NLP用户舆情的交易策略：手动标注的监督学习（SL）模型

## 筛选结论
保留 —— 展示把新闻标题转成情感因子的可执行路径（规则/情感词典/事件词三级打标 + 词向量 + 多模型交叉验证），且每条样本与收盘价、事件收益对齐，正是"情绪分析用作因子"的样本。

## 核心内容
- 项目总纲设定三种情感标签生成路线：预训练朴素贝叶斯；基于领域知识的标签（百分比值、超出预期、未达预期、手动标注）；以及用各股票的专家标记新闻构建可覆盖 12 只票之外标的的通用模型。
- 本篇聚焦手动标注的监督学习，用三类规则造标签：提取标题中的百分比值、TextBlob 情感极性、财报相关超出/未达预期。
- 百分比标签：从标题抓取涨跌幅百分比（如 "(WFMI) -5.2%"、"(NFLX +1.1%)"）；perc_sent 取值分布为 True 1429、NaN 1124、False 589。
- 情感兜底：对 1124 条未提取到百分比的样本，用 TextBlob 计算 blob_sent 极性值补标签。
- 事件标签：财报季的 beats/misses 报道共 177 行，用标题字符串包含 'misses' 标 0、包含 'beats' 标 1。
- 汇总口径：仅保留 perc_sent 非空的行构成 ldf（并删除 blob_sent 与 Close 列），把 True/False 映射为 1/0；样本量约 2146，词向量矩阵形状为 2146×96。
- 文本特征用 spaCy 静态词向量，把标题内词向量取平均得到 96 维句向量（简单平均句嵌入）。
- 评估：90/10 训练测试划分 + 四折交叉验证，模型池含逻辑回归、KNN、决策树、SVM、随机森林，输出 cv 均值/std、训练与测试准确率及混淆矩阵热力图。
- 结论：LR 与 SVM 的泛化优于其训练表现，DT 与 RF 训练-测试差距大（过拟合）；作者把错分归因于人工打标经验不足，并导出错分样本（含 ticker、headline、date、eventRet 与各模型预测）逐条复盘。

## 可复用要点
- 情感因子打标优先序可直接移植到 A 股股吧/新闻标题：先抽数值型情感（标题内的涨跌幅百分比，最客观）→ 用情感词典兜底 → 用 beats/misses 类事件词覆盖财报新闻。
- 数据表最小字段集：ticker、headline、date、eventRet（事件收益）、Close 与标签列；标签必须与收益字段对齐，情绪因子才具备进入策略的前提。
- 文本特征基线用平均句向量即可（静态词向量，96 维），无需微调大模型，适合快速建立基线。
- 模型评估最小组合：多模型横向对比 + 交叉验证 + 训练/测试差值，用后者识别过拟合；短文本小样本上线性模型（LR/SVM）通常比树模型更稳，可作默认首选。
- 错分样本人工复盘（导出原始字段 + 各模型预测并排查看）是提升标签质量的正反馈环节，尤其适合标注规则仍在迭代的阶段。
- 标签是上限：标签质量问题无法靠换模型解决，应先做标签一致性校验再谈模型优化。

## 关键实现
- `class make_labels(df, corpus)`：方法含 percentage_val（提百分比并标记）、sentiment_val（算情感极性）、create_label(option)（按选项分支生成标签）；静态工具 `isfloat(element)` 做浮点校验；依赖 nltk.tokenize.word_tokenize、copy.deepcopy、textblob.TextBlob。
- `class nlp_evals(df, corpus, label, ...)`：get_embedding（文档转 96 维平均词向量）、tts（默认 90/10 划分）、define_models（注入模型列表）、kfold（cross_val_score 四折，另存训练与测试得分及混淆矩阵）、plot_results（热力图）；结果表列名为 model、cv mean、cv std、train、test。
- 事件标签覆盖：`tdf.loc[tdf['headline'].str.contains('misses'), 'perc_sent'] = False`、`tdf.loc[tdf['headline'].str.contains('beats'), 'perc_sent'] = True`。
- 样本集构造：`ldf = tdf.loc[~tdf['perc_sent'].isna()]`；`ldf = ldf.drop(['blob_sent','Close'], axis=1)`；`ldf['perc_sent'].replace({False:0, True:1}, inplace=True)`。
- 错分分析：`all_test[all_test['LR_test'] != all_test['perc_sent']].index` 取错分索引，再 `pd.concat([ldf.loc[wrong_index,:], all_test.loc[wrong_index]], axis=1)` 合并查看。

## 数据与假设
- 数据为美股新闻标题集，覆盖 12 只股票（样本中出现 AMZN、NFLX、MSFT、GOOG、AMD、BA 等），样本日期从 2011 年到 2018 年。
- 字段：ticker、headline、date、eventRet、Close，以及派生的 perc_sent、blob_sent；eventRet 有正有负（如 +0.031269、-0.188974）。
- 假设：标题中的百分比数字代表方向性情感；TextBlob 极性可为剩余样本提供可用标签；beats/misses 关键词足以代表财报超预期/不及预期。

## 局限与风险
- perc_sent 语义混杂且判定不稳定：该列既承担"是否提取到百分比"又承担情感正负，示例中 "MSFT -1.4%" 被标为 True 而 "MSFT -1.2%" 被标为 False，方向判断前后矛盾；模型直接在含噪标签上做监督训练，精度上限被标签质量锁死。
- 1124 条待人工标注的样本，本篇只交代计划，未见实际标注结果与标注一致性（如 Kappa）校验。
- 训练/测试按 90/10 随机划分而非按时间切分，与"用历史预测未来"的实际用法不一致，同一事件的重复报道还可能跨集泄露。
- 全部为英文财经标题与美股，迁移到 A 股中文语境需重建抽取规则与情感词典。
- 平均句向量会抹平否定、程度副词等关键信息，原文未做与其他文本表示的对比。
- 树模型在 96 维平均向量上明显过拟合，原文未尝试正则化、降维或早停。
