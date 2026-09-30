---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-4-1）预训练模型：FinBERT.md
column: 金融大模型专栏系列-1
title: （9-4-1）预训练模型：FinBERT
distilled: 2026-09-30
relevance: 中
---

# （9-4-1）预训练模型：FinBERT

## 筛选结论
部分保留 —— 金融领域情感分类模型与微调流程可作为情绪因子的上游组件，但本篇只做标题分类，未接入预测或交易链路。

## 核心内容
- FinBERT 在 BERT 基础上用金融语料继续预训练并微调，在金融文本分类与情感分析上有优势。
- 结构沿用原生 BERT：Base 为 12 层 Transformer、Large 为 24 层。
- 预训练含字词级与任务级两类；任务级引入两个有监督目标：研报行业分类（约 40 万文档级语料，每行业 5k~20k）与财经新闻金融实体识别（约 50 万条语料）。
- 实例目标：加载预训练 FinBERT，对金融新闻标题做分词编码、微调，按验证集选最优模型后预测情感。
- 数据：FinancialNewsHeadline.csv 共 4846 条，标签分布 neutral 2879、positive 1363、negative 604，类别明显不平衡。
- 流程还包含：统计编码后标题长度分布（关注超过 512 上限的比例）、随机抽样查看标题与标签、情感标签数字化。
- 评估：加权 F1 与各类别准确率，训练中以 evaluate 返回验证损失与预测。

## 可复用要点
- 情感标签体系：正面/负面/中性三分类，可直接用作新闻情绪因子的标签定义。
- 输入长度控制：先统计 token 长度分布，再决定对超过 512 的样本截断或切分。
- 评估指标：加权 F1 + 每类准确率；类别不平衡时不能只看总体准确率。
- 微调选型：按验证集表现保存最佳模型再用于预测。

## 关键实现
函数：show_headline_distribution、show_random_headlines、get_headlines_len（用 finbert_tokenizer.encode(headline, add_special_tokens=True) 统计长度）、encode_sentiments_values（情感标签→数字映射表）、f1_score_func（argmax 后 weighted F1）、accuracy_per_class、evaluate（model.eval() 后算验证损失与预测）。依赖 transformers 的 FinBERT 与其 tokenizer、sklearn.metrics.f1_score、pandas、seaborn、tqdm。

## 数据与假设
FinancialNewsHeadline.csv，4846 条英文金融新闻标题，三分类情感标签；假设新闻标题情感可代表市场情绪倾向。

## 局限与风险
仅英文语料，中文 A 股新闻未验证；标签不平衡（负面仅 604 条），小类准确率易失真；未做情感与收益率/波动率的相关性检验，因子有效性未验证；未展示完整微调超参、训练轮数与最终准确率数值；对标题长度、行业与时间分布漂移未做检验。
