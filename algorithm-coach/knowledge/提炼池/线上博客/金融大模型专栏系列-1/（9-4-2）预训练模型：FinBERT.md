---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-4-2）预训练模型：FinBERT.md
column: 金融大模型专栏系列-1
title: （9-4-2）预训练模型：FinBERT
distilled: 2026-09-30
relevance: 中
---

# （9-4-2）预训练模型：FinBERT

## 筛选结论
部分保留 —— FinBERT 金融情感分类器可作为舆情因子的生产组件（情绪分析用作因子属我们的高相关方向），但本篇只覆盖微调与验证流程，未涉及情感分数如何变成因子并进入预测模型。

## 核心内容
- 用 Hugging Face 的 `ProsusAI/finbert` 对金融新闻标题做三分类情感识别：negative / neutral / positive。
- 数据划分：85% 训练 / 15% 验证，按标签分层（stratify=label），random_state=2022；训练集 negative 513、neutral 2447、positive 1159，验证集 91 / 432 / 204。
- 先用 BertTokenizer（do_lower_case=True）统计标题长度分布，确定编码最大长度 max_length=150。
- 编码参数：add_special_tokens=True、return_attention_mask=True、padding 到最大长度。
- 模型：AutoModelForSequenceClassification，num_labels=3，关闭 attention 与 hidden states 输出。
- 训练配置：batch_size=5，训练集 RandomSampler、验证集 SequentialSampler；AdamW(lr=1e-5, eps=1e-8)，线性 warmup 调度（warmup=0），epochs=2。
- 第 1 轮后训练损失下降、验证损失上升，判定为过拟合，最终取第 1 轮模型为最佳。
- 各类准确率：neutral 385/432、negative 81/91、positive 170/204。

## 可复用要点
- 情感标签体系：positive=2、neutral=0、negative=1（三分类），后续可映射为日频情绪因子取值。
- 训练超参参考：lr=1e-5、AdamW、batch_size=5、线性 warmup、分层切分。
- 早停判断依据：验证损失由降转升即停止；本例最优轮次为第 1 轮。
- 评估口径：按类别分别统计准确率，而非只看总体准确率。

## 关键实现
- 分词器：`BertTokenizer.from_pretrained("ProsusAI/finbert", do_lower_case=True)`。
- 模型：`AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=len(sentiment_dict), output_attentions=False, output_hidden_states=False)`。
- 编码：`finbert_tokenizer.batch_encode_plus(..., return_tensors='pt', add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=150)`。
- 数据加载：`DataLoader(dataset, sampler=RandomSampler/SequentialSampler, batch_size=5)`；优化器 `AdamW`；调度器 `get_linear_schedule_with_warmup`。
- 自定义函数：`get_headlines_len`、`show_headline_distribution`、`evaluate`、`accuracy_per_class`。
- 依赖：transformers、torch、tqdm、random/numpy（固定种子 2022，设备 CUDA 优先）。

## 数据与假设
- 数据：金融新闻标题（字段 NewsHeadline），已带 sentiment/label 标注，共约 4846 条标题。
- 划分：仅训练/验证（85:15），无独立测试集。
- 假设：标题文本本身足以判定情感；标签分布可通过分层抽样保持。

## 局限与风险
- 只跑 2 个 epoch 且第 1 轮即出现过拟合，既未充分训练也未调参，未使用早停回调保存最优权重。
- 没有独立测试集，类别准确率基于验证集，存在乐观偏差。
- 英文模型与英文语料，用于 A 股中文舆情需替换预训练模型与标注数据。
- 情感输出到因子（如日频情绪得分）的映射方式、与收益的相关性均未验证。
