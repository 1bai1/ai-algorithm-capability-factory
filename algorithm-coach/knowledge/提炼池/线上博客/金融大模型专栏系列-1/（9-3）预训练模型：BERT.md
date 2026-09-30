---
source: 原始池/线上博客/金融大模型专栏系列-1/（9-3）预训练模型：BERT.md
column: 金融大模型专栏系列-1
title: （9-3）预训练模型：BERT
distilled: 2026-09-30
relevance: 低
---

# （9-3）预训练模型：BERT

## 筛选结论
不适用 —— 本篇本质是 BERT 文本分类与微调教程，聚焦"NLP 模型怎么用"，不涉及价格、收益或因子有效性的检验；若后续要从新闻标题构造情绪因子，其中的关键词后处理做法与失败案例可作警示。

## 核心内容
- BERT 是预训练 Transformer，可作特征提取器或经微调直接做情感分类；使用流程为选框架（TensorFlow/PyTorch）→ 从 Hugging Face 等渠道下载权重 → 加载 → 用于编码或分类。
- 实例 9-2 用中文 BERT（chinese_wwm_ext_pytorch）对同花顺新闻快讯标题做情感分析，前 10 条标题几乎全部被判为"积极"。
- 作者将这种全积极输出归因于预训练模型未适应该任务或文本本身偏积极，而非模型缺陷。
- 改进手段是手写情感关键词表做后处理：消极词为利空、下跌、跌幅、亏损、超跌、下滑、强执、跌超；积极词为利好、增长、涨幅、上涨、涨停。
- 作者明确澄清：这些关键词与预训练模型参数无关，是使用者自行定义的列表，只能用于预处理打标或输出后处理。
- 微调流程（实例 9-4）：三分类标签映射（消极=0、积极=1、中性=2）→ tokenizer 编码（含 padding、truncation）→ TensorDataset 加 DataLoader 分 batch → CrossEntropyLoss 加 AdamW → 多轮训练 → save_pretrained 保存。

## 可复用要点
- 可起步的中文金融情感词典：positive = {利好, 增长, 涨幅, 上涨, 涨停}；negative = {利空, 下跌, 跌幅, 亏损, 超跌, 下滑, 强执, 跌超}。
- 工程顺序建议：先用规则词典做预处理/后处理兜底，再考虑微调模型；通用预训练模型直接套用到金融文本会系统性偏移。
- 微调标签体系：三分类映射（积极 1/消极 0/中性 2），可直接与行情标签对齐后再做因子有效性检验。

## 关键实现
- 关键调用：`BertTokenizer.from_pretrained(path)`、`BertForSequenceClassification.from_pretrained(path, num_labels=3)`、`model.save_pretrained("fine_tuned_model")`。
- 训练组件：TensorDataset 加 DataLoader（batch_size 可配）、`nn.CrossEntropyLoss()`、`torch.optim.AdamW`。
- 输入：同花顺快讯 CSV（news_10jqka.csv）的 title 列；微调数据为带情感标签的 news_with_sentiment.csv。
- 原文多处代码块缺失，超参数（学习率、epoch、batch_size）均未披露。

## 数据与假设
- 数据：同花顺新闻快讯标题（快照式取前 10 条）；微调用带标签标题的 CSV。
- 假设：新闻标题的情绪倾向可被三分类模型识别，且该标签对市场分析有意义（本篇未验证）。

## 局限与风险
- 实例输出全为"积极"，说明直接用通用中文 BERT 做金融标题情绪分类不可靠；作者未量化准确率，也未做人工标注校验。
- 情感关键词是人工枚举的小词表，覆盖有限，且与模型预测逻辑完全脱钩，只能覆盖少数样本。
- 全篇未把情绪标签与任何收益或波动指标关联，情绪因子有效性零验证。
- 代码不完整（多处"具体实现代码如下所示"后无代码），无法复现；模型版本、参数与训练轮数均未知。
- 与本项目的量价预测无直接耦合，暂不纳入特征管线。
