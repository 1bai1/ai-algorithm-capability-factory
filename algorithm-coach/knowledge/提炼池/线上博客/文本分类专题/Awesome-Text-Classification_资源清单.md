---
source: 原始池/线上博客/文本分类专题/Awesome-Text-Classification_资源清单.md
column: 线上博客
title: Awesome-Text-Classification
distilled: 2026-10-01
relevance: 中
---

# Awesome-Text-Classification

## 筛选结论

保留 —— 一份纯链接式资源清单，列出文本分类常用的开源实现、两篇奠基论文与一篇教程；本身无实验、无结论，价值在于"要落地一个文本分类基线时去哪找实现"的候选池。清单未分类整理，需自行甄别维护状态。

## 核心内容

1. **通用开源实现**：fastText（文本表示与分类库）、CNN_sentence（Kim 的句子分类 CNN）、cnn-text-classification-tf（TensorFlow 版 CNN 文本分类）、textClassifier（文档分类的层级注意力网络 HAN）、Crepe（字符级卷积网络文本分类）、text_classification（brightmart 的深度学习文本分类模型合集）、multi-class-text-classification-cnn-rnn（用 CNN/RNN + 词向量做 Kaggle 旧金山犯罪描述 39 分类）、klassify（基于 Redis 的贝叶斯文本分类服务）、sent-conv-torch（Torch 版卷积文本分类）。
2. **中文实现**：cnn-text-classification-tf-chinese（TensorFlow 中文 CNN 分类）、text-classification-cnn-rnn（基于 TensorFlow 的中文 CNN-RNN 分类）、Chinese-Text-Classification（TensorFlow CNN 中文分类）。
3. **奠基论文入口**：Convolutional Neural Networks for Sentence Classification（arXiv:1408.5882）、Character-level Convolutional Networks for Text Classification（arXiv:1509.01626）。
4. **教程**：wildml 的《Implementing a CNN for Text Classification in TensorFlow》。
5. **清单特征**：按项目/中文/论文/教程四节平铺，无数据、无基准数字、无维护状态标注。

## 可复用要点

- **按任务形态挑实现**：短文本/句子级分类 → Kim-CNN 一类卷积模型；长文档分类 → HAN（文档-句子两层注意力）；无分词或形态复杂语料 → Crepe 字符级 CNN；需要极快基线 → fastText。
- **中文任务**：优先在三个中文仓库里找带字/词级预处理的中文实现，避免直接用英文分词流程。
- **破冰顺序**：先跑通 fastText 或 Kim-CNN 拿到基线准确率，再上预训练模型对照；两个 arXiv 链接是 CNN/字符级 CNN 的原始出处，用于核对结构细节。
- 清单不含任何超参与精度数字，不能用于选型打分。

## 关键实现

- 仓库名与地址（清单原文给出）：facebookresearch/fastText、yoonkim/CNN_sentence、dennybritz/cnn-text-classification-tf、richliao/textClassifier、zhangxiangxiao/Crepe、brightmart/text_classification、jiegzhan/multi-class-text-classification-cnn-rnn、fatiherikli/klassify、harvardnlp/sent-conv-torch、indiejoseph/cnn-text-classification-tf-chinese、gaussic/text-classification-cnn-rnn、fendouai/Chinese-Text-Classification。
- 清单未给出任何函数的输入输出、依赖版本或关键参数。

## 数据与假设

- 清单中唯一提到具体数据集的条目是 Kaggle San Francisco Crime Description（39 类，用 CNN/RNN + 词向量分类）。
- 其余条目无数据说明；无实验假设与频率、区间等信息。

## 局限与风险

- **时效性差**：多为 2015–2018 年的 TensorFlow 1.x 时代仓库，多数已停止维护，与当前 PyTorch / transformers 生态不匹配，直接复用需重写训练与数据管道。
- **无质量信号**：没有精度、速度、维护状态等任何指标，无法据此判断实现质量。
- **清单维护质量有限**：条目夹杂社群推广链接，中文条目含 QQ 群/公众号信息，需甄别。
- 完全不覆盖预训练模型时代的资源（transformers、Hugging Face 等），不能作为当前技术栈的唯一清单。
