---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-4-01）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(1)OpenAI接口.md
column: 金融大模型专栏系列-1
title: （26-4-01）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(1)OpenAI接口
distilled: 2026-09-30
relevance: 低
---

# （26-4-01）基于OpenAI和LangChain的上市公司估值系统：OpenAI API和Langchain探索(1)OpenAI接口

## 筛选结论
不适用 —— 该篇是上市公司文档摘要系统中 OpenAI 接口封装的开头片段（消息构造、token 计数、调用封装），面向 EDGAR 财报文本理解，属规范中排除的大模型调用/文本处理方向，与个股量化预测无直接关系。

## 核心内容
- 不适用：内容是调用 OpenAI ChatCompletion 做金融文档摘要的接口层代码（系统提示、模型 token 上限表、token 计数与截断），不涉及价格预测、特征工程、交易信号或回测。

## 可复用要点
无。若仅作工程类比，只有"按模型维护 token 上限并在入参前做长度预算"这一约定可参考。

## 关键实现
文件 openai_interface.py：
- 用 ConfigParser 从 credentials.cfg 读取 api_key，再赋给 openai.api_key。
- INITIAL_CONTEXT_MESSAGE 定义 system 角色提示"作为证券分析助手，帮助理解美国上市公司在 EDGAR 上的财务信息"。
- MODEL_MAX_TOKENS 维护 {"gpt-3.5-turbo": 4097, "gpt-3.5-turbo-16k": 16384}。
- get_completion(messages, model="gpt-3.5-turbo") 封装 openai.ChatCompletion.create，temperature=0 降低输出随机性。
- num_tokens_from_messages 用 tiktoken.encoding_for_model（失败则回退 cl100k_base）估算输入 token，按每条消息约 4 token 的结构开销累加。
- 还包含摘要创建与输入 token 数量检查的函数（原文未展开）。

## 数据与假设
输入为美股公开公司 EDGAR 文档解析后的文本；假设财务文本摘要可由通用大模型完成，且必须在模型上下文预算内裁剪输入。

## 局限与风险
- 原文仅为片段，摘要质量、事实一致性与幻觉风险均未验证。
- temperature=0 并不保证输出确定；token 上限表只覆盖两个旧模型，模型更名即失效。
- 输出是自然语言摘要，与量化因子/信号之间没有结构化映射，接入预测流程需要额外抽取步骤。
