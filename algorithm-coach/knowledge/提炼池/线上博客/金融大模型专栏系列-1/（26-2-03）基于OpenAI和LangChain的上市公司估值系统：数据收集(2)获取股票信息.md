---
source: 原始池/线上博客/金融大模型专栏系列-1/（26-2-03）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)获取股票信息.md
column: 金融大模型专栏系列-1
title: （26-2-03）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)获取股票信息
distilled: 2026-09-30
relevance: 中
---

# （26-2-03）基于OpenAI和LangChain的上市公司估值系统：数据收集(2)获取股票信息

## 筛选结论
部分保留 —— 属于"股票价格数据获取"环节，主接口+降级接口的取价思路可迁移到我们的数据层；但它是估值系统的抓网页子步骤，实现依赖硬编码代理，工程上不能直接照搬。

## 核心内容
- yahoo_finance.py 负责从 Yahoo Finance 页面抓取股票价格信息，供后续估值分析使用。
- 提供三个函数：通用 HTTP 请求、盘前价获取、当前价获取。
- 当前价获取采取降级策略：先尝试盘前价格，失败则取当日历史价格数据中的最近收盘价作为"当前价"。
- 请求伪装：自定义 Accept / Accept-Encoding / User-Agent / Cache-Control 请求头并携带站点 Cookie，绕开反爬限制。
- 通过第三方商业代理（Bright Data 或 PacketStream）发起请求，代理地址与账号密码直接写在脚本里。

## 可复用要点
- 取数容错模式：主接口（盘前价）+ 备用接口（当日历史收盘价）两级 fallback，保证取价链路不中断。
- 接口字段：ticker 为入参，输出盘前价格或当日历史价格序列（取最新收盘价）。
- 反爬三件套：完整浏览器请求头 + Cookie + 代理 IP。

## 关键实现
- 文件 `yahoo_finance.py`；函数 `request_yahoo_url(url)`（带重试，处理连接错误与超时）、`get_premarket_price_yahoo(ticker)`、`get_current_price_from_yahoo(ticker, created_at=None)`。
- 脚本顶部定义 `bright_data_proxy` / `packet_stream_proxy` 并二选一赋给 `proxy`。
- 依赖 HTTP 请求库（requests 风格）与 HTML 解析（页面抽取盘前价）。

## 数据与假设
- 数据源：Yahoo Finance 网页；标的以美股 ticker 表示（示例代码出现 AAPL、MSFT、GOOG 等）。
- 获取内容：盘前价格、当日历史价格。
- 假设：代理稳定可用、页面结构不变、Yahoo 未封禁。

## 局限与风险
- 代理账号、Cookie 明文硬编码在源码中，已失效或存在泄露风险，无法直接复现。
- 依赖网页结构解析，页面改版即失效；未提及限频、缓存、失败重试上限与合规问题。
- 仅覆盖美股行情源，与 A 股数据接口（如 Tushare/AkShare）不能直接对接。
- 原文未给出任何取价结果的验证或准确率说明。
