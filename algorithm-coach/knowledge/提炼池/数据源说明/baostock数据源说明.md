---
source: 原始池/数据源说明/baostock数据源说明.md
column: 数据源说明
title: baostock 数据源说明
distilled: 2026-10-01
relevance: 高
---

# baostock 数据源说明

## 筛选结论
保留 —— baostock 是免费、免 token 的 A 股行情接入通道，直接对应「个股量化预测」链路的
第一步（取数）。本任务在 Tushare 无 token、akshare 连接失败时用它取回了 600519.SH 与
上证指数日线，是可复用的取数路线。

## 核心内容
1. baostock 是免费、免注册、免 token 的证券数据 Python SDK；依赖只有 `pip install baostock`，
   没有账号与积分门槛。
2. 连接方式为 `bs.login()` / `bs.logout()`，返回对象带 `error_code`（`'0'` 为成功）与 `error_msg`。
3. 核心取数接口是 `bs.query_history_k_data_plus(code, fields, start_date, end_date, frequency, adjustflag)`；
   返回结果集对象，不是 DataFrame，需用 `rs.next()` / `rs.get_row_data()` 自行循环成表，
   列名取 `rs.fields`。
4. 返回的每个单元格都是字符串，日期与数值列必须自行 `pd.to_datetime` / `pd.to_numeric`。
5. `adjustflag`：`1` 后复权、`2` 前复权、`3` 不复权（接口默认）。一次调用只能取一种口径，
   要同时拿原始价与前复权价必须调两次。
6. `frequency`：`d` 日、`w` 周、`m` 月、`5/15/30/60` 分钟；官方注明指数暂无分钟线。
7. 代码体系是「小写交易所前缀 + `.` + 代码」，如 `sh.600519`、`sz.000001`、`sh.000001`（上证指数），
   与统一写法 `600519.SH` 不同，需做显式映射。
8. 个股与指数可用字段不同：个股含 `turn/tradestatus/peTTM/pbMRQ/psTTM/isST`，指数字段止于
   OHLC + `volume/amount/pctChg`。

## 可复用要点
- 特征构造：直接提供 OHLCV + `amount` + `turn` + `pctChg` + `peTTM/pbMRQ/psTTM/isST`，
  以及由 `adjustflag=2` 得到的前复权价序列，可支撑量价、估值、ST 过滤三类特征。
- 模型结构：不涉及（本卡是数据接入，不是模型）。
- 训练与验证方式：不涉及。
- 策略规则：不涉及；可作为「大盘参照」接入指数日线做超额收益 / beta 对齐。
- 评价指标：不涉及；提供 `pctChg` 可用于收益口径自洽性检查（见「关键实现」）。
- 参数取值：日线取数固定用 `frequency='d'`；复权取 `adjustflag='2'`（前复权）与 `'3'`（不复权）成对拉取。
- 数据接口：`baostock.query_history_k_data_plus`；限速自行加 `time.sleep`。

## 关键实现
- 依赖：`baostock`、`pandas`（`time` 用于限速）。
- 登录：`lg = bs.login()`，判断 `lg.error_code == '0'`；结束 `bs.logout()`。
- 取数：`rs = bs.query_history_k_data_plus(code, fields, start_date=..., end_date=..., frequency='d', adjustflag=...)`；
  `while rs.next(): rows.append(rs.get_row_data())`；`pd.DataFrame(rows, columns=rs.fields)`。
- 代码归一：`sh.600519 -> 600519.SH`（拆分 `.`，交易所前缀大写后置）。
- 带重试封装：失败重试 3 次、每次间隔 `sleep_s * i`，用于抵御偶发网络/接口抖动。
- 复权与收益自洽校验：`(adj_close / adj_preclose - 1) * 100` 与接口自带 `pctChg` 比较。

## 数据与假设
- 数据源：baostock 在线接口（免费、免 token）。
- 频率：日线 `frequency='d'`（本任务实际使用的唯一频率）。
- 时间范围：请求 `2015-01-01 ~ 2026-10-01`，实际取回 `2015-01-05 ~ 2026-09-30`。
- 标的：`sh.600519`（贵州茅台）与 `sh.000001`（上证指数）。
- 前提假设：接口返回的字符串需自行转类型；同一标的前复权序列随新除权事件整体变化，
  跨快照不可增量拼接；一次调用只返回一种复权口径。

## 局限与风险
- 只实测了**单标的日线**；分钟级（`5/15/30/60`）、周线、月线均未验证。
- **复权基准日未验证**：官方未在脚本使用范围内暴露基准日，前复权序列跨时间复用是否可比未检验。
- **限频上限未验证**：官方无明确数值，任务只发了 3 次请求、每次 `sleep 0.3`，无法判断安全边界；
  全市场串行取数可能触发限流。
- **无 token 也就没有数据质量承诺**：baostock 无 SLA、无字段文档的版本化对照，长期可复现性弱于付费源。
- 本任务未做单只股票之外的多标的、跨市场（港股/美股）验证；本卡结论只覆盖 A 股沪深个股与指数。
- 未验证指标正确性（`peTTM` 等估值字段的来源与计算口径未核对），只验证了 OHLC 的逻辑自洽与
  复权收益一致性。
