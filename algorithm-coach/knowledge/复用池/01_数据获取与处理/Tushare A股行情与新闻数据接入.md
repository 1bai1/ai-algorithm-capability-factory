---
id: data.acquire.tushare
name: Tushare A股行情与新闻数据接入
category: 01_数据获取与处理
status: 已验证
sources:
  - 提炼池/线上博客/金融大模型专栏系列-1/（6-4）量化选股程序.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（5-3）制作贵州茅台的ARCH模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-2）基于人工智能的资产定价方法.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-3）交易策略优化.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（8-4-1）股票交易策略实战：制作股票交易策略模型.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（4-2-01）常用的时间序列分析方法（1）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（6-1-2）高频交易（HFT）.md
  - 提炼池/线上博客/金融大模型专栏系列-1/（9-1）金融市场情绪分析的概念与方法.md
  - 提炼池/线上博客/AI金融实战系列-2（持续更新）/（1-2）金融时间序列分析：移动平均法.md
---

# Tushare A股行情与新闻数据接入

## 能力说明
用 Tushare Pro 拉取 A 股个股日线/周线/月线、指数与市场成交量、上市标的清单以及同花顺新闻快讯，作为 A 股预测与策略链路的原始数据层。解决"从哪取 A 股数据、参数怎么传、取回字段是什么、复权怎么控"。

## 输入契约
- 前置：Tushare 账号 token，`ts.set_token(token)` 后 `ts.pro_api()` 得到 pro 客户端。
- 标的代码必须带交易所后缀：`600519.SH`、`000977.SZ`、`300479.SZ`、`002594.SZ`、`300308.SZ`、`601012.SH`。
- 日期参数 `start_date` / `end_date`：原文示例用 `'20200101'` 这类紧凑格式，也出现过 `'2020-01-01'`；未说明两者是否等价，按接口文档为准（原文未给）。
- 频率：`freq` 取 `D`/`W`/`M`（pro_bar）；经 FinRL 封装时改用 `time_interval='1d'`。
- 票池两种来源：接口枚举 `pro.stock_basic(exchange='', list_status='L', fields='ts_code')`，或外部给定清单（示例为 15 只沪市股票）。
- 新闻接口需带时分秒的时间串，且 `src` 指定来源（示例 `'10jqka'`）。
- 单次可取的日线长度、积分门槛与具体调用频次上限原文未给出。

## 输出契约
- `pro.daily` 返回字段：open / high / low / close / pre_close / change / pct_chg / vol / amount（另含 ts_code、trade_date）。`trade_date` 落到 DataFrame 后仍是字符串/对象，必须显式转 datetime 再排序。
- `ts.pro_bar(..., adj='qfq', ma=[短,长])` 直接返回前复权行情并附带所请求周期的均线列，省去自行 rolling。
- 经 FinRL `DataProcessor` 的下载+清洗+加指标流水线后，产物为 17 列：tic、time、index、open、high、low、close、adjusted_close、volume + 8 个技术指标；多标的以 date×tic 纵向堆叠。
- 新闻接口返回 datetime / content / title 三列，落盘为 UTF-8 CSV。
- 复权口径：本组来源只出现 `'qfq'`（前复权），未出现后复权；前复权序列会随新的除权事件整体变化，跨日复用的同一只股票历史值可能对不上。

## 调用方式
```python
import os, tushare as ts
ts.set_token(os.environ['TS_TOKEN'])        # 原文示例为明文硬编码 token，不要沿用
pro = ts.pro_api()
df  = pro.daily(ts_code='600519.SH', start_date='20200101', end_date='20230901')
df2 = ts.pro_bar(ts_code='600519.SH', freq='W', adj='qfq',
                 start_date='20000101', end_date='20230901', ma=[13, 55])
lst = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
news = pro.news(src='10jqka', start_date='2023-01-11 09:00:00', end_date='2023-08-22 10:10:00')
df.to_csv('maotai_stock_data.csv', index=False)   # 取到即落盘，后续脚本读本地 CSV
```
必要前后处理：`trade_date` → `pd.to_datetime` → 按日期升序 → 落盘快照；需要大盘指数/市场成交量作辅助特征时，把个股 `trade_date` 改名 `Date` 后与指数按日期对齐，并在入模前统一量纲（指数级别在数千量级，未标准化会导致损失量级失控）。

FinRL 封装路径：`DataProcessor(data_source='tushare', start_date=..., end_date=..., time_interval='1d', **kwargs).download_data(ticker_list=...)`，token 经 kwargs 注入。

## 关键参数
| 参数 | 含义 | 原文取值/说明 |
| --- | --- | --- |
| `ts_code` | 标的代码 | 600519.SH、000977.SZ、300479.SZ、002594.SZ、601012.SH、600725.SH、603259.SH、300308.SZ |
| `start_date`/`end_date` | 取数区间 | 2000-01-01~2023-09-01、2008-01-01~2023-09-01、2010-01-01~2020-01-01、2020-01-01~2023-09-01、2020-09-17~2023-09-17、2022-01-01~2023-01-01 |
| `freq` | 频率 | D / W / M |
| `adj` | 复权 | `'qfq'` 前复权（唯一出现过的取值） |
| `ma` | 顺带返回的均线 | [13,55]、[8,21] 等，均与调用方自己的策略参数绑定，不是通用推荐值 |
| `list_status` | 上市状态 | `'L'`（仅上市） |
| `time_interval` | FinRL 封装频率 | `'1d'` |
| 限频/重试参数 | — | 原文未给出 Tushare 的具体限额与重试配置 |

## 依赖
tushare（`ts.pro_api`、`ts.pro_bar`）；pandas / numpy；matplotlib（画图）；可选 FinRL（DataProcessor 封装）、os（token 注入）。

## 适用条件
- A 股个股、指数、市场成交量的日线及以上研究，需要本土数据源与本土代码体系时。
- 需要"上市标的清单"作为票池起点（可平移到沪深 300/中证 500 成分股思路）。
- 需要 A 股新闻快讯做舆情因子时（`pro.news` 直接给 datetime/content/title）。
- 与 FinRL 组合搭建 A 股多标的强化学习环境（15 只股票示例可直接套用）。

## 不适用条件
- 需要分钟级/秒级或订单簿数据：原文明确提示 Tushare 高频数据要付费，全部实例实际用的是日频，不能拿日频冒充高频。
- 需要美股、加密资产：换 yfinance/交易所 API，Tushare 票池不覆盖。
- 需要复权口径可追溯、可复现的研究：原文只给 `'qfq'`，未说明复权基准日与因子来源，跨时间复用的前复权序列不可比。
- 全市场逐股串行扫描的大批量任务：原文的实现是逐股请求 + 空循环延时，效率低且易触发限流。
- 需要凭据安全的工程化环境：原文示例把 token 明文写在脚本里，必须改为环境变量。

## 验证状态
已验证（数据获取环节有真实运行产物）：
- 600725.SH 与 603259.SH 在 2022-01-01~2023-01-01 取到 241 个交易日。
- 600519.SH 日线 2020-01-01~2023-09-01 落盘 CSV 共 891 条记录。
- FinRL+TuShare 15 只沪市股票：下载 (17960, 8)；清洗后 (18315, 8)；加 8 个技术指标后 (18270, 17)；训练集 (16695, 17)。（清洗后行数多于下载行数，原文未解释。）
- 同花顺快讯 2023-01-11~2023-08-22 成功落盘为 UTF-8 CSV；前 10 条标题用 TextBlob 判极性全为"中性"，等于未产出有效因子。
- 原文未给出接口失败率、限流触发情况与积分要求，属未验证。

## 来源
- （6-4）量化选股程序：`pro_bar(ts_code, freq, adj='qfq', ma=[短,长])` 参数组合、`stock_basic` 取上市清单、全市场遍历的限流风险。
- （5-3）制作贵州茅台的ARCH模型：`pro.daily` 字段清单、用 Tushare 另取大盘指数与市场成交量、取数与建模脚本分文件。
- （8-2）基于人工智能的资产定价方法：TuShare 日线字段（trade_date/pct_chg）与最小取数范式。
- （8-3）交易策略优化：`pro.daily(...)["close"]` 取收盘价序列的最小调用、票池单标的示例。
- （8-4-1）股票交易策略实战：FinRL DataProcessor 的 tushare 数据源配置、15 只票池、四段行数记录、产物 17 列字段。
- （4-2-01）常用的时间序列分析方法（1）：多标的取数区间与"241 个交易日"结果、token 硬编码风险。
- （6-1-2）高频交易（HFT）：`trade_date` 需转 datetime、Tushare 高频数据付费提示。
- （9-1）金融市场情绪分析：`pro.news(src='10jqka')` 的字段与落盘细节。
- AI金融实战系列-2/（1-2）移动平均法：取数后立即 `to_csv` 落盘复用的最小范式与 891 条记录结果。
