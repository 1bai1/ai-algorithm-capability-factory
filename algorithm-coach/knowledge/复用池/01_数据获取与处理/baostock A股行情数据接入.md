---
id: data.acquire.baostock
name: baostock A股行情数据接入
category: 01_数据获取与处理
status: 已验证
sources:
  - 提炼池/数据源说明/baostock数据源说明.md
---

# baostock A股行情数据接入

## 能力说明
用 baostock 免费、免 token 的 Python SDK 拉取 A 股个股与指数日线，一次任务内把**不复权
（raw）与前复权（qfq）两套价格**都取回并按日期对齐，得到可直接入模的行情快照。解决
「本机无 Tushare token、akshare 连接失败时，还能从哪里取 A 股行情」的问题。

## 输入契约
- 依赖 `baostock` 库，必须先 `bs.login()` 且判断 `lg.error_code == '0'`。
- 标的用 baostock 代码体系：沪市 `sh.`、深市 `sz.`，个股如 `sh.600519`，指数如 `sh.000001`。
- 字段以逗号分隔字符串一次性声明。个股与指数字段不同：
  - 个股：`date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST`
  - 指数：`date,code,open,high,low,close,preclose,volume,amount,pctChg`
- 区间用 `start_date` / `end_date`（`YYYY-MM-DD`）。
- 复权口径必须显式二选一或分别调用：`adjustflag='2'` 前复权、`'3'` 不复权。
- 无账号、无 token、无积分要求；接口不保证限频数值，调用方需自行限速。

## 输出契约
- 每次查询返回一个结果集对象（非 DataFrame）：`rs.error_code`、`rs.error_msg`、`rs.fields`、
  `rs.next()`、`rs.get_row_data()`。
- 自行循环得到 DataFrame 后，所有单元格仍是字符串，需 `pd.to_datetime` 转日期、
  `pd.to_numeric` 转数值。
- 本卡产物是「原始价 + 前复权价」对齐面板，列名显式区分口径：
  - 不复权列（`adjustflag='3'`）：`open, high, low, close, preclose`
  - 前复权列（`adjustflag='2'`）：`adj_open, adj_high, adj_low, adj_close, adj_preclose`
  - 公共列：`date, code, volume, amount, turn, pctChg, peTTM, pbMRQ, psTTM, isST`
- 代码体系统一：baostock `sh.600519` 映射为 `600519.SH`（主键列可被下游按统一代码对齐）。
- 日期升序、`date` 无重复、OHLC 无 NaN、`pctChg` 采用前复权口径（除权日收益连续）。

## 调用方式
```python
import time
import pandas as pd
import baostock as bs

STOCK_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

def to_std_code(bs_code):
    """sh.600519 -> 600519.SH"""
    ex, num = bs_code.split(".", 1)
    return "%s.%s" % (num, ex.upper())

def fetch(code, fields, start, end, adjustflag, sleep_s=0.3, attempts=3):
    """带重试的单次查询，返回 (DataFrame, fields)"""
    last_err = None
    for i in range(1, attempts + 1):
        try:
            rs = bs.query_history_k_data_plus(code, fields, start_date=start,
                                              end_date=end, frequency="d",
                                              adjustflag=adjustflag)
            if rs.error_code != "0":
                raise RuntimeError("baostock %s: %s" % (rs.error_code, rs.error_msg))
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            time.sleep(sleep_s)             # 自行限速
            return pd.DataFrame(rows, columns=rs.fields)
        except Exception as exc:
            last_err = exc
            time.sleep(sleep_s * i)
    raise RuntimeError("query failed after %d attempts: %r" % (attempts, last_err))

lg = bs.login()
assert lg.error_code == "0", lg.error_msg
try:
    qfq = fetch("sh.600519", STOCK_FIELDS, "2015-01-01", "2026-10-01", "2")  # 前复权
    raw = fetch("sh.600519", STOCK_FIELDS, "2015-01-01", "2026-10-01", "3")  # 不复权
    idx = fetch("sh.000001", INDEX_FIELDS, "2015-01-01", "2026-10-01", "3")  # 上证指数
finally:
    bs.logout()

# 原始价 + 前复权价对齐（date 内连接，保证两套价格同日）
q = qfq.rename(columns={"open": "adj_open", "high": "adj_high",
                        "low": "adj_low", "close": "adj_close",
                        "preclose": "adj_preclose"})
panel = raw[["date", "code", "open", "high", "low", "close", "preclose",
             "volume", "amount", "turn"]].merge(
    q[["date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_preclose"]],
    on="date", how="inner")
panel = panel.merge(qfq[["date", "pctChg", "peTTM", "pbMRQ", "psTTM", "isST"]],
                    on="date", how="left")
```

必要前后处理：`date` 转 datetime 并升序排序；数值列 `pd.to_numeric`；`code` 用
`to_std_code` 统一；取到即落盘 CSV 快照（记录区间、行数、列名、复权口径）。

## 关键参数
| 参数 | 含义 | 实测/文档取值 |
| --- | --- | --- |
| `code` | 标的代码 | `sh.600519`（个股）、`sh.000001`（上证指数）、`sz.000001`（深市个股） |
| `fields` | 字段清单 | 个股 17 字段见「输入契约」；指数 10 字段 |
| `start_date` / `end_date` | 区间 | `2015-01-01` ~ `2026-10-01`（实际取回 2015-01-05 ~ 2026-09-30） |
| `frequency` | 频率 | `d`（实测）；文档另列 `w/m/5/15/30/60`，指数无分钟线 |
| `adjustflag` | 复权 | `'2'` 前复权、`'3'` 不复权（实测）；`'1'` 后复权（文档） |
| `sleep_s` | 限速间隔 | `0.3` 秒（每次查询后） |
| `attempts` | 重试次数 | 3，退避 `sleep_s * i` |

## 依赖
`baostock`（`login` / `logout` / `query_history_k_data_plus`）；`pandas`（`DataFrame`、`to_datetime`、
`to_numeric`、`merge`）；`time`（限速）。

## 适用条件
- 本机无 Tushare token、akshare 不可用，需要一条免费、零配置的 A 股取数通道时。
- 研究需要**同时持有原始价与前复权价**（撮合用原始价、收益/标签用前复权价）时。
- A 股个股与大盘指数的日线级别取数，含估值（`peTTM/pbMRQ/psTTM`）与 ST 标记（`isST`）。
- 需要把非标准代码体系（`sh.600519`）归一为 `600519.SH` 再与其它源对齐时。

## 不适用条件
- 需要分钟级/高频数据：本卡只实测日线，分钟频率虽在文档中存在但未验证；标的高频研究不要照搬。
- 需要美股、港股、加密：baostock 只覆盖 A 股沪深市场，换 yfinance 或交易所 API。
- 需要**严格可复现的复权序列**：文档未暴露复权基准日，前复权序列跨时间复用是否可比未经验证；
  要可追溯口径应改用自带 `divCash/splitFactor` 的数据源自行复权。
- 需要 SLA 或字段级版本契约的工程化管道：baostock 免费、无限频承诺、无版本化文档。
- 全市场逐股串行扫描：限频上限未知，本任务只发了 3 次请求，大规模任务需先探明限流。

## 验证状态
已验证（有真实运行结果，任务 `20261001_600519_A股日线数据获取`，`validation.json` 的
`overall = pass`、0 项 fail）：
- 登录返回 `error_code='0'`、`error_msg='success'`。
- 600519.SH 前复权与不复权各 **2855 行**；上证指数 **2855 行**；合并面板 **2855 × 20**。
- 区间 `2015-01-05 ~ 2026-09-30`；`date` 无重复、升序；OHLC 无缺失（0 个 NaN）；
  `high ≥ max(open,close)` 与 `low ≤ min(open,close)` 违反 0 条。
- 不复权收盘 **169.64 ~ 2601.00**；前复权收盘 **123.29 ~ 2227.15**。
- 前复权与不复权日期集合**对称差 0**；相邻交易日无 >15 天缺口。
- `|pctChg − 复权收益|` 最大 **0.0001 个百分点**（原始值 5.5e-05），收益列与复权价自洽。
- 未验证：分钟级取数、复权基准日、限频上限、`peTTM` 等估值字段口径、多标的批量取数。

## 来源
- 提炼池/数据源说明/baostock数据源说明：`login/logout`、`query_history_k_data_plus` 签名、
  `adjustflag` 与 `frequency` 取值、代码体系与个股/指数字段清单、免费免 token 的事实。
- 本次任务脚本 `任务池/20261001_600519_A股日线数据获取/generated/fetch_baostock.py`：
  双口径取数 + 代码归一 + 落盘 + 结构化校验的可运行实现；`validation.json` 提供全部实测数字。

## 相关能力

- 并列/替代：[[Tushare A股行情与新闻数据接入|Tushare A股行情与新闻数据接入]] —— 同为 A 股取数通道；Tushare 需要账号 token（本机不可用），本卡免 token，两者按「有无凭据」二选一，代码体系与复权参数不可混用
- 下游用途：[[复权口径与复权价选用|复权口径与复权价选用]] —— 本卡一次取回 `adjustflag=2` 前复权与 `'3'` 不复权两套价并成对落盘，是该卡「原始价与复权价并存、声明口径」的落地通道；实测两者收盘区间 169.64~2601.00 与 123.29~2227.15
- 下游用途：[[行情字段契约与多标的面板组织|行情字段契约与多标的面板组织]] —— 本卡把 `sh.600519` 归一为 `600519.SH`，产出 date+code 主键、raw/adj 成对价格列的 2855×20 面板，直接满足该卡对统一主键与复权列名的要求
- 常见误用：[[高频定位与数据频率错配|高频定位与数据频率错配]] —— 本卡实测只覆盖日线（frequency='d'），文档中的分钟频率未验证，拿它做高频研究会踩频率错配的坑
