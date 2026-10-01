---
source: baostock 官方 Python API 文档 + 任务池 20261001_600519_A股日线数据获取/generated/fetch_baostock.py 实测
title: baostock 数据源说明
url: http://baostock.com/
collected: 2026-10-01
---

# baostock 数据源说明

## 一、这是什么

baostock 是一个免费、开源的证券历史行情数据平台，提供 Python SDK。**不需要注册账号、
不需要 token、不需要积分**，`pip install baostock` 后即可登录取数。这一点是本任务选它而
不选 Tushare（需账号 token）的直接原因。

本说明只收录两类内容：
1. 任务脚本 `fetch_baostock.py` 实际调用过、并成功取回数据的接口事实；
2. baostock 官方 Python API 文档中的确定事实。
未在文档中给出、或本任务未实测的项，一律在文末「未验证」处标明，不写成结论。

## 二、安装与连接

```bash
pip install baostock
```

```python
import baostock as bs

lg = bs.login()          # 登录
print(lg.error_code, lg.error_msg)   # 成功时为 '0' / 'success'
# ... 取数 ...
bs.logout()              # 登出
```

- `bs.login()` / `bs.logout()` 返回一个带 `error_code`、`error_msg` 属性的对象。
- 登录失败时 `error_code != '0'`，应先判断再取数。
- 取数进程结束前应调用 `bs.logout()`。

## 三、取数接口 `query_history_k_data_plus`

```python
rs = bs.query_history_k_data_plus(
    code,                 # 证券代码，如 'sh.600519'、'sz.000001'、'sh.000001'
    fields,               # 字段，逗号分隔的字符串
    start_date='',        # 'YYYY-MM-DD'
    end_date='',          # 'YYYY-MM-DD'
    frequency='d',        # 频率
    adjustflag='3',       # 复权类型
)
```

返回值 `rs` 是一个结果集对象，不是 DataFrame：

| 成员 | 含义 |
|---|---|
| `rs.error_code` | `'0'` 表示成功，否则为错误码 |
| `rs.error_msg` | 错误信息 |
| `rs.fields` | 列名列表，顺序与每行的 `get_row_data()` 一致 |
| `rs.next()` | 游标下移一行，返回 bool；为 False 表示取完 |
| `rs.get_row_data()` | 返回当前行，`list[str]` |

标准取数循环：

```python
rs = bs.query_history_k_data_plus(code, fields, start_date=start, end_date=end,
                                  frequency='d', adjustflag='2')
if rs.error_code != '0':
    raise RuntimeError('%s %s' % (rs.error_code, rs.error_msg))
rows = []
while rs.next():
    rows.append(rs.get_row_data())
df = pd.DataFrame(rows, columns=rs.fields)
```

**返回的每个单元格都是字符串**，数值列需自行 `pd.to_numeric`，日期列需自行
`pd.to_datetime`。

### 常用字段

个股日线：

```text
date,code,open,high,low,close,preclose,volume,amount,
adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST
```

指数日线（无 `turn`、`tradestatus`、`pe/pb/ps`、`isST` 等个股字段）：

```text
date,code,open,high,low,close,preclose,volume,amount,pctChg
```

常用字段含义：

| 字段 | 含义 |
|---|---|
| `date` | 交易日期 `YYYY-MM-DD` |
| `code` | 证券代码（baostock 体系，如 `sh.600519`） |
| `open/high/low/close` | 开高低收（按 `adjustflag` 决定是否为复权价） |
| `preclose` | 前收盘价 |
| `volume` | 成交量（股） |
| `amount` | 成交额（元） |
| `adjustflag` | 该行所属的复权类型 |
| `turn` | 换手率 |
| `tradestatus` | 交易状态（1 正常交易，0 停牌） |
| `pctChg` | 涨跌幅（%） |
| `peTTM` | 滚动市盈率 |
| `pbMRQ` | 市净率 |
| `psTTM` | 滚动市销率 |
| `isST` | 是否 ST |

### `adjustflag` 复权取值

```text
1 = 后复权
2 = 前复权
3 = 不复权   ← 接口默认值
```

一次调用只能取一种复权口径。要同时保留原始价与前复权价，必须调两次
（`adjustflag='3'` 一次、`adjustflag='2'` 一次）。指数没有复权概念。

### `frequency` 频率取值

```text
d  = 日K线（默认）
w  = 周K线
m  = 月K线
5  = 5分钟线
15 = 15分钟线
30 = 30分钟线
60 = 60分钟线
```

官方文档注明：**指数暂无分钟线**。

## 四、证券代码体系

baostock 的代码格式是「交易所前缀（小写）+ `.` + 数字代码」：

```text
sh.600519   沪市个股（贵州茅台）
sz.000001   深市个股（平安银行）
sh.000001   上证指数
sz.399001   深证成指
```

与常见写法 `600519.SH` 的区别是**交易所前缀在后且大写**，需要显式转换：

```python
def to_std_code(bs_code: str) -> str:
    """baostock 代码 sh.600519 -> 统一代码 600519.SH"""
    ex, num = bs_code.split(".", 1)
    return "%s.%s" % (num, ex.upper())
```

## 五、最小可运行示例（取自本任务脚本）

```python
import time
import pandas as pd
import baostock as bs

STOCK_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

lg = bs.login()
assert lg.error_code == "0", lg.error_msg
try:
    def fetch(code, fields, start, end, adjustflag):
        rs = bs.query_history_k_data_plus(code, fields, start_date=start,
                                          end_date=end, frequency="d",
                                          adjustflag=adjustflag)
        if rs.error_code != "0":
            raise RuntimeError("%s %s" % (rs.error_code, rs.error_msg))
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        time.sleep(0.3)                      # 自行限速
        return pd.DataFrame(rows, columns=rs.fields)

    qfq = fetch("sh.600519", STOCK_FIELDS, "2015-01-01", "2026-10-01", "2")  # 前复权
    raw = fetch("sh.600519", STOCK_FIELDS, "2015-01-01", "2026-10-01", "3")  # 不复权
    idx = fetch("sh.000001", INDEX_FIELDS, "2015-01-01", "2026-10-01", "3")  # 上证指数
finally:
    bs.logout()
```

## 六、其它可用接口（官方文档，未在本任务中实测）

- `bs.query_trade_dates(start_date, end_date)`：交易日历。
- `bs.query_all_stock(day)`：某交易日全部证券代码。
- `bs.query_stock_basic(code)`：证券基本信息。

## 七、未验证 / 需注意

- **限频数值**：官方文档未给出明确的每秒/每分钟请求上限；本任务用每次查询后
  `time.sleep(0.3)` 限速，一次任务只发了 3 次请求，是否接近上限无法判断。
- **分钟线的历史深度、复权基准日**：本任务只取了日线，分钟线可用区间与复权基准日未实测。
- **数据口径的具体计算方式**（成交量单位、复权因子来源）以官方文档为准，本任务未逐项核对。
