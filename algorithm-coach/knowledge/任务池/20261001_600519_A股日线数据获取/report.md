# 报告：600519.SH 日线数据获取

任务目录：`knowledge/任务池/20261001_600519_A股日线数据获取/`

## 一、产物在哪

| 文件 | 内容 | 行数 × 列数 |
|---|---|---|
| `data/600519_daily_merged.csv` | **主产物**：原始价 + 前复权价对齐面板 | 2855 × 20 |
| `data/600519_daily_qfq.csv` | 600519.SH 前复权日线（adjustflag=2） | 2855 × 17 |
| `data/600519_daily_raw.csv` | 600519.SH 不复权日线（adjustflag=3） | 2855 × 17 |
| `data/sh000001_index_daily.csv` | 上证指数日线（大盘参照） | 2855 × 10 |
| `validation/validation.json` | 结构化校验结果 | — |
| `generated/fetch_baostock.py` | 取数 + 落盘 + 校验脚本（可重跑） | — |

- 标的：600519.SH 贵州茅台（自选「任意 A 股」，换标的只需改脚本顶部的 `STOCK_CODE_BS`）。
- 区间：2015-01-05 ~ 2026-09-30（请求到 2026-10-01，取回最新可得交易日）。
- 频率：日频。字段：OHLCV + amount + turn + pctChg + peTTM/pbMRQ/psTTM/isST。

## 二、关键事实

**数据源**：baostock（免费、免 token）。
- 本机未装 `tushare` 且无 `TS_TOKEN`；`akshare` 已装但实测连接东方财富失败
  （`RemoteDisconnected`）；`baostock` 实测可登录取数。故知识库唯一的 A 股取数卡
  （`Tushare A股行情与新闻数据接入`，已验证）不可用，改用 baostock。

**复权口径**（对齐 `复权口径与复权价选用` 卡）：
- `sh.600519` / `adjustflag='2'` → 前复权（qfq），列名 `adj_open/adj_high/adj_low/adj_close/adj_preclose`；
- `adjustflag='3'` → 不复权（raw），列名 `open/high/low/close/preclose`；
- 两套并存、列名显式区分，`pctChg` 采用前复权口径（除权日收益连续）。
- 复权使历史价格整体下移：原始收盘区间 169.64 ~ 2601.00，前复权收盘区间 123.29 ~ 2227.15。

**校验结果**（`validation/validation.json`，`overall = pass`，0 项 fail）：
- 三份序列各 2855 行；`date` 无重复、严格升序、已转 datetime；
- OHLC 无缺失（0 个 NaN）；`high ≥ max(open,close)`、`low ≤ min(open,close)` 全部满足（违反 0 条）；
- 前复权与不复权的日期集合完全一致（对称差 0）；
- `|pctChg − 复权收益|` 最大 **0.0001 个百分点**，说明收益列与复权价自洽；
- 相邻交易日间隔无 >15 天缺口（无长期停牌/断档）。

**代码体系归一**：baostock 返回 `sh.600519`，已映射为知识库统一写法 `600519.SH`（`行情字段契约与多标的面板组织` 卡要求统一主键列）。

## 三、接着能做什么

这份快照是下游全部环节的输入，可直接接：

1. **特征工程**：`技术指标与量价特征构造`（用 `adj_*` 列算指标）、`波动率特征构造`、
   `滞后、差分与滚动统计特征构造`。
2. **标签构造**：`预测标签构造` —— 未来 N 日收益回归 / 涨跌分类，务必用 `adj_close` 算收益。
3. **平稳化与建模**：`价格序列平稳化` → `自回归族 ARIMA` / `GRU` / `LSTM` / `GARCH` 波动率。
4. **大盘对齐**：用 `交易日对齐与多源数据合并` 把个股与 `sh000001_index_daily.csv`
   按 `date` 对齐，做超额收益 / beta。
5. **回测**：`滚动回测引擎` + 买入持有基准；撮合用 `close`（真实成交价），净值用 `adj_close`。

**换标的**：改 `generated/fetch_baostock.py` 顶部 `STOCK_CODE_BS`（如 `sz.000001` 平安银行、
`sh.601012` 隆基绿能），重跑即可，脚本会自动按日期命名产物。

## 四、本次的知识缺口（已按闸门委派知识管理员）

知识库 `01_数据获取与处理` 中，A 股取数只有 Tushare 一张卡，且依赖账号 token。
**「免费 / 免 token 的 A 股取数通道」完全没有卡片**（baostock、akshare 均未覆盖），
属闸门里的「完全没有相关卡片」。本次已验证 baostock 可稳定取数并给出可复用脚本，
故委派知识管理员按 `外部资料 → 原始池 → 提炼池 → 复用池` 的路径补一张
baostock A 股数据接入能力卡。
