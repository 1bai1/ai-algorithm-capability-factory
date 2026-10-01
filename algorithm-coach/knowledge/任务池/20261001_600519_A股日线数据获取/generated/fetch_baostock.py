# -*- coding: utf-8 -*-
"""
A 股日线数据获取 + 落盘快照 + 结构化校验。

数据源: baostock（免费、免 token）。Tushare 需账号 token，本机不可用；
akshare 实测连接东方财富失败。

复权口径: adjustflag='2' 前复权(qfq)、'3' 不复权(raw)、'1' 后复权(未用)。

产出:
  data/600519_daily_qfq.csv       前复权日线
  data/600519_daily_raw.csv       不复权日线
  data/600519_daily_merged.csv    原始价 + 前复权价对齐面板
  data/sh000001_index_daily.csv   上证指数日线（大盘参照）
  validation/validation.json      结构化校验结果

用法:
  python fetch_baostock.py [--start 2015-01-01] [--end 2026-10-01] [--sleep 0.3]
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(TASK_DIR, "data")
VAL_DIR = os.path.join(TASK_DIR, "validation")

STOCK_CODE_BS = "sh.600519"      # baostock 代码体系
STOCK_CODE_STD = "600519.SH"     # 知识库统一代码体系
INDEX_CODE_BS = "sh.000001"
INDEX_CODE_STD = "000001.SH"

STOCK_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

REQUIRED_STOCK_COLS = {"date", "code", "open", "high", "low", "close",
                       "volume", "amount", "pctChg"}
REQUIRED_INDEX_COLS = {"date", "code", "open", "high", "low", "close",
                       "volume", "amount", "pctChg"}

NUMERIC_STOCK_COLS = ["open", "high", "low", "close", "preclose", "volume", "amount",
                      "turn", "pctChg", "peTTM", "pbMRQ", "psTTM"]

checks = []


def add_check(name, status, detail, value=None):
    checks.append({"name": name, "status": status, "detail": detail, "value": value})


def query_with_retry(bs, code, fields, start, end, adjustflag, freq="d",
                     attempts=3, sleep_s=0.3):
    """带重试的单次查询：返回 list[list[str]]。取不到返回空列表，不抛异常中断整批。"""
    last_err = None
    for i in range(1, attempts + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=start, end_date=end,
                frequency=freq, adjustflag=adjustflag)
            if rs.error_code != "0":
                raise RuntimeError("baostock error %s: %s" % (rs.error_code, rs.error_msg))
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            time.sleep(sleep_s)  # 限速
            return rows, rs.fields
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(sleep_s * i)
    raise RuntimeError("query failed after %d attempts: %r" % (attempts, last_err))


def rows_to_df(rows, fields):
    df = pd.DataFrame(rows, columns=fields)
    return df


def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_std_code(bs_code):
    """baostock 代码 sh.600519 -> 知识库统一代码 600519.SH。"""
    if not isinstance(bs_code, str) or "." not in bs_code:
        return bs_code
    ex, num = bs_code.split(".", 1)
    return "%s.%s" % (num, ex.upper())


def normalize_dates(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    return {"path": os.path.relpath(path, PROJECT_ROOT).replace("\\", "/"),
            "rows": int(len(df)), "cols": int(df.shape[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    result = {
        "task": "600519.SH 日线数据获取",
        "data_source": "baostock (adjustflag=2 qfq / 3 raw)",
        "requested_range": [args.start, args.end],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {},
        "snapshots": {},
        "checks": checks,
        "error_type": None,
        "overall": "fail",
    }

    import baostock as bs

    lg = bs.login()
    add_check("baostock_login", "pass" if lg.error_code == "0" else "fail",
              "%s %s" % (lg.error_code, lg.error_msg))
    if lg.error_code != "0":
        result["error_type"] = "runtime_error"
        _dump(result)
        return 1

    try:
        # ---------- 取数 ----------
        qfq_rows, sfields = query_with_retry(bs, STOCK_CODE_BS, STOCK_FIELDS,
                                             args.start, args.end, "2", sleep_s=args.sleep)
        raw_rows, _ = query_with_retry(bs, STOCK_CODE_BS, STOCK_FIELDS,
                                       args.start, args.end, "3", sleep_s=args.sleep)
        idx_rows, ifields = query_with_retry(bs, INDEX_CODE_BS, INDEX_FIELDS,
                                             args.start, args.end, "3", sleep_s=args.sleep)

        qfq = to_numeric(normalize_dates(rows_to_df(qfq_rows, sfields)), NUMERIC_STOCK_COLS)
        raw = to_numeric(normalize_dates(rows_to_df(raw_rows, sfields)), NUMERIC_STOCK_COLS)
        idx = to_numeric(normalize_dates(rows_to_df(idx_rows, ifields)),
                         ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"])

        # 代码体系统一：baostock sh.600519 -> 600519.SH（对齐知识库主键约定）
        for df in (qfq, raw, idx):
            if "code" in df.columns:
                df["code"] = df["code"].map(to_std_code)

        qfq = qfq.sort_values("date").reset_index(drop=True)
        raw = raw.sort_values("date").reset_index(drop=True)
        idx = idx.sort_values("date").reset_index(drop=True)

        for name, df in (("qfq", qfq), ("raw", raw), ("index", idx)):
            if df.empty:
                add_check("rows_%s" % name, "fail", "取回 0 行")
            else:
                add_check("rows_%s" % name, "pass", "%d 行" % len(df), int(len(df)))

        # ---------- 落盘 ----------
        result["artifacts"]["qfq"] = save_csv(qfq, os.path.join(DATA_DIR, "600519_daily_qfq.csv"))
        result["artifacts"]["raw"] = save_csv(raw, os.path.join(DATA_DIR, "600519_daily_raw.csv"))
        result["artifacts"]["index"] = save_csv(idx, os.path.join(DATA_DIR, "sh000001_index_daily.csv"))

        # 合并面板: 原始价 + 前复权价
        q = qfq.rename(columns={"open": "adj_open", "high": "adj_high",
                                "low": "adj_low", "close": "adj_close",
                                "preclose": "adj_preclose"})
        merged = raw[["date", "code", "open", "high", "low", "close", "preclose",
                      "volume", "amount", "turn"]].merge(
            q[["date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_preclose"]],
            on="date", how="inner")
        merged = merged.merge(qfq[["date", "pctChg", "peTTM", "pbMRQ", "psTTM", "isST"]],
                              on="date", how="left")
        merged = merged.sort_values("date").reset_index(drop=True)
        result["artifacts"]["merged"] = save_csv(merged, os.path.join(DATA_DIR, "600519_daily_merged.csv"))

        # 快照元信息（复权口径/区间/行数/字段）
        result["snapshots"]["stock_merged"] = {
            "code": STOCK_CODE_STD,
            "frequency": "daily",
            "adjust": {"open/high/low/close": "不复权(raw, adjustflag=3)",
                       "adj_open/.../adj_close": "前复权(qfq, adjustflag=2)"},
            "range": [str(merged["date"].min().date()), str(merged["date"].max().date())],
            "rows": int(len(merged)),
            "columns": list(merged.columns),
        }
        result["snapshots"]["index"] = {
            "code": INDEX_CODE_STD, "frequency": "daily", "adjust": "不复权",
            "range": [str(idx["date"].min().date()), str(idx["date"].max().date())],
            "rows": int(len(idx)), "columns": list(idx.columns),
        }

        # ---------- 校验 ----------
        for name, df, req in (("qfq", qfq, REQUIRED_STOCK_COLS),
                              ("raw", raw, REQUIRED_STOCK_COLS),
                              ("index", idx, REQUIRED_INDEX_COLS)):
            missing = req - set(df.columns)
            add_check("columns_%s" % name, "fail" if missing else "pass",
                      "缺失列: %s" % sorted(missing) if missing else "列齐全",
                      sorted(missing))
            dup = int(df["date"].duplicated().sum())
            add_check("date_unique_%s" % name, "fail" if dup else "pass",
                      "%d 个重复日期" % dup, dup)
            mono = bool(df["date"].is_monotonic_increasing)
            add_check("date_sorted_%s" % name, "pass" if mono else "fail",
                      "日期升序" if mono else "日期未升序")
            n_nan = int(df[["open", "high", "low", "close"]].isna().sum().sum())
            add_check("ohlc_no_nan_%s" % name, "fail" if n_nan else "pass",
                      "%d 个 OHLC 缺失" % n_nan, n_nan)

        for name, df in (("qfq", qfq), ("raw", raw)):
            if df.empty:
                continue
            bad_high = int((df["high"] < df[["open", "close"]].max(axis=1) - 1e-6).sum())
            bad_low = int((df["low"] > df[["open", "close"]].min(axis=1) + 1e-6).sum())
            add_check("ohlc_logic_%s" % name, "fail" if (bad_high or bad_low) else "pass",
                      "high违反%d条 / low违反%d条" % (bad_high, bad_low),
                      {"high": bad_high, "low": bad_low})

        # 复权与原始日期集合一致性
        set_q, set_r = set(qfq["date"]), set(raw["date"])
        add_check("qfq_raw_date_align", "pass" if set_q == set_r else "fail",
                  "对称差 %d 个日期" % len(set_q ^ set_r), len(set_q ^ set_r))

        # pctChg 与复权收盘价一致性（前复权口径）
        if not qfq.empty and "adj_close" in merged.columns:
            m = merged.dropna(subset=["adj_close", "adj_preclose", "pctChg"]).copy()
            calc = (m["adj_close"] / m["adj_preclose"] - 1.0) * 100.0
            diff = (calc - m["pctChg"]).abs()
            add_check("pctchg_consistency", "pass" if diff.max() < 0.5 else "fail",
                      "|pctChg - 复权收益| 最大 %.4f 个百分点" % diff.max(),
                      round(float(diff.max()), 6))

        # 缺失交易日粗查：相邻交易日间隔 > 15 自然日 视为可疑缺口
        if not merged.empty:
            gaps = merged["date"].diff().dt.days
            big = merged.loc[gaps > 15, "date"].dt.strftime("%Y-%m-%d").tolist()
            add_check("trading_gap", "pass" if not big else "warn",
                      "相邻>15天的缺口起点: %s" % (big[:10] if big else "无"), len(big))

        # 价格区间 sanity
        if not merged.empty:
            add_check("price_range_raw", "pass",
                      "原始收盘 %.2f ~ %.2f" % (merged["close"].min(), merged["close"].max()),
                      [round(float(merged["close"].min()), 2), round(float(merged["close"].max()), 2)])
            add_check("price_range_qfq", "pass",
                      "前复权收盘 %.2f ~ %.2f" % (merged["adj_close"].min(), merged["adj_close"].max()),
                      [round(float(merged["adj_close"].min()), 2), round(float(merged["adj_close"].max()), 2)])

    except Exception as exc:  # noqa: BLE001
        add_check("runtime", "fail", repr(exc))
        result["traceback"] = traceback.format_exc()
        result["error_type"] = "runtime_error"
        _dump(result)
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass

    fails = [c for c in checks if c["status"] == "fail"]
    result["overall"] = "pass" if not fails else "fail"
    if fails:
        result["error_type"] = "metric_fail"
    _dump(result)
    print("[overall] %s" % result["overall"])
    print("[fails] %s" % ([c["name"] for c in fails] or "none"))
    return 0 if not fails else 1


def _dump(result):
    path = os.path.join(VAL_DIR, "validation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[validation] -> %s" % path)


if __name__ == "__main__":
    sys.exit(main())
