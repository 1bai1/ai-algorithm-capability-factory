"""行情数据加载与标签参考值计算。

harness 负责读数据（算法不碰网络/文件），把干净的日线表喂给算法。
列名兼容英文与常见中文写法，大小写不敏感。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import LabelSpec

# 中文/常见别名 -> 标准列名
COLUMN_ALIASES = {
    "日期": "date", "时间": "date", "交易日期": "date", "date": "date",
    "开盘": "open", "开盘价": "open", "open": "open",
    "最高": "high", "最高价": "high", "high": "high",
    "最低": "low", "最低价": "low", "low": "low",
    "收盘": "close", "收盘价": "close", "close": "close",
    "成交量": "volume", "volume": "volume", "vol": "volume",
    "成交额": "amount", "amount": "amount",
    "换手率": "turn", "turn": "turn", "turnover": "turn",
    "涨跌幅": "pct_chg", "pct_chg": "pct_chg", "pctchg": "pct_chg",
}

REQUIRED_COLUMNS = ("date", "close")
MIN_ROWS = 120


class DataError(RuntimeError):
    """数据无法用于验证（列缺失、行数太少、日期不可解析等）。"""


def load_stock_csv(path: str | Path, min_rows: int = MIN_ROWS) -> tuple[pd.DataFrame, dict]:
    """读取日线 CSV，返回 (标准化的 DataFrame, 加载说明)。

    标准化：列名转小写标准名、date 解析为 datetime、数值列强制转换、
    按日期升序、去重复日期、丢弃 close 缺失的行。
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"数据文件不存在: {path}")

    raw = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            raw = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise DataError(f"无法解码数据文件: {path}")

    renamed = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        renamed[col] = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(str(col).strip(), key)
    raw = raw.rename(columns=renamed)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"数据缺少必需列 {missing}；实际列: {list(raw.columns)}")

    info: dict = {"path": str(path), "source_rows": int(len(raw))}

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    bad_dates = int(raw["date"].isna().sum())
    if bad_dates:
        info["dropped_bad_dates"] = bad_dates
        raw = raw.dropna(subset=["date"])

    for col in ("open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values("date").reset_index(drop=True)

    dup = int(raw["date"].duplicated().sum())
    if dup:
        info["dropped_duplicate_dates"] = dup
        raw = raw.drop_duplicates(subset="date", keep="last")

    nan_close = int(raw["close"].isna().sum())
    if nan_close:
        info["dropped_nan_close"] = nan_close
        raw = raw.dropna(subset=["close"])

    raw = raw.reset_index(drop=True)

    if len(raw) < min_rows:
        raise DataError(f"有效行数 {len(raw)} 少于下限 {min_rows}")

    info["rows"] = int(len(raw))
    info["start"] = str(raw["date"].iloc[0].date())
    info["end"] = str(raw["date"].iloc[-1].date())
    info["columns"] = list(raw.columns)
    return raw, info


# 文本分类数据集：常见列名别名 -> 标准列名
TEXT_COLUMN_ALIASES = {
    "text": "text", "文本": "text", "内容": "text", "正文": "text",
    "sentence": "text", "review": "text", "评论": "text", "标题": "text",
    "title": "text", "document": "text", "doc": "text",
    "label": "label", "标签": "label", "类别": "label", "分类": "label",
    "class": "label", "category": "label", "target": "label", "y": "label",
}

MIN_TEXT_ROWS = 200


def load_text_csv(path: str | Path, label_column: str = "label",
                  min_rows: int = MIN_TEXT_ROWS) -> tuple[pd.DataFrame, dict]:
    """读取文本分类数据集（必须有 text 与 label 两列），返回 (DataFrame, 加载说明)。

    标准化：列名转小写标准名、文本转字符串去空白、丢掉空文本与缺标签的行。
    info 里带上类数与类别分布——评测口径卡要求"数据集标识与类数"必须披露。
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"数据文件不存在: {path}")

    raw = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            raw = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise DataError(f"无法解码数据文件: {path}")

    renamed = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        renamed[col] = (TEXT_COLUMN_ALIASES.get(key)
                        or TEXT_COLUMN_ALIASES.get(str(col).strip(), key))
    raw = raw.rename(columns=renamed)

    missing = [c for c in ("text", label_column) if c not in raw.columns]
    if missing:
        raise DataError(f"数据缺少必需列 {missing}；实际列: {list(raw.columns)}")

    info: dict = {"path": str(path), "source_rows": int(len(raw))}

    raw["text"] = raw["text"].astype(str).str.strip()
    empty_text = int((raw["text"] == "").sum() + raw["text"].isna().sum())
    if empty_text:
        info["dropped_empty_text"] = empty_text
        raw = raw[raw["text"].notna() & (raw["text"] != "")]

    raw[label_column] = raw[label_column].astype(str).str.strip()
    bad_label = int(raw[label_column].isin(["", "nan", "None"]).sum())
    if bad_label:
        info["dropped_missing_label"] = bad_label
        raw = raw[~raw[label_column].isin(["", "nan", "None"])]

    raw = raw.reset_index(drop=True)
    if len(raw) < min_rows:
        raise DataError(f"有效行数 {len(raw)} 少于下限 {min_rows}")

    counts = raw[label_column].value_counts()
    if len(counts) < 2:
        raise DataError(f"标签只有 {len(counts)} 个取值，分类任务至少需要 2 类")

    info.update({
        "rows": int(len(raw)),
        "label_column": label_column,
        "classes": [str(c) for c in counts.index.tolist()],
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "majority_share": float(counts.iloc[0] / len(raw)),
        "mean_text_len": float(raw["text"].str.len().mean()),
        "columns": list(raw.columns),
    })
    return raw, info


def reference_label(df: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    """按声明口径独立重算标签，作为对账基准。"""
    close = df["close"].astype(float)
    h = spec.horizon
    if spec.type == "log":
        values = np.log(close.shift(-h) / close)
    else:
        values = close.shift(-h) / close - 1.0
    return pd.Series(values, index=df.index, name="label_ref")


def date_cutoff(df: pd.DataFrame, test_size: float = 0.2) -> pd.Timestamp:
    """按样本外比例取切分日：样本外约占尾部 test_size。"""
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 之间")
    idx = int(len(df) * (1 - test_size))
    idx = max(1, min(idx, len(df) - 1))
    return pd.Timestamp(df["date"].iloc[idx])


def corrupt(df: pd.DataFrame, nan_ratio: float = 0.01, drop_rows: int = 3,
            seed: int = 0) -> pd.DataFrame:
    """制造坏数据：随机注入缺失值 + 删除中间若干行（模拟停牌/缺日）。"""
    rng = np.random.default_rng(seed)
    bad = df.copy()
    numeric = [c for c in ("open", "high", "low", "close", "volume") if c in bad.columns]
    for c in numeric:
        bad[c] = bad[c].astype(float)   # 允许注入 NaN（整型列转浮点）
    n_cells = max(1, int(len(bad) * len(numeric) * nan_ratio))
    rows = rng.integers(len(bad) // 4, max(len(bad) // 4 + 1, len(bad) * 3 // 4), size=n_cells)
    cols = rng.choice(numeric, size=n_cells)
    for r, c in zip(rows, cols):
        bad.iloc[r, bad.columns.get_loc(c)] = np.nan
    if drop_rows > 0 and len(bad) > drop_rows + 100:
        start = len(bad) // 2
        bad = bad.drop(index=bad.index[start:start + drop_rows]).reset_index(drop=True)
    return bad


def corrupt_text(df: pd.DataFrame, blank_ratio: float = 0.01, drop_rows: int = 3,
                 seed: int = 0, text_column: str = "text") -> pd.DataFrame:
    """文本任务的坏数据：把一部分文本置空、删掉几行、再塞几条异常短文本。

    对应行情任务的 corrupt()——稳定性检查里的「坏数据不崩」在文本场景要有对应的脏法。
    """
    rng = np.random.default_rng(seed)
    bad = df.copy()
    if text_column in bad.columns:
        n_blank = max(1, int(len(bad) * blank_ratio))
        idx = rng.choice(bad.index.to_numpy(), size=n_blank, replace=False)
        bad.loc[idx, text_column] = ""                      # 空文本
        n_junk = max(1, n_blank // 3)
        jidx = rng.choice(bad.index.to_numpy(), size=n_junk, replace=False)
        bad.loc[jidx, text_column] = "???"                  # 无信息短文本
    if drop_rows > 0 and len(bad) > drop_rows + 100:
        start = len(bad) // 2
        bad = bad.drop(index=bad.index[start:start + drop_rows]).reset_index(drop=True)
    return bad
