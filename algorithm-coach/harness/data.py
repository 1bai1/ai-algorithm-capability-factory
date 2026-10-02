"""文本分类数据加载。

harness 负责读数据（算法不碰网络/文件），把干净的文本表喂给算法。
列名兼容英文与常见中文写法，大小写不敏感。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    """数据无法用于验证（列缺失、行数太少、类别不足等）。"""


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
    """读取文本分类数据集（必须有文本列与标签列），返回 (DataFrame, 加载说明)。

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


def corrupt_text(df: pd.DataFrame, blank_ratio: float = 0.01, drop_rows: int = 3,
                 seed: int = 0, text_column: str = "text") -> pd.DataFrame:
    """制造坏数据：把一部分文本置空、塞几条无信息短文本、再删掉几行。

    用于稳定性检查里的「坏数据不崩」——真实数据里空文本、乱码、缺行都会出现，
    算法得撑得住。
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
