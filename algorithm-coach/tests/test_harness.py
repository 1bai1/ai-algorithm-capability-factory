#!/usr/bin/env python3
"""harness 自测：用「正确方案 + 若干故意写错的方案」验证每个检查真的会触发。

验证工具自己不验证，就是个永远通过的摆设。这里对每个检查项都配一个
只犯该错误的样本——哪个检查漏报，对应变体就会失败，一眼能看出。

    正确方案            全部检查通过
    全样本标准化         lookahead 报错
    特征 shift(-1)       lookahead 报错
    预测恒为常数         prediction_valid 报错
    预测含 NaN           prediction_valid 报错
    缺 predict 函数      interface 报错

用法：
    python tests/test_harness.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_PY = ROOT / "harness" / "run.py"

# --------------------------------------------------------------------------
# 共用的合成数据与特征（纯 numpy/pandas，不依赖 sklearn，保证任何环境都能跑）
# --------------------------------------------------------------------------

COMMON = '''
import numpy as np
import pandas as pd

N_ROWS = 800
FEATURES = ["mom5", "mom20", "vol20"]


def _raw():
    """确定性合成行情：一个带轻微动量结构的对数价格序列。"""
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0002, 0.015, N_ROWS)
    close = 100.0 * np.exp(np.cumsum(ret))
    vol = rng.integers(1_000_000, 5_000_000, N_ROWS).astype(float)
    return pd.DataFrame(
        {"close": close, "volume": vol},
        index=pd.bdate_range("2020-01-01", periods=N_ROWS),
    )


def load_data():
    return _raw()
'''

# 只使用历史窗口的特征构造：可通过截断一致性
FEATURES_OK = '''
def build_features(df):
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["mom5"] = close.pct_change(5)
    out["mom20"] = close.pct_change(20)
    out["vol20"] = close.pct_change().rolling(20, min_periods=2).std()
    # 目标：未来 5 日对数收益。按定义就是前瞻的，故 harness 会把该列排除在
    # 截断一致性比较之外（见 profiles/regression.json 的 target_column）。
    out["target"] = np.log(close.shift(-5) / close)
    return out
'''

MODEL = '''
def fit(train_df):
    X = np.nan_to_num(train_df[FEATURES].to_numpy(float))
    y = np.nan_to_num(train_df["target"].to_numpy(float))
    A = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict(coef, test_df):
    X = np.nan_to_num(test_df[FEATURES].to_numpy(float))
    A = np.c_[X, np.ones(len(X))]
    return A @ coef
'''

# --- 各变体 -----------------------------------------------------------------

GOOD = COMMON + FEATURES_OK + MODEL

# 在 build_features 里用全样本均值方差标准化 —— 用了未来信息
LEAKY_SCALER = COMMON + '''
def build_features(df):
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    mom5 = close.pct_change(5)
    # 泄漏点：用全量样本的均值与标准差做标准化
    out["mom5"] = (mom5 - mom5.mean()) / mom5.std()
    out["mom20"] = close.pct_change(20)
    out["vol20"] = close.pct_change().rolling(20, min_periods=2).std()
    out["target"] = np.log(close.shift(-5) / close)
    return out
''' + MODEL

# 特征直接挪用下一期收盘价 —— 典型前视
LEAKY_SHIFT = COMMON + '''
def build_features(df):
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["mom5"] = close.pct_change(5)
    out["mom20"] = close.pct_change(20)
    # 泄漏点：把未来一期的价格当成特征
    out["vol20"] = df["close"].shift(-1).pct_change()
    out["target"] = np.log(close.shift(-5) / close)
    return out
''' + MODEL

CONSTANT = COMMON + FEATURES_OK + '''
def fit(train_df):
    return float(np.nanmean(train_df["target"].to_numpy(float)))


def predict(fitted, test_df):
    return np.full(len(test_df), fitted)
'''

NAN_PRED = COMMON + FEATURES_OK + '''
def fit(train_df):
    return None


def predict(fitted, test_df):
    out = np.full(len(test_df), np.nan)
    return out
'''

NO_PREDICT = COMMON + FEATURES_OK + '''
def fit(train_df):
    return None
'''

VARIANTS = [
    ("正确方案", GOOD, {}, None),
    ("全样本标准化", LEAKY_SCALER, {"lookahead": "fail"}, None),
    ("特征 shift(-1)", LEAKY_SHIFT, {"lookahead": "fail"}, None),
    ("预测恒为常数", CONSTANT, {"prediction_valid": "fail"}, None),
    ("预测含 NaN", NAN_PRED, {"prediction_valid": "fail"}, None),
    ("缺 predict 函数", NO_PREDICT, {"interface": "fail"}, None),
]


def run_variant(name: str, source: str, expect: dict[str, str]) -> tuple[bool, str]:
    tmp = Path(tempfile.mkdtemp(prefix="harness_test_"))
    try:
        generated = tmp / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "solution.py").write_text(source, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(RUN_PY), "--task", str(tmp)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300,
        )
        report_path = tmp / "validation" / "report.json"
        if not report_path.is_file():
            return False, f"未生成 report.json；stderr={proc.stderr[-300:]}"

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        got = {c["name"]: c["result"] for c in payload["checks"]}

        problems = []
        for check_name, want in expect.items():
            actual = got.get(check_name, "未注册")
            if actual != want:
                problems.append(f"{check_name} 期望 {want}，实际 {actual}")

        if not expect and payload["status"] != "passed":
            bad = [f"{c['name']}={c['result']}" for c in payload["checks"] if c["result"] != "pass"]
            problems.append(f"期望全部通过，实际 {bad}；errors={payload['errors']}")

        if expect and payload["status"] == "passed":
            problems.append("期望失败但整体判为通过")

        detail = "；".join(problems) if problems else "符合预期"
        return not problems, detail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print(f"harness 自测　共 {len(VARIANTS)} 个变体\n")
    n_pass = 0
    for name, source, expect, _ in VARIANTS:
        ok, detail = run_variant(name, source, expect)
        n_pass += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:16s}  {detail}")

    print()
    if n_pass == len(VARIANTS):
        print(f"全部通过：{n_pass}/{len(VARIANTS)}")
        return 0
    print(f"有 {len(VARIANTS) - n_pass} 个变体未达预期")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
