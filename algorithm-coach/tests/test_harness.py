"""harness 自测：每个检查都必须能被一个"故意埋 bug 的算法"触发。

做法：合成一段有已知结构的行情数据（未来 5 日收益可由过去动量预测，
单日方向近似噪声），然后为每个失败模式写一个算法变体，断言对应检查
未通过、而其它检查不受牵连。

运行::

    python -m unittest tests.test_harness -v
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.contract import parse_manifest
from harness.report import Report
from harness.validate import Validator

DEFAULT_MANIFEST = {
    "symbol": "601318",
    "label": {"horizon": 5, "type": "simple"},
    "seed": 42,
    "budget_seconds": 30,
}

CUTOFF = "2019-10-01"


def synthetic_ohlcv(n: int = 650, seed: int = 7):
    """合成日线：慢变状态 m 驱动未来 5 日收益；单日方向被噪声主导。

    这样「明日=今日」这类朴素基线只能拿到 ~0.5，而带动量特征的模型能明显更高。
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n)
    for i in range(1, n):
        state[i] = 0.95 * state[i - 1] + 0.05 * rng.normal()   # 慢变的动量状态
    # 单日看噪声占七成（「明日=今日」赢不了多少），但 5 日窗口上动量是相干叠加、
    # 噪声是 √5 放大，所以 5 日收益可被过去动量预测 —— 模型应当明显赢过基线
    ret = 0.018 * state + 0.004 * rng.normal(size=n)
    close = 50.0 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date": pd.bdate_range("2018-01-02", periods=n),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


# ---------------------------------------------------------------- 正常的算法
GOOD = '''
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HORIZON = 5
FEATURES = ["r1", "r5", "r20", "vol20", "ma20_dev", "amp"]


class Model:
    def __init__(self, medians, scaler, ridge):
        self.medians = medians
        self.scaler = scaler
        self.ridge = ridge


def build_features(df):
    out = df.copy()
    close = out["close"].astype(float)
    log_close = np.log(close)
    out["r1"] = log_close.diff(1)
    out["r5"] = log_close.diff(5)
    out["r20"] = log_close.diff(20)
    out["vol20"] = log_close.diff().rolling(20).std()
    out["ma20_dev"] = log_close - np.log(close.rolling(20).mean())
    out["amp"] = (out["high"].astype(float) - out["low"].astype(float)) / close
    out["label"] = close.shift(-HORIZON) / close - 1.0
    return out


def fit(train_df):
    X = train_df[FEATURES].astype(float)
    medians = X.median()
    X = X.fillna(medians)
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=10.0).fit(scaler.transform(X), train_df["label"].astype(float))
    return Model(medians, scaler, ridge)


def predict(model, test_df):
    X = test_df[FEATURES].astype(float).fillna(model.medians)
    return model.ridge.predict(model.scaler.transform(X))
'''

# ------------------------------------------------------------ 各失败模式变体
# 前视泄漏：用全样本均值/标准差做标准化
LEAKY = GOOD.replace(
    '''    out["amp"] = (out["high"].astype(float) - out["low"].astype(float)) / close''',
    '''    out["amp"] = (out["high"].astype(float) - out["low"].astype(float)) / close
    out["r1_z"] = (out["r1"] - out["r1"].mean()) / out["r1"].std()   # 全样本统计量'''
).replace('FEATURES = ["r1", "r5", "r20", "vol20", "ma20_dev", "amp"]',
          'FEATURES = ["r1", "r5", "r20", "vol20", "ma20_dev", "amp", "r1_z"]')

# 标签口径不符：声明 horizon=5，实际算的是当日收益
LABEL_MISMATCH = GOOD.replace(
    'out["label"] = close.shift(-HORIZON) / close - 1.0',
    'out["label"] = close.pct_change(1)')

# 偷偷删行：build_features 里 dropna 后重置索引
ROW_DROP = GOOD.replace(
    "    return out\n\n\ndef fit",
    "    return out.dropna().reset_index(drop=True)\n\n\ndef fit")

# 输出恒为常数
CONST_PRED = GOOD.replace(
    "    return model.ridge.predict(model.scaler.transform(X))",
    "    return np.zeros(len(test_df))")

# 不可复现：predict 里引入未受种子控制的随机源
NONDET = GOOD.replace(
    "    return model.ridge.predict(model.scaler.transform(X))",
    "    preds = model.ridge.predict(model.scaler.transform(X))\n"
    "    return preds + np.random.default_rng().normal(0, 1e-6, len(preds))")

# 预热窗口长：需要 60 日窗口，60 行切片整表 NaN（考验冒烟阶梯）
WARMUP_60 = GOOD.replace(
    '    out["amp"] = (out["high"].astype(float) - out["low"].astype(float)) / close',
    '    out["amp"] = (out["high"].astype(float) - out["low"].astype(float)) / close\n'
    '    out["ma60_dev"] = log_close - np.log(close.rolling(60).mean())'
).replace('FEATURES = ["r1", "r5", "r20", "vol20", "ma20_dev", "amp"]',
          'FEATURES = ["r1", "r5", "r20", "vol20", "ma20_dev", "amp", "ma60_dev"]')

# 慢：fit 里睡 5 秒
SLOW = GOOD.replace("def fit(train_df):", "def fit(train_df):\n    import time; time.sleep(5)")

# 小样本直接崩：要求至少 200 行
CRASH_SMALL = GOOD.replace(
    "def build_features(df):",
    'def build_features(df):\n'
    '    if len(df) < 200:\n'
    '        raise ValueError("需要至少 200 行历史，当前 %d 行" % len(df))')

# 坏数据直接崩：不接受缺失值
CRASH_ON_NAN = GOOD.replace(
    "def build_features(df):",
    'def build_features(df):\n'
    '    if df["close"].isna().any():\n'
    '        raise ValueError("输入含缺失收盘价")')


class HarnessTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="harness_test_"))
        cls.data_path = cls.tmp / "sample_daily.csv"
        synthetic_ohlcv().to_csv(cls.data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------- 工具
    def make_algo_dir(self, name: str, source: str, manifest: dict | None = None) -> Path:
        algo_dir = self.tmp / name
        algo_dir.mkdir(parents=True, exist_ok=True)
        (algo_dir / "algorithm.py").write_text(source, encoding="utf-8")
        (algo_dir / "manifest.json").write_text(
            json.dumps(manifest or DEFAULT_MANIFEST, ensure_ascii=False), encoding="utf-8")
        return algo_dir

    def validate(self, algo_dir: Path, **kwargs) -> tuple[Validator, Report]:
        validator = Validator(algo_dir=algo_dir, data_path=self.data_path,
                              cutoff=CUTOFF, **kwargs)
        return validator, validator.run()

    @staticmethod
    def check(report: Report, check_id: str):
        for item in report.checks:
            if item.id == check_id:
                return item
        raise AssertionError(f"报告里没有 {check_id}")


class TestNormalAlgorithm(HarnessTestCase):
    def test_good_algorithm_passes_everything(self):
        algo_dir = self.make_algo_dir("good", GOOD)
        validator, report = self.validate(algo_dir)
        failed = [c.id for c in report.failed]
        self.assertEqual(failed, [], f"正常算法不应有未通过项: "
                                     f"{[(c.id, c.detail) for c in report.failed]}")
        self.assertTrue(report.passed_all)
        # 报告落盘且可解析
        json_path, md_path = validator.save(self.tmp / "good" / "validation")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["conclusion"]["passed"])
        self.assertIn("功能正确性", md_path.read_text(encoding="utf-8"))

    def test_labels_and_baseline_reported(self):
        algo_dir = self.make_algo_dir("good2", GOOD)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        baseline = self.check(report, "correctness.naive_baseline")
        self.assertIsNotNone(baseline.passed)
        self.assertIn("方向准确率", baseline.detail)


class TestBugDetection(HarnessTestCase):
    def test_leakage_triggers_truncation_check(self):
        algo_dir = self.make_algo_dir("leaky", LEAKY)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        trunc = self.check(report, "correctness.truncation")
        self.assertIs(trunc.passed, False, trunc.detail)
        self.assertEqual(trunc.kb_card, "全样本统计量泄漏")
        self.assertIs(self.check(report, "correctness.row_conservation").passed, True)
        self.assertIs(self.check(report, "correctness.label_reconcile").passed, True)

    def test_label_mismatch_triggers_reconcile(self):
        algo_dir = self.make_algo_dir("label_mismatch", LABEL_MISMATCH)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.label_reconcile")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("不一致", item.detail)

    def test_row_drop_triggers_conservation(self):
        algo_dir = self.make_algo_dir("row_drop", ROW_DROP)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.row_conservation")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("行", item.detail)

    def test_constant_prediction_detected(self):
        algo_dir = self.make_algo_dir("const_pred", CONST_PRED)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.prediction_validity")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("常数", item.detail)
        self.assertIsNone(self.check(report, "correctness.naive_baseline").passed)

    def test_nondeterminism_detected(self):
        algo_dir = self.make_algo_dir("nondet", NONDET)
        _, report = self.validate(algo_dir, modules=("interface", "stability"))
        item = self.check(report, "stability.determinism")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("不一致", item.detail)

    def test_timeout_detected(self):
        algo_dir = self.make_algo_dir("slow", SLOW)
        _, report = self.validate(algo_dir, modules=("interface", "stability"), budget=2)
        self.assertIs(self.check(report, "interface.smoke").passed, True)
        item = self.check(report, "stability.timeout")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("预算", item.detail)

    def test_small_sample_crash_gates_at_interface(self):
        algo_dir = self.make_algo_dir("crash_small", CRASH_SMALL)
        _, report = self.validate(algo_dir)
        item = self.check(report, "interface.smoke")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("200", item.detail)
        # 闸门拦住后，后续模块全部跳过
        self.assertIsNone(self.check(report, "correctness.truncation").passed)
        self.assertIsNone(self.check(report, "stability.determinism").passed)

    def test_smoke_ladder_handles_long_warmup(self):
        algo_dir = self.make_algo_dir("warmup60", WARMUP_60)
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.smoke")
        self.assertIs(item.passed, True, item.detail)
        self.assertIn("150 行", item.detail)

    def test_bad_data_crash_detected(self):
        algo_dir = self.make_algo_dir("crash_on_nan", CRASH_ON_NAN)
        _, report = self.validate(algo_dir, modules=("interface", "correctness", "stability"))
        item = self.check(report, "stability.bad_data")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("缺失", item.detail)
        self.assertIs(self.check(report, "stability.small_sample").passed, True)


class TestInterfaceGate(HarnessTestCase):
    def test_missing_files(self):
        algo_dir = self.tmp / "empty"
        algo_dir.mkdir(exist_ok=True)
        _, report = self.validate(algo_dir)
        item = self.check(report, "interface.files")
        self.assertIs(item.passed, False)
        self.assertIn("manifest.json", item.detail)

    def test_syntax_error_reported_with_location(self):
        algo_dir = self.make_algo_dir("syntax_error", "def build_features(df)\n    return df\n")
        _, report = self.validate(algo_dir)
        item = self.check(report, "interface.import")
        self.assertIs(item.passed, False, item.detail)
        self.assertIsNotNone(item.location)
        self.assertEqual(item.location["file"], "algorithm.py")

    def test_wrong_arity(self):
        source = GOOD.replace("def predict(model, test_df):",
                              "def predict(model, test_df, extra):")
        algo_dir = self.make_algo_dir("bad_arity", source)
        _, report = self.validate(algo_dir)
        item = self.check(report, "interface.signatures")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("predict", item.detail)

    def test_missing_label_column(self):
        source = GOOD.replace('    out["label"] = close.shift(-HORIZON) / close - 1.0\n', "")
        algo_dir = self.make_algo_dir("no_label", source)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        self.assertIs(self.check(report, "interface.smoke").passed, False)
        self.assertIsNone(self.check(report, "correctness.label_reconcile").passed)


class TestManifestParsing(unittest.TestCase):
    def test_flat_form_accepted(self):
        tmp = Path(tempfile.mkdtemp(prefix="manifest_test_"))
        try:
            path = tmp / "manifest.json"
            path.write_text(json.dumps({"symbol": "601318", "horizon": 3,
                                        "return_type": "log"}), encoding="utf-8")
            manifest, errors = parse_manifest(path)
            self.assertEqual(errors, [])
            self.assertEqual(manifest.label.horizon, 3)
            self.assertEqual(manifest.label.type, "log")

            path.write_text(json.dumps({"symbol": "", "label": {"horizon": 0}}),
                            encoding="utf-8")
            manifest, errors = parse_manifest(path)
            self.assertIsNone(manifest)
            self.assertEqual(len(errors), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
