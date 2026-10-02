"""文本分类 harness 的自测：一个好算法 + 若干埋雷变体。

两条主线：
- **判据能不能抓到该抓的**——隐性泄漏（向量化器在全量语料上 fit）、恒常数预测、
  未见类别、删行、不可复现、超时、小样本崩、坏数据崩；
- **接口闸门与交付层**——缺文件、语法错带行号、签名不符、缺标签列、README 的
  示例命令能不能跑、文档有没有先讲清"它是什么"。

运行::

    python -m unittest tests.test_harness_classification -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness import metrics
from harness.report import Report
from harness.validate import Validator

RUN_PY = '''
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import algorithm

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="predictions.csv")
ap.add_argument("--train-ratio", type=float, default=0.8)
args = ap.parse_args()

df = pd.read_csv(args.data, encoding="utf-8-sig")
feats = algorithm.build_features(df)
cols = [c for c in feats.columns if c not in ("label", "date", "text", "symbol")]
usable = feats["label"].notna().to_numpy()
if cols:
    usable &= ~feats[cols].isna().any(axis=1).to_numpy()
pos = np.flatnonzero(usable)
n = max(1, int(len(pos) * args.train_ratio))
train_df, test_df = feats.iloc[pos[:n]], feats.iloc[pos[n:]]
model = algorithm.fit(train_df)
preds = np.asarray(algorithm.predict(model, test_df.drop(columns=["label"])))
pd.DataFrame({"prediction": preds}).to_csv(args.out, index=False, encoding="utf-8-sig")
print("ok", len(preds))
'''

CLS_MANIFEST = {
    "task": "classification",
    "subject": "synthetic_text",
    "label": {"column": "label", "classes": ["A", "B", "C"]},
    "text_column": "text",
    "seed": 42,
    "budget_seconds": 30,
}

CLS_GOOD = '''
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TEXT, LABEL, NORM = "text", "label", "text_norm"


def _norm(s):
    t = str(s).lower()
    return re.sub(r"\\s+", " ", re.sub(r"[^0-9a-z]+", " ", t)).strip()


def build_features(df):
    out = df.copy()
    out[NORM] = out[TEXT].map(_norm)      # 逐行规范化，不学任何东西
    return out


class Model:
    def __init__(self, pipe):
        self.pipe = pipe


def fit(train_df):
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2),
                         LogisticRegression(max_iter=500, C=4.0))
    pipe.fit(train_df[NORM].astype(str), train_df[LABEL].astype(str))
    return Model(pipe)


def predict(model, test_df):
    return np.asarray(model.pipe.predict(test_df[NORM].astype(str)))
'''

# 泄漏版：build_features 里就地把向量化器在传进来的数据上 fit（全量的词表/IDF 混入）
CLS_LEAKY = '''
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TEXT, LABEL, NORM = "text", "label", "text_norm"


def _norm(s):
    t = str(s).lower()
    return re.sub(r"\\s+", " ", re.sub(r"[^0-9a-z]+", " ", t)).strip()


def build_features(df):
    out = df.copy()
    out[NORM] = out[TEXT].map(_norm)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2)   # ← 在传入的数据上 fit
    dense = vec.fit_transform(out[NORM].astype(str)).toarray()
    feat = pd.DataFrame(dense, columns=["f%d" % i for i in range(dense.shape[1])])
    return pd.concat([out.reset_index(drop=True), feat], axis=1)


class Model:
    def __init__(self, clf, cols):
        self.clf, self.cols = clf, cols


def _cols(df):
    return [c for c in df.columns if c.startswith("f")]


def fit(train_df):
    cols = _cols(train_df)
    clf = LogisticRegression(max_iter=500, C=4.0)
    clf.fit(train_df[cols].to_numpy(), train_df[LABEL].astype(str))
    return Model(clf, cols)


def predict(model, test_df):
    return np.asarray(model.clf.predict(test_df[model.cols].to_numpy()))
'''

CLS_CONST = CLS_GOOD.replace(
    '    return np.asarray(model.pipe.predict(test_df[NORM].astype(str)))',
    '    return np.full(len(test_df), "A")')

CLS_UNSEEN = CLS_GOOD.replace(
    '    return np.asarray(model.pipe.predict(test_df[NORM].astype(str)))',
    '    preds = model.pipe.predict(test_df[NORM].astype(str)).astype(str)\n'
    '    preds[:5] = "Zzz"\n'
    '    return np.asarray(preds)')


# 偷偷删行：build_features 里丢掉首行再重置索引
# （文本表没有预热期 NaN，dropna 删不掉东西——得真删行才测得到这条检查）
CLS_ROW_DROP = CLS_GOOD.replace(
    "\n    return out\n\n\nclass Model:",
    "\n    return out.iloc[1:].reset_index(drop=True)\n\n\nclass Model:")

# 不可复现：predict 里引入未受种子控制的随机源（default_rng 不理会 np.random.seed）
CLS_NONDET = CLS_GOOD.replace(
    '    return np.asarray(model.pipe.predict(test_df[NORM].astype(str)))',
    '    preds = model.pipe.predict(test_df[NORM].astype(str)).astype(str)\n'
    '    rng = np.random.default_rng()\n'
    '    flip = rng.random(len(preds)) < 0.3\n'
    '    preds[flip] = rng.choice(["A", "B", "C"], size=int(flip.sum()))\n'
    '    return np.asarray(preds)')

# 慢：fit 里睡 5 秒
CLS_SLOW = CLS_GOOD.replace("def fit(train_df):",
                            "def fit(train_df):\n    import time; time.sleep(5)")

# 小样本直接崩：要求至少 300 行（冒烟阶梯最长只试到 250 行）
CLS_CRASH_SMALL = CLS_GOOD.replace(
    "def build_features(df):",
    'def build_features(df):\n'
    '    if len(df) < 300:\n'
    '        raise ValueError("需要至少 300 行文本，当前 %d 行" % len(df))')

# 坏数据直接崩：不接受空文本（corrupt_text 会注入空文本）
CLS_CRASH_ON_EMPTY = CLS_GOOD.replace(
    "def build_features(df):",
    'def build_features(df):\n'
    '    if (df["text"].astype(str).str.strip() == "").any():\n'
    '        raise ValueError("输入含空文本")')

# 丢了标签列：特征表里没有 label
CLS_NO_LABEL = CLS_GOOD.replace(
    "\n    return out\n\n\nclass Model:",
    "\n    return out.drop(columns=[LABEL])\n\n\nclass Model:")

# 参数个数不符：fit 要两个参数
CLS_WRONG_ARITY = CLS_GOOD.replace("def fit(train_df):", "def fit(train_df, extra):")

# 语法错：def 后面少了冒号
CLS_SYNTAX_ERROR = CLS_GOOD.replace("def predict(model, test_df):",
                                    "def predict(model, test_df)")

def synthetic_text_dataset(n_per_class: int = 200, seed: int = 11) -> pd.DataFrame:
    """三类可分的合成文本：每类有自己的词表，另加共享噪声词。"""
    rng = np.random.default_rng(seed)
    vocab = {"A": ["alpha", "bravo", "charlie", "delta"],
             "B": ["echo", "foxtrot", "golf", "hotel"],
             "C": ["india", "juliet", "kilo", "lima"]}
    noise = ["the", "a", "of", "and", "to", "in", "for", "on"]
    rows = []
    for cls, words in vocab.items():
        for _ in range(n_per_class):
            toks = list(rng.choice(words, size=6)) + list(rng.choice(noise, size=3))
            rng.shuffle(toks)
            rows.append({"text": " ".join(toks), "label": cls})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


class TestClassificationMetrics(unittest.TestCase):
    """分类指标纯函数：手算数值直接断言。"""

    def test_core_metrics(self):
        truth = np.array(["A", "A", "B", "B", "C"])
        pred = np.array(["A", "B", "B", "B", "C"])
        self.assertAlmostEqual(metrics.accuracy(truth, pred), 0.8)
        self.assertAlmostEqual(metrics.majority_share(truth), 0.4)
        self.assertAlmostEqual(metrics.class_distribution(["A", "A", "B"])["A"], 2 / 3)
        labels, matrix = metrics.confusion_counts(truth, pred)
        self.assertEqual(labels, ["A", "B", "C"])
        self.assertEqual(matrix[0], [1, 1, 0])
        rows = metrics.per_class_scores(truth, pred)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["recall"], 0.5)
        self.assertGreater(metrics.macro_f1(truth, pred), 0.0)


README_TEMPLATE = (
    "# 测试算法\n\n"
    "## 它是什么\n\n"
    "一个自测用的占位算法：读入数据、训练、预测并写出结果文件。这句话要够长，才能通过 harness 的「它是什么」检查。\n\n"
    "```bash\n"
    "python run.py --data {rel} --out 预测.csv\n"
    "```\n")


class TextHarnessTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="harness_cls_"))
        cls.data_path = cls.tmp / "text_sample.csv"
        synthetic_text_dataset().to_csv(cls.data_path, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make_algo_dir(self, name: str, source: str, manifest: dict | None = None) -> Path:
        algo_dir = self.tmp / name
        algo_dir.mkdir(parents=True, exist_ok=True)
        (algo_dir / "algorithm.py").write_text(source, encoding="utf-8")
        (algo_dir / "manifest.json").write_text(
            json.dumps(manifest or CLS_MANIFEST, ensure_ascii=False), encoding="utf-8")
        (algo_dir / "run.py").write_text(RUN_PY, encoding="utf-8")
        rel = os.path.relpath(self.data_path, algo_dir).replace(os.sep, "/")
        (algo_dir / "README.md").write_text(
            README_TEMPLATE.format(rel=rel), encoding="utf-8")
        return algo_dir

    def validate(self, algo_dir: Path, **kwargs) -> tuple[Validator, Report]:
        validator = Validator(algo_dir=algo_dir, data_path=self.data_path, **kwargs)
        return validator, validator.run()

    @staticmethod
    def check(report: Report, check_id: str):
        for item in report.checks:
            if item.id == check_id:
                return item
        raise AssertionError(f"报告里没有 {check_id}")


class TestClassificationPipeline(TextHarnessTestCase):
    def test_good_classifier_passes_all(self):
        algo_dir = self.make_algo_dir("cls_good", CLS_GOOD)
        validator, report = self.validate(algo_dir)
        failed = [(c.id, c.detail) for c in report.failed]
        self.assertEqual(failed, [], f"正常分类算法不应有未通过项: {failed}")
        json_path, md_path = validator.save(self.tmp / "cls_good" / "validation")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metrics"]["setup"]["task"], "classification")
        self.assertIn("macro_f1", payload["metrics"]["prediction"])
        self.assertIn("混淆矩阵", md_path.read_text(encoding="utf-8"))

    def test_leaky_vectorizer_caught(self):
        """向量化器在全量语料上 fit —— 必须被「拟合范围一致性」抓住。"""
        algo_dir = self.make_algo_dir("cls_leaky", CLS_LEAKY)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.feature_fit_scope")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("训练集以外的信息", item.detail)
        self.assertEqual(item.kb_card, "文本分类评估与交叉验证")

    def test_constant_prediction_caught(self):
        algo_dir = self.make_algo_dir("cls_const", CLS_CONST)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.prediction_validity")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("恒为同一类", item.detail)

    def test_unseen_label_caught(self):
        algo_dir = self.make_algo_dir("cls_unseen", CLS_UNSEEN)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.prediction_validity")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("没见过", item.detail)

    def test_missing_run_py_fails_deliverables(self):
        """交付物必须能独立运行：缺 run.py 要判负。

        生成的算法不只是给 harness 调用的零件，也是交给用户的东西；
        用户不该为了跑它去拉本项目的源码。
        """
        algo_dir = self.make_algo_dir("cls_no_run", CLS_GOOD)
        (algo_dir / "run.py").unlink()
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.deliverables")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("run.py", item.detail)

    def test_readme_without_runnable_command_fails(self):
        """README 的示例命令必须能直接跑：全是占位符要判负。"""
        algo_dir = self.make_algo_dir("cls_bad_readme", CLS_GOOD)
        (algo_dir / "README.md").write_text(
            README_TEMPLATE.replace("{rel}", "数据.csv"), encoding="utf-8")
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.deliverables")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("README", item.detail)

    def test_docs_must_open_with_what_it_is(self):
        """交付文档第一节必须先用一两句人话讲清"这是什么算法"。"""
        algo_dir = self.make_algo_dir("cls_no_whatis", CLS_GOOD)
        txt = (algo_dir / "README.md").read_text(encoding="utf-8")
        (algo_dir / "README.md").write_text(
            txt.replace("## 它是什么", "## 简介"), encoding="utf-8")
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.deliverables")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("它是什么", item.detail)

    def test_bad_manifest_rejected(self):
        algo_dir = self.make_algo_dir(
            "cls_bad", CLS_GOOD,
            manifest={"task": "classification", "subject": "x",
                      "label": {}, "text_column": ""})
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.manifest")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("text_column", item.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBugDetection(TextHarnessTestCase):
    """埋雷变体必须被对应的检查抓住——一条雷对一个检查 id。"""

    def test_row_drop_triggers_conservation(self):
        algo_dir = self.make_algo_dir("cls_row_drop", CLS_ROW_DROP)
        _, report = self.validate(algo_dir, modules=("interface", "correctness"))
        item = self.check(report, "correctness.row_conservation")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("行", item.detail)

    def test_nondeterminism_detected(self):
        algo_dir = self.make_algo_dir("cls_nondet", CLS_NONDET)
        _, report = self.validate(algo_dir, modules=("interface", "stability"))
        item = self.check(report, "stability.determinism")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("不一致", item.detail)

    def test_timeout_detected(self):
        algo_dir = self.make_algo_dir("cls_slow", CLS_SLOW)
        _, report = self.validate(algo_dir, modules=("interface", "stability"),
                                  budget=2)
        self.assertIs(self.check(report, "interface.smoke").passed, True)
        item = self.check(report, "stability.timeout")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("预算", item.detail)

    def test_small_sample_crash_gates_at_interface(self):
        algo_dir = self.make_algo_dir("cls_crash_small", CLS_CRASH_SMALL)
        _, report = self.validate(algo_dir)
        item = self.check(report, "interface.smoke")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("300", item.detail)
        # 闸门拦住后，后续模块全部跳过
        self.assertIsNone(self.check(report, "correctness.row_conservation").passed)
        self.assertIsNone(self.check(report, "stability.determinism").passed)

    def test_bad_data_crash_detected(self):
        algo_dir = self.make_algo_dir("cls_crash_empty", CLS_CRASH_ON_EMPTY)
        _, report = self.validate(algo_dir,
                                  modules=("interface", "correctness", "stability"))
        item = self.check(report, "stability.bad_data")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("空文本", item.detail)
        self.assertIs(self.check(report, "stability.small_sample").passed, True)


class TestInterfaceGate(TextHarnessTestCase):
    """闸门行为：拦不住的算法不该进后面的模块。"""

    def test_missing_files(self):
        algo_dir = self.tmp / "cls_empty"
        algo_dir.mkdir(exist_ok=True)
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.files")
        self.assertIs(item.passed, False, item.detail)
        self.assertIsNone(self.check(report, "interface.manifest").passed)

    def test_syntax_error_reported_with_location(self):
        algo_dir = self.make_algo_dir("cls_syntax", CLS_SYNTAX_ERROR)
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.import")
        self.assertIs(item.passed, False, item.detail)
        self.assertIsNotNone(item.location, "语法错误应带出错位置")
        self.assertIn("algorithm.py", str(item.location))

    def test_wrong_arity(self):
        algo_dir = self.make_algo_dir("cls_arity", CLS_WRONG_ARITY)
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.signatures")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("fit", item.detail)

    def test_missing_label_column(self):
        algo_dir = self.make_algo_dir("cls_no_label", CLS_NO_LABEL)
        _, report = self.validate(algo_dir, modules=("interface",))
        item = self.check(report, "interface.smoke")
        self.assertIs(item.passed, False, item.detail)
        self.assertIn("标签列", item.detail)

    def test_metrics_are_reported(self):
        algo_dir = self.make_algo_dir("cls_metrics", CLS_GOOD)
        _, report = self.validate(algo_dir)
        m = report.metrics
        self.assertIn("prediction", m)
        self.assertGreater(m["prediction"]["accuracy"], 0.8)
        self.assertEqual(len(m["prediction"]["per_class"]), 3)
        self.assertEqual(len(m["prediction"]["confusion"]), 3)
        md = report.to_markdown()
        self.assertIn("精度账", md)
        self.assertIn("混淆矩阵", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
