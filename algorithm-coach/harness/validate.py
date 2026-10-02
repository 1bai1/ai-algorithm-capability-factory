"""验证编排：按顺序跑四个模块，产出结构化报告。

流程::

    接口规范（闸门）
      └─ 不过 → 其余全部跳过
    加载全量数据 → build_features（子进程，带超时）
      └─ 分层随机切分训练/测试集（harness 控制）→ fit + predict（子进程，带超时）
    功能正确性 → 指标表现 → 运行稳定性
"""
from __future__ import annotations

import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import __version__, contract, data, runner
from .checks import correctness, interface, performance, stability
from .report import Report
from .split import Split, split_classification

ALL_MODULES = ("interface", "correctness", "performance", "stability")


class Validator:
    """一次完整验证的上下文与编排器。

    检查模块通过读取本对象的属性取用中间产物（features / split / chain 等）。
    """

    def __init__(self, algo_dir: str | Path, data_path: str | Path,
                 test_size: float = 0.2,
                 budget: float | None = None, seed: int | None = None,
                 module_name: str = "algorithm",
                 modules: tuple[str, ...] | None = None):
        self.algo_dir = Path(algo_dir)
        self.module_name = module_name
        self.data_path = str(data_path)
        self.test_size = test_size

        # 先窥一眼 manifest：拿它的标签列名去加载数据（读不到就按默认列名，
        # 接口规范那一关会把缺 manifest 拦下）。
        peek, _ = contract.parse_manifest(self.algo_dir / contract.MANIFEST_FILENAME)
        self.task = peek.task if peek else "classification"
        label_column = peek.label.column if peek else contract.LABEL_COLUMN
        self.raw, self.data_info = data.load_text_csv(self.data_path, label_column)

        self.manifest: contract.Manifest | None = None
        self.import_payload: dict | None = None
        self.features: pd.DataFrame | None = None
        self.features_result: runner.RunResult | None = None
        self.split: Split | None = None
        self.split_error: str | None = None
        self.chain: runner.RunResult | None = None
        self.insample: runner.RunResult | None = None
        self.metrics_payload: dict = {}

        self.modules = tuple(modules) if modules else ALL_MODULES
        self._budget = budget
        self._seed = seed
        self.report = Report()
        self._started = time.perf_counter()

    # ------------------------------------------------------------- 运行参数
    @property
    def budget(self) -> float:
        if self._budget is not None:
            return float(self._budget)
        if self.manifest is not None:
            return float(self.manifest.budget_seconds)
        return float(contract.DEFAULT_BUDGET_SECONDS)

    @property
    def smoke_budget(self) -> float:
        return max(60.0, self.budget)

    @property
    def seed(self) -> int:
        if self._seed is not None:
            return int(self._seed)
        return self.manifest.seed if self.manifest else contract.DEFAULT_SEED

    # ---------------------------------------------------------------- 编排
    def run(self) -> Report:
        results = []
        if "interface" in self.modules:
            results += interface.run(self)
        else:
            results += interface.skipped_all("未启用接口规范模块")

        gate_ok = all(r.passed is not False for r in results)
        needs_chain = any(m in self.modules for m in ("correctness", "stability"))
        if gate_ok and needs_chain:
            self._prepare()

        gate_reason = "接口规范未通过，跳过"
        if "correctness" in self.modules:
            results += (correctness.run(self) if gate_ok
                        else correctness.skipped_all(gate_reason))
        else:
            results += correctness.skipped_all("未启用功能正确性模块")

        if "performance" in self.modules:
            results += (performance.run(self) if gate_ok
                        else performance.skipped_all(gate_reason))
            if self.metrics_payload:
                self.report.metrics = self.metrics_payload
        else:
            results += performance.skipped_all("未启用指标表现模块")

        if "stability" in self.modules:
            results += (stability.run(self) if gate_ok
                        else stability.skipped_all(gate_reason))
        else:
            results += stability.skipped_all("未启用运行稳定性模块")

        self.report.checks = results
        self.report.meta = self._meta()
        return self.report

    def _prepare(self) -> None:
        """执行 build_features 与切分 + 主流程 fit/predict。"""
        res = runner.run_features(self.algo_dir, self.module_name, self.raw, self.budget)
        self.features_result = res
        if not res.ok or not isinstance(res.value, pd.DataFrame):
            return
        self.features = res.value
        split, error = split_classification(
            self.features, self.manifest.label.column,
            test_size=self.test_size, seed=self.seed)
        self.split, self.split_error = split, error
        if split is None:
            return
        self.chain = runner.run_chain(self.algo_dir, self.module_name, split.train_df,
                                      split.test_features, self.seed, self.budget)
        # 样本内预测（同一模型、训练集上），只用于「训练/测试差距」披露
        self.insample = runner.run_chain(
            self.algo_dir, self.module_name, split.train_df,
            split.train_df.drop(columns=[contract.LABEL_COLUMN]), self.seed, self.budget)

    # ---------------------------------------------------------------- 产出
    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir)
        json_path = self.report.write_json(out / "report.json")
        md_path = self.report.write_markdown(out / "report.md")
        return json_path, md_path

    def _meta(self) -> dict:
        counters = {
            "passed": sum(1 for c in self.report.checks if c.passed is True),
            "failed": sum(1 for c in self.report.checks if c.passed is False),
            "skipped": sum(1 for c in self.report.checks if c.passed is None),
        }
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "harness_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "algo_dir": str(self.algo_dir),
            "data_path": self.data_path,
            "data_rows": self.data_info.get("rows"),
            "task": self.task,
            "seed": self.seed,
            "budget_seconds": self.budget,
            "modules": list(self.modules),
            "elapsed_seconds": round(time.perf_counter() - self._started, 2),
            "counters": counters,
        }
        notes = {k: v for k, v in self.data_info.items()
                 if k.startswith("dropped_")}
        if notes:
            meta["data_notes"] = notes
        if self.manifest is not None:
            meta["manifest"] = self.manifest.to_dict()
        if self.split is not None:
            meta.update({
                "train_rows": int(len(self.split.train_df)),
                "test_rows": int(len(self.split.test_features)),
                "classes": self.data_info.get("classes"),
                "class_counts": self.data_info.get("class_counts"),
                "majority_share": self.data_info.get("majority_share"),
                "mean_text_len": self.data_info.get("mean_text_len"),
                "split": "stratified_random",
            })
        elif self.split_error:
            meta["split_error"] = self.split_error
        return meta


def format_console(report: Report) -> str:
    """控制台摘要。"""
    from .report import MODULE_TITLES

    icons = {True: "✓", False: "✗", None: "—"}
    lines = ["", "=" * 68]
    for module, title in MODULE_TITLES.items():
        items = report.by_module(module)
        if not items:
            continue
        lines.append(f"[{title}]")
        for c in items:
            detail = (c.detail or "").replace("\n", " ")
            if len(detail) > 96:
                detail = detail[:95] + "…"
            lines.append(f"  {icons[c.passed]} {c.name}：{detail}")
        lines.append("")
    m = report.meta.get("counters", {})
    verdict = "通过" if report.passed_all else "未通过"
    lines.append(f"结论：{verdict}（通过 {m.get('passed', 0)} / 未通过 {m.get('failed', 0)}"
                 f" / 跳过 {m.get('skipped', 0)}），耗时 {report.meta.get('elapsed_seconds', '-')}s")
    lines.append("=" * 68)
    return "\n".join(lines)
