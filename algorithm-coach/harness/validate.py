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
from .checks import ALL_MODULES, MODULES
from .report import Report
from .split import Split, split_classification


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

        # manifest 由 Validator 自己持有——它是整个验证的上下文，不该依赖
        # 「接口规范模块跑过了」这个副作用（只跑某一类检查时会拿不到）。
        # 读不到就按默认值走，接口规范那一关会把缺 manifest 拦下并报错。
        self.manifest, self.manifest_errors = contract.parse_manifest(
            self.algo_dir / contract.MANIFEST_FILENAME)
        self.task = self.manifest.task if self.manifest else "classification"
        label_column = (self.manifest.label.column if self.manifest
                        else contract.LABEL_COLUMN)
        self.raw, self.data_info = data.load_text_csv(self.data_path, label_column)

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
        """按注册表的 ORDER 依次跑各模块。

        闸门模块（GATE=True，当前是接口规范）不过，后面的模块全部记跳过；
        需要主流程产物的模块（NEEDS_CHAIN）在轮到自己时才触发 _prepare，
        这样单独跑某一类检查也不会白跑一遍 fit/predict。
        """
        results = []
        gate_ok = True
        prepared = False
        for name in ALL_MODULES:
            module = MODULES[name]
            if name not in self.modules:
                results += module.skipped_all(f"未启用「{module.TITLE}」模块")
                continue
            if not gate_ok:
                results += module.skipped_all("接口规范未通过，跳过")
                continue
            if module.NEEDS_CHAIN and not prepared:
                self._prepare()
                prepared = True
            results += module.run(self)
            if module.GATE and any(r.passed is False for r in results
                                   if r.module == name):
                gate_ok = False

        self.report.checks = results
        if self.metrics_payload:
            self.report.metrics = self.metrics_payload
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
    icons = {True: "✓", False: "✗", None: "—"}
    lines = ["", "=" * 68]
    from .checks import MODULE_TITLES

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
