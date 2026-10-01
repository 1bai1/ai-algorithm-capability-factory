"""接口规范检查：算法提交物能不能被 harness 正常调用。

这一模块是**闸门**：任何一项不过，功能正确性与运行稳定性全部跳过——
连不起来的东西不值得再往下查。

检查项
------
- 提交物齐全   algorithm.py + manifest.json 都在
- manifest 合法 字段类型正确（symbol / label.horizon / label.type / seed / budget）
- 模块可导入   能 import，语法错误与导入错误在此拦下，带行号
- 契约函数齐全 build_features / fit / predict 三个都有
- 函数签名正确 能以契约规定的位置参数个数调用
- 冒烟运行     60 行小数据上真跑一遍 build_features → fit → predict
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .. import contract, data, runner
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "interface"

CHECKS: list[tuple[str, str]] = [
    ("interface.files", "提交物齐全"),
    ("interface.manifest", "manifest 合法"),
    ("interface.import", "模块可导入"),
    ("interface.functions", "契约函数齐全"),
    ("interface.signatures", "函数签名正确"),
    ("interface.smoke", "冒烟运行"),
]

# 冒烟数据长度阶梯：先短后长。有些算法的特征窗口较长（如 60 日均线），
# 太短的切片整表都是预热 NaN，取不到可用行就换长一点的，直到能真正跑一遍。
SMOKE_LADDER = (60, 150, 250)
SMOKE_TEST_ROWS = 15
MIN_USABLE_ROWS = 5


def _result(cid: str, passed: bool, detail: str = "", location=None) -> CheckResult:
    name = dict(CHECKS)[cid]
    return CheckResult(id=cid, module=MODULE, name=name, passed=passed,
                       detail=detail, location=location)


def _all_skipped(reason: str, from_index: int = 0) -> list[CheckResult]:
    """CHECKS 里从 from_index 起（含）的检查项，记一条跳过。"""
    return [CheckResult.skipped(cid, MODULE, name, reason)
            for cid, name in CHECKS[from_index:]]


def skipped_all(reason: str) -> list[CheckResult]:
    return _all_skipped(reason)


def run(v: "Validator") -> list[CheckResult]:
    results: list[CheckResult] = []

    # ---- 1. 提交物齐全
    algo_file = v.algo_dir / f"{v.module_name}.py"
    manifest_file = v.algo_dir / contract.MANIFEST_FILENAME
    missing = [f.name for f in (algo_file, manifest_file) if not f.is_file()]
    if missing:
        results.append(_result("interface.files", False, f"缺少文件: {', '.join(missing)}"))
        results += _all_skipped("提交物不全，跳过", from_index=1)
        return results
    results.append(_result("interface.files", True, "algorithm.py 与 manifest.json 均在"))

    # ---- 2. manifest 合法
    manifest, errors = contract.parse_manifest(manifest_file)
    if errors:
        results.append(_result("interface.manifest", False, "；".join(errors)))
        results += _all_skipped("manifest 不合法，跳过", from_index=2)
        return results
    v.manifest = manifest
    if manifest.task == "classification":
        detail = (f"task=classification, subject={manifest.subject}, "
                  f"标签列={manifest.label.column}、"
                  f"声明 {len(manifest.label.classes or [])} 类，文本列={manifest.text_column}，"
                  f"seed={manifest.seed}, budget={manifest.budget_seconds}s")
    else:
        detail = (f"task=time_series, subject={manifest.subject}, "
                  f"label={manifest.label.type} h={manifest.label.horizon}, "
                  f"seed={manifest.seed}, budget={manifest.budget_seconds}s")
    results.append(_result("interface.manifest", True, detail))

    # ---- 3~5. 导入 + 函数 + 签名（同一个子进程里一次完成）
    payload = runner.run_import(v.algo_dir, v.module_name, budget=120)
    if not payload.ok:
        if payload.status == "timeout":
            results.append(_result("interface.import", False, "导入超时"))
        else:
            tail = _tail(payload.error)
            results.append(_result("interface.import", False, tail, payload.location))
        results += _all_skipped("模块无法导入，跳过", from_index=3)
        return results
    v.import_payload = payload.value
    results.append(_result("interface.import", True, "模块导入成功"))

    missing_funcs = payload.value.get("missing", [])
    if missing_funcs:
        results.append(_result("interface.functions", False,
                               f"缺少函数: {', '.join(missing_funcs)}"))
        results += _all_skipped("契约函数不全，跳过", from_index=4)
        return results
    results.append(_result("interface.functions", True,
                           "build_features / fit / predict 均已定义"))

    bad_arity = [(n, m["msg"]) for n, m in payload.value.get("functions", {}).items()
                 if not m["arity_ok"]]
    if bad_arity:
        detail = "；".join(f"{n}: {msg}" for n, msg in bad_arity)
        results.append(_result("interface.signatures", False, detail))
        results += _all_skipped("函数签名不符合契约，跳过", from_index=5)
        return results
    results.append(_result("interface.signatures", True, "三个函数的参数个数符合契约"))

    # ---- 6. 冒烟运行
    smoke = _smoke(v)
    results.append(smoke)
    return results


def _smoke(v: "Validator") -> CheckResult:
    last_problem = ""
    for rows in SMOKE_LADDER:
        if rows > len(v.raw) and rows != SMOKE_LADDER[0]:
            break
        sub = v.raw.head(min(rows, len(v.raw))).reset_index(drop=True)
        feats = runner.run_features(v.algo_dir, v.module_name, sub, v.smoke_budget)
        if not feats.ok:
            # 报错就是真报错，换长度也没用
            if feats.status == "timeout":
                return _result("interface.smoke", False, f"build_features 在 {len(sub)} 行数据上超时")
            return _result("interface.smoke", False,
                           f"build_features 报错: {_tail(feats.error)}", feats.location)

        frame = feats.value
        if not isinstance(frame, pd.DataFrame):
            return _result("interface.smoke", False,
                           f"build_features 返回 {type(frame).__name__}，契约要求 DataFrame")
        if contract.LABEL_COLUMN not in frame.columns:
            return _result("interface.smoke", False,
                           f"特征表缺少标签列 {contract.LABEL_COLUMN}")

        usable = frame.dropna()
        if len(usable) < MIN_USABLE_ROWS:
            last_problem = (f"{len(sub)} 行切片里只有 {len(usable)} 行可用"
                            f"（特征还在预热期）")
            continue

        test_rows = min(SMOKE_TEST_ROWS, max(1, len(usable) // 3))
        train_df = usable.iloc[:-test_rows]
        test_df = usable.iloc[-test_rows:].drop(columns=[contract.LABEL_COLUMN])
        if len(train_df) < 3:
            last_problem = f"{len(sub)} 行切片只能凑出 {len(train_df)} 行训练数据"
            continue

        chain = runner.run_chain(v.algo_dir, v.module_name, train_df, test_df,
                                 v.manifest.seed, v.smoke_budget)
        if not chain.ok:
            if chain.status == "timeout":
                return _result("interface.smoke", False, "fit/predict 冒烟超时")
            return _result("interface.smoke", False,
                           f"fit/predict 报错: {_tail(chain.error)}", chain.location)
        if len(chain.value) != len(test_df):
            return _result("interface.smoke", False,
                           f"predict 返回 {len(chain.value)} 个值，测试集 {len(test_df)} 行")
        return _result("interface.smoke", True,
                       f"{len(sub)} 行数据上 build_features→fit→predict 全通"
                       f"（训练 {len(train_df)} 行，预测 {len(test_df)} 行）")

    return _result("interface.smoke", False,
                   f"{last_problem}；最长试到 {SMOKE_LADDER[-1]} 行仍无可用特征，"
                   f"预热窗口过长或特征实现有问题")


def _tail(text: str | None, lines: int = 6) -> str:
    if not text:
        return "（无错误信息）"
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " ⏎ ".join(parts[-lines:])
