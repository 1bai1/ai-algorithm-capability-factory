"""算法执行器：每个动作都在独立子进程里跑，带超时强杀。

为什么要独立进程：

1. 超时可控——卡死的算法能被 terminate 掉，不会把 harness 一起拖死；
2. 状态隔离——每次运行都是全新解释器，上一次的全局状态（随机种子、
   缓存的拟合结果）不会污染下一次，确定性检查才有意义。

子进程通过 multiprocessing.Queue 回传结果；大数据（特征表/预测值）
在父进程用 get(timeout) 先收，再 join，避免管道缓冲区死锁。
"""
from __future__ import annotations

import multiprocessing as mp
import random
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import contract

_LOCATION_RE = re.compile(r'File "([^"]+)", line (\d+)')


@dataclass
class RunResult:
    """一次子进程执行的结果。status: ok | timeout | error。"""

    status: str
    value: Any = None            # ok 时为 DataFrame（特征）或 ndarray（预测）
    elapsed: float = 0.0
    error: str | None = None
    location: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _extract_location(text: str, algo_filename: str) -> dict | None:
    """从 traceback 文本里取最后一处算法文件的行号。"""
    matches = _LOCATION_RE.findall(text or "")
    if not matches:
        return None
    for file, line in reversed(matches):
        if Path(file).name == algo_filename:
            return {"file": Path(file).name, "line": int(line)}
    file, line = matches[-1]
    return {"file": Path(file).name, "line": int(line)}


def _features_worker(queue, algo_dir: str, module_name: str, df) -> None:
    started = time.perf_counter()
    try:
        module = contract.load_algorithm(algo_dir, module_name)
        features = getattr(module, "build_features")(df)
        queue.put({"status": "ok", "value": features,
                   "elapsed": time.perf_counter() - started})
    except BaseException:
        queue.put({"status": "error", "error": traceback.format_exc(),
                   "elapsed": time.perf_counter() - started})


def _import_worker(queue, algo_dir: str, module_name: str) -> None:
    """只做导入 + 契约函数盘点，不执行任何算法逻辑。"""
    started = time.perf_counter()
    try:
        module = contract.load_algorithm(algo_dir, module_name)
        found, missing = contract.collect_functions(module)
        functions = {}
        for name, func in found.items():
            ok, msg = contract.check_arity(func, contract.REQUIRED_FUNCS[name])
            functions[name] = {"arity_ok": ok, "msg": msg}
        queue.put({"status": "ok", "elapsed": time.perf_counter() - started,
                   "value": {"functions": functions, "missing": missing}})
    except BaseException:
        queue.put({"status": "error", "error": traceback.format_exc(),
                   "elapsed": time.perf_counter() - started})


def _chain_worker(queue, algo_dir: str, module_name: str, train_df, test_df, seed: int) -> None:
    started = time.perf_counter()
    try:
        random.seed(seed)
        np.random.seed(seed)
        module = contract.load_algorithm(algo_dir, module_name)
        model = getattr(module, "fit")(train_df)
        preds = getattr(module, "predict")(model, test_df)
        array = np.asarray(preds)
        if array.dtype.kind in "OUS":        # 字符串/对象：分类任务的类别标签
            array = array.astype(str).ravel()
        else:
            array = np.asarray(array, dtype=float).ravel()
        queue.put({"status": "ok", "value": array,
                   "elapsed": time.perf_counter() - started})
    except BaseException:
        queue.put({"status": "error", "error": traceback.format_exc(),
                   "elapsed": time.perf_counter() - started})


def _spawn(target, args: tuple, budget: float, algo_filename: str) -> RunResult:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(queue,) + args, daemon=True)
    started = time.perf_counter()
    proc.start()

    payload = None
    try:
        payload = queue.get(timeout=budget + 0.5)   # 多给 0.5s 让回传数据过管道
    except Exception:
        payload = None
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
        try:
            queue.close()
        except Exception:
            pass

    wall = time.perf_counter() - started
    if payload is None:
        return RunResult(status="timeout", elapsed=wall,
                         error=f"超过时间预算 {budget:.0f}s，子进程被终止")

    if payload["status"] == "error":
        text = payload.get("error", "")
        return RunResult(status="error", elapsed=wall, error=text,
                         location=_extract_location(text, algo_filename))

    elapsed = float(payload.get("elapsed", wall))
    if elapsed > budget:      # 跑完了但超预算，同样算超时
        return RunResult(status="timeout", elapsed=elapsed,
                         error=f"用时 {elapsed:.1f}s，超过时间预算 {budget:.0f}s")
    return RunResult(status="ok", value=payload["value"], elapsed=elapsed)


def run_import(algo_dir: str | Path, module_name: str, budget: float) -> RunResult:
    """在子进程里加载算法模块，返回契约函数盘点结果。"""
    return _spawn(_import_worker, (str(algo_dir), module_name),
                  budget, f"{module_name}.py")


def run_features(algo_dir: str | Path, module_name: str, df, budget: float) -> RunResult:
    """在子进程里执行 build_features。"""
    return _spawn(_features_worker, (str(algo_dir), module_name, df),
                  budget, f"{module_name}.py")


def run_chain(algo_dir: str | Path, module_name: str, train_df, test_df,
              seed: int, budget: float) -> RunResult:
    """在子进程里执行 fit + predict，返回预测数组。"""
    return _spawn(_chain_worker, (str(algo_dir), module_name, train_df, test_df, seed),
                  budget, f"{module_name}.py")
