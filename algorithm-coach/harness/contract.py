"""接口契约：生成的代码必须实现的四个函数，以及对其调用方式的封装。

契约存在的意义不是形式主义——它是"统一验证机制"得以成立的前提：
只要 fit 只拿得到训练段、predict 只拿得到测试段，"标准化用了全量数据"这类
泄漏就在结构上不可能发生，而不是靠事后检查去猜。

四个函数（详见 harness/README.md）：

    load_data()            -> pd.DataFrame        原始行情
    build_features(df)     -> pd.DataFrame        特征表（含目标列）
    fit(train_df)          -> Any                 只用训练段拟合
    predict(fitted, test_df) -> np.ndarray        只用测试段预测
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_FUNCTIONS = ["load_data", "build_features", "fit", "predict"]

# (名字, 位置参数个数)。check_interface 会核对函数能否接受这么多位置参数。
EXPECTED_ARITY = {
    "load_data": 0,
    "build_features": 1,
    "fit": 1,
    "predict": 2,
}


class ContractError(RuntimeError):
    """契约不满足，或调用契约函数时失败。"""


def load_solution(path: str | Path):
    """把 solution.py 当模块导入。导入期异常会被包装成 ContractError。"""
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"找不到方案文件: {path}")
    spec = importlib.util.spec_from_file_location("_harness_solution", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — 导入期任何异常都要如实上报
        raise ContractError(f"导入 solution.py 失败: {type(exc).__name__}: {exc}") from exc
    return module


def _arity_ok(fn, want: int) -> bool:
    """函数能否接受恰好 want 个位置参数。

    判定：有 *args 则永远可以；否则"必需位置参数 ≤ want ≤ 可接收位置参数总数"。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # 拿不到签名就不拦，交给实际调用去暴露问题
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is p.empty]
    return len(required) <= want <= len(positional)


def check_interface(module) -> dict:
    """检查四个必需函数是否齐备、能否接受约定数量的位置参数。

    只做能力检查（callable + 参数个数够用），不做类型注解强制——
    强行要求注解会把大量能跑的代码挡在外面，收益不抵成本。
    """
    missing: list[str] = []
    bad_arity: list[dict] = []

    for name in REQUIRED_FUNCTIONS:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            missing.append(name)
            continue
        if not _arity_ok(fn, EXPECTED_ARITY[name]):
            bad_arity.append({
                "name": name,
                "expected_args": EXPECTED_ARITY[name],
                "found_signature": _signature_text(fn),
            })

    return {
        "required": list(REQUIRED_FUNCTIONS),
        "missing": missing,
        "bad_arity": bad_arity,
        "ok": not missing and not bad_arity,
    }


def _signature_text(fn) -> str:
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(无法获取签名)"


def call_load_data(module) -> pd.DataFrame:
    df = _call(module, "load_data")
    if not isinstance(df, pd.DataFrame):
        raise ContractError(f"load_data 必须返回 DataFrame，实际返回 {type(df).__name__}")
    if df.empty:
        raise ContractError("load_data 返回了空表")
    return df


def call_build_features(module, df: pd.DataFrame) -> pd.DataFrame:
    out = _call(module, "build_features", df)
    if not isinstance(out, pd.DataFrame):
        raise ContractError(f"build_features 必须返回 DataFrame，实际返回 {type(out).__name__}")
    if out.empty:
        raise ContractError("build_features 返回了空表")
    if len(out) != len(df):
        raise ContractError(
            f"build_features 改变了行数：输入 {len(df)} 行，输出 {len(out)} 行"
            "（特征表必须与输入逐行对齐）"
        )
    return out


def call_fit(module, train_df: pd.DataFrame) -> Any:
    return _call(module, "fit", train_df)


def call_predict(module, fitted: Any, test_df: pd.DataFrame) -> np.ndarray:
    out = _call(module, "predict", fitted, test_df)
    arr = np.asarray(out, dtype=float).ravel()
    if arr.shape[0] != len(test_df):
        raise ContractError(
            f"predict 返回长度 {arr.shape[0]} 与测试段行数 {len(test_df)} 不一致"
        )
    return arr


def _call(module, name: str, *args):
    fn = getattr(module, name, None)
    if fn is None or not callable(fn):
        raise ContractError(f"缺少必需函数: {name}")
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 — 契约函数内部异常要带上函数名上报
        raise ContractError(f"{name}() 执行失败: {type(exc).__name__}: {exc}") from exc
