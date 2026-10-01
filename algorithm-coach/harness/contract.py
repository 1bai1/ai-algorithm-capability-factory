"""算法契约：manifest 解析、算法模块加载、函数签名校验。

生成的算法目录必须包含两个文件::

    <algo_dir>/algorithm.py    入口模块，定义 build_features / fit / predict
    <algo_dir>/manifest.json   声明标的、标签定义、随机种子、时间预算与成本口径
                               （可选字段 cost_per_side：单边手续费+滑点，默认 0.0005）

三个函数
--------
``build_features(df) -> DataFrame``
    输入原始日线（列含 date/open/high/low/close/volume 等）。
    输出**同长度、同顺序**的特征表：可以新增特征列，但不得增删行、
    不得改动 date 列。必须包含标签列（默认列名 ``label``）。
    预热期的 NaN 行允许保留——harness 自己负责在构造训练集时丢弃。

``fit(train_df) -> model``
    只用传入的 train_df（含特征列与 label 列）训练，返回任意模型对象。

``predict(model, test_df) -> ndarray``
    长度必须等于 ``len(test_df)``，与 test_df 行序对齐。
    test_df 已剥掉 label 列——预测时拿不到答案。

标签定义
--------
manifest 里声明 ``label.horizon`` 与 ``label.type``，harness 用收盘价**独立重算**
逐行核对（这是「标签口径对账」检查）::

    simple:  close.shift(-h) / close - 1
    log:     log(close.shift(-h) / close)

随机性约定
----------
harness 在调用 fit / predict 前会设置 ``random.seed(seed)`` 与 ``np.random.seed(seed)``。
算法内部若引入独立的随机源（如 ``np.random.default_rng()`` 不带种子），
同种子两次运行结果会不一致，被「确定性」检查判为不可复现。
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

ALGO_FILENAME = "algorithm.py"
MANIFEST_FILENAME = "manifest.json"
LABEL_COLUMN = "label"
RETURN_TYPES = ("simple", "log")

# 函数名 -> harness 调用时传入的位置参数个数
REQUIRED_FUNCS: dict[str, int] = {
    "build_features": 1,
    "fit": 1,
    "predict": 2,
}

DEFAULT_SEED = 42
DEFAULT_BUDGET_SECONDS = 300
DEFAULT_COST_PER_SIDE = 0.0005      # 单边手续费 + 滑点，进出各收一次


@dataclass
class LabelSpec:
    """标签定义：horizon 与收益类型。"""

    horizon: int
    type: str = "simple"


@dataclass
class Manifest:
    """manifest.json 的解析结果。"""

    symbol: str
    label: LabelSpec
    seed: int = DEFAULT_SEED
    budget_seconds: int = DEFAULT_BUDGET_SECONDS
    cost_per_side: float = DEFAULT_COST_PER_SIDE

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "label": {"horizon": self.label.horizon, "type": self.label.type},
            "seed": self.seed,
            "budget_seconds": self.budget_seconds,
            "cost_per_side": self.cost_per_side,
        }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_manifest(path: str | Path) -> tuple[Manifest | None, list[str]]:
    """解析 manifest.json。返回 (manifest, 错误列表)；错误非空时 manifest 为 None。"""
    path = Path(path)
    if not path.is_file():
        return None, [f"缺少 {MANIFEST_FILENAME}"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{MANIFEST_FILENAME} 不是合法 JSON: {exc}"]
    except OSError as exc:
        return None, [f"{MANIFEST_FILENAME} 读取失败: {exc}"]

    if not isinstance(raw, dict):
        return None, [f"{MANIFEST_FILENAME} 顶层必须是 JSON 对象"]

    errors: list[str] = []

    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        errors.append("symbol 必须是非空字符串")

    # label 支持嵌套写法 {"label": {"horizon":5,"type":"simple"}}
    # 也兼容扁平写法 {"horizon":5, "return_type":"simple"}
    label_raw = raw.get("label")
    if label_raw is None:
        label_raw = raw
    elif not isinstance(label_raw, dict):
        errors.append("label 必须是 JSON 对象（含 horizon / type）")
        label_raw = {}

    horizon = label_raw.get("horizon")
    if not _is_int(horizon) or horizon < 1:
        errors.append("label.horizon 必须是 >=1 的整数")
        horizon = None

    ret_type = label_raw.get("type", raw.get("return_type", "simple"))
    if ret_type not in RETURN_TYPES:
        errors.append(f"label.type 必须是 {RETURN_TYPES} 之一，当前为 {ret_type!r}")

    seed = raw.get("seed", DEFAULT_SEED)
    if not _is_int(seed):
        errors.append("seed 必须是整数")
        seed = DEFAULT_SEED

    budget = raw.get("budget_seconds", DEFAULT_BUDGET_SECONDS)
    if not _is_int(budget) or budget < 1:
        errors.append("budget_seconds 必须是 >=1 的整数")
        budget = DEFAULT_BUDGET_SECONDS

    cost = raw.get("cost_per_side", DEFAULT_COST_PER_SIDE)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
        errors.append("cost_per_side 必须是 >=0 的数字（单边费率，含手续费与滑点）")
        cost = DEFAULT_COST_PER_SIDE

    if errors:
        return None, errors
    return Manifest(symbol=symbol, label=LabelSpec(horizon=horizon, type=ret_type),
                    seed=seed, budget_seconds=budget,
                    cost_per_side=float(cost)), []


def load_algorithm(algo_dir: str | Path, module_name: str = "algorithm") -> ModuleType:
    """把 <algo_dir>/<module_name>.py 作为模块加载（子进程内调用）。

    算法目录会被加入 sys.path，因此算法可以拆分出同目录的辅助模块
    （例如 models/ 子包）。加载失败会抛出异常，由调用方记录 traceback。
    """
    algo_dir = Path(algo_dir).resolve()
    path = algo_dir / f"{module_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path.name}")

    if str(algo_dir) not in sys.path:
        sys.path.insert(0, str(algo_dir))

    spec = importlib.util.spec_from_file_location(f"_algo_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 建立导入规范")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def collect_functions(module: ModuleType) -> tuple[dict[str, Callable], list[str]]:
    """取出契约要求的函数。返回 (函数表, 缺失列表)。"""
    found: dict[str, Callable] = {}
    missing: list[str] = []
    for name in REQUIRED_FUNCS:
        func = getattr(module, name, None)
        if callable(func):
            found[name] = func
        else:
            missing.append(name)
    return found, missing


def check_arity(func: Callable, want: int) -> tuple[bool, str]:
    """检查函数能否以 want 个位置参数调用。

    可变参数（*args / **kwargs）一律视为可调用；否则要求
    必填位置参数个数 <= want <= 位置参数总个数。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 极少数内建函数
        return False, f"无法读取签名: {exc}"

    params = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return True, "接受可变参数"

    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in positional if p.default is p.empty]
    if len(required) <= want <= len(positional):
        return True, f"可接收 {want} 个位置参数"
    return False, (f"契约要求能以 {want} 个位置参数调用，"
                   f"实际需要 {len(required)}~{len(positional)} 个")


def has_algorithm_file(algo_dir: str | Path, module_name: str = "algorithm") -> bool:
    return (Path(algo_dir) / f"{module_name}.py").is_file()
