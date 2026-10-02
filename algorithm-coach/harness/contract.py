"""算法契约：manifest 解析、算法模块加载、函数签名校验。

生成的算法目录必须包含两个文件::

    <algo_dir>/algorithm.py    入口模块，定义 build_features / fit / predict
    <algo_dir>/manifest.json   声明对象、标签列与类别、随机种子、时间预算

交付形态（两层，都要齐）
------------------------
``generated/`` 目录是**交给用户的那一份**，必须自包含、可独立运行::

    algorithm.py    算法核心：下面那三个函数（harness 验证时调它）
    manifest.json   元数据：任务类型、标签口径、类别、随机种子
    run.py          运行入口：读数据 → 训练 → 预测 → 写结果（用户直接跑这个）
    README.md       使用说明。第一节必须是「## 它是什么」——用一两句人话讲清
                    做什么、怎么工作，术语放其后；然后才是依赖、可直接复制运行的
                    命令（--data 指向真实存在的文件）、输入输出格式与口径

``validation/`` 目录放 harness 的验证报告（给人看质量，不是运行的必需品）。
用户**不需要** harness——run.py 只依赖 pandas / numpy / scikit-learn。

harness 只调用 ``algorithm.py`` 里的三个函数，**从不调用 run.py**；
但「接口规范」模块会检查 run.py 存在、``--help`` 能起来、喂小数据能真产出预测。

三个函数
--------
``build_features(df) -> DataFrame``
    输入原始文本表（至少含 manifest 声明的文本列与标签列）。
    输出**同长度、同顺序**的特征表：可以新增特征列，但不得增删行、
    不得改动文本列与标签列。

``fit(train_df) -> model``
    只用传入的 train_df（含特征列与标签列）训练，返回任意模型对象。

``predict(model, test_df) -> ndarray``
    长度必须等于 ``len(test_df)``，与 test_df 行序对齐。
    test_df 已剥掉标签列——预测时拿不到答案。

标签定义
--------
分类任务的标签就是 manifest 里 ``label.column`` 那一列，取值集合由
``label.classes`` 声明（可选，声明后 harness 会核对预测是否落在集合内）。
切分由 harness 控制：分层随机，类别比例与随机种子原样写进报告。

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

# 当前只支持文本分类一种任务类型。
# 要加新类型（比如换场景），在这里加取值，再补一条数据加载、切分与检查路径；
# 其余部分（契约三函数、报告骨架、四个模块）是任务无关的。
TASKS = ("classification",)
DEFAULT_TEXT_COLUMN = "text"

# 函数名 -> harness 调用时传入的位置参数个数
REQUIRED_FUNCS: dict[str, int] = {
    "build_features": 1,
    "fit": 1,
    "predict": 2,
}

DEFAULT_SEED = 42
DEFAULT_BUDGET_SECONDS = 300


@dataclass
class LabelSpec:
    """标签定义：列名 + 可选的类别集合。"""

    column: str = LABEL_COLUMN
    classes: list[str] | None = None


@dataclass
class Manifest:
    """manifest.json 的解析结果。"""

    task: str
    subject: str                    # 对象标识：数据集名
    label: LabelSpec
    text_column: str = DEFAULT_TEXT_COLUMN
    seed: int = DEFAULT_SEED
    budget_seconds: int = DEFAULT_BUDGET_SECONDS

    def to_dict(self) -> dict:
        label: dict = {"column": self.label.column}
        if self.label.classes:
            label["classes"] = self.label.classes
        return {
            "task": self.task,
            "subject": self.subject,
            "label": label,
            "text_column": self.text_column,
            "seed": self.seed,
            "budget_seconds": self.budget_seconds,
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

    task = raw.get("task", "classification")
    if task not in TASKS:
        errors.append(f"task 必须是 {TASKS} 之一，当前为 {task!r}")
        task = "classification"

    subject = raw.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        errors.append("subject 必须是非空字符串")

    label_raw = raw.get("label")
    if label_raw is None:
        label_raw = {}
    elif not isinstance(label_raw, dict):
        errors.append("label 必须是 JSON 对象")
        label_raw = {}

    column = label_raw.get("column", LABEL_COLUMN)
    if not isinstance(column, str) or not column.strip():
        errors.append("label.column 必须是非空字符串")
        column = LABEL_COLUMN
    classes = label_raw.get("classes")
    if classes is not None:
        ok = (isinstance(classes, list) and classes and
              all(isinstance(c, (str, int, float)) and not isinstance(c, bool)
                  for c in classes))
        if not ok:
            errors.append("label.classes 必须是类别取值的非空数组")
            classes = None
        else:
            classes = [str(c) for c in classes]
    label = LabelSpec(column=column, classes=classes)

    text_column = raw.get("text_column", DEFAULT_TEXT_COLUMN)
    if not isinstance(text_column, str) or not text_column.strip():
        errors.append("text_column 必须是非空字符串（文本列名）")
        text_column = DEFAULT_TEXT_COLUMN

    seed = raw.get("seed", DEFAULT_SEED)
    if not _is_int(seed):
        errors.append("seed 必须是整数")
        seed = DEFAULT_SEED

    budget = raw.get("budget_seconds", DEFAULT_BUDGET_SECONDS)
    if not _is_int(budget) or budget < 1:
        errors.append("budget_seconds 必须是 >=1 的整数")
        budget = DEFAULT_BUDGET_SECONDS

    if errors:
        return None, errors
    return Manifest(task=task, subject=subject, label=label,
                    text_column=text_column, seed=seed,
                    budget_seconds=budget), []


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
