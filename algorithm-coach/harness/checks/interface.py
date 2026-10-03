"""接口规范检查：算法提交物能不能被 harness 正常调用。

本模块是**闸门**（`GATE = True`，`ORDER = 0`）——任何一项不过，后面的功能正确性、
指标表现、运行稳定性**全部记"跳过"**。理由：连不起来的东西不值得再往下查。
（"跳过"不等于"通过"：报告里那一栏是 `—`，不是 `✓`；只有真的跑过才可能通过。）

它检查什么（7 项，按 run() 的执行顺序）
--------------------------------------
1. `提交物齐全`   algorithm.py + manifest.json 在不在
2. `manifest 合法` 字段类型对不对（subject / label.column / label.classes / seed / budget）
3. `模块可导入`   能 import 吗——**语法错误与导入期报错在这里被拦下，并带出错行号**
4. `契约函数齐全` build_features / fit / predict 三个都有吗
5. `函数签名正确` 能以契约规定的位置参数个数调用吗（build_features 收 1 个、fit 收 1 个、predict 收 2 个）
6. `冒烟运行`     拿小切片真跑一遍 build_features → fit → predict，看会不会当场崩
7. `交付物完整性` run.py 能不能独立跑、README 的示例命令指向的文件在不在、
                 文档开头有没有用一两句人话讲清"这是什么算法"

为什么第 7 项也算"接口"
----------------------
交付物不只是给 harness 调用的零件，也是**要交给用户的东西**。用户不该为了跑它去拉本项目
的源码——所以 run.py / README.md 属于契约的一部分，缺了就是没交付完。
harness 在这里会**真起一个子进程跑 run.py**，而不是只检查文件存在。

本模块**不需要主流程产物**（`NEEDS_CHAIN = False`）：它自己造小切片来跑，
所以即使后面的模块被跳过，它也能独立完成判断。

判据都来自哪里
--------------
这一模块的 7 项是**工程规范**（能不能连起来、交付得完不完整），不是某条算法能力，
所以 `CHECKS` 里第三项（依据知识卡）统一是 `None`。判据本身写在 AGENTS.md 的
「算法契约」与「交付层」两张表里。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import pandas as pd

from .. import contract, data, runner
from ..report import CheckResult

if TYPE_CHECKING:  # pragma: no cover
    from ..validate import Validator

MODULE = "interface"
TITLE = "接口规范"
ORDER = 0          # 闸门排最前
GATE = True        # 这一关不过，后面的模块全部跳过
NEEDS_CHAIN = False

CHECKS: list[tuple[str, str, str | None]] = [
    # 第三项是"依据的知识卡"。接口规范这几项是**工程规范**（提交物齐不齐、能不能导入、
    # 签名对不对、交付物能不能独立跑），不是某条算法能力，所以没有卡可引——写 None。
    # 形状与其他三个模块保持一致（见 checks/__init__.py 的模块契约）。
    ("interface.files", "提交物齐全", None),
    ("interface.manifest", "manifest 合法", None),
    ("interface.import", "模块可导入", None),
    ("interface.functions", "契约函数齐全", None),
    ("interface.signatures", "函数签名正确", None),
    ("interface.smoke", "冒烟运行", None),
    ("interface.deliverables", "交付物完整性（可独立运行）", None),
]

CHECK_INFO = {cid: (name, card) for cid, name, card in CHECKS}

# 冒烟数据长度阶梯：先短后长。有些算法对样本量有隐含下限（比如向量化器要求
# 每类至少若干样本），太短的切片取不到可用行就换长一点的，直到能真正跑一遍。
SMOKE_LADDER = (60, 150, 250)
SMOKE_TEST_ROWS = 15
MIN_USABLE_ROWS = 5


def _result(cid: str, passed: bool, detail: str = "", location=None) -> CheckResult:
    """造一条"已判定"的检查结果。名字与依据知识卡从 CHECK_INFO 查，不用手写。"""
    name, card = CHECK_INFO[cid]
    return CheckResult(id=cid, module=MODULE, name=name, passed=passed,
                       detail=detail, location=location, kb_card=card)


def _all_skipped(reason: str, from_index: int = 0) -> list[CheckResult]:
    """CHECKS 里从 from_index 起（含）的检查项，记一条跳过。

    用在"闸门早早挂掉"的场合：比如提交物不全，那后面的 manifest 检查、导入检查
    都没意义，但报告里仍要出现这些条目（记 `—` 跳过 + 一句原因），
    否则读者会以为"检查项少了"。
    """
    return [CheckResult.skipped(cid, MODULE, name, reason)
            for cid, name, _ in CHECKS[from_index:]]


def skipped_all(reason: str) -> list[CheckResult]:
    """整个模块跳过（被 --modules 排除，或闸门在前面就挂了）。"""
    return _all_skipped(reason)


def run(v: "Validator") -> list[CheckResult]:
    """跑完 7 项接口检查，返回结果列表。

    执行方式是**阶梯式的**：每一步过不了就立刻把剩下的记"跳过"并返回——
    因为后一项都依赖前一项（文件都没有，谈不上去解析 manifest；导入不进来，
    谈不上查函数签名）。这样报告里看到的"跳过"永远有明确原因。
    """
    results: list[CheckResult] = []

    # ---- 1. 提交物齐全：algorithm.py 与 manifest.json 在不在
    # 缺文件是最彻底的失败，后面 6 项全部跳过。
    algo_file = v.algo_dir / f"{v.module_name}.py"
    manifest_file = v.algo_dir / contract.MANIFEST_FILENAME
    missing = [f.name for f in (algo_file, manifest_file) if not f.is_file()]
    if missing:
        results.append(_result("interface.files", False, f"缺少文件: {', '.join(missing)}"))
        results += _all_skipped("提交物不全，跳过", from_index=1)
        return results
    results.append(_result("interface.files", True, "algorithm.py 与 manifest.json 均在"))

    # ---- 2. manifest 合法：字段类型、枚举取值对不对
    # 注意这里会把解析结果**覆盖写回 v.manifest**——保证后续所有检查用的都是
    # 磁盘上那份真 manifest，而不是谁在内存里改过的。
    manifest, errors = contract.parse_manifest(manifest_file)
    if errors:
        results.append(_result("interface.manifest", False, "；".join(errors)))
        results += _all_skipped("manifest 不合法，跳过", from_index=2)
        return results
    v.manifest = manifest
    detail = (f"task=classification, subject={manifest.subject}, "
              f"标签列={manifest.label.column}、"
              f"声明 {len(manifest.label.classes or [])} 类，文本列={manifest.text_column}，"
              f"seed={manifest.seed}, budget={manifest.budget_seconds}s")
    results.append(_result("interface.manifest", True, detail))

    # ---- 3~5. 导入 + 函数齐全 + 签名正确
    # 三件事在**同一个子进程里一次做完**（runner.run_import）：反复起进程既慢，
    # 又可能撞上"第一次导入成功、第二次失败"这类环境噪声。
    payload = runner.run_import(v.algo_dir, v.module_name, budget=120)
    if not payload.ok:
        if payload.status == "timeout":
            results.append(_result("interface.import", False, "导入超时"))
        else:
            # _tail 取 traceback 最后几行——最有信息量的那几行（错误类型 + 原因）
            tail = _tail(payload.error)
            results.append(_result("interface.import", False, tail, payload.location))
        results += _all_skipped("模块无法导入，跳过", from_index=3)
        return results
    v.import_payload = payload.value
    results.append(_result("interface.import", True, "模块导入成功"))

    # 契约要求三个函数都存在。缺哪个报哪个，不一次性含糊说"函数不全"。
    missing_funcs = payload.value.get("missing", [])
    if missing_funcs:
        results.append(_result("interface.functions", False,
                               f"缺少函数: {', '.join(missing_funcs)}"))
        results += _all_skipped("契约函数不全，跳过", from_index=4)
        return results
    results.append(_result("interface.functions", True,
                           "build_features / fit / predict 均已定义"))

    # 签名检查：harness 会以固定个数的位置参数调用它们（build_features(df)、
    # fit(train_df)、predict(model, test_df)），参数个数不对就连不起来。
    bad_arity = [(n, m["msg"]) for n, m in payload.value.get("functions", {}).items()
                 if not m["arity_ok"]]
    if bad_arity:
        detail = "；".join(f"{n}: {msg}" for n, msg in bad_arity)
        results.append(_result("interface.signatures", False, detail))
        results += _all_skipped("函数签名不符合契约，跳过", from_index=5)
        return results
    results.append(_result("interface.signatures", True, "三个函数的参数个数符合契约"))

    # ---- 6. 冒烟运行：小切片上真跑一遍全链路
    results.append(_smoke(v))

    # ---- 7. 交付物完整性：run.py / README 能不能独立用起来
    results.append(_deliverables(v))
    return results


def _smoke(v: "Validator") -> CheckResult:
    """冒烟运行：拿一小段真实数据，真跑一遍 build_features → fit → predict。

    为什么要有这一关：前面 5 项都是"静态检查"（文件在不在、签名对不对），
    全过了也可能一跑就崩（比如 `fit` 里访问了不存在的列）。这一关是**第一次真执行**，
    在花大代价跑全量之前先把"当场就崩"的挡掉。

    为什么用"长度阶梯"（60 → 150 → 250 行）而不是固定一个长度：
    有些算法对样本量有隐含下限（比如要求每类至少若干样本），60 行的切片可能取不到
    可用行。这时不判它失败，而是换长一点的再试——**试到能跑通为止**，
    只有"报错"和"跑完仍取不到可用行"才算失败。

    判失败的两类情况：
    - `build_features` / `fit` / `predict` **报错或超时** → 真问题，换长度也没用，直接判负
    - 所有长度都试过仍凑不出可用行 → 判负，并提示"特征实现有问题或对样本量有隐含下限"
    """
    last_problem = ""
    for rows in SMOKE_LADDER:
        # 数据本身比这一档还短时不必再往上试（第一档除外，至少试一次）
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

        # 契约要求 build_features 返回 DataFrame，且必须保留标签列
        # （fit 要靠它训练，harness 也要靠它核对）。
        frame = feats.value
        if not isinstance(frame, pd.DataFrame):
            return _result("interface.smoke", False,
                           f"build_features 返回 {type(frame).__name__}，契约要求 DataFrame")
        if contract.LABEL_COLUMN not in frame.columns:
            return _result("interface.smoke", False,
                           f"特征表缺少标签列 {contract.LABEL_COLUMN}")

        # 挑"能用的行"：dropna 之后还要够跑（少于 MIN_USABLE_ROWS 就换长切片）
        usable = frame.dropna()
        if len(usable) < MIN_USABLE_ROWS:
            last_problem = (f"{len(sub)} 行切片里只有 {len(usable)} 行可用"
                            f"（特征还没成形）")
            continue

        # 留最后几行当"测试集"：契约要求 predict 的返回长度等于测试集行数，
        # 这里就能顺手验一次。测试集要剥掉标签列——模拟真实预测时拿不到答案。
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
                   f"{last_problem}；最长试到 {SMOKE_LADDER[-1]} 行仍跑不出可用特征，"
                   f"特征实现有问题（或对样本量有隐含下限）")


def _tail(text: str | None, lines: int = 6) -> str:
    """取报错信息的最后几行——traceback 里最有用的部分（错误类型 + 原因）。

    报告里只能放一行，所以把换行折成 `⏎`；完整 traceback 在报告 JSON 的
    其它字段里也有（`error`），这里只求"一眼看到是什么错"。
    """
    if not text:
        return "（无错误信息）"
    parts = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " ⏎ ".join(parts[-lines:])


DELIVERABLE_ROWS = 320     # 用小切片验证"能不能真跑出预测"


def _section_body(text: str, title: str) -> str | None:
    """取出某个小节（`## 它是什么`）的正文；没有这个小节返回 None。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^(#{{1,4}})\s*{re.escape(title)}\s*$", line.strip())
        if not m:
            continue
        level = len(m.group(1))
        body: list[str] = []
        for nxt in lines[i + 1:]:
            if re.match(rf"^#{{1,{level}}}\s", nxt):
                break
            body.append(nxt)
        return "\n".join(body)
    return None


def _deliverables(v: "Validator") -> CheckResult:
    """交付物能不能独立运行：run.py 在不在、--help 起不起得来、能不能真产出预测。

    生成的算法不只是给 harness 调用的零件，也是要交给用户的东西——
    用户不该为了跑它去拉本项目的源码。所以 run.py 属于契约的一部分。
    """
    cid = "interface.deliverables"
    # 用绝对路径：子进程的 cwd 就是这个目录，相对路径会被拼两遍
    run_py = (v.algo_dir / "run.py").resolve()
    if not run_py.is_file():
        return _result(cid, False,
                       "缺少 run.py：交付物应自带运行入口（读数据 → 训练 → 预测 → 写结果），"
                       "不能只作为被 harness 调用的模块存在")

    readme = v.algo_dir / "README.md"
    if not readme.is_file():
        return _result(cid, False,
                       "缺少 README.md：交付物应带使用说明（依赖 / 跑法 / 输入输出 / 口径）")
    text = readme.read_text(encoding="utf-8", errors="replace")
    cited = [m.group(1).strip("`\"'") for m in re.finditer(r"--data\s+([^\s`\"']+)", text)]
    if not cited:
        return _result(cid, False,
                       "README.md 里没有带 --data 的运行示例——用户拿到手第一步就卡住")
    if not any((v.algo_dir / c).exists() for c in cited):
        return _result(cid, False,
                       "README.md 里的 --data 路径都不存在（相对算法目录解析）："
                       + "、".join(cited[:3]) + "；示例命令必须能直接复制运行")

    # 交付文档必须先讲清「这是什么算法」，再谈依赖与跑法——不能一上来堆术语，
    # 也不能整篇不提。README 与任务报告用同一个固定小节名，便于机器检查。
    bad_docs: list[str] = []
    for doc_name, doc_path in (("README.md", readme),
                               ("report.md", v.algo_dir.parent / "report.md")):
        if not doc_path.is_file():
            if doc_name == "README.md":
                bad_docs.append("缺少 README.md")
            continue                      # 任务报告可选：存在才要求
        body = _section_body(doc_path.read_text(encoding="utf-8", errors="replace"),
                             "它是什么")
        if body is None:
            bad_docs.append(f"{doc_name} 没有「## 它是什么」一节")
            continue
        chars = len(re.sub(r"\s+", "", body))
        if chars < 50:
            bad_docs.append(f"{doc_name} 的「它是什么」只有 {chars} 字，没讲清")
    if bad_docs:
        return _result(cid, False,
                       "；".join(bad_docs)
                       + "。交付文档开头必须先用一两句人话讲清这是什么算法，再展开依赖与跑法")

    try:
        helped = subprocess.run(
            [sys.executable, str(run_py), "--help"], cwd=str(v.algo_dir),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60)
    except Exception as exc:                      # noqa: BLE001
        return _result(cid, False, f"run.py --help 起不来：{type(exc).__name__}: {exc}")
    if helped.returncode != 0:
        tail = (helped.stderr or helped.stdout or "").strip().splitlines()[-3:]
        return _result(cid, False, "run.py --help 非零退出：" + " ⏎ ".join(tail))

    tmpdir = tempfile.mkdtemp(prefix="harness_deliverables_")
    try:
        slice_path = os.path.join(tmpdir, "slice.csv")
        out_path = os.path.join(tmpdir, "predictions.csv")
        v.raw.head(min(DELIVERABLE_ROWS, len(v.raw))).to_csv(
            slice_path, index=False, encoding="utf-8-sig")
        try:
            ran = subprocess.run(
                [sys.executable, str(run_py), "--data", slice_path, "--out", out_path],
                cwd=str(v.algo_dir), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=v.smoke_budget)
        except subprocess.TimeoutExpired:
            return _result(cid, False, f"run.py 在 {v.smoke_budget:.0f}s 内没跑完")

        if ran.returncode != 0:
            tail = (ran.stderr or ran.stdout or "").strip().splitlines()[-3:]
            return _result(cid, False, "run.py 运行失败：" + " ⏎ ".join(tail))
        if not os.path.isfile(out_path):
            return _result(cid, False, "run.py 跑完了但没有产出结果文件（--out 指定的路径为空）")
        produced = pd.read_csv(out_path)
        if "prediction" not in produced.columns:
            return _result(cid, False,
                           f"结果文件缺少 prediction 列，实际列：{list(produced.columns)}")
        if produced.empty:
            return _result(cid, False, "结果文件是空的，没有任何预测")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return _result(cid, True,
                   f"可独立运行：run.py --help 正常；README 的示例命令指向真实数据（{cited[0]}）；"
                   f"{min(DELIVERABLE_ROWS, len(v.raw))} 行小数据上产出 {len(produced)} 条预测"
                   f"（不依赖本项目其它代码）")
