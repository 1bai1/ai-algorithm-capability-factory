"""命令行入口：解析参数 → 建 Validator → 跑四类检查 → 写报告 → 定退出码。

谁调用它
--------
用户在终端敲 `python -m harness validate ...`。`__main__.py` 负责把本包变成可执行入口，
再把控制权交给这里的 `main()`。除此之外没有别的入口——测试代码是直接构造 `Validator`
对象，不走命令行。

一句话职责
----------
**把命令行上的字符串，翻译成一次验证所需的上下文，跑完，把结果落盘并报告成败。**

本文件自己不检查任何东西：判定在 `checks/`，编排在 `validate.py`，执行在 `runner.py`。
它只管三件事——参数解析、参数校验、退出码。

退出码（脚本化调用靠它判断成败）
--------------------------------
- `0` 全部检查通过（`report.passed_all` 为真）
- `1` 有未通过的检查——报告**已经写盘**，失败明细在 `report.json` / `report.md` 里
- `2` 参数或数据有问题，压根没跑起来（数据文件不存在、`--modules` 写了不存在的模块名等）

`2` 与 `1` 的区别要记住：**`2` 是"你没让我跑成"，`1` 是"我跑了，你的算法没过"。**
脚本化调用时两者要分开处理——`2` 该修调用方式，`1` 该修算法。

用法
----
    python -m harness validate <算法目录> --data <数据集csv> [--out <报告目录>]

本仓库的真实调用：

    python -m harness validate knowledge/任务池/2026-10-02_agnews_四分类/generated \
        --data examples/text_cls_demo/data/agnews_sample.csv \
        --out knowledge/任务池/2026-10-02_agnews_四分类/validation/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import DataError
from .validate import ALL_MODULES, Validator, format_console


def build_parser() -> argparse.ArgumentParser:
    """拼出命令行参数表。

    目前只有 `validate` 一个子命令（harness 只做"验证一个算法目录"这一件事）。
    保留子命令结构是为了以后加别的动作（比如对比两个算法）时不用改调用方式。
    """
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="文本分类算法统一验证：接口规范 / 功能正确性 / 指标表现 / 运行稳定性")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="验证一个算法目录")

    # ── 被测对象：算法目录（位置参数，必填）
    # 目录里必须有 algorithm.py（三个契约函数）与 manifest.json（元数据）。
    # 缺了不会在这里报错，而是走到闸门那一关才判"提交物齐全"不过——
    # 因为"缺文件"本身就是一条要记录的检查结果，不是命令行错误。
    p.add_argument("algo_dir", help="算法目录（含 algorithm.py 与 manifest.json）")

    # ── 数据集（必填）
    # harness 自己读这个 CSV，标准化之后再喂给算法——算法不碰文件、不碰网络。
    # 读不了（文件不存在、缺文本列或标签列、行数太少、类别不足 2 类）会抛 DataError，
    # 由 main() 转成退出码 2。
    p.add_argument("--data", required=True,
                   help="文本分类数据集 CSV 路径（含文本列与标签列）")

    # ── 测试集占比（可选，默认 0.2）
    # 切分由 harness 控制、算法无法干预；4000 行数据按 0.2 切出 800 行测试集。
    # 想让测试集更大或更小，改这里；算法那边改不了这件事。
    p.add_argument("--test-size", type=float, default=0.2,
                   help="测试集占比，默认 0.2")

    # ── 时间预算（可选）
    # fit/predict 超过这个秒数就判「时间预算」不过。不填则取 manifest 里的
    # budget_seconds；两个都没有时用契约默认值 300 秒。
    p.add_argument("--budget", type=float, default=None,
                   help="fit/predict 时间预算（秒），默认取 manifest.budget_seconds")

    # ── 随机种子（可选）
    # 覆盖 manifest 里的 seed。典型用途：换个种子再跑一遍，复核"这个成绩是不是运气"。
    p.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")

    # ── 算法入口模块名（可选，默认 algorithm）
    # 对应 <算法目录>/algorithm.py。只有在算法把入口改了名字（或拆了包）时才需要传。
    p.add_argument("--module-name", default="algorithm", help="入口模块名，默认 algorithm")

    # ── 启用哪些检查模块（可选，默认全部）
    # 取值必须是注册表里的模块名（见 checks/__init__.py）。常用法：
    # 调试时只跑 interface,correctness 图快；只看成绩时只跑 performance。
    # 写错名字会返回退出码 2（属于"没让我跑成"）。
    p.add_argument("--modules", default=",".join(ALL_MODULES),
                   help="启用的检查模块，逗号分隔")

    # ── 报告输出目录（可选）
    # 默认写到 <算法目录>/validation/，跟任务产物放一起（report.json 与 report.md）。
    p.add_argument("--out", default=None,
                   help="报告输出目录，默认 <算法目录>/validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑一次完整验证，返回退出码（含义见模块 docstring 的 0 / 1 / 2）。"""
    args = build_parser().parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse 已限制取值
        return 2

    # ── 第一步：校验 --modules 的取值
    # 把逗号分隔的字符串拆成模块名，逐个对照注册表。写错模块名是"调用方式错了"，
    # 所以返回 2；顺手把可用的模块名列出来，省得去翻代码。
    modules = tuple(m.strip() for m in args.modules.split(",") if m.strip())
    unknown = [m for m in modules if m not in ALL_MODULES]
    if unknown:
        print(f"未知模块: {unknown}（可用: {list(ALL_MODULES)}）")
        return 2

    # ── 第二步：构造 Validator
    # 注意这一步就会**读数据**（为了拿 manifest 里的标签列名去加载 CSV）。
    # 数据不可用时抛 DataError —— 属于"没让我跑成"，返回 2，
    # 并且只打一行原因，不产出报告（没意义：连数据都没有）。
    try:
        validator = Validator(
            algo_dir=args.algo_dir,
            data_path=args.data,
            test_size=args.test_size,
            budget=args.budget,
            seed=args.seed,
            module_name=args.module_name,
            modules=modules,
        )
    except DataError as exc:
        print(f"数据不可用: {exc}")
        return 2

    # ── 第三步：真跑
    # 四类检查按注册表的 ORDER 依次执行：接口规范（闸门）→ 功能正确性 →
    # 指标表现 → 运行稳定性。闸门不过的话，后面的模块全部记"跳过"，
    # 不会拿一份连导入都失败的算法去算指标。
    report = validator.run()

    # ── 第四步：落盘
    # 默认写到 <算法目录>/validation/，两份报告各有用处：
    # report.json 给机器读（含每项检查的 id / detail / 出错位置 / 依据知识卡），
    # report.md   给人读（含精度账、逐类表现、混淆矩阵与未通过项明细）。
    out_dir = Path(args.out) if args.out else Path(args.algo_dir) / "validation"
    json_path, md_path = validator.save(out_dir)

    # ── 第五步：先给人看摘要，再给报告路径
    # 控制台摘要按模块分组列出每项检查的通过/未通过/跳过与一句话说明，
    # 目的是让人一眼看到"哪一项没过"，不用去翻 JSON。
    print(format_console(report))
    print(f"报告: {json_path}")
    print(f"      {md_path}")

    # ── 退出码
    # 全过 0，有未通过 1。报告此时已经写盘，脚本化调用可以按退出码分支：
    # 0 → 继续下一步；1 → 读 report.json 的未通过项去修算法。
    return 0 if report.passed_all else 1
