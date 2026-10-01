"""命令行入口。

    python -m harness validate <算法目录> --data <日线csv> [--cutoff 2024-01-01]

退出码：0 全部通过 / 1 有未通过项 / 2 输入数据或参数有误。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import DataError
from .validate import ALL_MODULES, Validator, format_console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="量化算法统一验证：接口规范 / 功能正确性 / 运行稳定性")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="验证一个算法目录")
    p.add_argument("algo_dir", help="算法目录（含 algorithm.py 与 manifest.json）")
    p.add_argument("--data", required=True, help="日线 CSV 路径")
    p.add_argument("--cutoff", default=None,
                   help="训练/样本外的截止日（如 2024-01-01）；不填则按 --test-size 取尾部")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="样本外占比，默认 0.2（仅在未指定 --cutoff 时生效）")
    p.add_argument("--budget", type=float, default=None,
                   help="fit/predict 时间预算（秒），默认取 manifest.budget_seconds")
    p.add_argument("--seed", type=int, default=None, help="覆盖 manifest 的随机种子")
    p.add_argument("--cost", type=float, default=None,
                   help="单边费率（手续费+滑点），默认取 manifest.cost_per_side 或 0.0005")
    p.add_argument("--module-name", default="algorithm", help="入口模块名，默认 algorithm")
    p.add_argument("--modules", default=",".join(ALL_MODULES),
                   help="启用的检查模块，逗号分隔")
    p.add_argument("--out", default=None,
                   help="报告输出目录，默认 <算法目录>/validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "validate":  # pragma: no cover - argparse 已限制
        return 2

    modules = tuple(m.strip() for m in args.modules.split(",") if m.strip())
    unknown = [m for m in modules if m not in ALL_MODULES]
    if unknown:
        print(f"未知模块: {unknown}（可用: {list(ALL_MODULES)}）")
        return 2

    try:
        validator = Validator(
            algo_dir=args.algo_dir,
            data_path=args.data,
            cutoff=args.cutoff,
            test_size=args.test_size,
            budget=args.budget,
            seed=args.seed,
            cost=args.cost,
            module_name=args.module_name,
            modules=modules,
        )
    except DataError as exc:
        print(f"数据不可用: {exc}")
        return 2

    report = validator.run()
    out_dir = Path(args.out) if args.out else Path(args.algo_dir) / "validation"
    json_path, md_path = validator.save(out_dir)

    print(format_console(report))
    print(f"报告: {json_path}")
    print(f"      {md_path}")
    return 0 if report.passed_all else 1
