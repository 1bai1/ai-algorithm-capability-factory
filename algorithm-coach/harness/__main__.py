"""python -m harness 入口。

注意：Windows 的 spawn 子进程会以 __mp_main__ 名字重新导入本模块，
因此这里只做分发，不写任何有副作用的模块级代码。
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
