# Validation Harness

对生成的文本分类算法做**四模块 19 项**检查，产出结构化报告（`report.json` 给机器、
`report.md` 给人），落在 `knowledge/任务池/<任务>/validation/`。

harness 由**验证者**运行，不是算法的一部分：它自己读数据、自己切分、在子进程里带
超时地调用算法的三个契约函数（`build_features` / `fit` / `predict`），再把结论写成报告。
交付物（`generated/`）不依赖 harness——用户跑 `run.py` 就够了。

## 四个模块

| 模块 | 检查项 | 管什么 |
|---|---|---|
| 接口规范 `checks/interface.py` | 7 | **闸门**：提交物齐全、manifest 合法、能导入、契约函数与签名、冒烟运行、交付物可独立运行。任一项不过，其余模块全部跳过 |
| 功能正确性 `checks/correctness.py` | 5 | 行数守恒、标签列有效性、**拟合范围一致性（反泄漏）**、预测输出有效性、多数类基线对比 |
| 指标表现 `checks/performance.py` | 3 | 精度账（对标多数类与 TF-IDF+线性参考基线）、类别分布与划分披露、训练/测试差距 |
| 运行稳定性 `checks/stability.py` | 4 | 时间预算、确定性（同种子可复现）、小样本不崩、坏数据不崩 |

指标数值由 `metrics.py` 计算（纯函数，便于单测），判定在 `checks/performance.py`——
计算与判据分开，数值可以单独测、判据可以单独改。

## 为什么切分由 harness 控制

训练集/测试集的切分在 harness 手里（分类任务用**分层随机**，种子取自 manifest），
意味着「随机切分」「切分前重采样」「把测试集混进训练集」这几类错误在结构上不可能
发生——不是靠检查事后抓，而是根本没机会犯。

最典型的隐性泄漏是「向量化器（词表/IDF）在全量语料上先 fit 再切分」：它躲得过所有
只看指标的眼睛，但躲不过 `correctness.feature_fit_scope`——那条检查把同一个
`build_features` 在全量数据与仅训练集上各跑一次，逐行比对训练集行上的特征。

## 报告

- `report.json`：机器读。`meta`（数据规模、划分方式、种子、耗时、计数）、
  `metrics`（精度账全量数字，含逐类分数与混淆矩阵）、`modules`（每项检查的
  通过/未通过/跳过 + 说明 + 出错位置 + 依据知识卡）。
- `report.md`：同一份内容的人读版，含精度账、逐类表现、混淆矩阵、类别分布、
  训练/测试差距与未通过项明细。

判据的出处写在检查项的 `kb_card` 字段里——每条判据都能指回复用池里的一张能力卡，
不是拍脑袋定的阈值。

## 用法

```bash
python -m harness validate <算法目录> --data <文本csv> [--out <报告目录>]
```

- `<算法目录>` 需含 `algorithm.py` 与 `manifest.json`（契约见 `contract.py` 顶部）
- `--modules interface,correctness` 可只跑部分模块（默认四个全跑）
- `--seed` / `--budget` / `--test-size` 可覆盖 manifest 里的对应值
- 退出码：`0` 全通过 / `1` 有未通过项 / `2` 输入数据或参数有误

自测：

```bash
python -m unittest tests.test_harness_classification
```

## 任务类型

当前只支持**文本分类**一种。`manifest.task` 目前只有 `classification` 一个取值，
保留这个字段是为了：换场景时在 `contract.TASKS` 里加取值、再补一条数据加载与切分
路径即可——契约三函数、报告骨架、四个模块是任务无关的。
