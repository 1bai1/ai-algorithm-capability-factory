# AG News 四分类算法 —— 使用说明

## 它是什么

这是一个**英文新闻标题/摘要的四分类**算法：给一段新闻文本，输出它属于
`Business`、`SciTech`、`Sports`、`World` 中的哪一类。

它怎么工作：先把每条文本做轻量规范化（转小写、去掉 HTML 与标点），然后用两套
TF-IDF 同时刻画它——一套看**词**（1/2-gram，比如 "stock market"），一套看
**词内字符片段**（3/5-gram，能抗拼写与词形变化）；两套特征拼在一起后交给
**ComplementNB** 朴素贝叶斯分类器打分，取分数最高的类别。全程 CPU 秒级完成，
不需要 GPU、不需要预训练模型。

## 1. 依赖

```bash
pip install pandas numpy scikit-learn
```

## 2. 数据格式

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 待分类的文本 |
| `label` | 类别标签（训练用），取值见 `manifest.json` 的 `label.classes` |

## 3. 跑起来

用本任务自带的示例数据集（路径相对本目录，可直接复制运行）：

```bash
python run.py --data ../data/agnews_sample.csv --out 预测结果.csv
```

换成你自己的数据，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out 预测结果.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`（随机种子）。
脚本按分层随机切分，训练前 80%、预测后 20%，并打印测试段准确率与宏 F1。

## 4. 输出

`预测结果.csv`，列：`row`（原数据行号）、`text`、`prediction`（预测类别）、
`actual`（真值）、`has_truth`（该行是否有真值）。

## 5. 口径

- 切分：分层随机，`seed` 取自 `manifest.json`（默认 42）；
- 特征：词 TF-IDF（1/2-gram，`sublinear_tf`、`min_df=2`）
  ∪ 词内字符 TF-IDF（`char_wb`，3/5-gram，`min_df=3`），**只在训练段拟合**；
- 分类器：`ComplementNB(alpha=0.3)`，由本任务选型实验在 seed=42 同划分下选定。

## 6. 完整验证

本目录由 Algorithm Coach 生成。更严格的检查（切分控制、反泄漏、确定性、
基线对照）在其 harness 里，报告见 `../validation/`：

```bash
python -m harness validate . --data ../data/agnews_sample.csv
```

**运行本算法不需要 harness**，它只是可选的质检工具。
