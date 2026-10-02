# 文本分类算法 —— 使用说明

## 它是什么

这是一个**文本分类**算法：输入一段文本，输出它所属的类别（类别集合见
`manifest.json` 的 `label.classes`）。

路线是经典浅层方案，不含神经网络、不需要 GPU：用 TF-IDF（词 1/2-gram）把文本
变成稀疏向量，再交给逻辑回归分类。训练在普通 CPU 上秒级完成。

拿到这个目录就能跑，不依赖生成它的项目。

## 1. 装依赖

```bash
pip install pandas numpy scikit-learn
```

## 2. 准备数据

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 待分类的文本 |
| `label` | 类别标签（训练用；类别见 `manifest.json` 的 `label.classes`） |

## 3. 跑

**直接跑**——用本任务的示例数据集，路径可直接复制：

```bash
python run.py --data ../data/agnews_sample.csv --out 预测.csv
```

**换成你自己的数据**，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out 预测.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`。
脚本会按 8:2 分层随机切分，训练前 80%、预测后 20%，并打印准确率与宏 F1。

## 4. 输出

`预测结果.csv`，列：`row`（原数据行号）、`text`、`prediction`（预测类别）、
`actual`（真值）、`has_truth`（该行是否用于评分）。

## 5. 口径

- 切分：分层随机，`seed` 取自 `manifest.json`（默认 42）
- 特征：TF-IDF（`sublinear_tf`、`min_df=2`、1-2 gram），**只在训练段上拟合**
- 类别：见 `manifest.json` 的 `label.classes`

## 6. 完整验证

本目录由 Algorithm Coach 生成，更严格的检查（切分控制、反泄漏、
基线对照与成本扫描）在其 harness 里，报告见 `../validation/`：

```bash
python -m harness validate . --data ../data/agnews_sample.csv
```

**运行本算法不需要 harness**——它只是一个可选的质检工具。
