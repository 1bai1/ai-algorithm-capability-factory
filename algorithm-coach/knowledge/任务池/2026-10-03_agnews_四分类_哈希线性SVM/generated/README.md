# AG News 四分类 —— 特征哈希 + TF-IDF 加权 + 线性 SVM

## 它是什么

这是一个**文本分类**算法：输入一段新闻文本，输出它属于四个类别中的哪一类
（Business / SciTech / Sports / World）。

它和常见的「先建词表再算 TF-IDF」做法不同：本算法**不维护词表**，而是把每段文本的
词 1/2-gram 和词内字符 3/5-gram 用**特征哈希**直接映射到固定维度的稀疏向量上，
再用 TF-IDF 加权，最后交给一个线性 SVM 做判决。好处是不怕没见过的词（哈希天然支持），
训练只需 CPU 秒级完成，不需要 GPU，也不需要任何预训练模型。代价是哈希会有碰撞，
所以维度取得较大（每支 2^18）。

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

本任务示例数据为 AG News 四分类采样集（4000 行、四类各 1000）。

## 3. 跑

**直接跑**——示例数据就在本仓库里，路径可复制（相对本目录）：

```bash
python run.py --data ../../../../examples/text_cls_demo/data/agnews_sample.csv --out 预测.csv
```

**换成你自己的数据**，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out 预测.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`。
脚本按 8:2 分层随机切分，训练前 80%、预测后 20%，并打印预测段准确率与宏 F1。

## 4. 输出

`预测.csv`，列：`row`（原数据行号）、`text`、`prediction`（预测类别）、
`actual`（真值）、`has_truth`（该行是否用于评分）。

## 5. 口径

- 切分：分层随机，`seed` 取自 `manifest.json`（默认 42）。
- 特征：词 1/2-gram 与词内字符 3/5-gram 各哈希到 2^18 维，再 `sublinear_tf` 的
  TF-IDF 加权；idf 只在训练段拟合，测试段只做变换。
- 分类器：`LinearSVC(C=0.5)`。
- 报告指标：准确率与宏 F1。特征只需逐行规范化，未做任何跨行统计。

## 6. 完整验证

本目录由 Algorithm Coach 生成，更严格的检查（切分控制、反泄漏、
基线对照与稳定性）在其 harness 里，报告见 `../validation/`：

```bash
python -m harness validate . --data ../../../../examples/text_cls_demo/data/agnews_sample.csv
```

**运行本算法不需要 harness**——它只是一个可选的质检工具。
