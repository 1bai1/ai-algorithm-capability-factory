# AG News 四分类算法 —— 使用说明

TF-IDF（词 1-gram 与 2-gram 两套词表）× 互补朴素贝叶斯 + 线性 SVM 的**硬投票集成**。
**拿到这个目录就能跑，不依赖生成它的项目。**

## 1. 装依赖

```bash
pip install pandas numpy scikit-learn
```

## 2. 准备数据

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 新闻标题 + 正文（英文） |
| `label` | 类别标签，取值 `World` / `Sports` / `Business` / `SciTech` |

## 3. 跑

```bash
python run.py --data 你的数据.csv --out 预测结果.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`。
脚本按 8:2 分层随机切分，训练前 80%、预测后 20%，并打印准确率与宏 F1。

## 4. 输出

`预测结果.csv`，列：`row`（原数据行号）、`text`、`prediction`（预测类别）、
`actual`（真值）、`has_truth`（该行是否用于评分）。

## 5. 口径

- 切分：分层随机，`seed=42`（见 `manifest.json`）
- 特征：两套 TF-IDF（`sublinear_tf=True`、`min_df=2`、`ngram_range=(1,1)` 与 `(1,2)`），
  **只在训练段上拟合**词表与 IDF
- 预处理：逐行小写 + 空白折叠；不做停用词过滤（避免词表与分词器口径错位）
- 集成：三个成员硬投票（多数票），成员数取奇数避免平票

## 6. 完整验证

本目录由 Algorithm Coach 生成，验证报告见 `../validation/`：

```bash
python -m harness validate . --data <数据csv>
```

报告里除了四模块结论，还包含**同一划分下的对照基线**（多数类基线、harness 自带的
TF-IDF+逻辑回归参考基线）。**运行本算法不需要 harness**——它只是可选的质检工具。
