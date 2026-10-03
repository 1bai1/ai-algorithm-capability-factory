# AG News 四分类算法 —— 使用说明

词 1/2-gram TF-IDF 与词内字符 3/5-gram TF-IDF 的并集，接 `ComplementNB(alpha=0.3)`。
面向**英文短文本、单标签、类别较均衡、无 GPU** 的场景。**拿到这个目录就能跑，
不依赖生成它的项目、不需要 harness。**

## 1. 装依赖

```bash
pip install pandas numpy scikit-learn
```

Python 3.9+（本项目在 `D:\environment\miniconda3\envs\math\python.exe` 上验证）。

## 2. 准备数据

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 待分类的文本（列名以 `manifest.json` 的 `text_column` 为准） |
| `label` | 类别标签（列名以 `manifest.json` 的 `label.column` 为准） |

类别由 `manifest.json` 的 `label.classes` 声明：`Business` / `SciTech` / `Sports` / `World`。

## 3. 跑

**直接用本任务的示例数据集**（命令可直接复制运行）：

```bash
python run.py --data ../../../../examples/text_cls_demo/data/agnews_sample.csv --out 预测.csv
```

**换成你自己的数据**，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out 预测.csv
```

可选参数：`--test-size 0.2`（测试段占比）、`--seed 42`、`--text-col`、`--label-col`。
脚本默认做 8:2 分层随机切分，在训练段上拟合词表 / IDF 与分类器，对测试段预测。

## 4. 输出

`预测.csv`，列为：

| 列 | 说明 |
|---|---|
| `row` | 原数据行号（0 基） |
| `text` | 原文本 |
| `prediction` | 预测类别 |
| `actual` | 真值（用于对照评分） |

## 5. 口径

- 切分：分层随机（`seed` 取自 `manifest.json`，默认 42）。
- 特征：词 `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2,
  strip_accents='unicode')` ∪ 词内字符 `TfidfVectorizer(analyzer='char_wb',
  ngram_range=(3,5), sublinear_tf=True, min_df=3)`；**词表与 IDF 只在训练段上拟合**。
- 分类器：`ComplementNB(alpha=0.3)`。
- 文本规范化只做逐行变换（小写 / 去 HTML / 非字母数字换空格 / 压缩空白），不引用全样本统计量。

## 6. 完整验证（可选）

harness 报告见 `../validation/`，它额外检查反泄漏、确定性与基线对照。
运行本算法**不需要**它：

```bash
python -m harness validate . --data ../../../../examples/text_cls_demo/data/agnews_sample.csv
```
