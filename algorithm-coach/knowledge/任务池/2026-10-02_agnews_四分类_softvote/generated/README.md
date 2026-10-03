# AG News 四分类（TF-IDF 词/字符 n-gram + NB·LogReg 软投票）

## 它是什么

这是一个**英文新闻的四分类**算法：给它一条新闻文本，它判断这条新闻属于
`Business`（财经）、`SciTech`（科技）、`Sports`（体育）还是 `World`（国际）中的哪一类。
工作方式是经典浅层流水线，不需要 GPU：先把文本逐行规范化（转小写、去 HTML 与标点），
再用两套 TF-IDF 同时表示它——词 1/2-gram 抓"股票、球队、选举"这类关键词，词内字符
3/5-gram 抓词形变化与拼写差异；两路特征拼接后，**同时**喂给两个互补的分类器
（互补朴素贝叶斯 ComplementNB 与逻辑回归），把两者预测各类别的概率等权平均，
取平均概率最高的那一类作为答案。训练在普通 CPU 上几秒完成。

## 依赖

```bash
pip install pandas numpy scikit-learn
```

Python 3.9+。本项目内置环境（可选）：`D:\environment\miniconda3\envs\math\python.exe`。

## 数据格式

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 待分类的新闻文本 |
| `label` | 类别标签（训练用；取值见 `manifest.json` 的 `label.classes`） |

## 怎么跑

**直接跑**（用仓库里的示例数据集，路径可直接复制）：

```bash
python run.py --data ../../../../examples/text_cls_demo/data/agnews_sample.csv --out predictions.csv
```

**换成你自己的数据**，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out predictions.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`（覆盖 manifest 的种子）。
脚本按 8:2 **分层随机**切分，训练前 80%、预测后 20%，并打印预测段准确率与宏 F1。

## 输出

`predictions.csv`，列：`row`（原数据行号）、`text`、`prediction`（预测类别）、
`actual`（真值）、`has_truth`（该行是否用于评分）。

## 口径

- 切分：分层随机，`seed` 取自 `manifest.json`（默认 42）。
- 特征：词 TF-IDF（`sublinear_tf`、`min_df=2`、1-2 gram）∪ 词内字符 TF-IDF
  （`char_wb`、`min_df=3`、3-5 gram），**词表与 IDF 只在训练段拟合**。
- 分类器：`ComplementNB(alpha=0.3)` 与 `LogisticRegression(C=8)` 软投票等权平均。
- 类别：见 `manifest.json` 的 `label.classes`（`Business` / `SciTech` / `Sports` / `World`）。

## 完整验证

本目录由 Algorithm Coach 生成，更严格的检查（切分控制、反泄漏、基线对照、
确定性、坏数据不崩）在其项目 harness 里，报告见 `../validation/`：

```bash
python -m harness validate . --data ../data/agnews_sample.csv
```

**运行本算法不需要 harness**——它只是一个可选的质检工具。
