# AG News 四分类算法 —— 使用说明

## 它是什么

这是一个**文本四分类**算法：输入一段英文新闻文本，输出四个类别之一 ——
`Business` / `SciTech` / `Sports` / `World`。

路线是经典浅层方案，不含神经网络、不需要 GPU：先用 TF-IDF 把文本变成稀疏向量
（词 1/2-gram 与词内字符 3/5-gram 两套词表取并集），再交给互补朴素贝叶斯
`ComplementNB(α=0.3)` 分类。训练在普通 CPU 上秒级完成，可解释、可复现。

拿到这个目录就能跑，不依赖生成它的项目或 harness。

## 1. 依赖

```bash
pip install pandas numpy scikit-learn
```

## 2. 数据格式

一个 CSV，至少两列：

| 列 | 说明 |
|---|---|
| `text` | 待分类文本 |
| `label` | 类别标签，取值为 `Business` / `SciTech` / `Sports` / `World` |

## 3. 跑

**直接跑**——用本任务的示例数据集（4000 行采样集），路径可直接复制：

```bash
python run.py --data ../../../../examples/text_cls_demo/data/agnews_sample.csv --out 预测.csv
```

**换成你自己的数据**，只改 `--data`：

```bash
python run.py --data 你的数据.csv --out 预测.csv
```

可选参数：`--train-ratio 0.8`（训练段占比）、`--seed 42`（覆盖 manifest 的种子）。

## 4. 输出

`预测结果.csv`，列：

| 列 | 含义 |
|---|---|
| `row` | 原数据行号（从 0 起） |
| `text` | 原文本 |
| `prediction` | 预测类别 |
| `actual` | 真值 |
| `has_truth` | 该行是否有真值（用于评分） |

同时在终端打印测试段准确率与宏 F1。

## 5. 算法口径

- **文本规范化**：小写、去 HTML 标签、非字母数字换空格、压缩空白。纯逐行变换。
- **特征**：两个 TF-IDF 的并集各建一套词表 —— 词 1/2-gram（`sublinear_tf`、`min_df=2`）
  与词内字符 3/5-gram（`sublinear_tf`、`min_df=3`，`analyzer='char_wb'`）。
- **分类器**：`ComplementNB(alpha=0.3)`。
- **切分**：分层随机 8:2，`seed` 取自 `manifest.json`（默认 42）；
  词表与 IDF **只在训练段拟合**，测试段只做 transform（反泄漏）。
- **类别**：见 `manifest.json` 的 `label.classes`。

## 6. 完整验证（可选）

本目录由 Algorithm Coach 生成。更严格的检查（切分控制、反泄漏、指标对标
多数类与 TF-IDF+线性参考基线、确定性等）由该项目 harness 负责，
验证报告见 `../validation/`。**运行本算法不需要 harness。**
