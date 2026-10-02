# Transformer 文本分类微调流程

## 能力说明
用预训练 Transformer（示例为 DistilBERT）在下游标注语料上做序列分类微调，端到端流程：
加载数据集 → 分词 → 动态 padding → 定义指标 → Trainer 训练 → pipeline 推理。是「数据足够、要精度」
时相对浅层特征工程的升级路线。

## 输入契约
- 标注数据集：`text` 列 + 整数 `label` 列（IMDb 为 0 负 / 1 正）；多分类需自行扩 `num_labels` 与标签映射。
- 与权重匹配的 tokenizer（同一 checkpoint 的 `AutoTokenizer.from_pretrained`）。
- 序列按 `truncation=True` 截到模型最大长度；原文未做长文本分段或滑窗。

## 输出契约
- 微调后的 checkpoint（config 内含 `id2label` / `label2id`），可 `push_to_hub` 或本地保存。
- 推理：`pipeline` 返回 `{"label": 标签名, "score": 概率}`；手动路径先取 logits，`argmax` 后经
  `model.config.id2label` 换成标签名。

## 调用方式
```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

imdb = load_dataset("stanfordnlp/imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
tokenized = imdb.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="my_awesome_model", learning_rate=2e-5,
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    num_train_epochs=2, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"],
                  eval_dataset=tokenized["test"], processing_class=tokenizer,
                  data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()

# 推理
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier(text)   # [{'label': 'POSITIVE', 'score': 0.9994940757751465}]

# 手动复现：tokenizer(text, return_tensors="pt") → model(**inputs).logits
#           → logits.argmax().item() → model.config.id2label[predicted_class_id]
```

## 关键参数
- 训练超参（原文固定一组）：`learning_rate=2e-5`、`per_device_train_batch_size=16`、
  `per_device_eval_batch_size=16`、`num_train_epochs=2`、`weight_decay=0.01`、
  `eval_strategy="epoch"`、`save_strategy="epoch"`、`load_best_model_at_end=True`；`output_dir` 是唯一必填项。
- 模型侧：`num_labels=2`、`id2label` / `label2id`。
- 分词侧：`truncation=True`；动态 padding（`DataCollatorWithPadding`，给 Trainer 传
  tokenizer/processing_class 时默认启用）。
- 原文没有调参过程，以上是一次固定取值，无敏感性分析。

## 依赖
transformers、datasets、evaluate、accelerate（原文安装命令 `pip install transformers datasets evaluate accelerate`）；
推理另需 torch。需要能联网拉取预训练权重。

## 适用条件
- 有足量标注文本、追求精度、需要上下文/语义表示而不是词表特征时。
- 二分类模板可直接套用；多分类改 `num_labels` 与标签映射即可（原文只示范了 2 类）。
- 需要把模型上传复用或保留标签语义时（`push_to_hub`、config 里的 id2label）。

## 不适用条件
- 没有独立验证集时不要开 `load_best_model_at_end`：原文把 `test` split 直接当 `eval_dataset`，
  等于用测试集挑 checkpoint。
- 长文本会被 `truncation=True` 截断，超出最大长度的部分直接丢弃——原文未讨论截断的影响与补救。
- 算力与数据门槛不明：原文没给训练耗时、显存占用与最小数据量，小数据低成本场景按另一份综述的口径
  应先用传统算法（见 [[文本分类模型谱系与选择]]）。
- 指标只用 accuracy，类别不平衡与多标签场景原文未涉及，不能照抄该评估口径。
- 原文没有给出训练后的任何评估数字，「微调有效」在本份资料里未经验证。

## 验证状态
- 待验证：原文给出完整可运行的流程与一次真实推理输出（POSITIVE，score = 0.9994940757751465，
  单条样例），但没有训练后的 accuracy / loss / 耗时，也没有与任何基线对比。
- 原文未验证：2 epochs 是否足够、lr 2e-5 是否最优、截断带来的信息损失——都没有实验支撑。

## 来源
- `HuggingFace_序列分类任务文档.md`：IMDb 数据字段、分词与 `batched=True` 的 map、动态 padding、
  `compute_metrics` 的 accuracy 口径、`id2label`/`label2id`、`TrainingArguments` 的全部取值、
  `Trainer` 组装与训练调用、pipeline 与手动推理路径、单条样例的推理分数、
  以及 `eval_dataset=tokenized_imdb["test"]` 与 `load_best_model_at_end=True` 同时出现的事实。
