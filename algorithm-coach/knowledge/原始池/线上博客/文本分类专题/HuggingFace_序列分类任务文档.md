---
source: huggingface.co/docs
title: Text classification (HuggingFace Transformers Docs)
url: https://huggingface.co/docs/transformers/tasks/sequence_classification
collected: 2026-10-01
---

[-1
]

[-1
]

Transformers documentation

Text classification

[0

]

# Transformers

🏡 View all docs
[
AWS Trainium & Inferentia
Accelerate
Argilla
AutoTrain
Bitsandbytes
CLI
Chat UI
Dataset viewer
Datasets
Deploying on AWS
Diffusers
Distilabel
Evaluate
Google Cloud
Google TPUs
Gradio
Hub
Hub Python Library
Huggingface.js
Inference Endpoints (dedicated)
Inference Providers
Kernels
LeRobot
Leaderboards
Lighteval
Microsoft Azure
OpenEnv
Optimum
PEFT
Reachy Mini
Safetensors
Sentence Transformers
TRL
Tasks
Text Embeddings Inference
Text Generation Inference
Tokenizers
Trackio
Transformers
Transformers.js
Xet
smolagents
timm
]

[0
[
main
v5.17.0
v5.15.1
v5.14.0
v5.13.1
v5.12.0
v5.11.0
v5.10.4
v5.9.0
v5.8.1
v5.7.0
v5.6.2
v5.5.4
v5.4.0
v5.3.0
v5.2.0
v5.1.0
v5.0.0
v4.57.6
v4.56.2
v4.55.4
v4.53.3
v4.52.3
v4.51.3
v4.50.0
v4.49.0
v4.48.2
v4.47.1
v4.46.3
v4.45.2
v4.44.2
v4.43.4
v4.42.4
v4.41.2
v4.40.2
v4.39.3
v4.38.2
v4.37.2
v4.36.1
v4.35.2
v4.34.1
v4.33.3
v4.32.1
v4.31.0
v4.30.0
v4.29.1
v4.28.1
v4.27.2
v4.26.1
v4.25.1
v4.24.0
v4.23.1
v4.22.2
v4.21.3
v4.20.1
v4.19.4
v4.18.0
v4.17.0
v4.16.2
v4.15.0
v4.14.1
v4.13.0
v4.12.5
v4.11.3
v4.10.1
v4.9.2
v4.8.2
v4.7.0
v4.6.0
v4.5.1
v4.4.2
v4.3.3
v4.2.2
v4.1.1
v4.0.1
v3.5.1
v3.4.0
v3.3.1
v3.2.0
v3.1.0
v3.0.2
v2.11.0
v2.10.0
v2.9.1
v2.8.0
v2.7.0
v2.6.0
v2.5.1
v2.4.1
v2.3.0
v2.2.2
v2.1.1
v2.0.0
v1.2.0
v1.1.0
v1.0.0
doc-builder-html
]
]

[
AR
DE
EN
ES
FR
HI
IT
JA
KO
PT
RO
TR
ZH
]
[-1
]

[-1

[-1
]

[-1
]

]

[-1
]

[-1
]

[0

Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces

Faster examples with accelerated inference

Switch between documentation themes

to get started

]

ss5ow4

[
[0
[
[0
[

[-1
]

[1

[-1
]

]

[1

[-1
]

]

[0
# Text classification

]

Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text classification in production for a wide range of practical applications. One of the most popular forms of text classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a sequence of text.

This guide will show you how to:

2. Finetune DistilBERT on the IMDb dataset to determine whether a movie review is positive or negative.

2. Use your finetuned model for inference.

To see all architectures and checkpoints compatible with this task, we recommend checking the
task-page
.

Before you begin, make sure you have all the necessary libraries installed:

```
    pip install transformers datasets evaluate accelerate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```
    >>> 
    from
     huggingface_hub 
    import
     notebook_login
    
    
    >>> 
    notebook_login()
```

[1
## Load IMDb dataset

]

Start by loading the IMDb dataset from the 🤗 Datasets library:

```
    >>> 
    from
     datasets 
    import
     load_dataset
    
    
    >>> 
    imdb = load_dataset(
    "stanfordnlp/imdb"
    )
```

Then take a look at an example:

```
    >>> 
    imdb[
    "test"
    ][
    0
    ]
    {
        
    "label"
    : 
    0
    ,
        
    "text"
    : 
    "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again."
    ,
    }
```

There are two fields in this dataset:

- text : the movie review text.

- label : a value that is either 0 for a negative review or 1 for a positive review.

[1
## Preprocess

]

The next step is to load a DistilBERT tokenizer to preprocess the
`text`
field:

```
    >>> 
    from
     transformers 
    import
     AutoTokenizer
    
    
    >>> 
    tokenizer = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
    )
```

Create a preprocessing function to tokenize
`text`
and truncate sequences to be no longer than DistilBERT’s maximum input length:

```
    >>> 
    def
     
    preprocess_function
    (
    examples
    ):
    
    ... 
        
    return
     tokenizer(examples[
    "text"
    ], truncation=
    True
    )
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets
map
function. You can speed up
`map`
by setting
`batched=True`
to process multiple elements of the dataset at once:

```
    tokenized_imdb = imdb.
    map
    (preprocess_function, batched=
    True
    )
```

Now create a batch of examples using
DataCollatorWithPadding
. It’s more efficient to
dynamically pad
the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

```
    >>> 
    from
     transformers 
    import
     DataCollatorWithPadding
    
    
    >>> 
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

[1
## Evaluate

]

Including a metric during training is often helpful for evaluating your model’s performance. You can quickly load a evaluation method with the 🤗
Evaluate
library. For this task, load the
accuracy
metric (see the 🤗 Evaluate
quick tour
to learn more about how to load and compute a metric):

```
    >>> 
    import
     evaluate
    
    
    >>> 
    accuracy = evaluate.load(
    "accuracy"
    )
```

Then create a function that passes your predictions and labels to
compute
to calculate the accuracy:

```
    >>> 
    import
     numpy 
    as
     np
    
    
    
    >>> 
    def
     
    compute_metrics
    (
    eval_pred
    ):
    
    ... 
        predictions, labels = eval_pred
    
    ... 
        predictions = np.argmax(predictions, axis=
    1
    )
    
    ... 
        
    return
     accuracy.compute(predictions=predictions, references=labels)
```

Your
`compute_metrics`
function is ready to go now, and you’ll return to it when you setup your training.

[1
## Train

]

Before you start training your model, create a map of the expected ids to their labels with
`id2label`
and
`label2id`
:

```
    >>> 
    id2label = {
    0
    : 
    "NEGATIVE"
    , 
    1
    : 
    "POSITIVE"
    }
    
    >>> 
    label2id = {
    "NEGATIVE"
    : 
    0
    , 
    "POSITIVE"
    : 
    1
    }
```

If you aren’t familiar with finetuning a model with the
Trainer
, take a look at the basic tutorial
here
!

You’re ready to start training your model now! Load DistilBERT with
AutoModelForSequenceClassification
along with the number of expected labels, and the label mappings:

```
    >>> 
    from
     transformers 
    import
     AutoModelForSequenceClassification, TrainingArguments, Trainer
    
    
    >>> 
    model = AutoModelForSequenceClassification.from_pretrained(
    
    ... 
        
    "distilbert/distilbert-base-uncased"
    , num_labels=
    2
    , id2label=id2label, label2id=label2id
    
    ... 
    )
```

At this point, only three steps remain:

3. Define your training hyperparameters in TrainingArguments . The only required parameter is output_dir which specifies where to save your model. You’ll push this model to the Hub by setting push_to_hub=True (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the Trainer will evaluate the accuracy and save the training checkpoint.

3. Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.

3. Call train() to finetune your model.

```
    >>> 
    training_args = TrainingArguments(
    
    ... 
        output_dir=
    "my_awesome_model"
    ,
    
    ... 
        learning_rate=
    2e-5
    ,
    
    ... 
        per_device_train_batch_size=
    16
    ,
    
    ... 
        per_device_eval_batch_size=
    16
    ,
    
    ... 
        num_train_epochs=
    2
    ,
    
    ... 
        weight_decay=
    0.01
    ,
    
    ... 
        eval_strategy=
    "epoch"
    ,
    
    ... 
        save_strategy=
    "epoch"
    ,
    
    ... 
        load_best_model_at_end=
    True
    ,
    
    ... 
        push_to_hub=
    True
    ,
    
    ... 
    )
    
    
    >>> 
    trainer = Trainer(
    
    ... 
        model=model,
    
    ... 
        args=training_args,
    
    ... 
        train_dataset=tokenized_imdb[
    "train"
    ],
    
    ... 
        eval_dataset=tokenized_imdb[
    "test"
    ],
    
    ... 
        processing_class=tokenizer,
    
    ... 
        data_collator=data_collator,
    
    ... 
        compute_metrics=compute_metrics,
    
    ... 
    )
    
    
    >>> 
    trainer.train()
```

Trainer
applies dynamic padding by default when you pass
`tokenizer`
to it. In this case, you don’t need to specify a data collator explicitly.

Once training is completed, share your model to the Hub with the
push_to_hub()
method so everyone can use your model:

```
    >>> 
    trainer.push_to_hub()
```

For a more in-depth example of how to finetune a model for text classification, take a look at the corresponding
PyTorch notebook
.

[1
## Inference

]

Great, now that you’ve finetuned a model, you can use it for inference!

Grab some text you’d like to run inference on:

```
    >>> 
    text = 
    "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

The simplest way to try out your finetuned model for inference is to use it in a
pipeline()
. Instantiate a
`pipeline`
for sentiment analysis with your model, and pass your text to it:

```
    >>> 
    from
     transformers 
    import
     pipeline
    
    
    >>> 
    classifier = pipeline(
    "sentiment-analysis"
    , model=
    "stevhliu/my_awesome_model"
    )
    
    >>> 
    classifier(text)
    [{
    'label'
    : 
    'POSITIVE'
    , 
    'score'
    : 
    0.9994940757751465
    }]
```

You can also manually replicate the results of the
`pipeline`
if you’d like:

Tokenize the text and return PyTorch tensors:

```
    >>> 
    from
     transformers 
    import
     AutoTokenizer
    
    
    >>> 
    tokenizer = AutoTokenizer.from_pretrained(
    "stevhliu/my_awesome_model"
    )
    
    >>> 
    inputs = tokenizer(text, return_tensors=
    "pt"
    )
```

Pass your inputs to the model and return the
`logits`
:

```
    >>> 
    import
     torch
    
    >>> 
    from
     transformers 
    import
     AutoModelForSequenceClassification
    
    
    >>> 
    model = AutoModelForSequenceClassification.from_pretrained(
    "stevhliu/my_awesome_model"
    )
    
    >>> 
    with
     torch.no_grad():
    
    ... 
        logits = model(**inputs).logits
```

Get the class with the highest probability, and use the model’s
`id2label`
mapping to convert it to a text label:

```
    >>> 
    predicted_class_id = logits.argmax().item()
    
    >>> 
    model.config.id2label[predicted_class_id]
    
    'POSITIVE'
```

Update
on GitHub

]

]
]
]

[-1
]
]

[-1
]

[-1
]

[0
←
Unsloth
]

[1
Token classification
→
]
