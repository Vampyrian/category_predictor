for _ in range(10):
    print("Hello World!")

################################################### Settings

checkpoint = 'amberoad/bert-multilingual-passage-reranking-msmarco'

# my_model_name = 'category_predictor_for_household_equipments'
my_model_name = 'category_predictor'

# dataset_name = 'Vampyrian/products_with_category_household_equipments'
dataset_name = 'Vampyrian/products_with_category'

################################################### Prepare dataset
from datasets import load_dataset, Dataset
dataset = load_dataset(dataset_name)

dataset = dataset.remove_columns('label')

unique_categories = set(dataset['train']['label_text'])
id2label = {idx: label for idx, label in enumerate(unique_categories)}
label2id = {label: idx for idx, label in enumerate(unique_categories)}

def replace_category_with_id(example):
    example['label'] = label2id[example.pop('label_text')]
    return example

dataset = dataset.map(replace_category_with_id)

dataset = dataset.shuffle(seed=42)

################################################### Tokenizing
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

################################################### Evaluate
import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

################################################### Train
tokenizer.push_to_hub(my_model_name)

from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
training_args = TrainingArguments(
    output_dir=my_model_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.push_to_hub()
# tokenizer.push_to_hub(my_model_name)