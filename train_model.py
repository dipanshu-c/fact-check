from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
df["text"] = df["title"] + " " + df["text"]

dataset = Dataset.from_pandas(df[["text", "label"]])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ✅ ADD THIS
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "fake", 1: "real"},
    label2id={"fake": 0, "real": 1}
)

training_args = TrainingArguments(
    output_dir="./bert_model",
    num_train_epochs = 4,
    learning_rate = 2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=200,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")