"""Stage 3: the focal run at the dual-H200 batch regime.

From the development notebook, unchanged in behavior. Same focal loss
(gamma 2.0) at global batch 256, 25 epochs, LR 1e-4, warmup ratio 0.15,
weight decay 0.05. Writes Final_Confusion_Matrix.png and
RANKED_SUBMISSION.csv. The Trainer picks up data parallelism from the
environment.
"""

import os, sys, torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

torch.manual_seed(42)
np.random.seed(42)

train_df = pd.read_csv("shared_task_train.csv")
test_df = pd.read_csv("shared_task_devtest_no_label.csv")

# Tail-42 weak classes
weak_indices = [2, 3, 4, 6, 7, 10, 12, 13, 14, 15, 18, 20, 25, 27, 28, 29, 30, 34,
                35, 37, 38, 39, 41, 42, 43, 44, 45, 48, 53, 56, 59, 63, 66, 68,
                70, 72, 73, 74, 76, 77, 78, 79]

train_df_boosted = pd.concat([
    train_df,
    train_df[train_df['label'].isin(weak_indices)]
]).sample(frac=1, random_state=42).reset_index(drop=True)

model_name = "CAMEL-Lab/bert-base-arabic-camelbert-da"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding=True)

raw_ds = Dataset.from_pandas(train_df_boosted[['text', 'label']])
split = raw_ds.train_test_split(test_size=0.1, seed=42)
tokenized_train = split["train"].map(tokenize_fn, batched=True).remove_columns(["text"])
tokenized_eval = split["test"].map(tokenize_fn, batched=True).remove_columns(["text"])
tokenized_test = Dataset.from_pandas(test_df[['text']]).map(tokenize_fn, batched=True).remove_columns(["text"])

class SupersonicTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (0.25 * (1 - pt)**2 * ce_loss).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=82, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./ACL_Final_Submission",
    num_train_epochs=25,
    per_device_train_batch_size=256,   # Global batch 256 on dual H200
    learning_rate=1e-4,
    warmup_ratio=0.15,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none",
    dataloader_num_workers=4
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

trainer = SupersonicTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# Confusion matrix artifact
print("\nGenerating evaluation artifacts")
eval_results = trainer.predict(tokenized_eval)
y_pred = np.argmax(eval_results.predictions, axis=-1)
y_true = eval_results.label_ids

plt.figure(figsize=(20, 15))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(np.log1p(cm), cmap='YlGnBu')
plt.title("82-class confusion matrix (log-scaled)")
plt.savefig("Final_Confusion_Matrix.png", dpi=300)
plt.show()

# Submission file
test_preds = trainer.predict(tokenized_test)
final_labels = np.argmax(test_preds.predictions, axis=-1)
submission = pd.DataFrame({"Id": test_df["Id"], "Predicted": final_labels})
submission.to_csv("RANKED_SUBMISSION.csv", index=False)
print("Done: RANKED_SUBMISSION.csv written.")
