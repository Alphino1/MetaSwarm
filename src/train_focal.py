"""Stage 2 development run: focal loss (gamma 2.0, alpha 0.25) on the oversampled set.

From the development notebook, unchanged in behavior. Introduces the Tail-42
oversampling list and replaces weighted CE with focal loss; the submitted
ensemble later uses gamma 2.5. Same data expectations as stage 1; writes
Confusion_Matrix_EACL.png and SUBMISSION.csv. CAMeLBERT-DA, max length 256,
batch 64, 12 epochs, LR 4e-5, cosine, warmup ratio 0.1, weight decay 0.01,
fp16, 90/10 split (seed 42). Like the notebook stage, no global torch seed is
set here; the other stages set seed 42.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

train_df = pd.read_csv("shared_task_train.csv")
test_df = pd.read_csv("shared_task_devtest_no_label.csv")

# Tail-42 weak classes (output of the per-class F1 audit in latent_manifold.py)
weak_indices = [2, 3, 4, 6, 7, 10, 12, 13, 14, 15, 18, 20, 25, 27, 28, 29, 30, 34,
                35, 37, 38, 39, 41, 42, 43, 44, 45, 48, 53, 56, 59, 63, 66, 68,
                70, 72, 73, 74, 76, 77, 78, 79]

# oversample weak classes, shuffle
train_df_boosted = pd.concat([
    train_df,
    train_df[train_df['label'].isin(weak_indices)]
]).sample(frac=1, random_state=42).reset_index(drop=True)

model_name = "CAMEL-Lab/bert-base-arabic-camelbert-da"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding=True)

full_ds = Dataset.from_pandas(train_df_boosted[['text', 'label']])
split = full_ds.train_test_split(test_size=0.1, seed=42)
tokenized_train = split["train"].map(preprocess_function, batched=True)
tokenized_eval = split["test"].map(preprocess_function, batched=True)
tokenized_test = Dataset.from_pandas(test_df[['text']]).map(preprocess_function, batched=True)

# focal loss: alpha_f (1 - p_t)^gamma CE
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
    output_dir="./ACL_Project_Final",
    num_train_epochs=12,
    per_device_train_batch_size=64,
    learning_rate=4e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

trainer = SupersonicTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_train, eval_dataset=tokenized_eval,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# confusion matrix (log-scaled)
print("\nGenerating evaluation artifacts")
eval_results = trainer.predict(tokenized_eval)
y_pred = np.argmax(eval_results.predictions, axis=-1)
y_true = eval_results.label_ids

plt.figure(figsize=(20, 15))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(np.log1p(cm), cmap='YlGnBu')
plt.title("82-class confusion matrix (log-scaled)")
plt.savefig("Confusion_Matrix_EACL.png", dpi=300)
plt.show()

# Submission file
test_preds = trainer.predict(tokenized_test)
final_labels = np.argmax(test_preds.predictions, axis=-1)
submission = pd.DataFrame({"Id": test_df["Id"], "Predicted": final_labels})
submission.to_csv("SUBMISSION.csv", index=False)
print("Done: SUBMISSION.csv written.")
