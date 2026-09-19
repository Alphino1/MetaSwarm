"""Stage 1 development run: weighted cross-entropy (Eq. 1), label smoothing 0.1.

From the development notebook for the AbjadNLP 2026 paper, unchanged in
behavior. Expects shared_task_train.csv (text, label) and
shared_task_devtest_no_label.csv (Id, text) in the working directory; writes
convergence_plot.png and submit_preds.csv. CAMeLBERT-DA, max length 256,
batch 32, 10 epochs, LR 3e-5, 200 warmup steps, weight decay 0.01, fp16,
seed 42, 90/10 split.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Data
train_df = pd.read_csv("shared_task_train.csv")
test_df = pd.read_csv("shared_task_devtest_no_label.csv")

# class weights, Eq. 1 (inverse-log frequency)
class_counts = train_df['label'].value_counts().sort_index().values
weights = 1.0 / np.log1p(class_counts)
weights = torch.tensor(weights, dtype=torch.float).to("cuda")

num_labels = 82

dataset = Dataset.from_pandas(train_df[['text', 'label']])
# 90/10 validation split, seed 42
split = dataset.train_test_split(test_size=0.1, seed=42)

model_name = "CAMEL-Lab/bert-base-arabic-camelbert-da"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding=True)

tokenized_train = split["train"].map(preprocess_function, batched=True)
tokenized_eval = split["test"].map(preprocess_function, batched=True)
test_dataset = Dataset.from_pandas(test_df[['text']]).map(preprocess_function, batched=True)

# Eq. 1: weighted CE
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # macro-F1
    f1 = f1_score(labels, predictions, average="macro")
    return {"macro_f1": f1}

training_args = TrainingArguments(
    output_dir="./medical_abjad_results",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_steps=200,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,
    logging_steps=50,
    report_to="none"
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

# convergence plot
log_history = trainer.state.log_history
epochs = [x['epoch'] for x in log_history if 'eval_macro_f1' in x]
f1_scores = [x['eval_macro_f1'] for x in log_history if 'eval_macro_f1' in x]

plt.figure(figsize=(10, 5))
sns.lineplot(x=epochs, y=f1_scores, marker='o')
plt.title("Validation macro-F1 by epoch")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.grid(True)
plt.savefig("convergence_plot.png")
plt.show()

# Submission file for the unlabeled evaluation split
print("\nGenerating submission file")
preds_output = trainer.predict(test_dataset)
final_preds = np.argmax(preds_output.predictions, axis=-1)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "Predicted": final_preds
})

submission.to_csv("submit_preds.csv", index=False)
print("Done: submit_preds.csv written.")
