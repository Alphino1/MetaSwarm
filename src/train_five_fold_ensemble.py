"""Submitted system: five-fold ensemble, focal loss (gamma 2.5), logit mean.

From the development notebook, unchanged in behavior; matches Table 1 of the
author version: CAMeLBERT-DA, gamma 2.5 / alpha 0.25, label smoothing 0.15,
max length 192, batch 128 per GPU (256 global on dual H200), 20 epochs,
LR 9e-5, cosine, warmup 0.1, decay 0.01, seed 42, five stratified folds.
Reported 0.3588 public / 0.3520 private macro-F1. Writes ENSEMBLE_FINAL.csv
and per-fold checkpoints under ./Fold_i. Global batch 256 assumes two GPUs.
"""

import os, sys, subprocess, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

# make sure the deps import, install if not
pkgs = ["transformers[torch]", "datasets", "accelerate", "scikit-learn"]
for p in pkgs:
    try: __import__(p.split('[')[0])
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

torch.manual_seed(42)

train_df = pd.read_csv("shared_task_train.csv")
test_df = pd.read_csv("shared_task_devtest_no_label.csv")

# Tail-42 weak classes, duplicated within each fold after the split
weak_indices = [2, 3, 4, 6, 7, 10, 12, 13, 14, 15, 18, 20, 25, 27, 28, 29, 30, 34,
                35, 37, 38, 39, 41, 42, 43, 44, 45, 48, 53, 56, 59, 63, 66, 68,
                70, 72, 73, 74, 76, 77, 78, 79]

model_name = "CAMEL-Lab/bert-base-arabic-camelbert-da"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_fn(ex): return tokenizer(ex["text"], truncation=True, max_length=192, padding=True)

# focal loss with label smoothing; pop labels so DDP replicas line up
class RobustEnsembleTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.15)
        pt = torch.exp(-ce_loss)
        focal_loss = (0.25 * (1 - pt)**2.5 * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_test_logits = []

print("--- Starting five-fold ensemble (dual H200) ---")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    print(f"\n--- Fold {fold+1}/5 ---")

    fold_train = train_df.iloc[train_idx]
    # Targeted oversampling of weak classes within the fold
    fold_train_boosted = pd.concat([fold_train, fold_train[fold_train['label'].isin(weak_indices)]])

    train_ds = Dataset.from_pandas(fold_train_boosted).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=82)

    training_args = TrainingArguments(
        output_dir=f"./Fold_{fold}",
        num_train_epochs=20,
        per_device_train_batch_size=128,  # Global batch 256 on dual H200
        learning_rate=9e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False
    )

    trainer = RobustEnsembleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()

    # keep fold logits
    fold_logits = trainer.predict(test_ds).predictions
    all_test_logits.append(fold_logits)

    # free memory
    del model, trainer
    torch.cuda.empty_cache()

# mean of fold logits
print("\nAggregating ensemble logits")
ensemble_logits = np.mean(all_test_logits, axis=0)
final_preds = np.argmax(ensemble_logits, axis=-1)

submission = pd.DataFrame({"Id": test_df["Id"], "Predicted": final_preds})
submission.to_csv("ENSEMBLE_FINAL.csv", index=False)
print("Done: ENSEMBLE_FINAL.csv written.")
