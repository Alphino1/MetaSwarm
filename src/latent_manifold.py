"""Analysis: UMAP of [CLS] embeddings, attention saliency, per-class F1 audit.

From the development notebook, unchanged in behavior. Runs inside the training
session and expects model, tokenizer, tokenized_eval, data_collator, split,
and trainer in scope, exactly as in the notebook where this file follows the
training cells. Part C prints the classes below the 0.40 threshold; that list
is the Tail-42 oversampling list used by the later training stages.
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import plotly.express as px
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix

# --- part A: UMAP of [CLS] embeddings ---
print("Extracting hidden states for manifold analysis...")
model.eval()
all_embeddings = []
all_labels = []

# drop the text column so the collator only sees tensors
eval_dataset_tensors = tokenized_eval.remove_columns(["text"])
eval_dataset_tensors.set_format("torch")

eval_loader = DataLoader(
    eval_dataset_tensors,
    batch_size=32,
    collate_fn=data_collator
)

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Encoding latent space"):
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

        # Access the underlying BERT encoder blocks
        outputs = model.bert(**inputs)

        # CLS embedding (index 0)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(batch['labels'].cpu().numpy())

embeddings_concat = np.concatenate(all_embeddings)
labels_concat = np.concatenate(all_labels)

# UMAP projection of the 82-class latent space
reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
projections = reducer.fit_transform(embeddings_concat)

df_viz = pd.DataFrame({
    'Dimension 1': projections[:, 0],
    'Dimension 2': projections[:, 1],
    'Category': labels_concat.astype(str)
})

fig = px.scatter(df_viz, x='Dimension 1', y='Dimension 2', color='Category',
                 title="Semantic clustering of 82 medical classes (UMAP of [CLS] embeddings)",
                 labels={'Category': 'Medical class ID'})
fig.update_traces(marker=dict(size=5))
fig.show()

# --- Part B: attention saliency ---
def plot_attention_map(sample_idx):
    # Retrieve raw text from the split (string column still present there)
    raw_text = split["test"][sample_idx]["text"]
    inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=128).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # last layer, mean over heads
    attention = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    plt.figure(figsize=(10, 8))
    limit = min(30, len(tokens))
    sns.heatmap(attention[:limit, :limit], xticklabels=tokens[:limit], yticklabels=tokens[:limit],
                cmap='magma')
    plt.title("Last-layer token attention map (Arabic medical query)")
    plt.show()

print("\nVisualizing attention focus for a medical query...")
plot_attention_map(0)

# --- part C: per-class F1 audit (Tail-42) ---
preds_eval = trainer.predict(tokenized_eval)
y_pred = np.argmax(preds_eval.predictions, axis=-1)
y_true = labels_concat

f1_per_class = f1_score(y_true, y_pred, average=None)

plt.figure(figsize=(15, 6))
plt.bar(range(len(f1_per_class)), f1_per_class, color='teal')
plt.axhline(y=0.4, color='r', linestyle='--', label='Diagnostic threshold (0.40)')
plt.title("Per-class F1 across 82 categories")
plt.xlabel("Class ID")
plt.ylabel("F1 score")
plt.legend()
plt.show()

low_perf_indices = np.where(f1_per_class < 0.4)[0]
print("\n--- Per-class F1 audit ---")
print(f"Classes below the 0.40 threshold ({len(low_perf_indices)}): {low_perf_indices.tolist()}")
