# Data

The AbjadMed shared task data is not redistributed in this repository. Obtain it
from the AbjadMed shared task / AbjadNLP 2026 organizers:

- Workshop page: https://wp.lancs.ac.uk/abjad/abjadnlp2026/
- Proceedings: https://aclanthology.org/events/abjadnlp-2026/

Place the following files in the working directory before running any script:

| File | Columns | Used by |
|---|---|---|
| `shared_task_train.csv` | `text`, `label` | all training scripts (82-class labels, integers 0-81) |
| `shared_task_devtest_no_label.csv` | `Id`, `text` | evaluation/submission generation |

Notes:

- All splits in this codebase are derived from `shared_task_train.csv` only
  (90/10 split with seed 42 in the single-run stages; stratified 5-fold with
  seed 42 in the ensemble). The unlabeled evaluation file is used only for
  generating predictions.
- The per-class F1 audit in `src/latent_manifold.py` is what produces the
  42-class oversampling list (`weak_indices`) used by the later stages. In this
  release the list is written out explicitly in the scripts, matching the runs
  reported in the paper.
