# Accuracy Metrics — Evaluation Scripts

Scripts for computing offline accuracy metrics on news recommendation models.
All scripts live in `experiments/recsys_2025/evaluation_scripts/check_accuracy/`.

---

## Scripts overview

| Script | Purpose |
|--------|---------|
| `run_accuracy_metrics.py` | **Primary entrypoint.** CLI runner for AUC + P@K + R@K + NDCG@K. Supports both `item_scores.pkl` (Mode A) and ranked `.txt` files (Mode B). Evaluates one config, a selection, or all configs in one go. |
| `compute_auc.py` | Legacy script — computes AUC only from `item_scores.pkl`. Superseded by `run_accuracy_metrics.py`. |
| `compute_precision_recall.py` | Standalone — Precision@K and Recall@K from a ranked `.txt` file. |
| `compute_ndcg.py` | Standalone — NDCG@K from a ranked `.txt` file. |

---

## Metrics computed

| Metric | Description |
|--------|-------------|
| **AUC** | Pairwise accuracy: fraction of (positive, negative) impression pairs where the positive item is scored higher. Only available in Mode A. |
| **Precision@K** | Fraction of the top-K recommended items that are relevant (clicked). |
| **Recall@K** | Fraction of all relevant items that appear in the top-K recommendations. |
| **NDCG@K** | Normalised Discounted Cumulative Gain at K. Uses binary relevance (click = 1, no-click = 0) with log₂ discount. |

K values default to `[5, 10, 20]`.

---

## Two evaluation modes

### Mode A — `item_scores.pkl` (D-RDW and similar models)

The model saves a dict `{user_idx: np.ndarray}` where each array holds one raw score per item in the impression pool. The script ranks items by score and evaluates all four metrics including AUC.

**Required files per config:**
```
experiment_results/{config_name}/D_RDW/item_scores.pkl
ebnerd_results_existing/article_pool.csv
ebnerd_results_existing/augmented_uir_top3similar.csv
ebnerd_results_existing/uir_impression_test.csv
```

### Mode B — ranked `.txt` recommendation file (NRMS, NPA, LSTUR, etc.)

The model produces a pre-ranked list per user. AUC is not available (no continuous scores).

**Expected file format:**
```
<userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>
```
Items must be ordered best-first; the script uses the list order directly.

**Required files:**
```
path/to/MODEL_ebnerd_recom_top20.txt
ebnerd_results_existing/augmented_uir_top3similar.csv
ebnerd_results_existing/uir_impression_test.csv
```

---

## Run Accuracy Metrics

Navigate to the `check_accuracy` directory

```bash
cd experiments/recsys_2025/evaluation_scripts/check_accuracy
```

### Mode A (default) — single config
```bash
python run_accuracy_metrics.py natural_aligned_rdw_5hops
# → saves: experiment_results/natural_aligned_rdw_5hops/D_RDW/all_accuracy_metrics.json
```

### Mode A — select configs
```bash
python run_accuracy_metrics.py --configs natural_aligned_rdw_5hops optimal_discriminative
```

### Mode A — all configs in results directory
```bash
python run_accuracy_metrics.py --all
```

### Mode A — custom data and results paths
```bash
python run_accuracy_metrics.py --all \
    --data-dir ./ebnerd_results_existing \
    --results-dir ./experiment_results_backup
```

### Mode B — single config with explicit rec file
```bash
python run_accuracy_metrics.py nrms_baseline \
    --mode txt \
    --rec-file path/to/nrms_ebnerd_recom_top20.txt
```

### Mode B — all configs, auto-discover .txt files
```bash
# Looks for a .txt under <rec-dir>/<config_name>/ for each config
python run_accuracy_metrics.py --all \
    --mode txt \
    --rec-dir ./final_recommendations/top20_ebnerd
```

### All CLI options
```
positional:
  config                Single config name to evaluate

optional:
  --configs NAME [NAME ...]   Select configs to evaluate
  --all                       Evaluate all configs in results directory
  --mode {pkl,txt}            pkl = Mode A (default), txt = Mode B
  --data-dir PATH             Data directory with CSVs (default: <repo>/ebnerd_results_existing)
  --results-dir PATH          Experiment results directory (default: <repo>/experiment_results)
  --rec-file PATH             [Mode B] Path to ranked recommendation .txt file
  --rec-dir PATH              [Mode B] Directory to auto-discover per-config .txt files
```

---

## Output files

### Per-config result
Saved to `experiment_results/{config_name}/D_RDW/all_accuracy_metrics.json`:

```json
{
    "config_name": "natural_aligned_rdw_5hops",
    "mode": "A",
    "auc": 0.6842,
    "auc_correct_pairs": 1482301,
    "auc_total_pairs": 2167450,
    "num_users": 4312,
    "k_values": [5, 10, 20],
    "precision_at_k": { "5": 0.1823, "10": 0.1541, "20": 0.1204 },
    "recall_at_k":    { "5": 0.2107, "10": 0.3215, "20": 0.4891 },
    "ndcg_at_k":      { "5": 0.2341, "10": 0.2518, "20": 0.2904 }
}
```

Mode B results have `"auc": "N/A (no item scores in Mode B)"` and omit the pair counts.

### Cross-config summary
Saved to `experiment_results/accuracy_summary.json` — one entry per evaluated config,
useful for comparing models side-by-side.

---

## Standalone scripts

For quick single-metric runs on a ranked `.txt` file:

```bash
# Precision@K + Recall@K
python compute_precision_recall.py

# NDCG@K
python compute_ndcg.py
```

Edit the path variables at the top of each script before running.
