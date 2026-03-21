## New scripts

### `compute_precision_recall.py`
Computes **Precision@K** and **Recall@K** from a ranked recommendation file (the `.txt` format your models already produce). Set `K_VALUES`, point it at your recommendation file and the input/output paths, then run.

### `compute_ndcg.py`
Computes **NDCG@K** using binary relevance (clicked = 1, not-clicked = 0). Same file-based interface as above.

### `compute_all_accuracy_metrics.py` ← recommended starting point
A combined script with **two modes**:

| Mode | `USE_ITEM_SCORES_PKL` | What it computes |
|------|----------------------|-----------------|
| **A** | `True` | AUC + Precision@K + Recall@K + NDCG@K — uses `item_scores.pkl` directly (same data as existing AUC script, so DRDW works immediately) |
| **B** | `False` | Precision@K + Recall@K + NDCG@K — reads a ranked `.txt` recommendation file (for NRMS, NPA, LSTUR, etc.) |

---

## Steps to run

**For models with `item_scores.pkl` (e.g., DRDW):**
```bash
cd experiments/recsys_2025/evaluation_scripts/check_accuracy

# Edit compute_all_accuracy_metrics.py:
#   USE_ITEM_SCORES_PKL = True
#   SAVE_PATH = "./experiment_ebnerd_drdw_results/D_RDW"
#   INPUT_PATH = "./ebnerd_results_existing"

python compute_all_accuracy_metrics.py
# → saves: <SAVE_PATH>/all_accuracy_metrics.json
```

**For models with ranked `.txt` files (e.g., NRMS, NPA, LSTUR):**
```bash
# Edit compute_all_accuracy_metrics.py:
#   USE_ITEM_SCORES_PKL = False
#   RECOMMENDATION_FILE = "path/to/NRMS_ebnerd_recom_top20.txt"
#   INPUT_PATH = "./ebnerd_results_existing"

python compute_all_accuracy_metrics.py
# → saves: <SAVE_PATH>/all_accuracy_metrics.json
```

Or use the standalone scripts for just one metric:
```bash
python compute_precision_recall.py   # Precision@K + Recall@K
python compute_ndcg.py               # NDCG@K
```