"""
compute_precision_recall.py
----------------------------
Computes Precision@K and Recall@K for recommendation lists.

The recommendation file format (same as all models in this project) is:
    <userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>

Items are ordered by descending predicted score (rank 1 = best).

The test set (uir_impression_test.csv) is loaded via cornac to extract ground-truth
positive items per user.  The train set (augmented_uir_top3similar.csv) is loaded only
to build the cornac BaseMethod split so internal user/item indices are consistent.

Usage
-----
Adjust the four path variables at the top of this script, then run:

    cd experiments/recsys_2025/evaluation_scripts/check_accuracy
    python compute_precision_recall.py

Outputs
-------
A JSON file  <save_path>/precision_recall_results.json  containing:
  - precision_at_k  : {k: mean Precision@k over all evaluated users}
  - recall_at_k     : {k: mean Recall@k over all evaluated users}
  - k_values        : list of K values evaluated
  - num_users       : number of users included in the evaluation
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from cornac.eval_methods.base_method import BaseMethod
from cornac.datasets import mind as mind_loader


# ── PATHS ──────────────────────────────────────────────────────────────────────
# Path to the recommendation file produced by your model.
# Format per line:  <userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>
RECOMMENDATION_FILE = "path/to/MODEL_ebnerd_recom_top20.txt"

# Directory that contains article_pool.csv,
# augmented_uir_top3similar.csv  and  uir_impression_test.csv
INPUT_PATH = "./ebnerd_results_existing"

# Directory where results will be saved
SAVE_PATH = "./experiment_ebnerd_drdw_results/D_RDW"

# K values for which Precision@K and Recall@K will be computed
K_VALUES = [5, 10, 20]
# ──────────────────────────────────────────────────────────────────────────────


def load_recommendations(filepath):
    """Load ranked recommendation lists from a text file.

    Returns
    -------
    dict  {user_id_str: [item_id_str, ...]}   (items ordered best-first)
    """
    recommendations = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user_part, items_part = line.split(":", 1)
            user_id = user_part.strip()
            items = [x.strip() for x in items_part.split(",") if x.strip()]
            recommendations[user_id] = items
    return recommendations


def get_user_positive_items(dataset):
    """Return {user_idx: [item_idx, ...]} for positively-rated (r > 0) entries."""
    user_positives = defaultdict(list)
    uids, iids, ratings = dataset.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r > 0:
            user_positives[uid].append(iid)
    return dict(user_positives)


def compute_precision_at_k(recommended_items, relevant_items, k):
    """Precision@K = |relevant ∩ top-K recommended| / K"""
    top_k = recommended_items[:k]
    hits = len(set(top_k) & set(relevant_items))
    return hits / k if k > 0 else 0.0


def compute_recall_at_k(recommended_items, relevant_items, k):
    """Recall@K = |relevant ∩ top-K recommended| / |relevant|"""
    if not relevant_items:
        return 0.0
    top_k = recommended_items[:k]
    hits = len(set(top_k) & set(relevant_items))
    return hits / len(relevant_items)


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    train_uir_path = os.path.join(INPUT_PATH, "augmented_uir_top3similar.csv")
    test_uir_path = os.path.join(INPUT_PATH, "uir_impression_test.csv")

    feedback_train = mind_loader.load_feedback(fpath=train_uir_path)
    feedback_test = mind_loader.load_feedback(fpath=test_uir_path)

    split = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5,
    )

    # Build lookup tables: string id ↔ cornac internal index
    uid_str2idx = split.global_uid_map          # {str_id: int_idx}
    iid_str2idx = split.global_iid_map          # {str_id: int_idx}
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}

    # Ground-truth positives (internal indices)
    positive_items_by_idx = get_user_positive_items(split.test_set)

    # ── Load recommendations ───────────────────────────────────────────────────
    recs = load_recommendations(RECOMMENDATION_FILE)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    precision_sums = {k: 0.0 for k in K_VALUES}
    recall_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user_str, rec_item_strs in tqdm(recs.items(), desc="Evaluating users"):
        # Map user string ID → internal index
        if user_str not in uid_str2idx:
            continue
        user_idx = uid_str2idx[user_str]

        # Ground-truth positive items as string IDs (for direct comparison)
        pos_idx_list = positive_items_by_idx.get(user_idx, [])
        if not pos_idx_list:
            continue  # Skip users with no positive items in test set

        # Convert positive item indices back to string IDs
        pos_str_set = {iid_idx2str[idx] for idx in pos_idx_list if idx in iid_idx2str}
        if not pos_str_set:
            continue

        for k in K_VALUES:
            precision_sums[k] += compute_precision_at_k(rec_item_strs, pos_str_set, k)
            recall_sums[k] += compute_recall_at_k(rec_item_strs, pos_str_set, k)

        num_evaluated += 1

    if num_evaluated == 0:
        print("No users were evaluated. Check that user IDs in the recommendation "
              "file match those in the train/test splits.")
        return

    # ── Aggregate and print ────────────────────────────────────────────────────
    precision_at_k = {k: precision_sums[k] / num_evaluated for k in K_VALUES}
    recall_at_k = {k: recall_sums[k] / num_evaluated for k in K_VALUES}

    print(f"\nResults over {num_evaluated} users:")
    for k in K_VALUES:
        print(f"  Precision@{k:2d} = {precision_at_k[k]:.4f}")
        print(f"  Recall@{k:2d}    = {recall_at_k[k]:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "k_values": K_VALUES,
        "num_users": num_evaluated,
        "precision_at_k": {str(k): v for k, v in precision_at_k.items()},
        "recall_at_k": {str(k): v for k, v in recall_at_k.items()},
    }
    output_path = os.path.join(SAVE_PATH, "precision_recall_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
