"""
compute_ndcg.py
---------------
Computes NDCG@K (Normalized Discounted Cumulative Gain) for recommendation lists.

The recommendation file format (same as all models in this project) is:
    <userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>

Items are ordered by descending predicted score (rank 1 = best).

The test set (uir_impression_test.csv) is used to derive binary relevance labels
(r > 0 ⇒ relevant).  The train set is loaded only to build the consistent
cornac BaseMethod split with the correct internal user/item indices.

NDCG formula used (binary relevance):
    DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i + 1)
    IDCG@K = DCG of ideal ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K

Usage
-----
Adjust the four path variables at the top of this script, then run:

    cd experiments/recsys_2025/evaluation_scripts/check_accuracy
    python compute_ndcg.py

Outputs
-------
A JSON file  <save_path>/ndcg_results.json  containing:
  - ndcg_at_k   : {k: mean NDCG@k over all evaluated users}
  - k_values    : list of K values evaluated
  - num_users   : number of users included in the evaluation
"""

import os
import json
import math
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

# Directory that contains augmented_uir_top3similar.csv and uir_impression_test.csv
INPUT_PATH = "./ebnerd_results_existing"

# Directory where results will be saved
SAVE_PATH = "./experiment_ebnerd_drdw_results/D_RDW"

# K values for which NDCG@K will be computed
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


def dcg_at_k(recommended_items, relevant_set, k):
    """Compute DCG@K with binary relevance.

    Parameters
    ----------
    recommended_items : list of str
        Ordered item IDs (best first).
    relevant_set : set of str
        Ground-truth relevant item IDs.
    k : int
        Cut-off.

    Returns
    -------
    float
        DCG@K score.
    """
    score = 0.0
    for rank, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_set:
            score += 1.0 / math.log2(rank + 1)
    return score


def idcg_at_k(num_relevant, k):
    """Compute Ideal DCG@K (assumes top-min(num_relevant, k) items are all relevant)."""
    ideal_hits = min(num_relevant, k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))


def ndcg_at_k(recommended_items, relevant_set, k):
    """NDCG@K with binary relevance."""
    if not relevant_set:
        return 0.0
    ideal = idcg_at_k(len(relevant_set), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(recommended_items, relevant_set, k) / ideal


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

    # Build lookup tables
    uid_str2idx = split.global_uid_map
    iid_str2idx = split.global_iid_map
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}

    # Ground-truth positives (internal indices)
    positive_items_by_idx = get_user_positive_items(split.test_set)

    # ── Load recommendations ───────────────────────────────────────────────────
    recs = load_recommendations(RECOMMENDATION_FILE)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    ndcg_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user_str, rec_item_strs in tqdm(recs.items(), desc="Evaluating users"):
        if user_str not in uid_str2idx:
            continue
        user_idx = uid_str2idx[user_str]

        pos_idx_list = positive_items_by_idx.get(user_idx, [])
        if not pos_idx_list:
            continue

        pos_str_set = {iid_idx2str[idx] for idx in pos_idx_list if idx in iid_idx2str}
        if not pos_str_set:
            continue

        for k in K_VALUES:
            ndcg_sums[k] += ndcg_at_k(rec_item_strs, pos_str_set, k)

        num_evaluated += 1

    if num_evaluated == 0:
        print("No users were evaluated. Check that user IDs in the recommendation "
              "file match those in the train/test splits.")
        return

    # ── Aggregate and print ────────────────────────────────────────────────────
    ndcg_at_k_scores = {k: ndcg_sums[k] / num_evaluated for k in K_VALUES}

    print(f"\nResults over {num_evaluated} users:")
    for k in K_VALUES:
        print(f"  NDCG@{k:2d} = {ndcg_at_k_scores[k]:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "k_values": K_VALUES,
        "num_users": num_evaluated,
        "ndcg_at_k": {str(k): v for k, v in ndcg_at_k_scores.items()},
    }
    output_path = os.path.join(SAVE_PATH, "ndcg_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
