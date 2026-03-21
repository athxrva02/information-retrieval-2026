"""
compute_all_accuracy_metrics.py
--------------------------------
Computes AUC, Precision@K, Recall@K, and NDCG@K in one pass.

This script can operate in two modes, controlled by USE_ITEM_SCORES_PKL:

  MODE A  (USE_ITEM_SCORES_PKL = True)
    Uses item_scores.pkl (a dict {user_idx: array-of-scores-over-impression-pool})
    to rank items, then evaluates all metrics against the test set.
    This is identical to how compute_auc.py works and is the primary mode for
    models that save raw score vectors (e.g., DRDW).

  MODE B  (USE_ITEM_SCORES_PKL = False)
    Reads ranked recommendation lists from a text file in the format:
        <userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>
    The ordering is used directly for ranking-based metrics.
    AUC cannot be computed in this mode (no continuous scores), so it is skipped.

Usage
-----
1. Set the paths and mode at the top of this script.
2. Run:
       cd experiments/recsys_2025/evaluation_scripts/check_accuracy
       python compute_all_accuracy_metrics.py

Outputs
-------
<save_path>/all_accuracy_metrics.json
"""

import os
import json
import math
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from cornac.eval_methods.base_method import BaseMethod
from cornac.datasets import mind as mind_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  – edit these variables before running
# ═══════════════════════════════════════════════════════════════════════════════

# Set True to use item_scores.pkl (MODE A), False to use a ranked recs file (MODE B)
USE_ITEM_SCORES_PKL = True

dataset_name = "ebnerd"

# ── Common paths ───────────────────────────────────────────────────────────────
# Directory that contains article_pool.csv (only needed in MODE A),
# augmented_uir_top3similar.csv, and uir_impression_test.csv
INPUT_PATH = f"./{dataset_name}_results_existing"

# Directory where results will be saved
SAVE_PATH = f"./experiment_{dataset_name}_drdw_results/D_RDW"

# ── MODE A paths ───────────────────────────────────────────────────────────────
# Path to the item-scores pickle file (dict {user_idx: np.ndarray})
ITEM_SCORES_PKL = os.path.join(SAVE_PATH, "item_scores.pkl")

# Path to the article pool CSV (must have a column named 'iid')
ARTICLE_POOL_CSV = os.path.join(INPUT_PATH, "article_pool.csv")

# ── MODE B path ────────────────────────────────────────────────────────────────
# Path to the ranked recommendation text file
RECOMMENDATION_FILE = "path/to/MODEL_ebnerd_recom_top20.txt"

# ── K values ───────────────────────────────────────────────────────────────────
K_VALUES = [5, 10, 20]
# ═══════════════════════════════════════════════════════════════════════════════


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_recommendations(filepath):
    """Load ranked rec lists from a text file → dict {user_id_str: [item_id_str, ...]}"""
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
    """Return {user_idx: [item_idx, ...]} for r > 0 entries in dataset."""
    result = defaultdict(list)
    uids, iids, ratings = dataset.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r > 0:
            result[uid].append(iid)
    return dict(result)


def get_user_negative_items(dataset):
    """Return {user_idx: set(item_idx)} for r <= 0 entries in dataset."""
    result = defaultdict(set)
    uids, iids, ratings = dataset.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r <= 0:
            result[uid].add(iid)
    return dict(result)


# ── Metric functions ────────────────────────────────────────────────────────────

def precision_at_k(ranked_list, relevant_set, k):
    top_k = ranked_list[:k]
    return len(set(top_k) & relevant_set) / k if k > 0 else 0.0


def recall_at_k(ranked_list, relevant_set, k):
    if not relevant_set:
        return 0.0
    top_k = ranked_list[:k]
    return len(set(top_k) & relevant_set) / len(relevant_set)


def ndcg_at_k(ranked_list, relevant_set, k):
    if not relevant_set:
        return 0.0
    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked_list[:k], start=1)
        if item in relevant_set
    )
    # IDCG
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Mode A: item-scores pkl ────────────────────────────────────────────────────

def evaluate_mode_a(split, positive_ratings, negative_ratings):
    """Full evaluation using item_scores.pkl (includes AUC)."""

    # Load impression pool
    impression_items_df = pd.read_csv(ARTICLE_POOL_CSV, dtype={"iid": str})
    impression_iid_list = impression_items_df["iid"].tolist()

    iid_str2idx = split.global_iid_map
    impression_items_idx = [
        iid_str2idx[iid] for iid in impression_iid_list if iid in iid_str2idx
    ]
    impression_idx2pos = {idx: pos for pos, idx in enumerate(impression_items_idx)}
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}

    # Load item scores
    with open(ITEM_SCORES_PKL, "rb") as f:
        item_scores = pickle.load(f)

    # Accumulators
    auc_correct = 0
    auc_total = 0
    user_auc_counts = {}
    prec_sums = {k: 0.0 for k in K_VALUES}
    rec_sums = {k: 0.0 for k in K_VALUES}
    ndcg_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user in tqdm(list(positive_ratings.keys()), desc="Evaluating (Mode A)"):
        pos_items = positive_ratings.get(user, [])
        neg_items = negative_ratings.get(user, set())

        if not pos_items or not neg_items:
            continue

        user_scores = item_scores.get(user)
        if user_scores is None:
            continue

        # ── AUC ──
        u_correct = 0
        u_total = 0
        for pos_item in pos_items:
            if pos_item not in impression_idx2pos:
                continue
            pos_score = user_scores[impression_idx2pos[pos_item]]
            for neg_item in neg_items:
                if neg_item not in impression_idx2pos:
                    continue
                neg_score = user_scores[impression_idx2pos[neg_item]]
                if pos_score > neg_score:
                    u_correct += 1
                    auc_correct += 1
                u_total += 1
                auc_total += 1
        user_auc_counts[user] = {"correct": u_correct, "total": u_total}

        # ── Ranking metrics ──
        # Sort impression items by score descending to get ranked list
        scored_pairs = [
            (impression_items_idx[i], float(user_scores[i]))
            for i in range(len(impression_items_idx))
        ]
        ranked_idx = [idx for idx, _ in sorted(scored_pairs, key=lambda x: -x[1])]
        ranked_strs = [iid_idx2str[idx] for idx in ranked_idx if idx in iid_idx2str]

        pos_str_set = {iid_idx2str[idx] for idx in pos_items if idx in iid_idx2str}
        if not pos_str_set:
            continue

        for k in K_VALUES:
            prec_sums[k] += precision_at_k(ranked_strs, pos_str_set, k)
            rec_sums[k] += recall_at_k(ranked_strs, pos_str_set, k)
            ndcg_sums[k] += ndcg_at_k(ranked_strs, pos_str_set, k)

        num_evaluated += 1

    auc_score = auc_correct / auc_total if auc_total > 0 else 0.0
    return {
        "auc": auc_score,
        "auc_correct_pairs": auc_correct,
        "auc_total_pairs": auc_total,
        "num_users": num_evaluated,
        "precision_at_k": {str(k): prec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "recall_at_k": {str(k): rec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "ndcg_at_k": {str(k): ndcg_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "user_auc_counts": {str(k): v for k, v in user_auc_counts.items()},
    }


# ── Mode B: ranked recommendation file ────────────────────────────────────────

def evaluate_mode_b(split, positive_ratings):
    """Evaluation from a ranked recommendation text file (no AUC)."""

    iid_str2idx = split.global_iid_map
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}
    uid_str2idx = split.global_uid_map

    recs = load_recommendations(RECOMMENDATION_FILE)

    prec_sums = {k: 0.0 for k in K_VALUES}
    rec_sums = {k: 0.0 for k in K_VALUES}
    ndcg_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user_str, rec_items in tqdm(recs.items(), desc="Evaluating (Mode B)"):
        if user_str not in uid_str2idx:
            continue
        user_idx = uid_str2idx[user_str]
        pos_idx_list = positive_ratings.get(user_idx, [])
        if not pos_idx_list:
            continue
        pos_str_set = {iid_idx2str[idx] for idx in pos_idx_list if idx in iid_idx2str}
        if not pos_str_set:
            continue

        for k in K_VALUES:
            prec_sums[k] += precision_at_k(rec_items, pos_str_set, k)
            rec_sums[k] += recall_at_k(rec_items, pos_str_set, k)
            ndcg_sums[k] += ndcg_at_k(rec_items, pos_str_set, k)

        num_evaluated += 1

    return {
        "auc": "N/A (no item scores available in Mode B)",
        "num_users": num_evaluated,
        "precision_at_k": {str(k): prec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "recall_at_k": {str(k): rec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "ndcg_at_k": {str(k): ndcg_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load train/test split
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

    positive_ratings = get_user_positive_items(split.test_set)

    if USE_ITEM_SCORES_PKL:
        negative_ratings = get_user_negative_items(split.test_set)
        results = evaluate_mode_a(split, positive_ratings, negative_ratings)
    else:
        results = evaluate_mode_b(split, positive_ratings)

    # ── Print summary ──────────────────────────────────────────────────────────
    n = results["num_users"]
    print(f"\n{'='*50}")
    print(f"Results over {n} users  (Mode {'A - item_scores.pkl' if USE_ITEM_SCORES_PKL else 'B - ranked recs file'}):")
    print(f"{'='*50}")
    if USE_ITEM_SCORES_PKL:
        print(f"  AUC = {results['auc']:.4f}")
    for k in K_VALUES:
        print(f"  Precision@{k:2d} = {results['precision_at_k'].get(str(k), 0.0):.4f}")
        print(f"  Recall@{k:2d}    = {results['recall_at_k'].get(str(k), 0.0):.4f}")
        print(f"  NDCG@{k:2d}      = {results['ndcg_at_k'].get(str(k), 0.0):.4f}")

    results["k_values"] = K_VALUES
    output_path = os.path.join(SAVE_PATH, "all_accuracy_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
