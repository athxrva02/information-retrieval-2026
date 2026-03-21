"""
run_accuracy_metrics.py
-----------------------
Computes all accuracy metrics (AUC, Precision@K, Recall@K, NDCG@K) for
NTD experiment configs using item_scores.pkl.

Usage:
    # Single config
    python run_accuracy_metrics.py natural_aligned_rdw_5hops

    # Select configs
    python run_accuracy_metrics.py --configs natural_aligned_rdw_5hops optimal_discriminative

    # All configs in experiment_results/
    python run_accuracy_metrics.py --all

    # Custom results directory
    python run_accuracy_metrics.py --all --results-dir ./experiment_results_backup

Results are saved per-config to:
    experiment_results/{config_name}/D_RDW/all_accuracy_metrics.json
A summary across configs is saved to:
    experiment_results/accuracy_summary.json
"""

import os
import sys
import json
import math
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from cornac.eval_methods.base_method import BaseMethod
from cornac.datasets import mind as mind_loader

# ── Resolve repo root (4 levels up from this script) ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "ebnerd_results_existing")
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "experiment_results")

# ── K values for ranking metrics ──────────────────────────────────────────────
K_VALUES = [5, 10, 20]


# ── Metric functions ──────────────────────────────────────────────────────────

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
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked_list[:k], start=1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Data loading (cached across configs) ─────────────────────────────────────

def load_split_and_pool(data_path):
    """Load train/test split and article pool. Returns reusable objects."""
    train_uir_path = os.path.join(data_path, "augmented_uir_top3similar.csv")
    test_uir_path = os.path.join(data_path, "uir_impression_test.csv")
    article_pool_path = os.path.join(data_path, "article_pool.csv")

    feedback_train = mind_loader.load_feedback(fpath=train_uir_path)
    feedback_test = mind_loader.load_feedback(fpath=test_uir_path)

    split = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5,
    )

    impression_items_df = pd.read_csv(article_pool_path, dtype={"iid": str})
    impression_iid_list = impression_items_df["iid"].tolist()

    iid_str2idx = split.global_iid_map
    impression_items_idx = [
        iid_str2idx[iid] for iid in impression_iid_list if iid in iid_str2idx
    ]
    impression_idx2pos = {idx: pos for pos, idx in enumerate(impression_items_idx)}
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}

    # Positive and negative ratings from test set
    positive_ratings = defaultdict(list)
    negative_ratings = defaultdict(set)
    uids, iids, ratings = split.test_set.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r > 0:
            positive_ratings[uid].append(iid)
        else:
            negative_ratings[uid].add(iid)

    return {
        "split": split,
        "impression_items_idx": impression_items_idx,
        "impression_idx2pos": impression_idx2pos,
        "iid_idx2str": iid_idx2str,
        "positive_ratings": dict(positive_ratings),
        "negative_ratings": dict(negative_ratings),
    }


# ── Per-config evaluation ────────────────────────────────────────────────────

def evaluate_config(config_name, cached_data, results_base):
    """Compute all accuracy metrics for a single config. Returns metrics dict or None."""
    save_path = os.path.join(results_base, config_name, "D_RDW")

    if not os.path.exists(save_path):
        print(f"  Skipping '{config_name}': {save_path} does not exist")
        return None

    scores_path = os.path.join(save_path, "item_scores.pkl")
    if not os.path.exists(scores_path):
        print(f"  Skipping '{config_name}': item_scores.pkl not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"{'='*60}")

    with open(scores_path, "rb") as f:
        item_scores = pickle.load(f)

    impression_items_idx = cached_data["impression_items_idx"]
    impression_idx2pos = cached_data["impression_idx2pos"]
    iid_idx2str = cached_data["iid_idx2str"]
    positive_ratings = cached_data["positive_ratings"]
    negative_ratings = cached_data["negative_ratings"]

    # Accumulators
    auc_correct = 0
    auc_total = 0
    prec_sums = {k: 0.0 for k in K_VALUES}
    rec_sums = {k: 0.0 for k in K_VALUES}
    ndcg_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user in tqdm(list(positive_ratings.keys()), desc=f"Evaluating {config_name}"):
        pos_items = positive_ratings.get(user, [])
        neg_items = negative_ratings.get(user, set())

        if not pos_items or not neg_items:
            continue

        user_scores = item_scores.get(user)
        if user_scores is None:
            continue

        # ── AUC (pairwise comparison) ──
        for pos_item in pos_items:
            if pos_item not in impression_idx2pos:
                continue
            pos_score = user_scores[impression_idx2pos[pos_item]]
            for neg_item in neg_items:
                if neg_item not in impression_idx2pos:
                    continue
                neg_score = user_scores[impression_idx2pos[neg_item]]
                if pos_score > neg_score:
                    auc_correct += 1
                auc_total += 1

        # ── Ranking metrics ──
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

    results = {
        "config_name": config_name,
        "auc": auc_score,
        "auc_correct_pairs": auc_correct,
        "auc_total_pairs": auc_total,
        "num_users": num_evaluated,
        "k_values": K_VALUES,
        "precision_at_k": {str(k): prec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "recall_at_k": {str(k): rec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "ndcg_at_k": {str(k): ndcg_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
    }

    # Print summary
    print(f"  AUC = {auc_score:.4f}  ({num_evaluated} users)")
    for k in K_VALUES:
        p = results["precision_at_k"].get(str(k), 0.0)
        r = results["recall_at_k"].get(str(k), 0.0)
        n = results["ndcg_at_k"].get(str(k), 0.0)
        print(f"  P@{k:2d}={p:.4f}  R@{k:2d}={r:.4f}  NDCG@{k:2d}={n:.4f}")

    # Save per-config results
    output_path = os.path.join(save_path, "all_accuracy_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved to {output_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics (AUC, P@K, R@K, NDCG@K) for NTD configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("config", nargs="?", default=None,
                       help="Single config name to evaluate")
    group.add_argument("--configs", nargs="+", metavar="NAME",
                       help="Select configs to evaluate")
    group.add_argument("--all", action="store_true",
                       help="Evaluate all configs in results directory")

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Path to data directory (default: <repo>/ebnerd_results_existing)")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Path to experiment results directory (default: <repo>/experiment_results)")
    return parser.parse_args()


def main():
    args = parse_args()
    results_base = args.results_dir
    data_path = args.data_dir

    # Determine which configs to run
    if args.all:
        if not os.path.exists(results_base):
            print(f"Results directory not found: {results_base}")
            sys.exit(1)
        config_names = sorted(
            d for d in os.listdir(results_base)
            if os.path.isdir(os.path.join(results_base, d))
        )
    elif args.configs:
        config_names = args.configs
    else:
        config_names = [args.config]

    print(f"Data directory:    {data_path}")
    print(f"Results directory: {results_base}")
    print(f"Configs to evaluate: {config_names}")

    # Load data once (shared across all configs)
    print("\nLoading dataset and article pool...")
    cached_data = load_split_and_pool(data_path)

    # Evaluate each config
    all_results = {}
    for config_name in config_names:
        result = evaluate_config(config_name, cached_data, results_base)
        if result is not None:
            all_results[config_name] = result

    if not all_results:
        print("\nNo configs were evaluated.")
        sys.exit(1)

    # Print cross-config summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Config':<40s} {'AUC':>6s}"
    for k in K_VALUES:
        header += f"  {'P@'+str(k):>6s}  {'R@'+str(k):>6s}  {'NDCG@'+str(k):>8s}"
    print(header)
    print("-" * len(header))

    for name in sorted(all_results.keys()):
        r = all_results[name]
        row = f"{name:<40s} {r['auc']:>6.4f}"
        for k in K_VALUES:
            p = r["precision_at_k"].get(str(k), 0.0)
            rc = r["recall_at_k"].get(str(k), 0.0)
            n = r["ndcg_at_k"].get(str(k), 0.0)
            row += f"  {p:>6.4f}  {rc:>6.4f}  {n:>8.4f}"
        print(row)

    # Save summary JSON
    summary = {
        config_name: {
            "auc": r["auc"],
            "num_users": r["num_users"],
            "precision_at_k": r["precision_at_k"],
            "recall_at_k": r["recall_at_k"],
            "ndcg_at_k": r["ndcg_at_k"],
        }
        for config_name, r in all_results.items()
    }
    summary_path = os.path.join(results_base, "accuracy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
