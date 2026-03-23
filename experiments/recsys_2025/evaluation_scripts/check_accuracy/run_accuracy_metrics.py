"""
run_accuracy_metrics.py
-----------------------
Computes all accuracy metrics (AUC, Precision@K, Recall@K, NDCG@K) for
experiment configs.

Two modes are supported, selected via --mode:

  MODE A  (default, --mode pkl)
    Uses item_scores.pkl — a dict {user_idx: array-of-scores-over-impression-pool} —
    to rank items, then evaluates all metrics (AUC + P@K + R@K + NDCG@K) against the
    test set. Designed for models that save raw score vectors (e.g., D-RDW).
    Supports multi-config evaluation with a shared data-loading step.

  MODE B  (--mode txt)
    Reads ranked recommendation lists from a text file per config:
        <userID>: <itemID_1>, <itemID_2>, ..., <itemID_N>
    Evaluates P@K, R@K, and NDCG@K. AUC is skipped (no continuous scores).
    Use --rec-file to point at a single file, or --rec-dir to auto-discover
    one .txt file per config folder.

Usage
-----
    # Single config — Mode A (item_scores.pkl)
    python run_accuracy_metrics.py natural_aligned_rdw_5hops

    # Single config — Mode B (ranked .txt file)
    python run_accuracy_metrics.py nrms_baseline --mode txt --rec-file path/to/nrms_ebnerd_recom_top20.txt

    # All configs — Mode A
    python run_accuracy_metrics.py --all

    # All configs — Mode B, auto-discover .txt files under each config folder
    python run_accuracy_metrics.py --all --mode txt --rec-dir ./experiment_results

    # Select configs — Mode A, custom directories
    python run_accuracy_metrics.py --configs config_a config_b \\
        --data-dir ./ebnerd_results_existing \\
        --results-dir ./experiment_results

Results
-------
Per-config:   experiment_results/{config_name}/D_RDW/all_accuracy_metrics.json
Summary:      experiment_results/accuracy_summary.json
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


# ── Data loading (shared across configs in Mode A) ────────────────────────────

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


def load_split_only(data_path):
    """Load train/test split without article pool. Used in Mode B."""
    train_uir_path = os.path.join(data_path, "augmented_uir_top3similar.csv")
    test_uir_path = os.path.join(data_path, "uir_impression_test.csv")

    feedback_train = mind_loader.load_feedback(fpath=train_uir_path)
    feedback_test = mind_loader.load_feedback(fpath=test_uir_path)

    split = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5,
    )

    iid_str2idx = split.global_iid_map
    iid_idx2str = {v: k for k, v in iid_str2idx.items()}
    uid_str2idx = split.global_uid_map

    positive_ratings = defaultdict(list)
    uids, iids, ratings = split.test_set.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r > 0:
            positive_ratings[uid].append(iid)

    return {
        "split": split,
        "iid_idx2str": iid_idx2str,
        "uid_str2idx": uid_str2idx,
        "positive_ratings": dict(positive_ratings),
    }


def load_recommendations(filepath):
    """Load ranked rec lists from a .txt file → dict {user_id_str: [item_id_str, ...]}"""
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


# ── Per-config evaluation: Mode A ─────────────────────────────────────────────

def evaluate_config_mode_a(config_name, cached_data, results_base):
    """Compute AUC + P@K + R@K + NDCG@K for a single config using item_scores.pkl."""
    save_path = os.path.join(results_base, config_name, "D_RDW")

    if not os.path.exists(save_path):
        print(f"  Skipping '{config_name}': {save_path} does not exist")
        return None

    scores_path = os.path.join(save_path, "item_scores.pkl")
    if not os.path.exists(scores_path):
        print(f"  Skipping '{config_name}': item_scores.pkl not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  [Mode A — item_scores.pkl]")
    print(f"{'='*60}")

    with open(scores_path, "rb") as f:
        item_scores = pickle.load(f)

    impression_items_idx = cached_data["impression_items_idx"]
    impression_idx2pos = cached_data["impression_idx2pos"]
    iid_idx2str = cached_data["iid_idx2str"]
    positive_ratings = cached_data["positive_ratings"]
    negative_ratings = cached_data["negative_ratings"]

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

        # AUC — pairwise comparison
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

        # Ranking metrics — sort impression pool by score descending
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
        "mode": "A",
        "auc": auc_score,
        "auc_correct_pairs": auc_correct,
        "auc_total_pairs": auc_total,
        "num_users": num_evaluated,
        "k_values": K_VALUES,
        "precision_at_k": {str(k): prec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "recall_at_k": {str(k): rec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "ndcg_at_k": {str(k): ndcg_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
    }

    _print_config_summary(results)
    _save_config_results(results, save_path)
    return results


# ── Per-config evaluation: Mode B ─────────────────────────────────────────────

def evaluate_config_mode_b(config_name, cached_data, results_base, rec_file):
    """Compute P@K + R@K + NDCG@K for a single config from a ranked .txt file."""
    save_path = os.path.join(results_base, config_name, "D_RDW")
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(rec_file):
        print(f"  Skipping '{config_name}': recommendation file not found: {rec_file}")
        return None

    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  [Mode B — ranked .txt]")
    print(f"  Rec file: {rec_file}")
    print(f"{'='*60}")

    iid_idx2str = cached_data["iid_idx2str"]
    uid_str2idx = cached_data["uid_str2idx"]
    positive_ratings = cached_data["positive_ratings"]

    recs = load_recommendations(rec_file)

    prec_sums = {k: 0.0 for k in K_VALUES}
    rec_sums = {k: 0.0 for k in K_VALUES}
    ndcg_sums = {k: 0.0 for k in K_VALUES}
    num_evaluated = 0

    for user_str, rec_items in tqdm(recs.items(), desc=f"Evaluating {config_name}"):
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

    results = {
        "config_name": config_name,
        "mode": "B",
        "auc": "N/A (no item scores in Mode B)",
        "num_users": num_evaluated,
        "k_values": K_VALUES,
        "precision_at_k": {str(k): prec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "recall_at_k": {str(k): rec_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
        "ndcg_at_k": {str(k): ndcg_sums[k] / num_evaluated for k in K_VALUES} if num_evaluated else {},
    }

    _print_config_summary(results)
    _save_config_results(results, save_path)
    return results


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _print_config_summary(results):
    n = results["num_users"]
    mode = results["mode"]
    print(f"  Users evaluated: {n}")
    if mode == "A":
        print(f"  AUC = {results['auc']:.4f}")
    for k in K_VALUES:
        p = results["precision_at_k"].get(str(k), 0.0)
        r = results["recall_at_k"].get(str(k), 0.0)
        n_ = results["ndcg_at_k"].get(str(k), 0.0)
        print(f"  P@{k:2d}={p:.4f}  R@{k:2d}={r:.4f}  NDCG@{k:2d}={n_:.4f}")


def _save_config_results(results, save_path):
    output_path = os.path.join(save_path, "all_accuracy_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved → {output_path}")


def _resolve_rec_file(config_name, results_base, rec_file_arg, rec_dir_arg):
    """Return the .txt recommendation file path for a given config in Mode B."""
    if rec_file_arg:
        return rec_file_arg  # explicit path — used for single-config runs

    if rec_dir_arg:
        # Auto-discover: look for any .txt in <rec_dir>/<config_name>/
        config_rec_dir = os.path.join(rec_dir_arg, config_name)
        if os.path.isdir(config_rec_dir):
            txt_files = [f for f in os.listdir(config_rec_dir) if f.endswith(".txt")]
            if txt_files:
                return os.path.join(config_rec_dir, txt_files[0])

        # Fallback: look for a .txt in <results_base>/<config_name>/D_RDW/
        drdw_path = os.path.join(results_base, config_name, "D_RDW")
        if os.path.isdir(drdw_path):
            txt_files = [f for f in os.listdir(drdw_path) if f.endswith(".txt")]
            if txt_files:
                return os.path.join(drdw_path, txt_files[0])

    return None


# ── Summary output ─────────────────────────────────────────────────────────────

def print_summary(all_results):
    has_auc = any(r.get("mode") == "A" for r in all_results.values())
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Config':<40s}"
    if has_auc:
        header += f" {'AUC':>6s}"
    for k in K_VALUES:
        header += f"  {'P@'+str(k):>6s}  {'R@'+str(k):>6s}  {'NDCG@'+str(k):>8s}"
    print(header)
    print("-" * len(header))

    for name in sorted(all_results.keys()):
        r = all_results[name]
        row = f"{name:<40s}"
        if has_auc:
            auc_val = r["auc"] if r.get("mode") == "A" else float("nan")
            row += f" {auc_val:>6.4f}" if isinstance(auc_val, float) else f" {'N/A':>6s}"
        for k in K_VALUES:
            p = r["precision_at_k"].get(str(k), 0.0)
            rc = r["recall_at_k"].get(str(k), 0.0)
            n = r["ndcg_at_k"].get(str(k), 0.0)
            row += f"  {p:>6.4f}  {rc:>6.4f}  {n:>8.4f}"
        print(row)


def save_summary(all_results, results_base):
    summary = {
        config_name: {
            "mode": r.get("mode", "A"),
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
    print(f"\nSummary saved → {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute accuracy metrics (AUC, P@K, R@K, NDCG@K) for experiment configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("config", nargs="?", default=None,
                       help="Single config name to evaluate")
    group.add_argument("--configs", nargs="+", metavar="NAME",
                       help="Select configs to evaluate")
    group.add_argument("--all", action="store_true",
                       help="Evaluate all configs in results directory")

    # Mode
    parser.add_argument("--mode", choices=["pkl", "txt"], default="pkl",
                        help="Evaluation mode: 'pkl' uses item_scores.pkl (default), "
                             "'txt' uses ranked recommendation .txt files")

    # Paths
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Path to data directory containing train/test CSVs and article_pool.csv "
                             "(default: <repo>/ebnerd_results_existing)")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Path to experiment results directory "
                             "(default: <repo>/experiment_results)")

    # Mode B — recommendation file options
    parser.add_argument("--rec-file", default=None,
                        help="[Mode B] Path to a single ranked recommendation .txt file. "
                             "Used when evaluating a single config.")
    parser.add_argument("--rec-dir", default=None,
                        help="[Mode B] Directory to auto-discover per-config .txt files. "
                             "Looks for a .txt file under <rec-dir>/<config_name>/. "
                             "Falls back to <results-dir>/<config_name>/D_RDW/ if not found.")

    return parser.parse_args()


def main():
    args = parse_args()
    results_base = args.results_dir
    data_path = args.data_dir
    mode = args.mode

    # ── Determine which configs to run ────────────────────────────────────────
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

    print(f"Mode:              {'A — item_scores.pkl' if mode == 'pkl' else 'B — ranked .txt files'}")
    print(f"Data directory:    {data_path}")
    print(f"Results directory: {results_base}")
    print(f"Configs:           {config_names}")

    # ── Load dataset once (shared across all configs) ─────────────────────────
    print("\nLoading dataset...")
    if mode == "pkl":
        cached_data = load_split_and_pool(data_path)
    else:
        cached_data = load_split_only(data_path)

    # ── Evaluate each config ──────────────────────────────────────────────────
    all_results = {}
    for config_name in config_names:
        if mode == "pkl":
            result = evaluate_config_mode_a(config_name, cached_data, results_base)
        else:
            rec_file = _resolve_rec_file(
                config_name, results_base, args.rec_file, args.rec_dir
            )
            if rec_file is None:
                print(f"  Skipping '{config_name}': no recommendation .txt file found. "
                      f"Provide --rec-file or --rec-dir.")
                continue
            result = evaluate_config_mode_b(config_name, cached_data, results_base, rec_file)

        if result is not None:
            all_results[config_name] = result

    if not all_results:
        print("\nNo configs were evaluated.")
        sys.exit(1)

    print_summary(all_results)
    save_summary(all_results, results_base)


if __name__ == "__main__":
    main()
