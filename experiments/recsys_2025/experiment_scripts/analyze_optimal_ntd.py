"""
Empirical NTD Optimizer
========================
Analyzes the dataset to find an optimal Normative Target Distribution (NTD)
that should maximize AUC.

Three analyses:
1. Oracle NTD: Distribution of POSITIVE test items (what users actually click)
2. Candidate Pool Coverage: What's available in each sentiment/party bin
3. Supply-Demand Gap: Where the paper NTD demands more than exists

Usage:
    python analyze_optimal_ntd.py
    python analyze_optimal_ntd.py --generate-config   # also writes optimal NTD config

Output saved to: ./ntd_analysis/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Add project root so cornac imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from cornac.eval_methods.base_method import BaseMethod
from cornac.datasets import mind as mind

DATA_PATH = os.path.join(project_root, "ebnerd_results_existing")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ntd_analysis")

# Danish political parties (from existing configs)
GOV_PARTIES = ["Social Democrats", "Venstre", "Moderate Party", "Union Party",
               "Social Democratic Party", "Inuit Ataqatigiit", "Naleraq"]
OPP_PARTIES = ["Denmark Democrats - Inger Støjberg", "Green Left", "Liberal Alliance",
               "Conservative People's Party", "Conservative Party", "Red–Green Alliance",
               "Danish People's Party", "Danish Social Liberal Party", "The Alternative"]
ALL_PARTIES = GOV_PARTIES + OPP_PARTIES


def load_data():
    """Load train/test splits and item features."""
    print("Loading data...")

    train_path = os.path.join(DATA_PATH, 'augmented_uir_top3similar.csv')
    test_path = os.path.join(DATA_PATH, 'uir_impression_test.csv')

    feedback_train = mind.load_feedback(fpath=train_path)
    feedback_test = mind.load_feedback(fpath=test_path)

    split = BaseMethod.from_splits(
        train_data=feedback_train, test_data=feedback_test,
        exclude_unknowns=False, verbose=False, rating_threshold=0.5
    )

    # Load article pool
    pool_df = pd.read_csv(os.path.join(DATA_PATH, "article_pool.csv"), dtype={'iid': str})
    pool_iids = pool_df['iid'].tolist()

    # Load sentiment
    with open(os.path.join(DATA_PATH, "sentiment.json")) as f:
        sentiment_raw = json.load(f)

    # Load party
    with open(os.path.join(DATA_PATH, "party.json")) as f:
        party_raw = json.load(f)

    return split, pool_iids, sentiment_raw, party_raw


def classify_party(parties_dict):
    """Classify an article's party mentions into NTD categories."""
    if not parties_dict or not isinstance(parties_dict, dict):
        return "no_parties"

    mentioned = set(parties_dict.keys())
    if not mentioned:
        return "no_parties"

    has_gov = bool(mentioned & set(GOV_PARTIES))
    has_opp = bool(mentioned & set(OPP_PARTIES))
    has_other = bool(mentioned - set(ALL_PARTIES))

    if has_gov and has_opp:
        return "composition"
    elif has_gov and not has_opp and not has_other:
        return "gov_only"
    elif has_opp and not has_gov and not has_other:
        return "opp_only"
    elif has_other:
        return "minority_other"
    else:
        return "gov_only" if has_gov else "opp_only"


def get_sentiment_bin(val, bins):
    """Return which bin a sentiment value falls into."""
    for i, (lo, hi) in enumerate(bins):
        if lo <= val < hi:
            return i
    return len(bins) - 1  # last bin


def analyze_distributions(split, pool_iids, sentiment_raw, party_raw):
    """Analyze sentiment & party distributions for positive/negative/pool items."""

    item_id2idx = dict(split.global_iid_map.items())
    item_idx2id = {v: k for k, v in split.global_iid_map.items()}

    # Standard 4-bin sentiment bins
    SENTIMENT_BINS = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.01)]
    BIN_LABELS = ["[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1.01)"]

    # Get positive and negative items from test set
    pos_items_per_user = defaultdict(list)
    neg_items_per_user = defaultdict(list)
    uids, iids, ratings = split.test_set.uir_tuple
    for uid, iid, r in zip(uids, iids, ratings):
        if r > 0:
            pos_items_per_user[uid].append(iid)
        else:
            neg_items_per_user[uid].append(iid)

    # Flatten to get all positive/negative item indices
    all_pos_items = set()
    all_neg_items = set()
    for items in pos_items_per_user.values():
        all_pos_items.update(items)
    for items in neg_items_per_user.values():
        all_neg_items.update(items)

    # Pool items (mapped to indices)
    pool_indices = set()
    for iid_str in pool_iids:
        if iid_str in item_id2idx:
            pool_indices.add(item_id2idx[iid_str])

    print(f"\nDataset stats:")
    print(f"  Total items in global map: {len(item_id2idx)}")
    print(f"  Impression pool items: {len(pool_indices)}")
    print(f"  Positive test items (unique): {len(all_pos_items)}")
    print(f"  Negative test items (unique): {len(all_neg_items)}")
    print(f"  Overlap (pos ∩ neg): {len(all_pos_items & all_neg_items)}")

    # ---- SENTIMENT ANALYSIS ----
    def sentiment_distribution(item_indices, label):
        counts = [0] * len(SENTIMENT_BINS)
        missing = 0
        values = []
        for idx in item_indices:
            iid_str = item_idx2id.get(idx)
            if iid_str and iid_str in sentiment_raw:
                val = sentiment_raw[iid_str]
                values.append(val)
                b = get_sentiment_bin(val, SENTIMENT_BINS)
                counts[b] += 1
            else:
                missing += 1

        total = sum(counts)
        probs = [c / total if total > 0 else 0 for c in counts]

        print(f"\n  {label} sentiment distribution (n={total}, missing={missing}):")
        for i, (bl, p) in enumerate(zip(BIN_LABELS, probs)):
            print(f"    {bl}: {p:.3f} ({counts[i]} items)")

        if values:
            print(f"    Mean: {np.mean(values):.3f}, Median: {np.median(values):.3f}")

        return probs, counts

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    pool_sent_probs, pool_sent_counts = sentiment_distribution(pool_indices, "Article Pool")
    pos_sent_probs, pos_sent_counts = sentiment_distribution(all_pos_items, "Positive Test Items")
    neg_sent_probs, neg_sent_counts = sentiment_distribution(all_neg_items, "Negative Test Items")

    # Paper NTD sentiment
    paper_sent = [0.20, 0.30, 0.30, 0.20]
    natural_sent = [0.10, 0.45, 0.35, 0.10]

    print(f"\n  Paper NTD sentiment:     {[f'{p:.2f}' for p in paper_sent]}")
    print(f"  Natural-aligned NTD:     {[f'{p:.2f}' for p in natural_sent]}")
    print(f"  Pool distribution:       {[f'{p:.3f}' for p in pool_sent_probs]}")
    print(f"  Positive items dist:     {[f'{p:.3f}' for p in pos_sent_probs]}")
    print(f"  Negative items dist:     {[f'{p:.3f}' for p in neg_sent_probs]}")

    # ---- PARTY ANALYSIS ----
    def party_distribution(item_indices, label):
        cats = Counter()
        missing = 0
        for idx in item_indices:
            iid_str = item_idx2id.get(idx)
            if iid_str and iid_str in party_raw:
                cat = classify_party(party_raw[iid_str])
                cats[cat] += 1
            else:
                missing += 1

        total = sum(cats.values())
        print(f"\n  {label} party distribution (n={total}, missing={missing}):")
        for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]:
            c = cats.get(cat, 0)
            p = c / total if total > 0 else 0
            print(f"    {cat:20s}: {p:.3f} ({c} items)")

        return {cat: cats.get(cat, 0) / total if total > 0 else 0
                for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]}

    print("\n" + "=" * 60)
    print("PARTY ANALYSIS")
    print("=" * 60)

    pool_party = party_distribution(pool_indices, "Article Pool")
    pos_party = party_distribution(all_pos_items, "Positive Test Items")
    neg_party = party_distribution(all_neg_items, "Negative Test Items")

    # ---- SUPPLY-DEMAND GAP ----
    print("\n" + "=" * 60)
    print("SUPPLY-DEMAND GAP (Paper NTD vs Pool)")
    print("=" * 60)

    print("\n  Sentiment bins (target_size=20):")
    for i, bl in enumerate(BIN_LABELS):
        demand = paper_sent[i]
        supply = pool_sent_probs[i]
        gap = demand - supply
        status = "OVERSUPPLY" if gap < -0.02 else ("UNDERSUPPLY" if gap > 0.02 else "OK")
        print(f"    {bl}: demand={demand:.2f}, supply={supply:.3f}, gap={gap:+.3f} [{status}]")

    print("\n  Party categories:")
    paper_party = {"gov_only": 0.15, "opp_only": 0.15, "composition": 0.15,
                   "minority_other": 0.15, "no_parties": 0.40}
    for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]:
        demand = paper_party[cat]
        supply = pool_party.get(cat, 0)
        gap = demand - supply
        status = "OVERSUPPLY" if gap < -0.02 else ("UNDERSUPPLY" if gap > 0.02 else "OK")
        print(f"    {cat:20s}: demand={demand:.2f}, supply={supply:.3f}, gap={gap:+.3f} [{status}]")

    # ---- DISCRIMINATIVE ANALYSIS ----
    print("\n" + "=" * 60)
    print("DISCRIMINATIVE POWER (positive vs negative)")
    print("=" * 60)

    print("\n  Sentiment bins where positive items are MORE concentrated than negative:")
    for i, bl in enumerate(BIN_LABELS):
        diff = pos_sent_probs[i] - neg_sent_probs[i]
        direction = "↑ POSITIVE" if diff > 0.005 else ("↓ NEGATIVE" if diff < -0.005 else "≈ NEUTRAL")
        print(f"    {bl}: pos={pos_sent_probs[i]:.3f}, neg={neg_sent_probs[i]:.3f}, diff={diff:+.3f} [{direction}]")

    print("\n  Party categories where positive items are MORE concentrated:")
    for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]:
        p = pos_party.get(cat, 0)
        n = neg_party.get(cat, 0)
        diff = p - n
        direction = "↑ POSITIVE" if diff > 0.005 else ("↓ NEGATIVE" if diff < -0.005 else "≈ NEUTRAL")
        print(f"    {cat:20s}: pos={p:.3f}, neg={n:.3f}, diff={diff:+.3f} [{direction}]")

    return {
        "sentiment": {
            "bins": BIN_LABELS,
            "pool": pool_sent_probs,
            "positive": pos_sent_probs,
            "negative": neg_sent_probs,
            "pool_counts": pool_sent_counts,
            "pos_counts": pos_sent_counts,
            "neg_counts": neg_sent_counts,
        },
        "party": {
            "pool": pool_party,
            "positive": pos_party,
            "negative": neg_party,
        }
    }


def compute_oracle_ntd(analysis):
    """
    Compute the 'oracle' NTD — the distribution that best matches positive test items.

    Three strategies:
    1. Pure oracle: exact positive item distribution
    2. Discriminative: weight bins by (positive_rate - negative_rate)
    3. Feasibility-aware: oracle clamped to pool supply
    """
    sent = analysis["sentiment"]
    party = analysis["party"]

    SENTIMENT_BINS = [(-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.01)]

    strategies = {}

    # Strategy 1: Pure oracle (match positive item distribution)
    oracle_sent = sent["positive"]
    oracle_party = party["positive"]
    strategies["oracle_pure"] = {
        "sentiment": oracle_sent,
        "party": oracle_party,
        "description": "Exact distribution of positive test items"
    }

    # Strategy 2: Discriminative weighting
    # Upweight bins where pos > neg, downweight where neg > pos
    disc_sent = []
    for i in range(4):
        # Use ratio: pos/(pos+neg), then normalize
        p = sent["positive"][i]
        n = sent["negative"][i]
        disc_sent.append(p / (p + n) if (p + n) > 0 else 0.25)
    # Normalize to sum to 1
    total = sum(disc_sent)
    disc_sent = [s / total for s in disc_sent]

    disc_party = {}
    for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]:
        p = party["positive"].get(cat, 0)
        n = party["negative"].get(cat, 0)
        disc_party[cat] = p / (p + n) if (p + n) > 0 else 0.2
    total = sum(disc_party.values())
    disc_party = {k: v / total for k, v in disc_party.items()}

    strategies["discriminative"] = {
        "sentiment": disc_sent,
        "party": disc_party,
        "description": "Weighted by pos/(pos+neg) ratio per bin"
    }

    # Strategy 3: Feasibility-aware oracle
    # Clamp oracle to pool supply (can't demand more than exists)
    feasible_sent = []
    for i in range(4):
        feasible_sent.append(min(oracle_sent[i], sent["pool"][i]))
    # Redistribute excess to other bins proportionally
    deficit = 1.0 - sum(feasible_sent)
    if deficit > 0.001:
        unclamped = [i for i in range(4) if feasible_sent[i] < sent["pool"][i]]
        if unclamped:
            extra_each = deficit / len(unclamped)
            for i in unclamped:
                feasible_sent[i] = min(feasible_sent[i] + extra_each, sent["pool"][i])
    # Final normalize
    total = sum(feasible_sent)
    feasible_sent = [s / total for s in feasible_sent]

    feasible_party = {}
    for cat in ["gov_only", "opp_only", "composition", "minority_other", "no_parties"]:
        feasible_party[cat] = min(oracle_party.get(cat, 0), party["pool"].get(cat, 0))
    deficit = 1.0 - sum(feasible_party.values())
    if deficit > 0.001:
        unclamped = [c for c in feasible_party if feasible_party[c] < party["pool"].get(c, 0)]
        if unclamped:
            extra_each = deficit / len(unclamped)
            for c in unclamped:
                feasible_party[c] = min(feasible_party[c] + extra_each, party["pool"].get(c, 0))
    total = sum(feasible_party.values())
    feasible_party = {k: v / total for k, v in feasible_party.items()}

    strategies["feasibility_aware"] = {
        "sentiment": feasible_sent,
        "party": feasible_party,
        "description": "Oracle clamped to pool supply (ILP-feasible)"
    }

    # Print all strategies
    print("\n" + "=" * 60)
    print("COMPUTED NTD STRATEGIES")
    print("=" * 60)

    for name, strat in strategies.items():
        print(f"\n  {name}: {strat['description']}")
        print(f"    Sentiment: {[f'{p:.3f}' for p in strat['sentiment']]}")
        print(f"    Party: {{{', '.join(f'{k}: {v:.3f}' for k, v in strat['party'].items())}}}")

    return strategies


def generate_ntd_config(strategy_name, strategy, output_dir):
    """Generate a JSON NTD config from a computed strategy."""
    sent = strategy["sentiment"]
    party = strategy["party"]

    config = {
        "name": f"optimal_{strategy_name}",
        "description": f"Empirically computed NTD: {strategy['description']}",
        "target_distribution": {
            "sentiment": {
                "type": "continuous",
                "distr": [
                    {"min": -1, "max": -0.5, "prob": round(sent[0], 3)},
                    {"min": -0.5, "max": 0, "prob": round(sent[1], 3)},
                    {"min": 0, "max": 0.5, "prob": round(sent[2], 3)},
                    {"min": 0.5, "max": 1.01, "prob": round(sent[3], 3)}
                ]
            },
            "entities": {
                "type": "parties",
                "distr": [
                    {
                        "description": "only mention",
                        "contain": GOV_PARTIES,
                        "prob": round(party.get("gov_only", 0), 3)
                    },
                    {
                        "description": "only mention",
                        "contain": OPP_PARTIES,
                        "prob": round(party.get("opp_only", 0), 3)
                    },
                    {
                        "description": "composition",
                        "contain": [GOV_PARTIES, OPP_PARTIES],
                        "prob": round(party.get("composition", 0), 3)
                    },
                    {
                        "description": "minority but can also mention",
                        "contain": ALL_PARTIES,
                        "prob": round(party.get("minority_other", 0), 3)
                    },
                    {
                        "description": "no parties",
                        "contain": [],
                        "prob": round(party.get("no_parties", 0), 3)
                    }
                ]
            }
        },
        "model_params": {
            "maxHops": 5,
            "rankingType": "rdw_score"
        }
    }

    # Ensure probabilities sum to 1 (fix rounding)
    sent_distr = config["target_distribution"]["sentiment"]["distr"]
    sent_total = sum(d["prob"] for d in sent_distr)
    if abs(sent_total - 1.0) > 0.001:
        sent_distr[-1]["prob"] = round(sent_distr[-1]["prob"] + (1.0 - sent_total), 3)

    party_distr = config["target_distribution"]["entities"]["distr"]
    party_total = sum(d["prob"] for d in party_distr)
    if abs(party_total - 1.0) > 0.001:
        party_distr[-1]["prob"] = round(party_distr[-1]["prob"] + (1.0 - party_total), 3)

    config_path = os.path.join(output_dir, f"optimal_{strategy_name}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {config_path}")

    return config_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Empirical NTD Optimizer")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate optimal NTD config files")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    split, pool_iids, sentiment_raw, party_raw = load_data()
    analysis = analyze_distributions(split, pool_iids, sentiment_raw, party_raw)
    strategies = compute_oracle_ntd(analysis)

    # Save analysis
    analysis_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    # Convert for JSON serialization
    serializable = {
        "sentiment": {
            "bins": analysis["sentiment"]["bins"],
            "pool": analysis["sentiment"]["pool"],
            "positive": analysis["sentiment"]["positive"],
            "negative": analysis["sentiment"]["negative"],
            "pool_counts": analysis["sentiment"]["pool_counts"],
            "pos_counts": analysis["sentiment"]["pos_counts"],
            "neg_counts": analysis["sentiment"]["neg_counts"],
        },
        "party": {
            "pool": analysis["party"]["pool"],
            "positive": analysis["party"]["positive"],
            "negative": analysis["party"]["negative"],
        },
        "strategies": {
            name: {
                "description": s["description"],
                "sentiment": s["sentiment"],
                "party": s["party"]
            } for name, s in strategies.items()
        }
    }
    with open(analysis_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")

    if args.generate_config:
        print("\n" + "=" * 60)
        print("GENERATING NTD CONFIGS")
        print("=" * 60)
        config_dir = os.path.join(os.path.dirname(__file__), "ntd_configs")
        for name, strat in strategies.items():
            generate_ntd_config(name, strat, config_dir)

    # Print recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
  The 'feasibility_aware' strategy is recommended as the starting point.
  It matches the positive item distribution while ensuring the ILP can
  actually satisfy the constraints given the available article pool.

  To run:
    1. Generate configs:
       python analyze_optimal_ntd.py --generate-config

    2. Run experiment:
       python drdw_ntd_runner.py --config ntd_configs/optimal_feasibility_aware.json

    3. Compute AUC:
       python ../evaluation_scripts/compute_auc_for_config.py optimal_feasibility_aware

  For ablation, also try:
    - optimal_oracle_pure:   Theoretical best (may have ILP failures)
    - optimal_discriminative: Maximizes separation between pos/neg items
""")


if __name__ == "__main__":
    main()