"""
D-RDW NTD Experiment Runner
============================
Run D-RDW with different Normative Target Distributions without re-training.

Usage:
    # Run a specific NTD config:
    python drdw_ntd_runner.py --config configs/uniform_sentiment.json

    # Run all configs in a directory:
    python drdw_ntd_runner.py --config-dir configs/

    # List available configs:
    python drdw_ntd_runner.py --list-configs configs/

NTD Config Format (JSON):
    {
        "name": "uniform_sentiment",
        "description": "Equal probability across all sentiment bins",
        "target_distribution": {
            "sentiment": {"type": "continuous", "distr": [
                {"min": -1, "max": 0, "prob": 0.5},
                {"min": 0, "max": 1.01, "prob": 0.5}
            ]},
            "entities": {"type": "parties", "distr": [
                {"description": "only mention", "contain": ["Social Democrats", "Venstre"], "prob": 0.15},
                {"description": "no parties", "contain": [], "prob": 0.85}
            ]}
        },
        "model_params": {
            "maxHops": 3,
            "targetSize": 20,
            "rankingType": "graph_coloring",
            "rankingObjectives": "category",
            "sampleObjective": "rdw_score"
        }
    }

    "model_params" is optional — defaults are used if omitted.

Results are saved to: ./experiment_results/{config_name}/D_RDW/
Each run also saves a copy of the config used, so results are always traceable.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import pickle

from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.experiment.experiment import Experiment
from cornac.datasets import mind as mind
from cornac.metrics import Recall
from cornac.models import D_RDW


def load_data(input_path, top_n=3):
    """Load and prepare all data. This is the expensive part that only needs to run once."""
    print("Loading data...")
    train_uir_path = os.path.join(input_path, f'augmented_uir_top{top_n}similar.csv')
    feedback_train = mind.load_feedback(fpath=train_uir_path)

    test_uir_path = os.path.join(input_path, 'uir_impression_test.csv')
    feedback_test = mind.load_feedback(fpath=test_uir_path)

    article_pool_path = os.path.join(input_path, "article_pool.csv")
    impression_items_df = pd.read_csv(article_pool_path, dtype={'iid': str})
    impression_iid_list = impression_items_df['iid'].tolist()

    rs = BaseMethod.from_splits(
        train_data=feedback_train,
        test_data=feedback_test,
        exclude_unknowns=False,
        verbose=True,
        rating_threshold=0.5
    )

    sentiment = mind.load_sentiment(fpath=os.path.join(input_path, 'sentiment.json'))
    category = mind.load_category(fpath=os.path.join(input_path, 'category.json'))
    entities = mind.load_entities(fpath=os.path.join(input_path, 'party.json'), keep_empty=True)

    Item_sentiment = mind.build(data=sentiment, id_map=rs.global_iid_map)
    Item_category = mind.build(data=category, id_map=rs.global_iid_map)
    Item_entities = mind.build(data=entities, id_map=rs.global_iid_map)

    article_feature_dataframe = (
        pd.Series(Item_category).to_frame('category')
        .join(pd.Series(Item_entities).to_frame('entities'), how='outer')
        .join(pd.Series(Item_sentiment).to_frame('sentiment'), how='outer')
    )

    # Fill missing items (cold-start/augmented items not in enrichment JSONs)
    all_item_indices = pd.RangeIndex(len(rs.global_iid_map))
    article_feature_dataframe = article_feature_dataframe.reindex(all_item_indices)
    article_feature_dataframe['category'] = article_feature_dataframe['category'].fillna('unknown')
    article_feature_dataframe['sentiment'] = article_feature_dataframe['sentiment'].fillna(0.0)
    article_feature_dataframe['entities'] = article_feature_dataframe['entities'].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    return rs, article_feature_dataframe, impression_iid_list


def get_default_ebnerd_ntd():
    """Return the default EB-NeRD NTD from the original paper."""
    gov_parties = [
        "Social Democrats", "Venstre", "Moderate Party", "Union Party",
        "Social Democratic Party", "Inuit Ataqatigiit", "Naleraq"
    ]
    opp_parties = [
        "Denmark Democrats - Inger Støjberg", "Green Left", "Liberal Alliance",
        "Conservative People's Party", "Conservative Party", "Red–Green Alliance",
        "Danish People's Party", "Danish Social Liberal Party", "The Alternative"
    ]
    gov_opp_combined = gov_parties + opp_parties

    return {
        "sentiment": {"type": "continuous", "distr": [
            {"min": -1, "max": -0.5, "prob": 0.2},
            {"min": -0.5, "max": 0, "prob": 0.3},
            {"min": 0, "max": 0.5, "prob": 0.3},
            {"min": 0.5, "max": 1.01, "prob": 0.2}
        ]},
        "entities": {"type": "parties", "distr": [
            {"description": "only mention", "contain": gov_parties, "prob": 0.15},
            {"description": "only mention", "contain": opp_parties, "prob": 0.15},
            {"description": "composition", "contain": [gov_parties, opp_parties], "prob": 0.15},
            {"description": "minority but can also mention", "contain": gov_opp_combined, "prob": 0.15},
            {"description": "no parties", "contain": [], "prob": 0.4}
        ]}
    }


def run_experiment(rs, article_feature_dataframe, impression_iid_list,
                   target_distribution, model_params, output_path):
    """Run D-RDW with a specific NTD and save results."""

    max_hops = model_params.get("maxHops", 3)
    target_size = model_params.get("targetSize", 20)
    ranking_type = model_params.get("rankingType", "graph_coloring")
    ranking_objectives = model_params.get("rankingObjectives", "category")
    sample_objective = model_params.get("sampleObjective", "rdw_score")

    model = D_RDW(
        item_dataframe=article_feature_dataframe,
        diversity_dimension=["sentiment", "entities"],
        target_distributions=target_distribution,
        targetSize=target_size,
        maxHops=max_hops,
        filteringCriteria=None,
        rankingType=ranking_type,
        rankingObjectives=ranking_objectives,
        sampleObjective=sample_objective,
        verbose=True,
        article_pool=impression_iid_list
    )

    metrics = [Recall(k=target_size)]

    Experiment(
        eval_method=rs,
        models=[model],
        metrics=metrics,
        save_dir=output_path
    ).run()


def load_config(config_path):
    """Load an NTD config from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run D-RDW with different NTD configurations")
    parser.add_argument("--config", type=str, help="Path to a single NTD config JSON file")
    parser.add_argument("--config-dir", type=str, help="Path to directory of NTD config JSON files (run all)")
    parser.add_argument("--list-configs", type=str, help="List available configs in a directory")
    parser.add_argument("--data-path", type=str, default="./ebnerd_results_existing",
                        help="Path to dataset results (default: ./ebnerd_results_existing)")
    parser.add_argument("--output-base", type=str, default="./experiment_results",
                        help="Base output directory (default: ./experiment_results)")
    parser.add_argument("--default", action="store_true",
                        help="Run with the default paper NTD")
    args = parser.parse_args()

    if args.list_configs:
        configs = sorted(f for f in os.listdir(args.list_configs) if f.endswith('.json'))
        if not configs:
            print(f"No JSON configs found in {args.list_configs}")
        for c in configs:
            cfg = load_config(os.path.join(args.list_configs, c))
            print(f"  {c}: {cfg.get('description', '(no description)')}")
        return

    # Collect configs to run
    configs_to_run = []

    if args.default:
        configs_to_run.append({
            "name": "paper_default",
            "description": "Default NTD from the D-RDW paper",
            "target_distribution": get_default_ebnerd_ntd(),
            "model_params": {}
        })

    if args.config:
        configs_to_run.append(load_config(args.config))

    if args.config_dir:
        for f in sorted(os.listdir(args.config_dir)):
            if f.endswith('.json'):
                configs_to_run.append(load_config(os.path.join(args.config_dir, f)))

    if not configs_to_run:
        parser.print_help()
        print("\nError: Specify --config, --config-dir, or --default")
        sys.exit(1)

    # Load data once
    rs, article_feature_dataframe, impression_iid_list = load_data(args.data_path)

    # Run each config
    for i, config in enumerate(configs_to_run):
        config_name = config.get("name", f"experiment_{i}")
        description = config.get("description", "")
        target_distribution = config["target_distribution"]
        model_params = config.get("model_params", {})

        output_path = os.path.join(args.output_base, config_name)
        os.makedirs(output_path, exist_ok=True)

        # Save config alongside results for reproducibility
        config_save_path = os.path.join(output_path, "ntd_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Running config {i+1}/{len(configs_to_run)}: {config_name}")
        if description:
            print(f"Description: {description}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        start_time = time.time()
        run_experiment(
            rs, article_feature_dataframe, impression_iid_list,
            target_distribution, model_params, output_path
        )
        elapsed = time.time() - start_time
        print(f"\nCompleted '{config_name}' in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()