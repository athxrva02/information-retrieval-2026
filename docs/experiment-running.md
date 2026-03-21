# Experiment Running Guide

Reproduce D-RDW experiments on EB-NeRD. Tested on macOS (Apple Silicon) with Python 3.10.

---

## Directory Layout

```
<repo>/
├── ebnerd_input/                          # Raw EB-NeRD dataset (you provide)
├── ebnerd_results_existing/               # Preprocessed data (scripts produce)
├── experiment_results/                    # D-RDW outputs per NTD config
│   ├── accuracy_summary.json
│   └── {config_name}/
│       ├── ntd_config.json
│       └── D_RDW/
│           ├── item_scores.pkl
│           ├── recommendations.pkl
│           └── all_accuracy_metrics.json
├── experiments/recsys_2025/               # All scripts
│   ├── article_enrichment_scripts/
│   ├── neural_preparation/ebnerd/
│   ├── graph_preparation/
│   ├── experiment_scripts/
│   │   ├── drdw_ntd_runner.py             # Main experiment runner
│   │   ├── analyze_optimal_ntd.py
│   │   └── ntd_configs/                   # NTD JSON configs
│   └── evaluation_scripts/
│       ├── check_accuracy/
│       │   └── run_accuracy_metrics.py    # AUC, P@K, R@K, NDCG@K
│       ├── check_diversity/
│       │   ├── check_diversity.py         # ILD, Gini
│       │   └── check_radio.py            # RADio metrics
│       ├── generate_senti_one_hot.py
│       ├── generate_party_one_hot.py
│       └── party_binary.py
└── Recommenders/                          # Cornac framework (dependency)
```

---

## 1. Environment Setup

### 1.1 Python 3.10

Python 3.10 is required. 3.13+ causes Eigen compilation errors in Cornac.

```bash
brew install python@3.10
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
```

### 1.2 Patch Eigen for Apple clang 17+

In `Recommenders/cornac/utils/external/eigen/Eigen/src/Core/Transpositions.h` line 387:

```diff
-      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt.derived());
+      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt);
```

### 1.3 Install Cornac + Dependencies

```bash
cd Recommenders && pip install -e . && cd ..
iconv -f UTF-16 -t UTF-8 requirements.txt | grep -v "^-e file:///" | grep -v "^$" > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

### 1.4 Verify

```bash
python -c "import cornac; print(cornac.__version__)"   # 2.3.3
python -c "import torch; print(torch.__version__)"     # 2.10.0
python -c "import spacy; print(spacy.__version__)"     # 3.8.11
```

---

## 2. Dataset Setup

Download from [recsys.eb.dk](https://recsys.eb.dk/index.html):
- `ebnerd_small` — interactions
- `ebnerd_roberta_base` — article embeddings

Prepare `ebnerd_input/`:

```bash
mkdir -p ebnerd_input
cp ebnerd_small/articles.parquet                       ebnerd_input/articles.parquet
cp ebnerd_small/train/behaviors.parquet                ebnerd_input/behaviors-train.parquet
cp ebnerd_small/train/history.parquet                  ebnerd_input/history-train.parquet
cp ebnerd_small/validation/behaviors.parquet           ebnerd_input/behaviors-val.parquet
cp ebnerd_small/validation/history.parquet             ebnerd_input/history-val.parquet
cp FaceBookAI_xlm_roberta_base/xlm_roberta_base.parquet ebnerd_input/xlm_roberta_base.parquet
```

**Expected files in `ebnerd_input/`:**
`articles.parquet`, `behaviors-train.parquet`, `behaviors-val.parquet`, `history-train.parquet`, `history-val.parquet`, `xlm_roberta_base.parquet`

---

## 3. Preprocessing

All commands from repo root with venv activated. Output goes to `ebnerd_results_existing/`.

### 3.1 Data Cleaning

```bash
python experiments/recsys_2025/article_enrichment_scripts/data_cleaning_ebnerd.py
```

- **Input:** `ebnerd_input/articles.parquet`
- **Output:** `ebnerd_results_existing/cleaned_articles.csv` (19,194 articles), `ebnerd_results_existing/incomplete_article_ids.txt`

### 3.2 Article Enrichment

**Before running:** Remove the test limit in `article_enrich.py` line 262:
```python
# df = df[:10]  # REMOVE THIS LINE
```

```bash
python experiments/recsys_2025/article_enrichment_scripts/article_enrich.py
```

- **Input:** `ebnerd_results_existing/cleaned_articles.csv`
- **Output (in `ebnerd_results_existing/`):**
  - `sentiment.json` — sentiment scores per article ([-1, 1])
  - `category.json` — article categories
  - `party.json` — political party mentions per article
  - `enriched.csv`, `total.json`, `readability.json`, `named_entities.json`, `enriched_named_entities.json`, `region.json`, `min_maj_ratio.json`, `story.json`
- **Runtime:** Several hours (Wikidata lookups are the bottleneck)
- **Requires:** Internet connection

### 3.3 Neural/UIR Preparation

```bash
python experiments/recsys_2025/neural_preparation/ebnerd/combine_train_test_user_history_ebnerd.py
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_train_impression.py
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_test_impression.py
```

- **Input:** `ebnerd_input/` (behaviors + history parquets)
- **Output (in `ebnerd_results_existing/`):**
  - `combined_user_history.json` — user → [item_ids]
  - `uir_impression_train.csv` — train UIR triplets
  - `uir_impression_test.csv` — test UIR triplets
  - `article_pool.csv` — candidate item pool (~2,200 items)

### 3.4 Graph Augmentation

```bash
python experiments/recsys_2025/graph_preparation/generate_uir_augmentation_top3_combined_history.py
```

- **Input:** `ebnerd_input/xlm_roberta_base.parquet`, `ebnerd_results_existing/uir_impression_train.csv`, `ebnerd_results_existing/combined_user_history.json`
- **Output:** `ebnerd_results_existing/augmented_uir_top3similar.csv`

### 3.5 Feature Vectors (for diversity evaluation)

These produce one-hot vectors needed by the diversity evaluation scripts (step 5.2).

```bash
python experiments/recsys_2025/evaluation_scripts/generate_senti_one_hot.py
python experiments/recsys_2025/evaluation_scripts/generate_party_one_hot.py
python experiments/recsys_2025/evaluation_scripts/party_binary.py
```

- **Input:** `ebnerd_results_existing/sentiment.json`, `ebnerd_results_existing/party.json`
- **Output (in `ebnerd_results_existing/`):**
  - `sentiment_vectors.json` — 4-bin one-hot sentiment
  - `party_vectors.json` — 5-class one-hot party
  - `entities_binary_count.json` — binary gov/opp

---

## 4. Running D-RDW Experiments

### 4.1 Required Preprocessed Files

Verify these exist in `ebnerd_results_existing/` before running:

| File | From Step |
|------|-----------|
| `augmented_uir_top3similar.csv` | 3.4 |
| `uir_impression_test.csv` | 3.3 |
| `article_pool.csv` | 3.3 |
| `sentiment.json` | 3.2 |
| `category.json` | 3.2 |
| `party.json` | 3.2 |

### 4.2 NTD Config Format

NTD configs are JSON files in `experiments/recsys_2025/experiment_scripts/ntd_configs/`:

```json
{
  "name": "config_name",
  "description": "What this NTD tests",
  "target_distribution": {
    "sentiment": {"type": "continuous", "distr": [
      {"min": -1, "max": -0.5, "prob": 0.2},
      {"min": -0.5, "max": 0, "prob": 0.3},
      {"min": 0, "max": 0.5, "prob": 0.3},
      {"min": 0.5, "max": 1.01, "prob": 0.2}
    ]},
    "entities": {"type": "parties", "distr": [
      {"description": "gov only", "contain": ["Social Democrats", ...], "prob": 0.15},
      {"description": "no parties", "contain": [], "prob": 0.4}
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
```

`model_params` is optional — paper defaults are used if omitted.

### 4.3 Available NTD Configs

| Config | File | Key Difference |
|--------|------|----------------|
| Paper default | `paper_default.json` | Original NTD from D-RDW paper |
| Natural aligned | `natural_aligned.json` | NTD matches actual data distribution |
| Natural + 5 hops | `natural_aligned_rdw_5hops.json` | + `rdw_score` ranking, 5 hops |
| Uniform sentiment | `uniform_sentiment.json` | Equal 25% per sentiment bin |
| More opposition | `more_opposition.json` | 30% opposition exposure |
| RDW score ranking | `rdw_score_ranking.json` | `rdw_score` instead of `graph_coloring` |
| Oracle pure | `optimal_oracle_pure.json` | Exact positive item distribution |
| Discriminative | `optimal_discriminative.json` | Weighted by pos/(pos+neg) ratio |
| Feasibility-aware | `optimal_feasibility_aware.json` | Oracle clamped to pool supply |

The `optimal_*` configs are generated by `analyze_optimal_ntd.py --generate-config`.

### 4.4 Running Experiments

```bash
# Run paper default:
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py --default

# Run a specific config:
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --config experiments/recsys_2025/experiment_scripts/ntd_configs/uniform_sentiment.json

# Run ALL configs (data loaded once, configs run sequentially):
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --config-dir experiments/recsys_2025/experiment_scripts/ntd_configs/

# List available configs:
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --list-configs experiments/recsys_2025/experiment_scripts/ntd_configs/
```

**Options:**
- `--data-path <dir>` — override input data directory (default: `./ebnerd_results_existing`)
- `--output-base <dir>` — override output directory (default: `./experiment_results`)

**Runtime:** ~2 min data loading + ~77 min per config (15,342 users).

### 4.5 Output

Per config, saved to `experiment_results/{config_name}/`:

```
{config_name}/
├── ntd_config.json                    # Config used (for reproducibility)
├── CornacExp-YYYY-MM-DD_HH-MM-SS.log
└── D_RDW/
    ├── item_scores.pkl                # {user_idx: score_array}
    ├── item_scores_mapped_indices.pkl
    └── recommendations.pkl            # {user_idx: [item_indices]} top-20
```

### 4.6 Parallelisation

Run separate configs in different terminals (each loads data independently, ~8GB RAM per process):

```bash
# Terminal 1:
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --config ntd_configs/uniform_sentiment.json

# Terminal 2:
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --config ntd_configs/more_opposition.json
```

### 4.7 Generating Optimal NTD Configs

Empirically analyze the test data to compute optimal NTDs:

```bash
# Analyze only (prints results):
python experiments/recsys_2025/experiment_scripts/analyze_optimal_ntd.py

# Analyze + generate optimal_*.json configs:
python experiments/recsys_2025/experiment_scripts/analyze_optimal_ntd.py --generate-config
```

- **Input:** `ebnerd_results_existing/` (augmented UIR, test UIR, article pool, sentiment, party)
- **Output:** `experiments/recsys_2025/experiment_scripts/ntd_analysis/analysis_results.json`
- **Generated configs:** `ntd_configs/optimal_oracle_pure.json`, `optimal_discriminative.json`, `optimal_feasibility_aware.json`

---

## 5. Evaluation

### 5.1 Accuracy Metrics (AUC, Precision@K, Recall@K, NDCG@K)

Uses `item_scores.pkl` from each config.

```bash
# All configs:
python experiments/recsys_2025/evaluation_scripts/check_accuracy/run_accuracy_metrics.py --all

# Specific configs:
python experiments/recsys_2025/evaluation_scripts/check_accuracy/run_accuracy_metrics.py \
  --configs optimal_oracle_pure optimal_discriminative

# Single config:
python experiments/recsys_2025/evaluation_scripts/check_accuracy/run_accuracy_metrics.py optimal_oracle_pure

# Custom directories:
python experiments/recsys_2025/evaluation_scripts/check_accuracy/run_accuracy_metrics.py --all \
  --data-dir ./ebnerd_results_existing --results-dir ./experiment_results
```

- **Input:** `experiment_results/{config}/D_RDW/item_scores.pkl` + `ebnerd_results_existing/` (UIR files, article pool)
- **Output per config:** `experiment_results/{config}/D_RDW/all_accuracy_metrics.json`
- **Summary:** `experiment_results/accuracy_summary.json`

K values: 5, 10, 20.

### 5.2 Diversity Metrics (ILD, Gini)

**Prerequisite:** Run step 3.5 (one-hot vector generation) first.

```bash
# Default (reads from experiment_ebnerd_drdw_results/D_RDW/):
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py

# Specific config from experiment_results/:
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py \
  --config optimal_oracle_pure

# Custom path:
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py \
  --save-path ./experiment_results/optimal_oracle_pure/D_RDW
```

- **Input:** `{save_path}/recommendations.pkl`, `ebnerd_results_existing/` (sentiment, category, party vectors)
- **Output:** `{save_path}/average_diversity_ild_gini.json`
- **Metrics:** ILD (category, sentiment, party), Gini (category, sentiment, party) — all @20

### 5.3 RADio Metrics

**Prerequisite:** Requires `entities_binary_count.json` from step 3.5, and `story.json`, `readability.json`, `min_maj_ratio.json` from step 3.2.

```bash
# Default:
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_radio.py

# Specific config:
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_radio.py \
  --config optimal_oracle_pure
```

- **Input:** `{save_path}/recommendations.pkl`, `ebnerd_results_existing/` (all feature files + user history)
- **Output:** `{save_path}/average_radio.json`
- **Metrics:** Activation, Representation, Calibration (category, complexity), Fragmentation, Alternative Voices

---

## 6. Complete Reproduction Walkthrough

Run these in order from repo root:

```bash
# 1. Preprocessing (skip if ebnerd_results_existing/ already populated)
python experiments/recsys_2025/article_enrichment_scripts/data_cleaning_ebnerd.py
python experiments/recsys_2025/article_enrichment_scripts/article_enrich.py
python experiments/recsys_2025/neural_preparation/ebnerd/combine_train_test_user_history_ebnerd.py
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_train_impression.py
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_test_impression.py
python experiments/recsys_2025/graph_preparation/generate_uir_augmentation_top3_combined_history.py
python experiments/recsys_2025/evaluation_scripts/generate_senti_one_hot.py
python experiments/recsys_2025/evaluation_scripts/generate_party_one_hot.py
python experiments/recsys_2025/evaluation_scripts/party_binary.py

# 2. (Optional) Generate optimal NTD configs from data
python experiments/recsys_2025/experiment_scripts/analyze_optimal_ntd.py --generate-config

# 3. Run all NTD experiments
python experiments/recsys_2025/experiment_scripts/drdw_ntd_runner.py \
  --config-dir experiments/recsys_2025/experiment_scripts/ntd_configs/

# 4. Evaluate accuracy (all configs at once)
python experiments/recsys_2025/evaluation_scripts/check_accuracy/run_accuracy_metrics.py --all

# 5. Evaluate diversity (for each config)
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py --config optimal_oracle_pure
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_radio.py --config optimal_oracle_pure
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Eigen `no member named 'derived'` | Apply patch in step 1.2, use Python 3.10 |
| D-RDW `KeyError: "None of [...] are in the [index]"` | Enrichment JSONs have only 10 entries. Remove `df = df[:10]` in `article_enrich.py` and re-run |
| `requirements.txt` encoding error | Use the `iconv` conversion in step 1.3 |
| Wikidata SPARQL 400 errors | Non-fatal. Caused by entity names with quotes. Enrichment continues |
| HuggingFace 429 rate limiting | `export HF_TOKEN=your_token_here` |
