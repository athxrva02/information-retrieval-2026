# Experiment Running Guide

Step-by-step instructions for reproducing the D-RDW experiment on the EB-NeRD dataset.
Tested on macOS (Apple Silicon, ARM64) with Apple clang 17 and Python 3.10.

---

## 1. Environment Setup

### 1.1 Install Python 3.10

Python 3.10 is required. Python 3.14 (and likely 3.13+) causes Eigen compilation errors in the
bundled Cornac C++ extensions that cannot be patched.

```bash
brew install python@3.10
```

### 1.2 Create Virtual Environment

From the repository root:

```bash
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
```

### 1.3 Patch Eigen for Apple clang 17+

Apple clang 17 introduces a breaking change for the bundled Eigen library. Before installing Cornac,
apply this patch to `Recommenders/cornac/utils/external/eigen/Eigen/src/Core/Transpositions.h` (line 387):

```diff
-      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt.derived());
+      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt);
```

### 1.4 Install Cornac (Recommenders)

```bash
cd Recommenders
pip install -e .
cd ..
```

Expected output: `Successfully built cornac` and `Successfully installed cornac-2.3.3`.

### 1.5 Install Python Dependencies

The `requirements.txt` in this repo is UTF-16 encoded and contains a Windows-specific path.
Convert it and remove the editable cornac line before installing:

```bash
iconv -f UTF-16 -t UTF-8 requirements.txt | grep -v "^-e file:///" | grep -v "^$" > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

### 1.6 Verify Installation

```bash
python -c "import cornac; print('cornac', cornac.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import tensorflow; print('tensorflow', tensorflow.__version__)"
python -c "import spacy; print('spacy', spacy.__version__)"
```

Expected versions (as tested):
- cornac 2.3.3
- torch 2.10.0
- tensorflow 2.21.0
- spacy 3.8.11

---

## 2. Dataset Setup

### 2.1 Download EB-NeRD

Download from [recsys.eb.dk](https://recsys.eb.dk/index.html):
- `ebnerd_small` — user-item interactions
- `ebnerd_roberta_base` — article embeddings (`xlm_roberta_base.parquet`)

### 2.2 Prepare `ebnerd_input/` Directory

The scripts expect a flat `ebnerd_input/` directory in the repo root with 6 files:

```bash
mkdir -p ebnerd_input

# From ebnerd_small download:
cp ebnerd_small/articles.parquet ebnerd_input/articles.parquet
cp ebnerd_small/train/behaviors.parquet ebnerd_input/behaviors-train.parquet
cp ebnerd_small/train/history.parquet ebnerd_input/history-train.parquet
cp ebnerd_small/validation/behaviors.parquet ebnerd_input/behaviors-val.parquet
cp ebnerd_small/validation/history.parquet ebnerd_input/history-val.parquet

# From ebnerd_roberta_base download:
cp FaceBookAI_xlm_roberta_base/xlm_roberta_base.parquet ebnerd_input/xlm_roberta_base.parquet
```

Final contents of `ebnerd_input/`:
```
articles.parquet
behaviors-train.parquet
behaviors-val.parquet
history-train.parquet
history-val.parquet
xlm_roberta_base.parquet
```

---

## 3. Preprocessing (Step 2)

All commands are run from the repository root with the virtual environment activated.

### 3.1 Data Cleaning (Step 2-1)

```bash
python experiments/recsys_2025/article_enrichment_scripts/data_cleaning_ebnerd.py
```

This reads all files from `ebnerd_input/`, removes articles with missing fields (title, body,
published_time, category, article_id), and outputs:
- `ebnerd_results/cleaned_articles.csv` — 19,194 clean articles
- `ebnerd_results/incomplete_article_ids.txt` — 1,544 removed article IDs

### 3.2 Article Enrichment (Step 2-2)

**Important:** Before running, edit `experiments/recsys_2025/article_enrichment_scripts/article_enrich.py`
and comment out or remove the test limit on line 262:

```python
# df = df[:10]  # for test — REMOVE THIS LINE for full processing
```

Then run:

```bash
python experiments/recsys_2025/article_enrichment_scripts/article_enrich.py
```

This enriches all 19,194 articles with the following attributes (in order):
1. **Category** — from existing metadata
2. **Readability** — Flesch-style readability scores
3. **Sentiment** — via `xlm-roberta-base-sentiment-multilingual` model
4. **Named Entities (NER)** — via spaCy `da_core_news_sm` (Danish), extracts PER, LOC, ORG, MISC
5. **Enriched Named Entities** — Wikidata lookup for each PER/ORG entity (slowest step)
6. **Region** — geographic regions from named entities
7. **Political Party** — party affiliations from enriched entities
8. **Min/Maj Ratio** — minority/majority representation ratios
9. **Story** — article clustering by text similarity, date, and category

**Runtime:** Several hours (the Wikidata lookups in step 5 are the bottleneck).

**Requires:** Active internet connection for Wikidata API and HuggingFace model downloads.

**Config:** The enrichment configuration is in
`experiments/recsys_2025/article_enrichment_scripts/config_ebnerd.py`. Default settings:
- Language: `da` (Danish)
- Category method: `metadata`
- All enrichment options enabled
- Majority demographics: male, Denmark, white people, Denmark (for min/maj ratio)

**Output** (all in `ebnerd_results/`):
- `total.json` — all enrichments combined
- `enriched.csv` — full enriched DataFrame
- Individual JSON files: `category.json`, `readability.json`, `sentiment.json`,
  `named_entities.json`, `enriched_named_entities.json`, `region.json`, `party.json`,
  `min_maj_ratio.json`, `story.json`

### 3.3 Neural Preparation (Step 3A-1 to 3A-3)

These steps prepare data for neural models AND random walk models (they produce the UIR files
and user history needed by all models).

```bash
# Step 3A-1: Combine user history from train and validation sets
python experiments/recsys_2025/neural_preparation/ebnerd/combine_train_test_user_history_ebnerd.py

# Step 3A-2: Generate UIR impression files (train and test)
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_train_impression.py
python experiments/recsys_2025/neural_preparation/ebnerd/generate_uir_test_impression.py
```

**Output** (in `ebnerd_results/`):
- `combined_user_history.json`
- `uir_impression_train.csv`
- `uir_impression_test.csv`
- `article_pool.csv`

### 3.4 Graph Preparation (Step 3B-1)

Build the augmented bipartite graph for random walk models:

```bash
python experiments/recsys_2025/graph_preparation/generate_uir_augmentation_top3_combined_history.py
```

This uses `xlm_roberta_base.parquet` embeddings to find the top-3 most similar articles for
cold-start items and augments the user-item graph.

**Output:** `ebnerd_results/augmented_uir_top3similar.csv`

---

## 4. Running D-RDW (Step 3B-2)

### 4.1 Required Files

Before running, verify all required files exist in `ebnerd_results/`:

```bash
ls ebnerd_results_existing/
```

Required files:
- `augmented_uir_top3similar.csv` — augmented graph (from step 3.4)
- `uir_impression_test.csv` — test impressions (from step 3.3)
- `article_pool.csv` — article pool (from step 3.3)
- `sentiment.json` — must contain all ~19k articles (from step 3.2)
- `category.json` — must contain all ~19k articles (from step 3.2)
- `party.json` — must contain all ~19k articles (from step 3.2)

**Common pitfall:** If the enrichment was run with the `df = df[:10]` test limit, these JSON files
will only contain 10 entries and D-RDW will fail with a `KeyError` during sampling
(`None of [...] are in the [index]`). Re-run the enrichment with the limit removed.

### 4.2 Run the Experiment

```bash
python experiments/recsys_2025/experiment_scripts/drdw_experiment.py
```

### 4.3 D-RDW Model Parameters

As configured in `drdw_experiment.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `dataset_name` | `ebnerd` | Danish EB-NeRD dataset |
| `TopN` | `3` | Top-3 similar items for graph augmentation |
| `maxHops` | `3` | Random walk depth (use 5 for NeMig) |
| `targetSize` | `20` | Number of recommendations per user |
| `rankingType` | `graph_coloring` | Re-ranking method |
| `rankingObjectives` | `category` | Objective for re-ranking |
| `sampleObjective` | `rdw_score` | Sampling objective |
| `diversity_dimension` | `["sentiment", "entities"]` | Dimensions for NTD |
| `filteringCriteria` | `None` | No additional filtering |

### 4.4 Target Distribution (NTD)

The Normative Target Distribution for EB-NeRD is configured as:

**Sentiment:**
| Range | Probability |
|-------|-------------|
| [-1, -0.5) | 0.20 |
| [-0.5, 0) | 0.30 |
| [0, 0.5) | 0.30 |
| [0.5, 1.01) | 0.20 |

**Political Parties:**
| Description | Probability |
|-------------|-------------|
| Government parties only | 0.15 |
| Opposition parties only | 0.15 |
| Both government and opposition | 0.15 |
| Minor/other parties | 0.15 |
| No parties mentioned | 0.40 |

Government parties: Social Democrats, Venstre, Moderate Party, Union Party, Social Democratic Party,
Inuit Ataqatigiit, Naleraq

Opposition parties: Denmark Democrats, Green Left, Liberal Alliance, Conservative People's Party,
Conservative Party, Red–Green Alliance, Danish People's Party, Danish Social Liberal Party,
The Alternative

### 4.5 Output

Results are saved to `./experiment_ebnerd_drdw_results/` containing:
- Model recommendations
- Recall@20 evaluation results

---

## 5. Evaluation (Step 4)

After the experiment completes, evaluate with:

```bash
# Compute top-20 recommendations (skip for D-RDW, already limited to 20)
# python experiments/recsys_2025/evaluation_scripts/check_diversity/compute_top20_list.py

# Generate one-hot vectors for ILD calculation
python experiments/recsys_2025/evaluation_scripts/generate_party_one_hot.py
python experiments/recsys_2025/evaluation_scripts/generate_senti_one_hot.py
python experiments/recsys_2025/evaluation_scripts/party_binary.py

# Compute diversity metrics (RADio, Gini, ILD)
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_radio.py
python experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py

# Compute AUC
python experiments/recsys_2025/evaluation_scripts/check_accuracy/compute_auc.py
```

---

## Troubleshooting

### Eigen compilation error on macOS

```
error: no member named 'derived' in 'Transpose<TranspositionsBase<type-parameter-0-0>>'
```

**Fix:** Apply the patch in step 1.3 and use Python 3.10.

### D-RDW `KeyError: "None of [...] are in the [index]"`

The enrichment JSON files (`sentiment.json`, `category.json`, `party.json`) only contain 10 entries
instead of ~19k. This happens when `article_enrich.py` was run with the `df = df[:10]` test limit.
Comment out that line and re-run the enrichment.

### `requirements.txt` encoding error

The file is UTF-16 encoded with Windows line endings. Convert it:

```bash
iconv -f UTF-16 -t UTF-8 requirements.txt | grep -v "^-e file:///" > requirements_fixed.txt
pip install -r requirements_fixed.txt
```

### HuggingFace rate limiting

If you see slow downloads or 429 errors during enrichment, set a HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```



# Test 1
[D_RDW] Evaluation started!
Ranking: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15342/15342 [1:17:28<00:00,  3.30it/s]
Recommendations saved to ./experiment_ebnerd_drdw_results/D_RDW/recommendations.pkl
Item scores saved to ./experiment_ebnerd_drdw_results/D_RDW/item_scores.pkl
Item scores mapped index saved to ./experiment_ebnerd_drdw_results/D_RDW/item_scores_mapped_indices.pkl

TEST:
...
      | Recall@20 | Train (s) |  Test (s)
----- + --------- + --------- + ---------
D_RDW |    0.0056 |    1.7707 | 4648.4456


(venv) atharva@Atharvas-Mac-mini information-retrieval-2026 % python experiments/recsys_2025/evaluation_scripts/check_accuracy/compute_auc.py

rating_threshold = 0.5
exclude_unknowns = False
---
Training data:
Number of users = 18827
Number of items = 11693
Number of ratings = 3578104
Max rating = 1.0
Min rating = 1.0
Global mean = 1.0
---
Test data:
Number of users = 18827
Number of items = 11693
Number of ratings = 1985817
Number of unknown users = 0
Number of unknown items = 0
---
Total users = 18827
Total items = 11693
Processing Users: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15342/15342 [09:36<00:00, 26.62user/s]
correct_pairs: 30656806
total_pairs: 55314390
AUC Score: 0.5542
Total users evaluated:15342