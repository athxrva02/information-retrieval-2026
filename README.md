# Information Retrieval 2026 - Group 36

Reproduction and experimentation repository for [**D-RDW: Diversity-Driven Random Walks for News Recommender Systems**](https://dl.acm.org/doi/10.1145/3705328.3748016), 
presented at RecSys 2025 (Li, Heitz, Inel, Bernstein — University of Zurich).

The code in this repository is based on the code developed for the original D-RDW algorithm, which can be found at the
[Informfully/Experiments](https://github.com/Informfully/Experiments/tree/main) and [Informfully/Recommenders](https://github.com/Informfully/Recommenders/tree/main#)
repos.

## What is D-RDW?

D-RDW is a lightweight, diversity-aware news recommender that utilizes random walks on a user-item bipartite graph. 
Unlike neural models that treat diversity as a post-processing afterthought, D-RDW enforces a **Normative Target 
Distribution (NTD)** of article properties (sentiment, political party mentions, category) during the recommendation 
step itself.

The pipeline has three stages:

1. **Pre-processing** — Build a bipartite graph from user-item interactions; augment it with new/cold items using 
semantic similarity (via [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).
2. **In-processing** — Run random walks (3 hops), apply heuristic filters, then sample a candidate list that satisfies 
the NTD constraints using binary integer programming.
3. **Post-processing** — Re-rank the candidate list by random walk score, multiple objectives, or graph coloring (for 
homogeneous category display).

**Key results (EB-NeRD, top-20 recommendations):** D-RDW ranks in the top 3 across all four evaluation dimensions — 
normative RADio diversity, traditional Gini/ILD diversity, energy cost, and AUC — while being ~100× cheaper to run 
than neural baselines.

## Setup

After cloning this repository, create a virtual environment in the root directory. For example by running:
```
uv venv --python 3.10
```

Then `cd` into the `Recommenders` directory, which contains the Cornac package with the necessary extensions. To install,
run:
```
# Without uv
pip install -e .
# With uv
uv pip install -e .
```

Once this has been done, navigate back to the root directory and install the dependencies found in `requirements.txt`.

Following these steps should be sufficient to set up the environment.

### macOS (Apple Silicon) Notes

On macOS with Apple clang 17+, the bundled Eigen library in Cornac fails to compile. Before installing Cornac, apply the
following patch to `Recommenders/cornac/utils/external/eigen/Eigen/src/Core/Transpositions.h` (line 387):

```diff
-      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt.derived());
+      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt);
```

Additionally, Python 3.10 is recommended. Python 3.14 triggers the same Eigen compilation error regardless of the patch.
If you don't have Python 3.10 installed, you can install it via Homebrew:
```
brew install python@3.10
```

Then create the virtual environment with:
```
/opt/homebrew/bin/python3.10 -m venv venv
```


## Datasets

This project uses the [EB-NeRD dataset](https://recsys.eb.dk/index.html). This project requires ebnerd_small 
(iter-item interactions) and ebnerd_roberta_base (article embeddings). To set up the folder, create a directory called
`ebnerd_input` in the root directory, containing 5 files:
- `articles.parquet`: Contains the article data
- `behaviors-train.parquet`: Contains the behaviors data from the train set
- `behaviors-val.parquet`: Contains the behaviors data from the validation set
- `history-train.parquet`: Contains the history data from the train set
- `history-val.parquet`: Contains the history data from the validation set

Once the dataset has been set up, data processing can be executed by following the instructions in the experiments 
[README.md](experiments/recsys_2025/README.md).

## Repository Structure

```
experiments/recsys_2025/
├── article_enrichment_scripts/   # Clean + enrich articles (sentiment, party, NER, etc.)
├── neural_preparation/           # Convert EB-NeRD behaviors to Cornac UIR format
├── graph_preparation/            # Build augmented bipartite graphs for random walk models
├── PLD_EPD_preparation/          # Prepare data for PLD/EPD filtering baselines
├── experiment_scripts/           # Run models (one script per model)
├── experiment_reranking_scripts/ # Apply re-ranking (MMR, GreedyKL, PM-2, DYN) to neural outputs
├── evaluation_scripts/           # Compute AUC, Gini, ILD, RADio metrics
└── final_recommendations/        # Pre-computed top-20 recommendation lists

Recommenders/                     # The Recommenders package - an extension of Cornac
```

[//]: # (## Baselines Compared)

[//]: # ()
[//]: # (| Type | Models |)

[//]: # (|------|--------|)

[//]: # (| Neural | LSTUR, NPA, NRMS |)

[//]: # (| Neural + re-ranking | + Greedy KL, PM-2, MMR, DYN-ATT, DYN-POS |)

[//]: # (| Random walk | RP³β, RWE-D, D-RDW |)

[//]: # (| Baseline | Random |)

## Key Links

- [Informfully Recommenders](https://github.com/Informfully/Recommenders) — extended Cornac framework (required dependency)
- [D-RDW model source](https://github.com/Informfully/Recommenders/tree/main/cornac/models/drdw)
- [Graph preparation guide](https://github.com/Informfully/Experiments/blob/main/experiments/recsys_2025/graph_preparation/README.md)
- [EB-NeRD dataset](https://recsys.eb.dk/)
- [Sentence embeddings model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
