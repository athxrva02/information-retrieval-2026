# Information Retrieval 2026 - Group 36

Reproduction and experimentation repository for **D-RDW: Diversity-Driven Random Walks for News Recommender Systems**, presented at RecSys 2025 (Li, Heitz, Inel, Bernstein — University of Zurich).

## What is D-RDW?

D-RDW is a lightweight, diversity-aware news recommender that runs random walks on a user-item bipartite graph. Unlike neural models that treat diversity as a post-processing afterthought, D-RDW enforces a **Normative Target Distribution (NTD)** of article properties (sentiment, political party mentions, category) during the recommendation step itself.

The pipeline has three stages:

1. **Pre-processing** — Build a bipartite graph from user-item interactions; augment it with new/cold items using semantic similarity (via [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).
2. **In-processing** — Run random walks (3 hops), apply heuristic filters, then sample a candidate list that satisfies the NTD constraints using binary integer programming.
3. **Post-processing** — Re-rank the candidate list by random walk score, multiple objectives, or graph coloring (for homogeneous category display).

**Key results (EB-NeRD, top-20 recommendations):** D-RDW ranks in the top 3 across all four evaluation dimensions — normative RADio diversity, traditional Gini/ILD diversity, energy cost, and AUC — while being ~100× cheaper to run than neural baselines (2.81 W·s vs. 326–523 W·s).

## Datasets

| Dataset | Domain | Language |
|---------|--------|----------|
| [EB-NeRD](https://recsys.eb.dk/) | Danish news | Danish |
| [MIND](https://msnews.github.io) | Microsoft news | English |
| [NeMig](https://github.com/andreeaiana/nemig) | German news | German |

Data files go under `dataset/` and are not included in this repo for legal reasons.

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

recommenders/models/drdw/         # D-RDW model implementation (Cornac extension)
dataset/ebnerd_small/             # Raw parquet files (not committed)
```

## Baselines Compared

| Type | Models |
|------|--------|
| Neural | LSTUR, NPA, NRMS |
| Neural + re-ranking | + Greedy KL, PM-2, MMR, DYN-ATT, DYN-POS |
| Random walk | RP³β, RWE-D, D-RDW |
| Baseline | Random |

## Key Links

- [Informfully Recommenders](https://github.com/Informfully/Recommenders) — extended Cornac framework (required dependency)
- [D-RDW model source](https://github.com/Informfully/Recommenders/tree/main/cornac/models/drdw)
- [Graph preparation guide](https://github.com/Informfully/Experiments/blob/main/experiments/recsys_2025/graph_preparation/README.md)
- [EB-NeRD dataset](https://recsys.eb.dk/)
- [Sentence embeddings model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
