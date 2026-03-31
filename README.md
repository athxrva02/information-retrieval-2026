# Information Retrieval 2026 - Group 36

This is the reproduction and experimentation repository for [**D-RDW: Diversity-Driven Random Walks for News Recommender Systems**](https://dl.acm.org/doi/10.1145/3705328.3748016),
used for our group project for the Information Retrieval course in 2026.

The code in this repository is based on the code developed for the original D-RDW algorithm, which can be found at the
[Informfully/Experiments](https://github.com/Informfully/Experiments/tree/main) and [Informfully/Recommenders](https://github.com/Informfully/Recommenders/tree/main#)
repos.

## What is D-RDW?

D-RDW is a diversity-aware news recommender that utilizes random walks on a user-item bipartite graph. 
Unlike neural models that treat diversity as a post-processing afterthought, D-RDW enforces a **Normative Target 
Distribution (NTD)** of article properties (sentiment, political party mentions, category) during the recommendation 
step itself.

The pipeline has three stages:

1. Pre-processing: Build a bipartite graph from user-item interactions; augment it with new/cold items using 
semantic similarity.
2. In-processing: Run random walks (3 hops), apply heuristic filters, then sample a candidate list that satisfies 
the NTD constraints using binary integer programming.
3. Post-processing: Re-rank the candidate list.

## Setup

### Datasets

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

### Dependencies

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

### D-RDW Setup

There are many files that need to be generated prior to running the D-RDW algorithm. Which files to run, and how to run them,
is further explained in the [experiments README](./experiments/recsys_2025/README.md). Note that the code for running
the D-RDW experiment itself is different from this README. More information on how to run the D-RDW experiment can be seen below.

### macOS (Apple Silicon) Notes

On macOS with Apple clang 17+, the bundled Eigen library in Cornac fails to compile. Before installing Cornac, apply the
following patch to `Recommenders/cornac/utils/external/eigen/Eigen/src/Core/Transpositions.h` (line 387):

```diff
-      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt.derived());
+      return Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt);
```

Python 3.10 is recommended. Python 3.14 triggers the same Eigen compilation error regardless of the patch.

In addition, the file found at `Recommenders/cornac/augmentation/enrich_ne.py` differs depending on the OS. If
using macOS, uncomment the code at lines `110` and `173`, and comment the indicated code below them.

## Running D-RDW

Our project involved running the D-RDW algorithm on different NTDs. First we ran `experiments/recsys_2025/experiment_scripts/analyze_optimal_ntd.py`
to get different NTDs that might be useful to experiment with. More information on how these NTDs are calculated is in the project
paper. Different NTD configurations can be found in the `experiments/recsys_2025/experiment_scripts/ntd_configs` directory. The
NTDs that were used for our project paper can be found in `experiment_results`, each NTD having its own directory.

Once the NTD configs are created, the D-RDW algorithm can be run on the NTD configs. The original repository used the `drdw_experiment.py`
file in the `experiments/recsys_2025/experiment_scripts` directory. We modified the code to work with our NTD configs, resulting
in the `drdw_ntd_runner.py` file in the same directory. To run D-RDW, run the `drdw_ntd_runner.py`, with the desired config
as an argument. For example:
```
# Run a specific NTD config:
python drdw_ntd_runner.py --config ntd_configs/drdw_base_paper_config.json
```

The results are then saved to `./experiment_results/{config_name}/D_RDW/`. These results can be used for running accuracy
and diversity metrics.

## Accuracy Metrics

Refer to the accuracy metrics [README](./experiments/recsys_2025/evaluation_scripts/check_accuracy/README) for more information 
about the metrics used and how to compute them.

## Diversity Metrics

The diversity metrics code did not change from the original repository. Refer to the [experiments README](./experiments/recsys_2025/README.md)
for more information on how to run the diversity metrics. Make sure that the paths in the file that is being run point to
the correct model.

## Repository Structure

```
experiments/recsys_2025/
├── article_enrichment_scripts/   # Clean + enrich articles (sentiment, party, NER, etc.)
├── neural_preparation/           # Convert EB-NeRD behaviors to Cornac UIR format
├── graph_preparation/            # Build augmented bipartite graphs for random walk models
├── PLD_EPD_preparation/          # Prepare data for PLD/EPD filtering baselines
├── experiment_scripts/           # Run models (one script per model)
├── experiment_reranking_scripts/ # Apply re-ranking to neural outputs
├── evaluation_scripts/           # Compute accuracy and diversity metrics
└── final_recommendations/        # Pre-computed top-20 recommendation lists

Recommenders/                     # The Recommenders package - an extension of Cornac
```



## Key Links

- [Informfully Recommenders](https://github.com/Informfully/Recommenders)
- [D-RDW model source](https://github.com/Informfully/Recommenders/tree/main/cornac/models/drdw)
- [Graph preparation guide](https://github.com/Informfully/Experiments/blob/main/experiments/recsys_2025/graph_preparation/README.md)
- [EB-NeRD dataset](https://recsys.eb.dk/)
- [Sentence embeddings model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
