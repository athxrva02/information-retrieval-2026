#!/bin/bash
set -e

SCRIPTS=experiments/recsys_2025/experiment_scripts
CONFIGS=$SCRIPTS/ntd_configs

python $SCRIPTS/drdw_ntd_runner.py --config $CONFIGS/drdw_base_paper_config.json
python $SCRIPTS/drdw_ntd_runner.py --config $CONFIGS/optimal_oracle_pure.json
python $SCRIPTS/drdw_ntd_runner.py --config $CONFIGS/optimal_discriminative.json
python $SCRIPTS/drdw_ntd_runner.py --config $CONFIGS/optimal_feasibility_aware.json
python $SCRIPTS/drdw_ntd_runner.py --config $CONFIGS/natural_aligned_rdw_5hops.json
python experiments/recsys_2025/evaluation_scripts/compute_auc_for_config.py --all