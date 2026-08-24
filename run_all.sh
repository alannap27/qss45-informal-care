#!/usr/bin/env bash
# Rebuild every figure and table, in order.
# Script 00 is not in this loop. It reconstructs the analysis file from the raw
# HRS Section G helper records, which are not in this repository and cannot be
# redistributed under the HRS data use agreement. Register for free at
# hrs.isr.umich.edu, place h22g_hp.csv in data/raw/, and run it by hand:
# python3 code/00_build_analysis_file.py
# Scripts 01 through 09 run from data/processed/, which is also gitignored for
# the same reason. See the Data section of README.md.
set -e
cd "$(dirname "$0")"
for script in 01_explore 02_baseline_ols_vs_gbm 03_two_part_model \
              04_weights_and_clustered_cv 05_care_networks 06_neural_net \
              07_final_model_and_leakage 08_crossnational_embeddings \
              09_interactions; do
  echo "=== code/${script}.py ==="
  python3 "code/${script}.py" > "output/log_${script}.txt" 2>&1
done
echo
echo "figures:"; ls -1 output/figures/*.png
echo "tables:";  ls -1 output/tables
