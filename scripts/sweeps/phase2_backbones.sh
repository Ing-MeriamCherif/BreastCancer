#!/usr/bin/env bash
# Phase 2 (E2): backbone benchmark. First set data.input in configs/base.yaml to the E1 winner.
# Run each config with 3 seeds; quick LR check on fold 1 first, e.g.:
#   for lr in 1e-4 3e-4 1e-3; do python scripts/run_cv.py --config X --folds 1 --run-dir runs/lr/X_$lr optim.lr=$lr; done
set -euo pipefail
CONFIGS=${CONFIGS:-"e2_resnet50_in1k e2_effb3 e2_effb5 e2_convnext_t e2_radimagenet_fair e2_clip_b32_224 \
  e2_clip_b16_448 e2_biomedclip_224 e2_dinov2_b14_448 e2_mammoclip_b5"}
for c in $CONFIGS; do
  for s in 42 43 44; do python scripts/run_cv.py --config configs/exp/$c.yaml --seed $s; done
done
python scripts/evaluate.py --run runs/e2_*/seed* --out results/e2_summary.csv
python scripts/compare.py --runs runs/e2_*/seed42 --out results/e2_compare.csv
