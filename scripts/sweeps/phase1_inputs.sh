#!/usr/bin/env bash
# Phase 1 (E1): which input works best? Same backbone (EfficientNet-B3), same recipe.
set -euo pipefail
for v in crop ctx1.5 ctx2.0 full; do
  python scripts/build_cache.py --records data/records.csv --variant $v --size 512 --workers ${WORKERS:-4}
done
for c in e1_crop_effb3 e1_ctx15_effb3 e1_ctx20_effb3 e1_ctx15_dualfull_effb3; do
  python scripts/run_cv.py --config configs/exp/$c.yaml
done
python scripts/evaluate.py --run runs/e1_*/seed42 --out results/e1_summary.csv
python scripts/compare.py --runs runs/e1_*/seed42 --out results/e1_compare.csv
