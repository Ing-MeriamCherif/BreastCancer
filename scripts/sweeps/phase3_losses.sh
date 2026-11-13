#!/usr/bin/env bash
# Phase 3 (E3): loss / multi-task ablation on the best backbone. Set BEST to its config.
set -euo pipefail
BEST=${BEST:-configs/exp/e2_effb3.yaml}
run(){ name=$1; shift; python scripts/run_cv.py --config "$BEST" --run-dir runs/e3_$name/seed42 "$@"; }
run plain_ce
run classweight      'heads.pathology.loss={name: ce, class_weight: [1.0, 3.0]}'
run focal            'heads.pathology.loss={name: focal, gamma: 2.0}'
for s in 0.5 0.9 1.5; do
  run gauss_$s "heads.assessment.loss={name: gaussian, sigma: $s}" "heads.subtlety.loss={name: gaussian, sigma: $s}" \
               "heads.density.loss={name: gaussian, sigma: $s}"
done
run corn 'heads.assessment.loss={name: corn}' 'heads.subtlety.loss={name: corn}' 'heads.density.loss={name: corn}'
run uncertainty train.mtl_weighting=uncertainty
run single_task 'heads.assessment.enabled=false' 'heads.subtlety.enabled=false' 'heads.density.enabled=false' \
  'heads.shape.enabled=false' 'heads.margins.enabled=false' 'heads.calc_type.enabled=false' \
  'heads.calc_distribution.enabled=false'
python scripts/evaluate.py --run runs/e3_*/seed42 --out results/e3_summary.csv
python scripts/compare.py --runs runs/e3_plain_ce/seed42 runs/e3_*/seed42 --out results/e3_compare.csv
