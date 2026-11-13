#!/usr/bin/env bash
# Phase 0: reproduce the paper (configs A/B/C) with the new code, then quantify the mask bug.
set -euo pipefail
DATA=${DATA:-data/cbis-ddsm}
python scripts/audit_data.py --data-root "$DATA" --out data/records_legacy.csv --resolver legacy
python scripts/audit_data.py --data-root "$DATA" --out data/records.csv
for c in A_legacy_clip_custom B_legacy_clip_ce C_legacy_radimagenet; do
  python scripts/run_cv.py --config configs/exp/$c.yaml
done
# Same legacy recipe, but with correctly resolved crops (isolates the effect of bug #2)
python scripts/run_cv.py --config configs/exp/A_legacy_clip_custom.yaml --run-dir runs/A_fixed_crops/seed42 \
  data.records=data/records.csv
python scripts/evaluate.py --run runs/A_legacy_clip_custom/seed42 runs/B_legacy_clip_ce/seed42 \
  runs/C_legacy_radimagenet/seed42 runs/A_fixed_crops/seed42 --out results/phase0_summary.csv
python scripts/compare.py --runs runs/A_legacy_clip_custom/seed42 runs/B_legacy_clip_ce/seed42 \
  runs/A_fixed_crops/seed42 --out results/phase0_compare.csv
