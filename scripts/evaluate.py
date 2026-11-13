"""Step 4: evaluate one or more run directories (OOF-calibrated, no test tuning).

    python scripts/evaluate.py --run runs/A_legacy_clip_custom/seed42 runs/B_legacy_clip_ce/seed42
"""
import argparse
import json

import _path  # noqa: F401
import pandas as pd

from bcr.evaluate import evaluate_run, summary_row

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs="+", required=True)
ap.add_argument("--target-sens", type=float, default=0.90)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", default="results/summary.csv")
a = ap.parse_args()

rows = []
for r in a.run:
    res = evaluate_run(r, a.target_sens, a.n_boot)
    rows.append(summary_row(res))
    print(f"\n### {r}")
    print(json.dumps(res["heads"]["pathology"]["test"], indent=1, default=float)[:1500])
    print("report:", json.dumps(res["report"], default=float))
df = pd.DataFrame(rows)
import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print("\n" + df.round(4).to_string(index=False))
print(f"\nsaved {a.out}")
