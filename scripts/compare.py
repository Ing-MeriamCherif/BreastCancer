"""Step 5: pairwise significance between evaluated runs (same official test set).

    python scripts/compare.py --runs runs/A/seed42 runs/B/seed42 runs/C/seed42 --out results/compare.csv

Pathology AUC: paired DeLong. Specificity at 90% sensitivity: paired patient-level
bootstrap. Ordinal heads: paired bootstrap on QWK. Holm correction per metric.
"""
import argparse
import itertools

import _path  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from bcr import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", default="results/compare.csv")
a = ap.parse_args()

preds = {r: pd.read_csv(f"{r}/test_predictions.csv") for r in a.runs}
base = preds[a.runs[0]]
for r, df in preds.items():
    assert (df["lesion_id"].values == base["lesion_id"].values).all(), f"{r}: different test rows"

y = base["y_pathology"].values
g = base["patient_id"].values
rows = []
for r1, r2 in itertools.combinations(a.runs, 2):
    p1, p2 = preds[r1]["p_malignant"].values, preds[r2]["p_malignant"].values
    auc1, auc2, p = M.delong_test(y, p1, p2)
    rows.append({"a": r1, "b": r2, "metric": "path_auc", "a_val": auc1, "b_val": auc2,
                 "diff": auc1 - auc2, "p": p})
    d, lo, hi, pb = M.paired_bootstrap_diff(lambda yy, pp: M.spec_at_sens(yy, pp, 0.9), g, y, p1, p2, a.n_boot)
    rows.append({"a": r1, "b": r2, "metric": "spec@sens0.90", "diff": d, "ci_lo": lo, "ci_hi": hi, "p": pb})
    for col in [c for c in base.columns if c.startswith("pred_")]:
        head = col[5:]
        yt = base[f"y_{head}"].values
        m = yt != -1
        f = lambda yy, pp: cohen_kappa_score(yy, pp, weights="quadratic")
        d, lo, hi, pb = M.paired_bootstrap_diff(f, g[m], yt[m], preds[r1][col].values[m],
                                                preds[r2][col].values[m], a.n_boot)
        rows.append({"a": r1, "b": r2, "metric": f"{head}_qwk", "diff": d, "ci_lo": lo, "ci_hi": hi, "p": pb})

df = pd.DataFrame(rows)
df["p_holm"] = np.nan
for metric, sub in df.groupby("metric"):
    adj = M.holm(dict(zip(sub.index, sub["p"])))
    for i, v in adj.items():
        df.loc[i, "p_holm"] = v
import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(df.round(4).to_string(index=False))
