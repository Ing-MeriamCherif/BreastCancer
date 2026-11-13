"""Step 1: resolve every CBIS-DDSM lesion to (crop, mask, full) and audit the data.

    python scripts/audit_data.py --data-root data/cbis-ddsm --out data/records.csv
    python scripts/audit_data.py --data-root data/cbis-ddsm --out data/records_legacy.csv --resolver legacy

The legacy run additionally reports how many images the OLD notebooks fed to the model
that are actually binary ROI masks.
"""
import argparse
import json

import _path  # noqa: F401
import pandas as pd
from tqdm import tqdm

from bcr.data.cbis import build_records, image_stats

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--out", default="data/records.csv")
ap.add_argument("--resolver", choices=["robust", "legacy"], default="robust")
ap.add_argument("--no-pixel-check", action="store_true")
a = ap.parse_args()

rec = build_records(a.data_root, resolver=a.resolver, check_pixels=not a.no_pixel_check)

if a.resolver == "legacy":  # how many legacy "crops" are masks?
    flags = []
    for p in tqdm(rec["crop_path"], desc="checking legacy crops"):
        flags.append(image_stats(p)["is_binary"] if isinstance(p, str) else None)
    rec["legacy_crop_is_mask"] = flags

rec.to_csv(a.out, index=False)

def flat(d):
    return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): v for k, v in d.items()}


summary = {
    "rows": len(rec),
    "by_split_type": flat(rec.groupby(["split", "lesion_type"]).size().to_dict()),
    "crop_missing": int(rec["crop_path"].isna().sum()),
    "mask_missing": int(rec["mask_path"].isna().sum()),
    "full_missing": int(rec["full_path"].isna().sum()),
    "issues": rec["resolve_issues"].fillna("").str.split(";").explode().value_counts().drop("", errors="ignore").to_dict(),
    "pathology_by_split": flat(rec.groupby(["split", "pathology"]).size().to_dict()),
}
if "legacy_crop_is_mask" in rec:
    summary["legacy_crops_that_are_masks"] = int(rec["legacy_crop_is_mask"].fillna(False).sum())
print(json.dumps(summary, indent=2, default=str))
json.dump(summary, open(a.out.replace(".csv", "_audit.json"), "w"), indent=2, default=str)
