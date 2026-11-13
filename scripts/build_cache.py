"""Step 2: pre-process images once into an 8-bit PNG cache.

    python scripts/build_cache.py --records data/records.csv --variant crop   --size 512
    python scripts/build_cache.py --records data/records.csv --variant ctx1.5 --size 512
    python scripts/build_cache.py --records data/records.csv --variant full   --size 512
"""
import argparse

import _path  # noqa: F401
import pandas as pd

from bcr.data.preprocess import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--records", required=True)
ap.add_argument("--variant", required=True, help="crop | ctx<factor> (e.g. ctx1.5) | full")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--cache-root", default="cache")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

rec = pd.read_csv(a.records)
rec = rec[rec["crop_path"].notna()]
df = build_cache(rec, a.variant, a.size, a.cache_root, a.workers)
print(df["cache_note"].fillna("").value_counts().to_string())
