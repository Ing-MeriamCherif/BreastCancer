"""Step 3: train with patient-level CV.

    python scripts/run_cv.py --config configs/exp/e1_crop512_effb3.yaml
    python scripts/run_cv.py --config configs/exp/A_legacy_clip_custom.yaml train.epochs=2
    python scripts/run_cv.py --config ... --seed 1 --folds 1 2
"""
import argparse
from pathlib import Path

import _path  # noqa: F401

from bcr.config import load_config
from bcr.train import run_cv

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run-dir", default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. optim.lr=3e-5")
a = ap.parse_args()

cfg = load_config(a.config, a.overrides)
if a.seed is not None:
    cfg["seed"] = a.seed
name = cfg.get("name", Path(a.config).stem)
run_dir = a.run_dir or f"runs/{name}/seed{cfg.get('seed', 42)}"
run_cv(cfg, run_dir, a.folds)
