"""Create a tiny synthetic dataset with the Kaggle CBIS-DDSM layout (for tests only).

Reproduces the pitfall that broke the legacy loader: for half of the lesions the cropped
image and the binary ROI mask live in the SAME series folder, with the mask sorted first.
Malignant lesions are brighter/spikier so a model can learn a non-trivial signal.
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SHAPES = ["ROUND", "OVAL", "IRREGULAR", "LOBULATED", "IRREGULAR-ARCHITECTURAL_DISTORTION"]
MARGINS = ["CIRCUMSCRIBED", "SPICULATED", "ILL_DEFINED", "OBSCURED", "CIRCUMSCRIBED-OBSCURED"]
CTYPES = ["PLEOMORPHIC", "AMORPHOUS", "PUNCTATE", "FINE_LINEAR_BRANCHING", "PLEOMORPHIC-AMORPHOUS"]
CDIST = ["CLUSTERED", "LINEAR", "SEGMENTAL", "REGIONAL"]


def uid():
    return "1.3.6.1.4.1.9590.100.1.2." + str(uuid.uuid4().int)[:30]


def make(root: Path, n_patients: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    (root / "csv").mkdir(parents=True, exist_ok=True)
    jpeg = root / "jpeg"
    rows = {k: [] for k in ["mass_train", "mass_test", "calc_train", "calc_test"]}
    info = []
    for i in range(n_patients):
        pid = f"P_{i:05d}"
        split = "test" if i % 5 == 0 else "train"
        for j in range(rng.integers(1, 3)):
            ltype = "mass" if rng.random() < 0.5 else "calc"
            mal = rng.random() < 0.4
            side, view = rng.choice(["LEFT", "RIGHT"]), rng.choice(["CC", "MLO"])
            W, H = 250, 400
            full = (rng.normal(70, 20, (H, W))).clip(0, 255)
            cx, cy, r = int(rng.integers(60, 190)), int(rng.integers(60, 340)), int(rng.integers(12, 30))
            yy, xx = np.ogrid[:H, :W]
            blob = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
            full[blob] += 80 if mal else 40
            if mal:
                full[blob] += rng.normal(0, 35, blob.sum())
            full = full.clip(0, 255).astype(np.uint8)
            mask = np.zeros((H, W), np.uint8)
            mask[blob] = 255
            crop = full[max(0, cy - r - 5):cy + r + 5, max(0, cx - r - 5):cx + r + 5]

            fu, cu, mu, st = uid(), uid(), uid(), uid()
            (jpeg / fu).mkdir(parents=True)
            Image.fromarray(full).save(jpeg / fu / "1-1.jpg")
            info.append({"image_path": f"CBIS-DDSM/jpeg/{fu}/1-1.jpg", "SeriesDescription": "full mammogram images"})
            same_folder = rng.random() < 0.5
            (jpeg / cu).mkdir(parents=True)
            if same_folder:  # mask sorts first -> legacy files[0] returns the MASK
                Image.fromarray(mask).save(jpeg / cu / "1-1.jpg")
                Image.fromarray(crop).save(jpeg / cu / "1-2.jpg")
                info += [{"image_path": f"CBIS-DDSM/jpeg/{cu}/1-1.jpg", "SeriesDescription": "ROI mask images"},
                         {"image_path": f"CBIS-DDSM/jpeg/{cu}/1-2.jpg", "SeriesDescription": "cropped images"}]
                mu = cu
            else:
                Image.fromarray(crop).save(jpeg / cu / "1-1.jpg")
                (jpeg / mu).mkdir(parents=True)
                Image.fromarray(mask).save(jpeg / mu / "1-1.jpg")
                info += [{"image_path": f"CBIS-DDSM/jpeg/{cu}/1-1.jpg", "SeriesDescription": "cropped images"},
                         {"image_path": f"CBIS-DDSM/jpeg/{mu}/1-1.jpg", "SeriesDescription": "ROI mask images"}]
            case = f"{'Mass' if ltype == 'mass' else 'Calc'}-{'Training' if split == 'train' else 'Test'}_{pid}_{side}_{view}"
            row = {
                "patient_id": pid, "left or right breast": side, "image view": view, "abnormality id": j + 1,
                "abnormality type": "mass" if ltype == "mass" else "calcification",
                "assessment": int(rng.choice([4, 5]) if mal else rng.choice([2, 3, 4])),
                "pathology": "MALIGNANT" if mal else rng.choice(["BENIGN", "BENIGN_WITHOUT_CALLBACK"]),
                "subtlety": int(rng.integers(1, 6)),
                "image file path": f"{case}/{st}/{fu}/000000.dcm",
                "cropped image file path": f"{case}_{j + 1}/{st}/{cu}/000000.dcm",
                "ROI mask file path": f"{case}_{j + 1}/{st}/{mu}/000001.dcm",
            }
            if ltype == "mass":
                row.update({"breast_density": int(rng.integers(1, 5)),
                            "mass shape": SHAPES[2 if mal else int(rng.integers(0, 5))],
                            "mass margins": MARGINS[1 if mal else int(rng.integers(0, 5))]})
            else:
                row.update({"breast density": int(rng.integers(1, 5)),
                            "calc type": CTYPES[int(rng.integers(0, 5))],
                            "calc distribution": CDIST[int(rng.integers(0, 4))]})
            rows[f"{ltype}_{split}"].append(row)
    names = {"mass_train": "mass_case_description_train_set.csv", "mass_test": "mass_case_description_test_set.csv",
             "calc_train": "calc_case_description_train_set.csv", "calc_test": "calc_case_description_test_set.csv"}
    for k, f in names.items():
        pd.DataFrame(rows[k]).to_csv(root / "csv" / f, index=False)
    pd.DataFrame(info).to_csv(root / "csv" / "dicom_info.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--patients", type=int, default=80)
    a = ap.parse_args()
    make(Path(a.out), a.patients)
    print("synthetic dataset written to", a.out)
