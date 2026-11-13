# Structured BI-RADS Reporting from Mammography (CBIS-DDSM)

Multi-task prediction of the components of a BI-RADS report from mammography lesions:
pathology, BI-RADS assessment, subtlety, breast density, mass shape, mass margins,
calcification type and distribution. The predictions are assembled into a structured report.

This repository is being refactored from the original notebooks (`notebooks/legacy/`) into a
reproducible benchmark (`src/bcr/`). See **Roadmap** at the bottom.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Linux / WSL2 recommended
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m pytest -q tests/                               # ~1 min on CPU, uses synthetic data
```

## Data

Download the Kaggle JPEG release of CBIS-DDSM and place it as:

```
data/cbis-ddsm/csv/{mass,calc}_case_description_{train,test}_set.csv
data/cbis-ddsm/csv/dicom_info.csv
data/cbis-ddsm/jpeg/<SeriesInstanceUID>/*.jpg
```

Optional weights: `weights/RadImageNet_pytorch/ResNet50.pt` (RadImageNet) and a
Mammo-CLIP checkpoint (github.com/batmanlab/Mammo-CLIP, needs `pip install efficientnet_pytorch`).
All other backbones download automatically from Hugging Face / timm.

## Workflow

```bash
# 1. Resolve every lesion to (crop, ROI mask, full mammogram) and audit the data
python scripts/audit_data.py --data-root data/cbis-ddsm --out data/records.csv

# 2. Pre-process once (aspect-preserving, 8-bit PNG cache)
python scripts/build_cache.py --records data/records.csv --variant crop   --size 512
python scripts/build_cache.py --records data/records.csv --variant ctx1.5 --size 512   # lesion + context
python scripts/build_cache.py --records data/records.csv --variant full   --size 512   # whole breast (density)

# 3. Train with 5-fold patient-level CV (any value can be overridden on the command line)
python scripts/run_cv.py --config configs/exp/e2_effb3.yaml
python scripts/run_cv.py --config configs/exp/e2_effb3.yaml --seed 43 optim.lr=1e-4

# 4. Evaluate (temperature + threshold fitted on out-of-fold validation, never on test)
python scripts/evaluate.py --run runs/e2_effb3/seed42 runs/e2_clip_b16_448/seed42

# 5. Significance (paired DeLong, patient-level bootstrap, Holm correction)
python scripts/compare.py --runs runs/e2_effb3/seed42 runs/e2_clip_b16_448/seed42
```

Each phase of the roadmap has a ready-made script in `scripts/sweeps/`.

## Evaluation protocol

* Official CBIS-DDSM test split kept **intact**; train patients that also appear in test are removed from train.
* Model selection: stratified 5-fold GroupKFold by patient on the official train split (3 seeds).
* Test prediction: mean of the fold models' logits, calibrated with a per-head temperature fitted on pooled
  out-of-fold (OOF) logits.
* Pathology operating point: chosen on OOF validation with the rule *sensitivity ≥ 0.90*, then frozen.
* Metrics:

| Field | Metrics |
|---|---|
| Pathology | AUROC (DeLong 95% CI), specificity @ 90/95% sensitivity, sensitivity @ 80% specificity, AUPRC, Brier, ECE (fixed + adaptive), decision-curve net benefit |
| Assessment, subtlety, density (ordinal) | quadratic weighted κ (bootstrap CI), MAE, ±1 accuracy, macro-F1, macro-AUROC |
| Shape, margins, calc type/distribution (multi-label) | macro-AUROC, per-descriptor AUROC, macro-F1 |
| Report | exact match of all fields, per-field accuracy, consistency rate (e.g. malignant + BI-RADS ≤ 3) |

## Configurations

| Config | Description |
|---|---|
| `A_legacy_clip_custom` | Paper config A reproduced (CLIP ViT-B/32, class-weighted + Gaussian ordinal + label smoothing) |
| `B_legacy_clip_ce` | Paper config B reproduced (plain CE) |
| `C_legacy_radimagenet` | Paper config C reproduced (RadImageNet ResNet50 with the CLIP hyper-parameters) |
| `e1_*` | Input study: crop vs context crops vs whole-breast stream for density |
| `e2_*` | Backbone benchmark: ResNet50, EfficientNet-B3/B5, ConvNeXt-T, RadImageNet (fair), CLIP B/32 & B/16@448, BiomedCLIP, DINOv2, Mammo-CLIP |
| `scripts/sweeps/phase3_losses.sh` | Loss / multi-task ablation (class weight, focal, Gaussian σ, CORN, uncertainty weighting, single-task) |

## Fixes relative to the legacy notebooks

1. Notebook names corrected: `ResnetNotebook.ipynb` contained plain CLIP (config B) and `ClipPlainNotebook.ipynb`
   contained ResNet50-RadImageNet (config C).
2. Image paths: the legacy loader took `files[0]` from each series folder, which can be the **binary ROI mask**
   instead of the crop. `scripts/audit_data.py --resolver legacy` counts how many training images were masks.
3. Official test set no longer modified (legacy dropped patient+laterality "conflicts" from test).
4. Thresholds and temperature fitted on pooled OOF validation only (legacy: fixed τ=0.45, temperature on fold-1 val).
5. TTA in the legacy notebook did not augment; it is not used.
6. ECE includes p = 1.0; adaptive ECE and Brier added.
7. RadImageNet baseline: BatchNorm statistics frozen and its own learning rate (legacy reused CLIP settings at batch 8).
8. Shape/margins as multi-label over base BI-RADS descriptors instead of 16–18 compound classes;
   calcification type/distribution heads added.
9. Breast density can be predicted from the whole mammogram (`full` stream) instead of a lesion crop.

## Project layout

```
src/bcr/data/      cbis.py (path resolution), labels.py, preprocess.py (cache), dataset.py, splits.py
src/bcr/models/    backbones.py (factory), multihead.py (heads, fine-tuning modes, layer-wise LR decay)
src/bcr/           losses.py, train.py (CV engine), evaluate.py, metrics.py (DeLong, bootstrap, QWK, ECE)
configs/           base.yaml, legacy_base.yaml, exp/*.yaml
scripts/           audit_data, build_cache, run_cv, evaluate, compare, sweeps/
tests/             synthetic CBIS-DDSM generator + unit / end-to-end tests
notebooks/legacy/  original notebooks (renamed) and the legacy package freeze
```

## Roadmap

- [x] Phase 0: refactor, bug fixes, reproducible configs, evaluation suite, tests
- [ ] Phase 0b: reproduce A/B/C on the real data; measure the effect of the mask bug
- [ ] Phase 1 (E1): input study (resolution, context, whole-breast density stream)
- [ ] Phase 2 (E2): backbone benchmark incl. Mammo-CLIP, 3 seeds
- [ ] Phase 3 (E3): loss / multi-task ablation at matched operating points
- [ ] Phase 4 (E4/E5): text-anchored heads, zero-shot, free-text VLM baseline, reader study
- [ ] Phase 5 (E6/E7): calibration, Grad-CAM pointing game, external validation (VinDr-Mammo, CMMD, INbreast)
- [ ] Phase 6: paper (MICCAI 2027), code and model release
