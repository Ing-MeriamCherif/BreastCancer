"""Offline evaluation of a CV run directory.

Protocol (no test-set tuning):
  1. Out-of-fold (OOF) validation logits are pooled across folds.
  2. A temperature per head is fitted on the OOF logits (NLL).
  3. The pathology decision threshold is chosen on the calibrated OOF probabilities
     with the rule  sensitivity >= target  (default 0.90).
  4. Test predictions = mean of the fold models' logits, calibrated with (2),
     thresholded with (3). Single-fold test metrics are also reported (mean ± SD).
  5. 95% CIs: DeLong for AUC, patient-level bootstrap for everything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from bcr import metrics as M
from bcr.data.labels import HeadSpace
from bcr.losses import logits_to_probs


def load_run(run_dir: Path):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    summ = json.load(open(run_dir / "data_summary.json"))
    spaces = {n: HeadSpace(n, h["type"], h["column"], h["classes"]) for n, h in summ["heads"].items()}

    def targets(split):
        z = np.load(run_dir / f"targets_{split}.npz")
        out = {}
        for key in z.files:
            h, k = key.split("__")
            out.setdefault(h, {})[k] = z[key]
        return out

    folds = sorted(run_dir.glob("predictions/fold*.npz"))
    preds = [dict(np.load(f)) for f in folds]
    return cfg, spaces, targets("train"), targets("test"), preds


def fit_temperature(logits: np.ndarray, y: np.ndarray, space: HeadSpace, loss_name: str,
                    bounds=(0.05, 20.0)) -> float:
    """Single temperature minimising validation NLL (bounded 1-D search on log T)."""
    from scipy.optimize import minimize_scalar

    z = torch.from_numpy(np.asarray(logits, np.float32))
    yt = torch.from_numpy(np.asarray(y))
    F = torch.nn.functional

    @torch.no_grad()
    def nll(log_t: float) -> float:
        T = float(np.exp(log_t))
        if space.type == "multilabel":
            return float(F.binary_cross_entropy_with_logits(z / T, yt.float()))
        probs = logits_to_probs(z, space, loss_name, T).clamp_min(1e-8)
        return float(F.nll_loss(probs.log(), yt.long()))

    res = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    T = float(np.exp(res.x))
    return T if np.isfinite(T) and nll(np.log(T)) <= nll(0.0) else 1.0


def _valid(t: dict) -> np.ndarray:
    return t["valid"] if "valid" in t else t["y"] != -1


def _probs(logits, space, loss_name, T):
    return logits_to_probs(torch.from_numpy(logits), space, loss_name, T).numpy()


def head_metrics(space: HeadSpace, y, P, threshold=None) -> dict:
    if space.type == "binary":
        return M.binary_metrics(y, P[:, 1], threshold)
    if space.type == "ordinal":
        return M.ordinal_metrics(y, P)
    if space.type == "categorical":
        return M.categorical_metrics(y, P)
    return M.multilabel_metrics(y, P, space.classes)


def evaluate_run(run_dir: str | Path, target_sens: float = 0.90, n_boot: int = 2000) -> dict:
    run_dir = Path(run_dir)
    cfg, spaces, tg_tr, tg_te, preds = load_run(run_dir)
    if not preds:
        raise FileNotFoundError(f"no predictions in {run_dir}/predictions")
    loss_names = {n: cfg["heads"][n].get("loss", {}).get("name", "ce") for n in spaces}
    test_idx = pd.read_csv(run_dir / "test_index.csv")
    groups_te = test_idx["patient_id"].values
    n_train = len(tg_tr["pathology"]["y"])

    results = {"run": str(run_dir), "n_folds_found": len(preds), "heads": {}}

    # ---- pool OOF logits ------------------------------------------------------
    oof = {n: None for n in spaces}
    covered = np.zeros(n_train, bool)
    for p in preds:
        vi = p["val_idx"]
        covered[vi] = True
        for n in spaces:
            arr = p[f"val__{n}"]
            if oof[n] is None:
                oof[n] = np.full((n_train, arr.shape[1]), np.nan, np.float32)
            oof[n][vi] = arr

    temps, thr = {}, None
    for n, sp in spaces.items():
        m = _valid(tg_tr[n]) & covered
        y = tg_tr[n]["y"][m]
        temps[n] = fit_temperature(oof[n][m], y, sp, loss_names[n]) if cfg.get("eval", {}).get(
            "temperature", True) else 1.0
        P = _probs(oof[n][m], sp, loss_names[n], temps[n])
        if n == "pathology":
            thr = M.threshold_for_sensitivity(y, P[:, 1], target_sens)
        results["heads"].setdefault(n, {})["oof"] = head_metrics(sp, y, P, thr)
    results["temperatures"] = temps
    results["pathology_threshold"] = thr

    # ---- test: ensemble + single folds ------------------------------------------
    ens_probs = {}
    for n, sp in spaces.items():
        m = _valid(tg_te[n])
        y = tg_te[n]["y"][m]
        ens_logits = np.mean([p[f"test__{n}"] for p in preds], 0)
        P = _probs(ens_logits, sp, loss_names[n], temps[n])
        ens_probs[n] = P
        res = head_metrics(sp, y, P[m], thr)
        single = [head_metrics(sp, y, _probs(p[f"test__{n}"], sp, loss_names[n], temps[n])[m], thr) for p in preds]
        key = {"binary": "auc", "ordinal": "qwk"}.get(sp.type, "macro_auc")
        vals = [s[key] for s in single]
        res[f"single_fold_{key}_mean"], res[f"single_fold_{key}_sd"] = float(np.mean(vals)), float(np.std(vals))

        g = groups_te[m]
        if sp.type == "binary":
            p1 = P[m][:, 1]
            res["auc_delong_ci"] = M.delong_ci(y, p1)[1:]
            res["spec@sens0.90_ci"] = M.bootstrap_ci(lambda a, b: M.spec_at_sens(a, b, 0.9), g, y, p1, n_boot=n_boot)
            res["sensitivity_ci"] = M.bootstrap_ci(lambda a, b: float(np.mean(b[a == 1] >= thr)), g, y, p1,
                                                   n_boot=n_boot)
            res["specificity_ci"] = M.bootstrap_ci(lambda a, b: float(np.mean(b[a == 0] < thr)), g, y, p1,
                                                   n_boot=n_boot)
        elif sp.type == "ordinal":
            labels = np.arange(P.shape[1])
            res["qwk_ci"] = M.bootstrap_ci(
                lambda a, b: M.cohen_kappa_score(a, b.argmax(1), weights="quadratic", labels=labels),
                g, y, P[m], n_boot=n_boot)
        results["heads"][n]["test"] = res

    # ---- report-level (structured report) ---------------------------------------
    results["report"] = report_level(spaces, tg_te, ens_probs, thr, test_idx)
    out = run_dir / "metrics.json"
    json.dump(results, open(out, "w"), indent=2, default=float)

    # flat prediction file for model comparison
    pred_df = test_idx.copy()
    pred_df["y_pathology"] = tg_te["pathology"]["y"]
    pred_df["p_malignant"] = ens_probs["pathology"][:, 1]
    for n, sp in spaces.items():
        if sp.type in {"ordinal", "categorical"}:
            pred_df[f"y_{n}"] = tg_te[n]["y"]
            pred_df[f"pred_{n}"] = ens_probs[n].argmax(1)
    pred_df.to_csv(run_dir / "test_predictions.csv", index=False)
    return results


def report_level(spaces, tg_te, probs, thr, test_idx) -> dict:
    """Exact-match of the discrete report and internal-consistency rate."""
    n = len(test_idx)
    correct = np.ones(n, bool)
    fields = {}
    for name, sp in spaces.items():
        t = tg_te[name]
        v = _valid(t)
        if sp.type == "binary":
            pred = (probs[name][:, 1] >= thr).astype(int)
            ok = pred == t["y"]
        elif sp.type == "multilabel":
            pred = (probs[name] >= 0.5).astype(int)
            ok = (pred == t["y"]).all(1)
        else:
            pred = probs[name].argmax(1)
            ok = pred == t["y"]
        ok = np.where(v, ok, True)  # fields that do not apply cannot be wrong
        fields[name] = float(ok[v].mean()) if v.any() else float("nan")
        correct &= ok
    out = {"field_accuracy": fields, "exact_match": float(correct.mean())}

    if "assessment" in spaces:
        classes = [int(c) for c in spaces["assessment"].classes]
        birads = np.array(classes)[probs["assessment"].argmax(1)]
        mal = probs["pathology"][:, 1] >= thr
        contradiction = (mal & (birads <= 3)) | (~mal & (birads >= 5))
        out["consistency_rate"] = float(1 - contradiction.mean())
    return out


def summary_row(results: dict) -> dict:
    h = results["heads"]
    row = {"run": results["run"]}
    p = h["pathology"]["test"]
    row.update({
        "path_auc": p["auc"], "path_auc_lo": p["auc_delong_ci"][0], "path_auc_hi": p["auc_delong_ci"][1],
        "spec@sens0.90": p["spec@sens0.90"], "sens": p["sensitivity"], "spec": p["specificity"],
        "auprc": p["auprc"], "brier": p["brier"], "ece": p["ece"],
        "path_auc_single_mean": p["single_fold_auc_mean"], "path_auc_single_sd": p["single_fold_auc_sd"],
    })
    for n, v in h.items():
        t = v["test"]
        if "qwk" in t:
            row[f"{n}_qwk"] = t["qwk"]
        elif n != "pathology":
            row[f"{n}_macro_auc"] = t.get("macro_auc")
    row["report_exact_match"] = results["report"]["exact_match"]
    row["consistency_rate"] = results["report"].get("consistency_rate")
    return row
