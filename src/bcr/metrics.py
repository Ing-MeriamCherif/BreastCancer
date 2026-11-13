"""Evaluation metrics, confidence intervals and significance tests.

All functions take numpy arrays. Missing labels (-1 / valid=False) must be filtered
by the caller (see ``evaluate_heads``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score,
                             f1_score, roc_auc_score, roc_curve)


# --------------------------------------------------------------------------- #
# Binary / pathology
# --------------------------------------------------------------------------- #
def ece_fixed(p, y, n_bins=10):
    """Expected calibration error for P(positive); includes p == 1.0 in the last bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def ece_adaptive(p, y, n_bins=10):
    """Equal-mass bins (more stable with skewed confidence distributions)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(ece)


def spec_at_sens(y, p, target=0.90):
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= target
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def sens_at_spec(y, p, target=0.80):
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_for_sensitivity(y, p, target=0.90):
    """Largest threshold that still reaches the target sensitivity (choose on VALIDATION)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    return float(min(thr[ok[0]], 1.0)) if len(ok) else 0.0


def net_benefit(y, p, thresholds=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = {}
    for t in thresholds:
        pred = p >= t
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        out[f"nb@{t}"] = float(tp / n - fp / n * t / (1 - t))
    return out


def binary_metrics(y, p, threshold: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tp, tn = np.sum((pred == 1) & (y == 1)), np.sum((pred == 0) & (y == 0))
    fp, fn = np.sum((pred == 1) & (y == 0)), np.sum((pred == 0) & (y == 1))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "spec@sens0.90": spec_at_sens(y, p, 0.90),
        "spec@sens0.95": spec_at_sens(y, p, 0.95),
        "sens@spec0.80": sens_at_spec(y, p, 0.80),
        "threshold": float(threshold),
        "sensitivity": float(sens), "specificity": float(spec), "precision": float(prec),
        "f1": float(2 * prec * sens / max(prec + sens, 1e-12)),
        "balanced_acc": float((sens + spec) / 2),
        "accuracy": float(np.mean(pred == y)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece_fixed(p, y), "ece_adaptive": ece_adaptive(p, y),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        **net_benefit(y, p),
    }


# --------------------------------------------------------------------------- #
# Ordinal / categorical / multi-label
# --------------------------------------------------------------------------- #
def ordinal_metrics(y, probs) -> dict:
    y = np.asarray(y).astype(int)
    probs = np.asarray(probs, float)
    pred = probs.argmax(1)
    labels = np.arange(probs.shape[1])
    out = {
        "qwk": float(cohen_kappa_score(y, pred, weights="quadratic", labels=labels)),
        "mae": float(np.mean(np.abs(pred - y))),
        "mae_expected": float(np.mean(np.abs(probs @ labels - y))),
        "acc": float(np.mean(pred == y)),
        "acc_pm1": float(np.mean(np.abs(pred - y) <= 1)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)),
    }
    out["macro_auc"] = _macro_ovr_auc(y, probs)
    return out


def _macro_ovr_auc(y, probs):
    aucs = []
    for k in range(probs.shape[1]):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            aucs.append(roc_auc_score(yk, probs[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def categorical_metrics(y, probs) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(probs).argmax(1)
    return {
        "acc": float(np.mean(pred == y)),
        "balanced_acc": float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)])),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)),
        "macro_auc": _macro_ovr_auc(y, np.asarray(probs)),
    }


def multilabel_metrics(Y, P, classes, threshold=0.5) -> dict:
    Y, P = np.asarray(Y).astype(int), np.asarray(P, float)
    pred = (P >= threshold).astype(int)
    out, aucs = {}, []
    for k, c in enumerate(classes):
        if 0 < Y[:, k].sum() < len(Y):
            a = roc_auc_score(Y[:, k], P[:, k])
            aucs.append(a)
            out[f"auc_{c}"] = float(a)
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_f1"] = float(f1_score(Y, pred, average="macro", zero_division=0))
    out["exact_match"] = float(np.mean((pred == Y).all(1)))
    return out


# --------------------------------------------------------------------------- #
# DeLong (fast implementation, Sun & Xu 2014)
# --------------------------------------------------------------------------- #
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _delong_cov(preds: np.ndarray, y: np.ndarray):
    """preds: (n_models, n_samples). Returns (aucs, covariance)."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.concatenate([pos, neg], 1)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01).reshape(k, k) / m + np.cov(v10).reshape(k, k) / n
    return aucs, cov


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y).astype(int)
    auc, cov = _delong_cov(np.asarray(p, float)[None], y)
    se = np.sqrt(cov[0, 0])
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc[0]), float(max(0, auc[0] - z * se)), float(min(1, auc[0] + z * se))


def delong_test(y, p1, p2):
    """Two-sided paired DeLong test. Returns (auc1, auc2, p_value)."""
    y = np.asarray(y).astype(int)
    aucs, cov = _delong_cov(np.vstack([p1, p2]).astype(float), y)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
# Patient-level bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(metric_fn, groups, *arrays, n_boot=2000, seed=0, alpha=0.05):
    """Resample PATIENTS (not images) with replacement. metric_fn(*arrays_subset)->float."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            vals.append(metric_fn(*[np.asarray(a)[idx] for a in arrays]))
        except ValueError:  # e.g. a resample with a single class
            continue
    vals = np.asarray(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(metric_fn, groups, y, p1, p2, n_boot=2000, seed=0):
    """CI and two-sided p-value for metric(p1) - metric(p2), patient-level resampling."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    obs = metric_fn(y, p1) - metric_fn(y, p2)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            diffs.append(metric_fn(y[idx], p1[idx]) - metric_fn(y[idx], p2[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(obs), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(min(p, 1.0))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
