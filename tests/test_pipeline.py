"""Unit + end-to-end smoke tests. Run:  python -m pytest -q tests/"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from bcr import metrics as M  # noqa: E402
from bcr.data.cbis import apply_split_policy, build_records  # noqa: E402
from bcr.data.labels import HeadSpace  # noqa: E402
from bcr.losses import corn_loss, logits_to_probs  # noqa: E402


# --------------------------------------------------------------------------- #
def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p_good = y + rng.normal(0, 0.8, 400)
    p_bad = y + rng.normal(0, 3.0, 400)
    auc, lo, hi = M.delong_ci(y, p_good)
    assert abs(auc - roc_auc_score(y, p_good)) < 1e-9 and lo < auc < hi
    a1, a2, p = M.delong_test(y, p_good, p_bad)
    assert a1 > a2 and p < 0.01
    _, _, p_same = M.delong_test(y, p_good, p_good + 1e-9 * rng.normal(size=400))
    assert p_same > 0.5


def test_threshold_rule_reaches_target_sensitivity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(0, 1.2, 500))))
    t = M.threshold_for_sensitivity(y, p, 0.9)
    assert np.mean(p[y == 1] >= t) >= 0.9


def test_ece_includes_probability_one():
    assert M.ece_fixed(np.array([1.0, 1.0]), np.array([0, 0])) == pytest.approx(1.0)


def test_corn_probabilities_are_valid_distribution():
    sp = HeadSpace("a", "ordinal", "a", ["1", "2", "3", "4"])
    z = torch.randn(16, 3)
    p = logits_to_probs(z, sp, "corn")
    assert p.shape == (16, 4) and torch.allclose(p.sum(1), torch.ones(16), atol=1e-5)
    assert corn_loss(z, torch.randint(0, 4, (16,))) > 0


def test_holm_monotone():
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from make_synthetic import make

    root = tmp_path_factory.mktemp("cbis")
    make(root, n_patients=60, seed=0)
    return root


def test_robust_resolver_never_returns_masks(synth):
    from bcr.data.cbis import image_stats

    rec = build_records(synth, resolver="robust", progress=False)
    assert rec["crop_path"].notna().all()
    assert not any(image_stats(p)["is_binary"] for p in rec["crop_path"])
    assert rec["mask_path"].notna().all()
    assert all(image_stats(p)["is_binary"] for p in rec["mask_path"])


def test_legacy_resolver_reproduces_the_mask_bug(synth):
    from bcr.data.cbis import image_stats

    rec = build_records(synth, resolver="legacy", progress=False)
    n_masks = sum(image_stats(p)["is_binary"] for p in rec["crop_path"])
    assert n_masks > 0  # the bug we fixed


def test_official_test_set_kept_intact(synth):
    rec = build_records(synth, progress=False)
    tr, te, log = apply_split_policy(rec, legacy=False)
    assert len(te) == (rec["split"] == "test").sum()
    assert not set(tr["patient_id"]) & set(te["patient_id"])


def _run(args, cwd):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_end_to_end_cv_evaluate_compare(synth, tmp_path):
    rec = tmp_path / "records.csv"
    _run(["scripts/audit_data.py", "--data-root", str(synth), "--out", str(rec)], ROOT)
    for v in ["crop", "ctx1.5", "full"]:
        _run(["scripts/build_cache.py", "--records", str(rec), "--variant", v, "--size", "64",
              "--cache-root", str(tmp_path / "cache"), "--workers", "1"], ROOT)
    common = ["--config", "configs/base.yaml", f"data.records={rec}", f"data.cache_root={tmp_path / 'cache'}",
              "data.cache_size=64", "model.backbone=timm:resnet18", "model.pretrained=false", "model.image_size=64",
              "cv.n_folds=2", "train.epochs=2", "train.batch_size=8", "train.grad_accum=1", "train.num_workers=0",
              "train.patience=5", "heads.shape.min_count=1", "heads.margins.min_count=1",
              "heads.calc_type.min_count=1", "heads.calc_distribution.min_count=1"]
    runs = []
    for name, extra in [("ce", []),
                        ("corn_dual", ["data.input=ctx1.5", "data.full_input=full", "heads.density.source=full",
                                       "heads.assessment.loss={name: corn}", "train.mtl_weighting=uncertainty"])]:
        rd = tmp_path / "runs" / name
        _run(["scripts/run_cv.py", *common[:2], "--run-dir", str(rd), *common[2:], *extra], ROOT)
        runs.append(str(rd))
    out = _run(["scripts/evaluate.py", "--run", *runs, "--n-boot", "50", "--out", str(tmp_path / "s.csv")], ROOT)
    assert "path_auc" in out
    _run(["scripts/compare.py", "--runs", *runs, "--n-boot", "50", "--out", str(tmp_path / "c.csv")], ROOT)


def test_legacy_input_pipeline_runs(synth, tmp_path):
    rec = tmp_path / "records_legacy.csv"
    _run(["scripts/audit_data.py", "--data-root", str(synth), "--out", str(rec), "--resolver", "legacy"], ROOT)
    rd = tmp_path / "runs" / "legacy"
    _run(["scripts/run_cv.py", "--config", "configs/exp/A_legacy_clip_custom.yaml", "--run-dir", str(rd),
          f"data.records={rec}", "model.backbone=timm:resnet18", "model.pretrained=false",
          "train.epochs=1", "train.num_workers=0"], ROOT)
    _run(["scripts/evaluate.py", "--run", str(rd), "--n-boot", "20", "--out", str(tmp_path / "s.csv")], ROOT)


def test_finetune_modes_and_frozen_bn():
    import torch.nn as nn
    from bcr.models.backbones import RadImageNetResNet50, TimmBackbone
    from bcr.models.multihead import MultiHeadModel, configure_finetuning, param_groups

    sp = {"pathology": HeadSpace("pathology", "binary", "pathology", ["B", "M"])}
    hc = {"pathology": {"loss": {"name": "ce"}}}
    m = MultiHeadModel(TimmBackbone("resnet18", pretrained=False), sp, hc)
    n_all = sum(p.numel() for p in m.parameters())
    configure_finetuning(m, "last_k", 1)
    n_last = sum(p.numel() for p in m.parameters() if p.requires_grad)
    configure_finetuning(m, "full")  # must fully unfreeze again
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == n_all > n_last
    configure_finetuning(m, "linear")
    assert all(not p.requires_grad for p in m.backbone.parameters())

    r = MultiHeadModel(RadImageNetResNet50(None, freeze_bn=True), sp, hc)
    configure_finetuning(r, "full")
    bn = [x for x in r.modules() if isinstance(x, nn.BatchNorm2d)]
    r.train()
    assert all(not x.training for x in bn) and all(not p.requires_grad for x in bn for p in x.parameters())
    groups = param_groups(r, 1e-4)
    assert sum(len(g["params"]) for g in groups) == sum(1 for p in r.parameters() if p.requires_grad)
