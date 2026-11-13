"""Cross-validated training. Saves logits only; all metrics are computed offline by
``bcr.evaluate`` so thresholds/temperatures can be fitted on validation predictions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from bcr.config import save_config, seed_everything
from bcr.data.cbis import apply_split_policy
from bcr.data.dataset import CachedDataset, LegacyDataset, TensorMammoAugment
from bcr.data.labels import encode_all, fit_label_space
from bcr.data.splits import make_folds
from bcr.losses import MultiTaskLoss, logits_to_probs
from bcr.models.multihead import build_model, param_groups


# --------------------------------------------------------------------------- #
def subset_targets(targets: dict, idx) -> dict:
    return {n: {k: v[idx] for k, v in t.items()} for n, t in targets.items()}


def attach_cache(df: pd.DataFrame, cache_root: str, variant: str, size: int, col: str) -> pd.DataFrame:
    index = pd.read_csv(Path(cache_root) / f"{variant}_{size}" / "index.csv")
    index = index.rename(columns={"cache_path": col})[["lesion_id", col]]
    return df.merge(index, on="lesion_id", how="left")


def prepare_data(cfg: dict):
    d = cfg["data"]
    rec = pd.read_csv(d["records"])
    train, test, log = apply_split_policy(rec, legacy=d.get("legacy_split", False))
    if d["input"] != "legacy":
        for frame_name in ("train", "test"):
            df = train if frame_name == "train" else test
            df = attach_cache(df, d["cache_root"], d["input"], d["cache_size"], "cache_path")
            if d.get("full_input"):
                df = attach_cache(df, d["cache_root"], d["full_input"], d["cache_size"], "full_cache_path")
            n0 = len(df)
            keep = df["cache_path"].notna() & (df["full_cache_path"].notna() if d.get("full_input") else True)
            df = df[keep].reset_index(drop=True)
            log[f"{frame_name}_dropped_missing_cache"] = int(n0 - len(df))
            if frame_name == "train":
                train = df
            else:
                test = df
    spaces = fit_label_space(train, cfg["heads"])
    return train, test, spaces, log


def make_dataset(cfg, df, targets, mean, std, train: bool):
    d = cfg["data"]
    if d["input"] == "legacy":
        return LegacyDataset(df, targets, mean, std, augment=train)
    aug = TensorMammoAugment(cfg.get("augment")) if train else None
    return CachedDataset(df, targets, cfg["model"]["image_size"], mean, std, augment=aug,
                         full_path_col="full_cache_path" if d.get("full_input") else None)


def make_sampler(df: pd.DataFrame, mode: str | None):
    if not mode:
        return None
    if mode == "pathology":
        y = (df["pathology"] == "MALIGNANT").astype(int).values
    elif mode == "pathology_x_type":
        y = ((df["pathology"] == "MALIGNANT").astype(int) * 2 + (df["lesion_type"] == "mass")).values
    else:
        raise ValueError(mode)
    counts = np.bincount(y)
    w = 1.0 / counts[y]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)


def multilabel_pos_weights(spaces, targets, device):
    out = {}
    for n, sp in spaces.items():
        if sp.type == "multilabel":
            Y = targets[n]["y"][targets[n]["valid"]]
            pos = Y.sum(0)
            neg = len(Y) - pos
            out[n] = torch.as_tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32,
                                     device=device)
    return out


@torch.no_grad()
def predict(model, loader, device, amp_dtype) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list] = {}
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            o = model(batch["pixel_values"].to(device),
                      batch.get("full_pixel_values").to(device) if "full_pixel_values" in batch else None)
        for k, v in o.items():
            out.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def quick_val_score(logits: dict, targets: dict, spaces, heads_cfg, metric: str) -> float:
    """Model-selection score on the validation fold (default: pathology AUC)."""
    from sklearn.metrics import roc_auc_score

    if metric == "pathology_auc":
        y = targets["pathology"]["y"]
        p = logits_to_probs(torch.from_numpy(logits["pathology"]), spaces["pathology"], "ce")[:, 1].numpy()
        m = y != -1
        return float(roc_auc_score(y[m], p[m]))
    raise ValueError(metric)


# --------------------------------------------------------------------------- #
def run_cv(cfg: dict, run_dir: str | Path, folds_to_run: list[int] | None = None) -> None:
    run_dir = Path(run_dir)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = cfg["train"]
    amp_dtype = None
    if device.type == "cuda" and t.get("amp", True):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    train_df, test_df, spaces, split_log = prepare_data(cfg)
    tgt_train, tgt_test = encode_all(train_df, spaces), encode_all(test_df, spaces)
    folds = make_folds(train_df, cfg["cv"]["n_folds"], seed, cfg["cv"].get("method", "stratified_group"))

    train_df[["lesion_id", "patient_id", "lesion_type", "pathology"]].assign(
        fold=-1).to_csv(run_dir / "train_index.csv", index=False)
    test_df[["lesion_id", "patient_id", "lesion_type", "pathology"]].to_csv(run_dir / "test_index.csv", index=False)
    for name, tg in (("train", tgt_train), ("test", tgt_test)):
        np.savez_compressed(run_dir / f"targets_{name}.npz",
                            **{f"{h}__{k}": v for h, d in tg.items() for k, v in d.items()})
    json.dump({"split_log": split_log,
               "heads": {n: {"type": s.type, "column": s.column, "classes": s.classes} for n, s in spaces.items()},
               "folds": [{"train": len(a), "val": len(b)} for a, b in folds]},
              open(run_dir / "data_summary.json", "w"), indent=2, default=str)
    print(json.dumps(split_log, default=str))

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        if folds_to_run and k not in folds_to_run:
            continue
        print(f"\n===== fold {k}/{len(folds)}  train={len(tr_idx)} val={len(va_idx)} =====")
        seed_everything(seed + k)
        model = build_model(cfg["model"], spaces, cfg["heads"]).to(device)
        mean, std = model.backbone.mean, model.backbone.std

        df_tr, df_va = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)
        tg_tr, tg_va = subset_targets(tgt_train, tr_idx), subset_targets(tgt_train, va_idx)
        nw = t.get("num_workers", 4)
        sampler = make_sampler(df_tr, t.get("sampler", "pathology"))
        dl_tr = DataLoader(make_dataset(cfg, df_tr, tg_tr, mean, std, True), batch_size=t["batch_size"],
                           sampler=sampler, shuffle=sampler is None, num_workers=nw, drop_last=True,
                           pin_memory=device.type == "cuda", persistent_workers=nw > 0)
        eval_bs = t.get("eval_batch_size", t["batch_size"] * 2)
        dl_va = DataLoader(make_dataset(cfg, df_va, tg_va, mean, std, False), batch_size=eval_bs, num_workers=nw)
        dl_te = DataLoader(make_dataset(cfg, test_df, tgt_test, mean, std, False), batch_size=eval_bs,
                           num_workers=nw)

        crit = MultiTaskLoss(spaces, cfg["heads"], t.get("mtl_weighting", "fixed"),
                             multilabel_pos_weights(spaces, tg_tr, device)).to(device)
        o = cfg["optim"]
        groups = param_groups(model, o["lr"], o.get("backbone_lr_mult", 0.1), o.get("neck_lr_mult", 5.0),
                              o.get("head_lr_mult", 10.0), o.get("llrd", 1.0), o.get("weight_decay", 0.01))
        extra = [p for p in crit.parameters() if p.requires_grad]
        if extra:
            groups.append({"params": extra, "lr": o["lr"] * o.get("head_lr_mult", 10.0), "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]

        accum = t.get("grad_accum", 1)
        steps_per_epoch = max(1, len(dl_tr) // accum)
        total = steps_per_epoch * t["epochs"]
        warm = int(t.get("warmup_epochs", 0) * steps_per_epoch)
        min_ratio = t.get("min_lr_ratio", 0.01)

        def lr_factor(step):
            if step < warm:
                return (step + 1) / max(warm, 1)
            prog = (step - warm) / max(total - warm, 1)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

        best, bad, step, history = -1.0, 0, 0, []
        ckpt = run_dir / f"fold{k}_best.pt"
        for epoch in range(1, t["epochs"] + 1):
            model.train()
            t0, running, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(dl_tr):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(batch["pixel_values"].to(device, non_blocking=True),
                                batch["full_pixel_values"].to(device) if "full_pixel_values" in batch else None)
                loss, _ = crit(out, batch, device)
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip_grad", 1.0))
                    for g in opt.param_groups:
                        g["lr"] = g["initial_lr"] * lr_factor(step)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                running += float(loss.detach())
                nb += 1
            val_logits = predict(model, dl_va, device, amp_dtype)
            score = quick_val_score(val_logits, tg_va, spaces, cfg["heads"], t.get("select_metric", "pathology_auc"))
            history.append({"fold": k, "epoch": epoch, "train_loss": running / max(nb, 1), "val_score": score,
                            "sec": round(time.time() - t0, 1)})
            print(f"  epoch {epoch:3d}  loss {running / max(nb, 1):.4f}  val {score:.4f}  ({history[-1]['sec']}s)")
            if score > best:
                best, bad = score, 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= t.get("patience", 5):
                    print(f"  early stop at epoch {epoch} (best {best:.4f})")
                    break

        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        val_logits = predict(model, dl_va, device, amp_dtype)
        test_logits = predict(model, dl_te, device, amp_dtype)
        np.savez_compressed(run_dir / "predictions" / f"fold{k}.npz", val_idx=va_idx,
                            **{f"val__{n}": v for n, v in val_logits.items()},
                            **{f"test__{n}": v for n, v in test_logits.items()})
        pd.DataFrame(history).to_csv(run_dir / f"history_fold{k}.csv", index=False)
        if not t.get("keep_checkpoints", True):
            ckpt.unlink(missing_ok=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nDone. Evaluate with: python scripts/evaluate.py --run {run_dir}")
