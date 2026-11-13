"""Per-head losses, multi-task weighting, and logits -> probabilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bcr.data.labels import HeadSpace


# --------------------------------------------------------------------------- #
# Individual losses
# --------------------------------------------------------------------------- #
def gaussian_soft_labels(y: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    c = torch.arange(k, device=y.device, dtype=torch.float32)[None]
    soft = torch.exp(-0.5 * ((c - y.float()[:, None]) / sigma) ** 2)
    return soft / soft.sum(1, keepdim=True)


def corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CORN (Shi et al., 2023): task k is trained only on samples with y >= k."""
    total, count = logits.new_zeros(()), 0
    for k in range(logits.shape[1]):
        m = y >= k
        if m.any():
            total = total + F.binary_cross_entropy_with_logits(
                logits[m, k], (y[m] > k).float(), reduction="sum")
            count += int(m.sum())
    return total / max(count, 1)


def focal_loss(logits, y, gamma=2.0, weight=None):
    logp = F.log_softmax(logits, 1)
    logpt = logp.gather(1, y[:, None]).squeeze(1)
    w = weight[y] if weight is not None else 1.0
    return (-w * (1 - logpt.exp()) ** gamma * logpt).mean()


class HeadLoss(nn.Module):
    """Configured by heads.<name>.loss in the YAML, e.g.
    {name: ce, class_weight: [1, 3], label_smoothing: 0.0}
    {name: gaussian, sigma: 0.9} | {name: corn} | {name: focal, gamma: 2}
    {name: bce, pos_weight: auto, label_smoothing: 0.05}
    """

    def __init__(self, space: HeadSpace, cfg: dict, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.space, self.name = space, cfg.get("name", "ce")
        self.ls = float(cfg.get("label_smoothing", 0.0))
        self.sigma = float(cfg.get("sigma", 0.9))
        self.gamma = float(cfg.get("gamma", 2.0))
        cw = cfg.get("class_weight")
        self.register_buffer("cw", torch.tensor(cw, dtype=torch.float32) if cw else None)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, y):
        n = self.name
        if self.space.type == "multilabel":
            t = y * (1 - self.ls) + 0.5 * self.ls
            return F.binary_cross_entropy_with_logits(logits, t, pos_weight=self.pos_weight)
        if n == "ce":
            return F.cross_entropy(logits, y, weight=self.cw, label_smoothing=self.ls)
        if n == "focal":
            return focal_loss(logits, y, self.gamma, self.cw)
        if n == "gaussian":
            soft = gaussian_soft_labels(y, logits.shape[1], self.sigma)
            return -(soft * F.log_softmax(logits, 1)).sum(1).mean()
        if n == "corn":
            return corn_loss(logits, y)
        raise ValueError(n)


# --------------------------------------------------------------------------- #
# Multi-task combination
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """weighting: 'fixed' (heads.<name>.weight) or 'uncertainty' (Kendall et al., 2018):
    L = sum_h exp(-s_h) * L_h + s_h, with learnable s_h = log sigma_h^2."""

    def __init__(self, spaces: dict[str, HeadSpace], heads_cfg: dict, weighting: str = "fixed",
                 pos_weights: dict | None = None):
        super().__init__()
        self.fns = nn.ModuleDict({
            n: HeadLoss(sp, heads_cfg[n].get("loss", {}), (pos_weights or {}).get(n))
            for n, sp in spaces.items()
        })
        self.w = {n: float(heads_cfg[n].get("weight", 1.0)) for n in spaces}
        self.weighting = weighting
        if weighting == "uncertainty":
            self.log_vars = nn.ParameterDict({n: nn.Parameter(torch.zeros(())) for n in spaces})

    def forward(self, outputs: dict, batch: dict, device) -> tuple[torch.Tensor, dict]:
        total = None
        parts = {}
        for n, fn in self.fns.items():
            y = batch[n].to(device)
            valid = batch[f"{n}__valid"].to(device) if f"{n}__valid" in batch else (y != -1)
            if not valid.any():  # e.g. an all-calcification batch for the shape head
                continue
            l = fn(outputs[n][valid].float(), y[valid])
            parts[n] = float(l.detach())
            if self.weighting == "uncertainty":
                s = self.log_vars[n]
                term = torch.exp(-s) * l + s
            else:
                term = self.w[n] * l
            total = term if total is None else total + term
        if total is None:
            total = next(iter(outputs.values())).sum() * 0.0
        return total, parts


# --------------------------------------------------------------------------- #
def logits_to_probs(logits: torch.Tensor, space: HeadSpace, loss_name: str,
                    temperature: float = 1.0) -> torch.Tensor:
    z = logits.float() / temperature
    if space.type == "multilabel":
        return torch.sigmoid(z)
    if space.type == "ordinal" and loss_name in {"corn", "coral"}:
        cond = torch.sigmoid(z)
        surv = torch.cumprod(cond, 1) if loss_name == "corn" else cond  # P(y > k)
        ones = torch.ones_like(surv[:, :1])
        zeros = torch.zeros_like(surv[:, :1])
        upper = torch.cat([ones, surv], 1)
        lower = torch.cat([surv, zeros], 1)
        return (upper - lower).clamp_min(0)
    return torch.softmax(z, 1)
