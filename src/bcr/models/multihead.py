"""Shared backbone + one head per structured BI-RADS field."""
from __future__ import annotations

import torch
import torch.nn as nn

from bcr.data.labels import HeadSpace
from bcr.models.backbones import Backbone, build_backbone


def head_out_dim(space: HeadSpace, loss_name: str) -> int:
    if space.type == "ordinal" and loss_name in {"corn", "coral"}:
        return space.n_out - 1  # K-1 rank-consistent binary tasks
    return space.n_out


class MultiHeadModel(nn.Module):
    def __init__(self, backbone: Backbone, spaces: dict[str, HeadSpace], heads_cfg: dict,
                 neck_dim: int | None = None, dropout: float = 0.3, neck_act: bool = True):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_dim
        self.neck = nn.Identity()
        if neck_dim:
            self.neck = nn.Sequential(nn.Linear(d, neck_dim), nn.GELU() if neck_act else nn.Identity())
            d = neck_dim
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(d, head_out_dim(sp, heads_cfg[name].get("loss", {}).get("name", "ce")))
            for name, sp in spaces.items()
        })
        # heads that read the whole-breast stream instead of the lesion stream
        self.full_heads = {n for n, hc in heads_cfg.items() if hc.get("source") == "full" and n in spaces}

    def embed(self, x):
        return self.dropout(self.neck(self.backbone(x)))

    def forward(self, pixel_values, full_pixel_values=None):
        z = self.embed(pixel_values)
        zf = self.embed(full_pixel_values) if (self.full_heads and full_pixel_values is not None) else None
        out = {}
        for name, head in self.heads.items():
            src = zf if (name in self.full_heads and zf is not None) else z
            out[name] = head(src)
        return out


# --------------------------------------------------------------------------- #
# Freezing and optimiser parameter groups
# --------------------------------------------------------------------------- #
def configure_finetuning(model: MultiHeadModel, mode: str, last_k: int = 4) -> None:
    """mode: 'linear' (frozen backbone), 'last_k' (legacy: last k blocks), 'full'."""
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad = mode == "full"
    if mode == "last_k":
        for blk in bb.blocks()[-last_k:]:
            for p in blk.parameters():
                p.requires_grad = True
    for p in bb.extra_params():
        p.requires_grad = mode != "linear"  # CLIP projection trained in last_k (legacy) and full
    for p in bb.always_frozen():
        p.requires_grad = False


def param_groups(model: MultiHeadModel, lr: float, backbone_lr_mult: float = 0.1,
                 neck_lr_mult: float = 5.0, head_lr_mult: float = 10.0,
                 llrd: float = 1.0, weight_decay: float = 0.01) -> list[dict]:
    """Layer-wise LR decay: block i of n gets lr*backbone_lr_mult*llrd**(n-1-i);
    parameters outside the blocks (stem/embeddings) get the deepest decay."""
    bb = model.backbone
    blocks = bb.blocks()
    n = len(blocks)
    seen, groups = set(), []

    def add(params, group_lr, tag):
        ps = [p for p in params if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in ps)
        if ps:
            groups.append({"params": ps, "lr": group_lr, "weight_decay": weight_decay, "tag": tag})

    base = lr * backbone_lr_mult
    for i in reversed(range(n)):
        add(blocks[i].parameters(), base * llrd ** (n - 1 - i), f"block{i}")
    add(bb.extra_params(), lr * neck_lr_mult, "projection")
    add(model.neck.parameters(), lr * neck_lr_mult, "neck")
    add(model.heads.parameters(), lr * head_lr_mult, "heads")
    add(bb.parameters(), base * llrd ** n, "stem")  # whatever remains in the backbone
    return groups


def build_model(model_cfg: dict, spaces: dict[str, HeadSpace], heads_cfg: dict) -> MultiHeadModel:
    bb = build_backbone(model_cfg)
    m = MultiHeadModel(bb, spaces, heads_cfg, neck_dim=model_cfg.get("neck_dim"),
                       dropout=model_cfg.get("dropout", 0.3), neck_act=model_cfg.get("neck_act", True))
    configure_finetuning(m, model_cfg.get("finetune", "last_k"), model_cfg.get("last_k", 4))
    if model_cfg.get("grad_checkpointing"):
        bb.enable_grad_checkpointing()
    return m
