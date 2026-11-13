"""Backbone factory.

Every backbone returns a (B, D) embedding and exposes:
  out_dim, mean, std             normalisation to use for its input
  blocks()                       ordered shallow -> deep list of modules (for last-k / LLRD)
  extra_params()                 projection-like params trained with `neck_lr_mult`
  enable_grad_checkpointing()

Spec strings (``model.backbone`` in the YAML):
  hf_clip:openai/clip-vit-base-patch32         HuggingFace CLIP vision tower (+projection)
  open_clip:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  timm:resnet50.a1_in1k | timm:efficientnet_b3.ra2_in1k | timm:convnext_tiny.fb_in22k
  timm:vit_base_patch14_dinov2.lvd142m
  radimagenet_resnet50:<path to RadImageNet_pytorch/ResNet50.pt>
  mammoclip:<path to Mammo-CLIP checkpoint>   (EfficientNet-B2/B5 image encoder)
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Backbone(nn.Module):
    out_dim: int
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def blocks(self) -> list[nn.Module]:
        raise NotImplementedError

    def extra_params(self) -> list[nn.Parameter]:
        return []

    def always_frozen(self) -> list[nn.Parameter]:
        """Parameters that must never be trained (e.g. frozen BatchNorm)."""
        return []

    def enable_grad_checkpointing(self) -> None:
        pass


# --------------------------------------------------------------------------- #
class HFClipBackbone(Backbone):
    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, name: str, image_size: int = 224, use_projection: bool = True):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        m = CLIPVisionModelWithProjection.from_pretrained(name)
        self.vision_model = m.vision_model
        self.visual_projection = m.visual_projection if use_projection else None
        self.native = m.config.image_size
        self.image_size = image_size
        self.out_dim = m.config.projection_dim if use_projection else m.config.hidden_size

    def forward(self, x):
        kw = {"interpolate_pos_encoding": True} if self.image_size != self.native else {}
        out = self.vision_model(pixel_values=x, **kw).pooler_output
        return self.visual_projection(out) if self.visual_projection is not None else out

    def blocks(self):
        return list(self.vision_model.encoder.layers)

    def extra_params(self):
        return list(self.visual_projection.parameters()) if self.visual_projection is not None else []

    def enable_grad_checkpointing(self):
        self.vision_model.encoder.gradient_checkpointing = True


class OpenClipBackbone(Backbone):
    def __init__(self, name: str):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(name)
        self.visual = model.visual
        self.out_dim = model.visual.output_dim
        cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
        self.mean = tuple(cfg.get("mean", CLIP_MEAN))
        self.std = tuple(cfg.get("std", CLIP_STD))
        del model

    def forward(self, x):
        return self.visual(x)

    def blocks(self):
        v = self.visual
        if hasattr(v, "trunk") and hasattr(v.trunk, "blocks"):
            return list(v.trunk.blocks)
        if hasattr(v, "transformer"):
            return list(v.transformer.resblocks)
        raise AttributeError("cannot locate blocks in open_clip visual tower")

    def enable_grad_checkpointing(self):
        if hasattr(self.visual, "set_grad_checkpointing"):
            self.visual.set_grad_checkpointing(True)


class TimmBackbone(Backbone):
    def __init__(self, name: str, pretrained: bool = True, image_size: int | None = None):
        super().__init__()
        import timm

        kw = {}
        if "vit" in name or "dinov2" in name or "deit" in name:
            kw["dynamic_img_size"] = True
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
        self.out_dim = self.model.num_features
        cfg = self.model.pretrained_cfg or {}
        self.mean = tuple(cfg.get("mean", IMAGENET_MEAN))
        self.std = tuple(cfg.get("std", IMAGENET_STD))

    def forward(self, x):
        return self.model(x)

    def blocks(self):
        m = self.model
        if hasattr(m, "blocks"):          # ViT / EfficientNet (stages)
            return list(m.blocks)
        if hasattr(m, "stages"):          # ConvNeXt, etc.
            return list(m.stages)
        if hasattr(m, "layer4"):          # ResNet
            return [m.layer1, m.layer2, m.layer3, m.layer4]
        raise AttributeError(f"cannot locate blocks in {type(m).__name__}")

    def enable_grad_checkpointing(self):
        self.model.set_grad_checkpointing(True)


class RadImageNetResNet50(Backbone):
    """torchvision ResNet50 with RadImageNet weights.

    ``freeze_bn=True`` keeps every BatchNorm in eval mode with frozen affine params: with
    batch size 8 the running statistics are otherwise destroyed (the main reason the
    legacy RadImageNet baseline under-performed).
    """

    def __init__(self, weights_path: str | None, freeze_bn: bool = True):
        super().__init__()
        import torchvision

        r = torchvision.models.resnet50(weights=None)
        self.body = nn.Sequential(*list(r.children())[:-1])
        self.out_dim = 2048
        self.freeze_bn = freeze_bn
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            sd = {k.removeprefix("backbone."): v for k, v in sd.items()}
            missing, unexpected = self.body.load_state_dict(sd, strict=False)
            if missing:
                raise RuntimeError(f"RadImageNet: {len(missing)} missing keys, e.g. {missing[:3]}")
        if freeze_bn:
            for mod in self.body.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    for p in mod.parameters():
                        p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            for mod in self.body.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        return self

    def forward(self, x):
        return self.body(x).flatten(1)

    def always_frozen(self):
        if not self.freeze_bn:
            return []
        return [p for m in self.body.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters()]

    def blocks(self):
        return [self.body[4], self.body[5], self.body[6], self.body[7]]


class MammoCLIPBackbone(Backbone):
    """Image encoder of Mammo-CLIP (Ghosh et al., MICCAI 2024).

    Checkpoints: see https://github.com/batmanlab/Mammo-CLIP (EfficientNet-B2 / B5).
    Requires ``pip install efficientnet_pytorch``. The key layout of released checkpoints
    can differ between versions, so this loader strips known prefixes and FAILS LOUDLY if
    more than 2% of encoder weights are missing. Check the printed report on first use.
    """

    def __init__(self, ckpt_path: str, arch: str = "efficientnet-b5",
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        self.enc = EfficientNet.from_name(arch)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        cleaned = {}
        for k, v in sd.items():
            for pre in ("module.", "image_encoder.", "encoder.", "model."):
                k = k.removeprefix(pre)
            cleaned[k] = v
        missing, unexpected = self.enc.load_state_dict(cleaned, strict=False)
        n = len(self.enc.state_dict())
        print(f"[Mammo-CLIP] loaded {n - len(missing)}/{n} tensors; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0.02 * n:
            raise RuntimeError(f"Mammo-CLIP checkpoint does not match {arch}: e.g. {missing[:5]}")
        self.out_dim = self.enc._fc.in_features
        self.mean, self.std = tuple(mean), tuple(std)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.enc.extract_features(x)).flatten(1)

    def blocks(self):
        return list(self.enc._blocks)


# --------------------------------------------------------------------------- #
def build_backbone(cfg: dict) -> Backbone:
    spec = cfg["backbone"]
    kind, _, arg = spec.partition(":")
    size = cfg.get("image_size", 224)
    if kind == "hf_clip":
        return HFClipBackbone(arg, image_size=size, use_projection=cfg.get("use_projection", True))
    if kind == "open_clip":
        return OpenClipBackbone(arg)
    if kind == "timm":
        return TimmBackbone(arg, pretrained=cfg.get("pretrained", True), image_size=size)
    if kind == "radimagenet_resnet50":
        return RadImageNetResNet50(arg or None, freeze_bn=cfg.get("freeze_bn", True))
    if kind == "mammoclip":
        return MammoCLIPBackbone(arg, arch=cfg.get("arch", "efficientnet-b5"),
                                 mean=cfg.get("mean", (0.5, 0.5, 0.5)), std=cfg.get("std", (0.5, 0.5, 0.5)))
    raise ValueError(f"unknown backbone spec {spec}")
