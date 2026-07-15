"""
Mammography Preprocessing Pipeline - Step-by-Step PNG Export
--------------------------------------------------------------
Takes one or more mammography images from a folder and saves a PNG
for EVERY stage of the pipeline, exactly as it actually happens
(no forced/fake steps - if noise or gamma get randomly skipped,
that stage's PNG is just a copy of the previous stage, and the log
says so).

Stages saved per image:
    step0_original.png              - as loaded (RGB)
    step1_geometric.png              - after flip/rotation/crop/shear
    step2_intensity.png              - after brightness/contrast jitter
    step3_gaussian_noise.png         - after noise (or copy of step2 if skipped)
    step4_gamma.png                  - after gamma correction (or copy of step3 if skipped)
    step5_clip_normalized_viewable.png - CLIP-normalized tensor, denormalized back
                                          to a viewable image
    step5_clip_normalized_tensor.pt  - the actual raw tensor fed to the model

Usage:
    python mammography_pipeline_export.py --input_dir /path/to/images --output_dir ./pipeline_output --seed 42
"""

import argparse
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

try:
    from transformers import CLIPProcessor
except ImportError:
    CLIPProcessor = None

SUPPORTED_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# ============================================================
# EDIT THESE TO POINT TO YOUR OWN FILE/FOLDER
# ============================================================
# Note: you gave a single .jpg file path, not a folder. This script
# processes whole folders (so it can handle 1 or many images), so
# INPUT_DIR should be the FOLDER containing that image - the script
# will pick up every supported image inside it automatically.
INPUT_DIR = r"D:\BreastCancer - Copie\data\jpeg\1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"
OUTPUT_DIR = r"D:\BreastCancer - Copie\pipeline_output"
SEED = 42  # set to None for different random augmentation each run
# ============================================================

# ============================================================
# EDIT THESE TWO LINES TO POINT TO YOUR OWN FOLDERS
# ============================================================
INPUT_DIR = r"D:\BreastCancer - Copie\data\jpeg\1.3.6.1.4.1.9590.100.1.2.100131208110604806117271735422083351547"
OUTPUT_DIR = r"D:\BreastCancer - Copie\pipeline_output"       # <-- where the step PNGs will be saved
# ============================================================

# Fallback constants (used only if CLIPProcessor can't be downloaded,
# e.g. no internet access). These are CLIP's well-known published
# normalization constants for openai/clip-vit-base-patch32.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def log(msg):
    print(msg)


def process_one_image(path, out_dir, processor, seed=None):
    base = os.path.splitext(os.path.basename(path))[0]
    stage_dir = os.path.join(out_dir, base)
    os.makedirs(stage_dir, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    log("\n" + "=" * 70)
    log(f"PROCESSING: {os.path.basename(path)}")
    log("=" * 70)

    try:
        img = Image.open(path).convert('RGB')
        log(f"[0] Loaded image. size={img.size}, mode={img.mode}")
    except Exception as e:
        log(f"[0] ERROR loading image ({e}). Using gray placeholder.")
        img = Image.new('RGB', (224, 224), color=128)
    img.save(os.path.join(stage_dir, "step0_original.png"))

    # ---- Step 1: Geometric augmentation ----
    geometric = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=5),
    ])
    img = geometric(img)
    log(f"[1] Geometric augmentation applied (flip/rotation/resized-crop/shear). size={img.size}")
    img.save(os.path.join(stage_dir, "step1_geometric.png"))

    # ---- Step 2: Intensity jitter ----
    intensity = transforms.Compose([
        transforms.ColorJitter(brightness=0.35, contrast=0.45, saturation=0.0, hue=0.0),
    ])
    img = intensity(img)
    log("[2] Intensity jitter applied (brightness/contrast; saturation/hue skipped - grayscale).")
    img.save(os.path.join(stage_dir, "step2_intensity.png"))

    # ---- Step 3: Gaussian noise (50% chance, matching training code) ----
    if random.random() < 0.5:
        tensor = TF.to_tensor(img)
        noise = torch.randn_like(tensor) * 0.015
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        img = TF.to_pil_image(tensor)
        log("[3] Gaussian noise APPLIED (std=0.015).")
    else:
        log("[3] Gaussian noise SKIPPED (50/50 random draw came up skip). Stage image = step2 unchanged.")
    img.save(os.path.join(stage_dir, "step3_gaussian_noise.png"))

    # ---- Step 4: Gamma correction (40% chance, matching training code) ----
    if random.random() < 0.4:
        gamma = random.uniform(0.8, 1.25)
        tensor = TF.to_tensor(img)
        tensor = torch.clamp(tensor ** gamma, 0.0, 1.0)
        img = TF.to_pil_image(tensor)
        log(f"[4] Gamma correction APPLIED (gamma={gamma:.3f}).")
    else:
        log("[4] Gamma correction SKIPPED (40% random draw came up skip). Stage image = step3 unchanged.")
    img.save(os.path.join(stage_dir, "step4_gamma.png"))

    # ---- Step 5: CLIP normalization ----
    if processor is not None:
        img_in = processor(images=img, return_tensors='pt')
        final_tensor = img_in['pixel_values'].squeeze(0)
        log("[5] CLIPProcessor normalization applied (REAL, downloaded from Hugging Face).")
    else:
        # Manual fallback using CLIP's published constants.
        tensor = TF.to_tensor(img)
        final_tensor = TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)
        log("[5] CLIPProcessor unavailable -> applied CLIP's published mean/std constants "
            "MANUALLY as a fallback (not verified against the real processor).")

    log(f"    Final tensor shape={tuple(final_tensor.shape)}, "
        f"range=[{final_tensor.min().item():.3f}, {final_tensor.max().item():.3f}], "
        f"mean={final_tensor.mean().item():.4f}, std={final_tensor.std().item():.4f}")

    torch.save(final_tensor, os.path.join(stage_dir, "step5_clip_normalized_tensor.pt"))

    # Denormalize just so it can be viewed as a PNG
    mean_t = torch.tensor(CLIP_MEAN if processor is None else CLIP_MEAN).view(3, 1, 1)
    std_t = torch.tensor(CLIP_STD if processor is None else CLIP_STD).view(3, 1, 1)
    viewable = torch.clamp(final_tensor * std_t + mean_t, 0, 1)
    TF.to_pil_image(viewable).save(os.path.join(stage_dir, "step5_clip_normalized_viewable.png"))
    log(f"    Saved viewable (denormalized) version for visual inspection.")

    log(f"\nAll stage PNGs for '{base}' saved to: {stage_dir}")
    return stage_dir


def main():
    parser = argparse.ArgumentParser(description="Export every preprocessing stage as a real PNG")
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                         help='Folder containing one or more mammography images')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                         help='Where to save the stage PNGs (one subfolder per image)')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch32',
                         help='CLIP model name/path used by CLIPProcessor')
    parser.add_argument('--seed', type=int, default=SEED,
                         help='Random seed for reproducible augmentation (recommended: set this)')
    args, _unknown = parser.parse_known_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input_dir '{args.input_dir}' does not exist.")
        sys.exit(1)

    image_paths = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])
    if not image_paths:
        print(f"No supported images found in '{args.input_dir}'. Supported: {SUPPORTED_EXTS}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s):")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    processor = None
    if CLIPProcessor is not None:
        print(f"\nAttempting to load real CLIPProcessor ('{args.model_name}')...")
        try:
            processor = CLIPProcessor.from_pretrained(args.model_name)
            print("Loaded real CLIPProcessor successfully - step 5 will use it.")
        except Exception as e:
            print(f"Could not load CLIPProcessor ({e}).")
            print("Falling back to manual normalization using CLIP's published mean/std constants.")
    else:
        print("\n'transformers' not installed - install it with: pip install transformers")
        print("Falling back to manual normalization using CLIP's published mean/std constants.")

    os.makedirs(args.output_dir, exist_ok=True)
    for path in image_paths:
        process_one_image(path, args.output_dir, processor, seed=args.seed)

    print("\n" + "=" * 70)
    print(f"DONE. Output saved under: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()