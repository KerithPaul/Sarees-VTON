"""
Step 2: auto-detect the saree region in each selected pose photo.

Uses SegFormer B2 clothes (already downloaded in models/segformer_clothes).
It understands 18 clothing labels. For a saree it looks for:
  - upper-clothes (the blouse + draped body section)
  - skirt          (the pleated lower section)
  - dress          (sometimes the full saree is tagged as this)

AFTER RUNNING, you must manually clean each mask:
  1. Open poses/masks/pose_XX_mask_overlay.jpg to see what was detected
  2. Open poses/masks/pose_XX_mask_auto.png in GIMP or Photopea
     - Erase any face, hair, hands, or feet pixels that leaked in
     - Fill any holes inside the saree region (paint white)
     - Use a hard round brush at the edges for a clean border
  3. Save (overwrite) as poses/masks/pose_XX_mask.png  (no '_auto' suffix)

Run:
  python scripts/02_mask_generator.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_DIR = Path("models/segformer_clothes")
SEL_DIR   = Path("poses/selected")
MASK_DIR  = Path("poses/masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png"}

SAREE_LABELS = [4, 5, 7]
# 4=upper-clothes  5=skirt  7=dress
# Excluded (intentional): 2=hair  9=left-shoe  10=right-shoe
#                         11=face  12=left-leg  13=right-leg
#                         14=left-arm  15=right-arm

pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])
if not pose_files:
    print(f"No pose images in {SEL_DIR}/. Run 01_pose_selector.py first.")
    sys.exit(1)

print("Loading SegFormer B2 clothes...")
processor = SegformerImageProcessor.from_pretrained(str(MODEL_DIR))
model = SegformerForSemanticSegmentation.from_pretrained(str(MODEL_DIR))
model.eval()
print("Model ready.\n")

for pose_path in pose_files:
    name = pose_path.stem
    print(f"Processing {name} ...")

    pil = Image.open(pose_path).convert("RGB")
    W, H = pil.size

    inp = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits

    up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1)[0].numpy()

    mask = np.zeros((H, W), dtype=np.uint8)
    for lbl in SAREE_LABELS:
        mask[pred == lbl] = 255

    # Morphological cleanup
    k5  = np.ones((5, 5),  np.uint8)
    k11 = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k11, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)

    # Keep two largest connected components (body + pallu are sometimes separate blobs)
    n_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_cc > 3:
        areas = sorted(range(1, n_cc), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        clean = np.zeros_like(mask)
        for lbl in areas[:2]:
            clean[cc_map == lbl] = 255
        mask = clean

    # Save the auto mask (will be manually cleaned)
    auto_path = MASK_DIR / f"{name}_mask_auto.png"
    cv2.imwrite(str(auto_path), mask)

    # Coloured overlay so you can see what was detected at a glance
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[mask > 127] = (
        overlay[mask > 127].astype(float) * 0.45
        + np.array([20, 120, 255]) * 0.55
    ).astype(np.uint8)
    cv2.imwrite(str(MASK_DIR / f"{name}_mask_overlay.jpg"), overlay)

    # Feathered version (pre-computed for later compositing)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (21, 21), 0)
    cv2.imwrite(
        str(MASK_DIR / f"{name}_mask_feathered.png"),
        (soft * 255).astype(np.uint8),
    )

    coverage = mask.sum() // 255
    print(f"  Detected {coverage:,} pixels  → {auto_path.name}")

print("\n--- Done ---")
print("\nNow manually clean each mask:")
print("  1. Open poses/masks/*_mask_overlay.jpg to see the auto-detected region")
print("  2. Open poses/masks/*_mask_auto.png in GIMP or Photopea")
print("     - Erase face, hair, hands, feet that leaked in")
print("     - Fill gaps in the saree fabric")
print("  3. Save as poses/masks/NAME_mask.png  (drop '_auto' from the name)")
print("\nAfter cleaning ALL poses, run:")
print("  python scripts/03_shading_extractor.py")
