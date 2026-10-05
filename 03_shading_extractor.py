"""
Step 3: extract the shading map (fold and lighting structure) from each pose.

WHY THIS MATTERS:
  At runtime, when you warp a new saree texture onto her body, it looks like
  a flat decal — no folds, no depth. The shading map fixes this.

  It records, for every pixel inside the saree region, how bright that point
  is relative to the average brightness of the whole saree:
    value = 1.0 → average brightness (flat area)
    value < 1.0 → shadow inside a fold (dark crease)
    value > 1.0 → highlight on a raised pleat

  At runtime: new_texture × shading_map → texture with real folds.

REQUIRES:
  poses/masks/pose_XX_mask.png  (the CLEANED mask, not _auto)

Run:
  python scripts/03_shading_extractor.py
"""
from pathlib import Path

import cv2
import numpy as np

SEL_DIR     = Path("poses/selected")
MASK_DIR    = Path("poses/masks")
SHADING_DIR = Path("poses/shading")
SHADING_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png"}

pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])

for pose_path in pose_files:
    name = pose_path.stem

    mask_path = MASK_DIR / f"{name}_mask.png"
    if not mask_path.exists():
        print(f"[SKIP] {name}: no clean mask at {mask_path}")
        print(f"       Finish manual cleanup in GIMP first, then rerun.")
        continue

    print(f"Processing {name} ...")

    bgr  = cv2.imread(str(pose_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    H, W = mask.shape
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H))

    # Convert to LAB: L channel captures brightness, independent of colour
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    inside = mask > 127
    mean_L = float(L[inside].mean())

    if mean_L < 2.0:
        print(f"  [WARN] Mean brightness is near zero — is the mask correct?")
        continue

    # Shading map: ratio of each pixel's brightness to the mean
    shading = np.zeros((H, W), dtype=np.float32)
    shading[inside] = np.clip(L[inside] / mean_L, 0.3, 2.2)

    # Optional: lightly smooth the shading map to reduce JPEG noise
    shading_smooth = cv2.GaussianBlur(shading, (5, 5), 0)
    # Restore zeros outside mask after smoothing
    shading_smooth[~inside] = 0.0

    np.save(str(SHADING_DIR / f"{name}_shading.npy"), shading_smooth)

    # Visual preview: 0-2.2 range → 0-255 greyscale
    preview = np.clip(shading_smooth / 2.2 * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(SHADING_DIR / f"{name}_shading_preview.jpg"), preview)

    vmin = float(shading[inside].min())
    vmax = float(shading[inside].max())
    print(f"  mean_L={mean_L:.1f}  shading range=[{vmin:.2f}, {vmax:.2f}]")
    print(f"  Saved: {name}_shading.npy + _shading_preview.jpg")

print("\nDone. Check poses/shading/*_shading_preview.jpg")
print("Bright areas = raised pleats (highlights), dark areas = fold shadows.")
print("That pattern should match the visible folds in her photo.\n")
print("Next: python scripts/04_region_labeler.py")
