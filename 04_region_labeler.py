"""
Step 4: split the saree mask into three regions.

WHY THIS MATTERS:
  A saree has three visually distinct zones that come from different parts
  of the flat saree fabric:

    BODY   (blue)  — the main draped section, which uses the centre fabric tile
    BORDER (green) — the decorative edge strip that runs along the pleat edge
                     and the bottom hem (usually 8-12% of saree width)
    PALLU  (red)   — the decorative end piece draped over the left shoulder
                     (typically the last 20-25% of the saree length)

  At runtime the system maps the matching part of the uploaded flat saree photo
  onto each zone. Without these labels, you would put body fabric where the
  pallu should be, and the result looks completely wrong.

HOW TO USE:
  An OpenCV window opens showing the pose with colour-coded regions overlaid.
  Adjust the auto-split with these keys:

    +  / -   increase / decrease border thickness (in pixels)
    P        move the pallu boundary DOWN (more pallu)
    O        move the pallu boundary UP (less pallu)
    S        save and move to the next pose
    Q        skip this pose

  The coloured preview updates live as you press keys.
  Once it looks right, press S to save.

  For fine-tuned edits: open the saved _regions.png in GIMP.
  The file uses flat pixel values:  0=background  1=body  2=border  3=pallu

Run:
  python scripts/04_region_labeler.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SEL_DIR  = Path("poses/selected")
MASK_DIR = Path("poses/masks")
REG_DIR  = Path("poses/regions")
REG_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png"}

LABEL_COLORS = {
    1: (160, 70,  10),   # body   — blue-brown (BGR)
    2: ( 20, 160,  20),  # border — green
    3: ( 20,  20, 180),  # pallu  — red
}


def auto_label(mask: np.ndarray, border_px: int, pallu_frac: float) -> np.ndarray:
    """Return a label map (uint8) from a binary mask."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.uint8)
    inside = mask > 127
    labels[inside] = 1  # start: everything is body

    # Border: pixels within border_px of the mask boundary
    eroded = cv2.erode(mask, np.ones((border_px * 2 + 1, border_px * 2 + 1), np.uint8))
    labels[inside & (eroded == 0)] = 2

    # Pallu: upper portion of the mask bounding box
    ys = np.where(inside)[0]
    if ys.size > 0:
        y_min, y_max = ys.min(), ys.max()
        pallu_cut = int(y_min + (y_max - y_min) * pallu_frac)
        row_idx = np.arange(H)[:, None]
        labels[inside & (row_idx < pallu_cut)] = 3

    return labels


def draw_overlay(bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    color_img = np.zeros_like(bgr)
    for lbl, col in LABEL_COLORS.items():
        color_img[labels == lbl] = col
    mask_any = labels > 0
    out[mask_any] = (
        out[mask_any].astype(float) * 0.35
        + color_img[mask_any].astype(float) * 0.65
    ).astype(np.uint8)
    return out


pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])
if not pose_files:
    print("No poses found. Run 01_pose_selector.py first.")
    sys.exit(1)

for pose_path in pose_files:
    name = pose_path.stem
    mask_path = MASK_DIR / f"{name}_mask.png"

    if not mask_path.exists():
        print(f"[SKIP] {name}: mask not found. Run 02 and clean masks first.")
        continue

    bgr  = cv2.imread(str(pose_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr.shape[:2] != mask.shape:
        bgr = cv2.resize(bgr, (mask.shape[1], mask.shape[0]))

    # Scale to a sensible display size
    disp_h = 750
    scale = disp_h / bgr.shape[0]
    disp_bgr  = cv2.resize(bgr, (int(bgr.shape[1] * scale), disp_h))
    disp_mask = cv2.resize(mask, (disp_bgr.shape[1], disp_h), interpolation=cv2.INTER_NEAREST)

    border_px  = 30
    pallu_frac = 0.25

    win = f"Region labeler: {name}  |  +/-=border  P/O=pallu  S=save  Q=skip"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    print(f"\nLabeling {name}  (press S when satisfied)")

    while True:
        labels  = auto_label(disp_mask, border_px, pallu_frac)
        overlay = draw_overlay(disp_bgr, labels)

        H, W = overlay.shape[:2]
        bar = np.zeros((60, W, 3), np.uint8)
        cv2.putText(bar,
            f"border={border_px}px   pallu=top {pallu_frac*100:.0f}%   "
            f"[+/-] border   [P/O] pallu   S=save   Q=skip",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(bar,
            "  Blue=body   Green=border   Red=pallu",
            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.imshow(win, np.vstack([overlay, bar]))
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            # Save at original resolution, not display resolution
            labels_full = auto_label(mask, border_px, pallu_frac)
            out_path = REG_DIR / f"{name}_regions.png"
            cv2.imwrite(str(out_path), labels_full)
            print(f"  Saved: {out_path.name}")
            break
        elif key in (ord("q"), 27):
            print(f"  Skipped {name}")
            break
        elif key in (ord("+"), ord("=")):
            border_px = min(border_px + 5, 120)
        elif key == ord("-"):
            border_px = max(border_px - 5, 5)
        elif key == ord("p"):
            pallu_frac = min(pallu_frac + 0.02, 0.60)
        elif key == ord("o"):
            pallu_frac = max(pallu_frac - 0.02, 0.05)

    cv2.destroyWindow(win)

print("\nDone. Next: python scripts/05_drape_mapper.py")
