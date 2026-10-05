"""
Step 1: pick your 3 best poses from the 45 model photos.

BEFORE RUNNING:
  Copy all 45 model photos into:  poses/raw/

HOW TO USE:
  - A grid of all your photos opens in a window
  - Left-click any photo to select it (green border appears)
  - Click again to deselect
  - When you have 3 selected, press Q to confirm
  - Selected photos are copied to poses/selected/

Run:
  python scripts/01_pose_selector.py
"""
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path("poses/raw")
SEL_DIR = Path("poses/selected")
SEL_DIR.mkdir(parents=True, exist_ok=True)

THUMB = 180
MAX_COLS = 6
EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

image_files = sorted([f for f in RAW_DIR.iterdir() if f.suffix in EXTS])

if not image_files:
    print(f"No images found in {RAW_DIR}/")
    print("Add your 45 model photos there and rerun.")
    sys.exit(1)

n = len(image_files)
cols = min(n, MAX_COLS)
rows = math.ceil(n / cols)

print(f"Found {n} images. Loading thumbnails...")

thumbs = []
for f in image_files:
    img = cv2.imread(str(f))
    if img is None:
        thumbs.append(np.zeros((THUMB, THUMB, 3), np.uint8))
        continue
    h, w = img.shape[:2]
    s = THUMB / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    pad = np.zeros((THUMB, THUMB, 3), np.uint8)
    yo = (THUMB - img.shape[0]) // 2
    xo = (THUMB - img.shape[1]) // 2
    pad[yo : yo + img.shape[0], xo : xo + img.shape[1]] = img
    thumbs.append(pad)

selected = set()
CAP_H = 22

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        col = x // THUMB
        row = y // (THUMB + CAP_H)
        idx = row * cols + col
        if 0 <= idx < n:
            selected.discard(idx) if idx in selected else selected.add(idx)

win = "Pose selector — click to select, Q when done"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, mouse_cb)

while True:
    rows_disp = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                cell = np.zeros((THUMB + CAP_H, THUMB, 3), np.uint8)
            else:
                cell = thumbs[idx].copy()
                if idx in selected:
                    cv2.rectangle(cell, (0, 0), (THUMB - 1, THUMB - 1), (0, 220, 80), 4)
                cap = np.zeros((CAP_H, THUMB, 3), np.uint8)
                label = image_files[idx].name[:20]
                color = (0, 220, 80) if idx in selected else (160, 160, 160)
                cv2.putText(cap, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cell = np.vstack([cell, cap])
            row_cells.append(cell)
        while len(row_cells) < cols:
            row_cells.append(np.zeros((THUMB + CAP_H, THUMB, 3), np.uint8))
        rows_disp.append(np.hstack(row_cells))

    grid = np.vstack(rows_disp)
    bar = np.zeros((36, grid.shape[1], 3), np.uint8)
    cv2.putText(
        bar,
        f"Selected: {len(selected)}/3  |  Left-click to toggle  |  Q to confirm",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
    )
    cv2.imshow(win, np.vstack([grid, bar]))

    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27):
        break

cv2.destroyAllWindows()

if not selected:
    print("Nothing selected. Run again.")
    sys.exit(1)

for f in SEL_DIR.iterdir():
    if f.is_file():
        f.unlink()

for i, idx in enumerate(sorted(selected)):
    src = image_files[idx]
    dest = SEL_DIR / f"pose_{i + 1:02d}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    print(f"  Saved: {dest.name}  (from {src.name})")

print(f"\n{len(selected)} pose(s) saved to poses/selected/")
print("Next: python scripts/02_mask_generator.py")
