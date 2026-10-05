"""
Phase 4: Saree image preprocessing

Takes the admin's flat saree photo, removes the background, finds the
rectangular fabric, and splits it into body / border / pallu zones.

Run once per new saree before generating:
    python scripts/phase4_preprocess.py input/my_saree.jpg

Output: input/processed/
    saree_normalized.png   full normalized saree rectangle
    body_tile.png          512×512 tileable body texture
    border_h.png           representative horizontal border strip
    pallu.png              pallu section crop
    meta.json              zone measurements + mean LAB per zone

Interactive window lets you adjust zone boundaries with trackbars.
Press S to save, Q to use defaults.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

PROC_DIR = Path("input/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BORDER_PCT = 8   # percent of short side
DEFAULT_PALLU_PCT  = 28  # percent of long side from right end


# ------------------------------------------------------------------ helpers

def order_points(pts):
    """Order four corner points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def remove_bg(img_path: Path) -> np.ndarray:
    """Strip background, return BGR numpy array."""
    session = new_session("isnet-general-use")
    pil     = Image.open(img_path).convert("RGBA")
    result  = remove(pil, session=session)
    arr     = np.array(result)                           # RGBA

    # Wherever alpha < 10 → make white (background)
    alpha = arr[:, :, 3]
    rgb   = arr[:, :, :3].copy()
    rgb[alpha < 10] = [255, 255, 255]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def find_and_unwarp(bgr: np.ndarray) -> np.ndarray:
    """
    Find the saree quadrilateral and perspective-warp it to a clean rectangle.
    Falls back to bounding-box crop if a clean quad cannot be found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               np.ones((15, 15), np.uint8), iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr

    largest  = max(contours, key=cv2.contourArea)
    peri     = cv2.arcLength(largest, True)
    approx   = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        pts  = approx.reshape(4, 2).astype(np.float32)
        rect = order_points(pts)
        w    = int(max(np.linalg.norm(rect[0] - rect[1]),
                       np.linalg.norm(rect[3] - rect[2])))
        h    = int(max(np.linalg.norm(rect[0] - rect[3]),
                       np.linalg.norm(rect[1] - rect[2])))
        dst  = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]],
                         dtype=np.float32)
        M    = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(bgr, M, (w, h))

    # Fallback: tight bounding box
    x, y, w, h = cv2.boundingRect(largest)
    pad = 8
    return bgr[max(0, y-pad): y+h+pad, max(0, x-pad): x+w+pad]


def ensure_landscape(bgr: np.ndarray) -> np.ndarray:
    """Rotate to landscape if the image is taller than it is wide."""
    H, W = bgr.shape[:2]
    if H > W:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    return bgr


def split_zones(bgr: np.ndarray, border_pct: float, pallu_pct: float) -> dict:
    """Return zone crops from the normalised landscape saree."""
    H, W    = bgr.shape[:2]
    border  = int(min(H, W) * border_pct)
    pallu_x = int(W * (1 - pallu_pct))

    body   = bgr[border:H-border, border:pallu_x-border]
    border_top = bgr[:border, :]
    pallu  = bgr[border:H-border, pallu_x:]

    return {
        "body":     body,
        "border_h": border_top,
        "pallu":    pallu,
        "border_px": border,
        "pallu_x":   pallu_x,
        "H": H, "W": W,
    }


def make_tile(body: np.ndarray, size: int = 512) -> np.ndarray:
    """Central square crop of the body, resized to size×size."""
    if body.size == 0 or min(body.shape[:2]) < 4:
        return np.full((size, size, 3), 128, dtype=np.uint8)
    H, W   = body.shape[:2]
    side   = min(H, W)
    cy, cx = H // 2, W // 2
    half   = side // 2
    crop   = body[cy-half:cy+half, cx-half:cx+half]
    if crop.shape[0] < size or crop.shape[1] < size:
        crop = cv2.resize(crop, (size, size))
    else:
        crop = cv2.resize(crop, (size, size))
    return crop


def lab_mean(bgr_crop: np.ndarray) -> list:
    if bgr_crop.size == 0:
        return [128.0, 128.0, 128.0]
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB).astype(float)
    return [float(lab[:, :, c].mean()) for c in range(3)]


# ------------------------------------------------------- interactive adjuster

class ZoneAdjuster:
    def __init__(self, saree: np.ndarray):
        self.saree      = saree.copy()
        self.border_pct = DEFAULT_BORDER_PCT
        self.pallu_pct  = DEFAULT_PALLU_PCT
        self._flipped   = False  # track flip state

    def _draw(self) -> np.ndarray:
        H, W   = self.saree.shape[:2]
        b      = int(min(H, W) * self.border_pct / 100)
        px     = int(W * (1 - self.pallu_pct / 100))
        img    = self.saree.copy()
        green  = (0, 200, 50)
        red    = (0, 50, 220)
        lw     = 2
        # border box
        cv2.rectangle(img, (0, 0), (W-1, b),    green, lw)
        cv2.rectangle(img, (0, H-b), (W-1, H-1), green, lw)
        cv2.rectangle(img, (0, 0), (b, H-1),    green, lw)
        # pallu divider
        cv2.line(img, (px, 0), (px, H-1), red, lw)
        fnt = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "BODY",   (b+10, H//2),   fnt, 0.9, (255,255,255), 2)
        cv2.putText(img, "PALLU",  (px+8, H//2),   fnt, 0.8, (100,200,255), 2)
        cv2.putText(img, "BORDER", (5, b//2+8),    fnt, 0.5, (100,255,100), 1)
        flip_hint = "F=flipped (pallu on right)" if self._flipped else "F=flip horizontal (if pallu on wrong side)"
        cv2.putText(img, flip_hint, (b+10, H-20),  fnt, 0.45, (200,200,100), 1)
        return img

    def run(self):
        win = "Adjust zones  |  S = save  Q = use defaults"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        disp_h = min(500, self.saree.shape[0])
        disp_w = int(self.saree.shape[1] * disp_h / self.saree.shape[0])
        cv2.resizeWindow(win, disp_w, disp_h + 80)

        def _set_b(v): self.border_pct = max(1, v)
        def _set_p(v): self.pallu_pct  = max(5, v)

        cv2.createTrackbar("Border %", win, self.border_pct, 20, _set_b)
        cv2.createTrackbar("Pallu %",  win, self.pallu_pct,  50, _set_p)

        while True:
            frame = cv2.resize(self._draw(), (disp_w, disp_h))
            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                cv2.destroyWindow(win)
                return self.border_pct / 100, self.pallu_pct / 100
            if key == ord("f"):
                # Flip horizontally: moves pallu from left→right or right→left
                self.saree = cv2.flip(self.saree, 1)
                self._flipped = not self._flipped
                print("  Flipped horizontally — pallu should now be on the RIGHT side")
            if key in (ord("q"), 27):
                cv2.destroyWindow(win)
                return DEFAULT_BORDER_PCT / 100, DEFAULT_PALLU_PCT / 100


# ----------------------------------------------------------------------- main

def main():
    if len(sys.argv) < 2:
        candidates = (list(Path("input").glob("*.jpg")) +
                      list(Path("input").glob("*.jpeg")) +
                      list(Path("input").glob("*.png")))
        candidates = [p for p in candidates if "processed" not in str(p)]
        if not candidates:
            print("Usage: python scripts/phase4_preprocess.py input/your_saree.jpg")
            sys.exit(1)
        img_path = candidates[0]
        print(f"Auto-detected: {img_path}")
    else:
        img_path = Path(sys.argv[1])

    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    print(f"Processing: {img_path.name}")

    # ---- Step 1: background removal
    print("  Removing background (rembg isnet)...")
    bgr_raw = remove_bg(img_path)
    print(f"  Raw size: {bgr_raw.shape[1]}×{bgr_raw.shape[0]}")

    # ---- Step 2: find rectangle + perspective unwarp
    print("  Finding saree rectangle...")
    bgr_rect = find_and_unwarp(bgr_raw)

    # ---- Step 3: ensure landscape
    bgr_rect = ensure_landscape(bgr_rect)
    print(f"  Normalised: {bgr_rect.shape[1]}×{bgr_rect.shape[0]}")

    # ---- Step 4: interactive zone split
    print("  Opening zone adjuster (press S when ready)...")
    print("  TIP: If the dark patterned section (pallu) appears on the LEFT,")
    print("       press F to flip the image so pallu moves to the RIGHT.")
    adjuster = ZoneAdjuster(bgr_rect)
    border_pct, pallu_pct = adjuster.run()
    # Use adjuster.saree — may have been flipped by the user
    bgr_rect = adjuster.saree
    print(f"  Border: {border_pct*100:.0f}%   Pallu: {pallu_pct*100:.0f}%"
          + ("  [FLIPPED]" if adjuster._flipped else ""))

    # ---- Step 5: compute zones
    zones = split_zones(bgr_rect, border_pct, pallu_pct)

    # ---- Step 6: save all outputs
    cv2.imwrite(str(PROC_DIR / "saree_normalized.png"), bgr_rect)
    cv2.imwrite(str(PROC_DIR / "body_tile.png"),  make_tile(zones["body"]))
    cv2.imwrite(str(PROC_DIR / "border_h.png"),   zones["border_h"])
    cv2.imwrite(str(PROC_DIR / "pallu.png"),       zones["pallu"])
    cv2.imwrite(str(PROC_DIR / "body.png"),        zones["body"])

    meta = {
        "source":      str(img_path),
        "H":           int(zones["H"]),
        "W":           int(zones["W"]),
        "border_px":   int(zones["border_px"]),
        "pallu_x":     int(zones["pallu_x"]),
        "border_pct":  border_pct,
        "pallu_pct":   pallu_pct,
        "lab_means": {
            "body":     lab_mean(zones["body"]),
            "border":   lab_mean(zones["border_h"]),
            "pallu":    lab_mean(zones["pallu"]),
        },
    }
    with open(PROC_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 4 done. Files in {PROC_DIR}/")
    print("  saree_normalized.png  body_tile.png  border_h.png  pallu.png  meta.json")
    print("\nNext: python scripts/generate.py")


if __name__ == "__main__":
    main()
