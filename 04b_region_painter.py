"""
Step 4b: interactive region painter — accurate pallu / body / border labeling.

Replaces the auto-labeler (04_region_labeler.py) for accurate results.
You paint directly on the photo with a brush. The brush only marks pixels
inside the saree mask, so you cannot accidentally paint outside it.

WHERE TO PAINT (Nivi drape):

  FRONT VIEW (pose_02):
    PALLU (3) — Paint diagonally from upper-right (shoulder area) across
                the chest. The boundary runs diagonally, NOT horizontally.
                Stop where the draping fabric ends and the plain wrap begins.
    BORDER (2) — Paint the narrow right vertical edge (pleat edge visible
                from front) and the bottom hem strip only.
    BODY (1)  — Everything else: the main skirt and wrap section.

  SIDE VIEW (pose_01):
    PALLU (3) — Paint the shoulder area and the fabric hanging from it.
                Roughly the upper 30% of the saree region.
    BORDER (2) — Paint the right edge (which is the front-facing edge of
                the saree in this side view) and the bottom hem.
    BODY (1)  — Everything else.

  BACK VIEW (pose_03):
    PALLU (3) — Paint ONLY the narrow vertical strip on the LEFT side of the
                image where the pallu fabric hangs down the back.
                It is a thin strip, not a wide zone.
    BORDER (2) — Paint both left and right edge strips and the bottom hem.
    BODY (1)  — The large central back section.

WORKFLOW (do in this order for each pose):
  1. Press 1 — paint BODY over the entire mask first (fills everything blue)
               This gives you a clean slate. Use a large brush (+/=).
  2. Press 3 — paint PALLU over the pallu region accurately.
  3. Press 2 — paint BORDER along the edge strips and hem.
  Right-click anywhere = instantly paint BODY (quick correction).

CONTROLS:
  1     paint BODY (blue)
  2     paint BORDER (green)
  3     paint PALLU (red)
  +/=   increase brush size
  -     decrease brush size
  Z     undo last stroke
  R     reset: fill entire mask as BODY and start over
  S     save and move to next pose
  Q     skip this pose without saving

Run:
  python scripts/04b_region_painter.py
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

LABEL_COLORS_BGR = {
    1: (160, 80,  40),   # body   — displayed as blue-ish overlay
    2: ( 30, 160,  30),  # border — green
    3: ( 40,  40, 200),  # pallu  — red
}
LABEL_NAMES = {1: "BODY", 2: "BORDER", 3: "PALLU"}

INSTRUCTIONS = {
    "pose_01": [
        "SIDE VIEW: paint PALLU (3) = upper shoulder drape (~top 30%)",
        "          paint BORDER (2) = right edge strip + bottom hem",
        "          paint BODY (1)   = everything else",
    ],
    "pose_02": [
        "FRONT VIEW: paint PALLU (3) = diagonal from upper-right across chest",
        "            paint BORDER (2) = right pleat edge strip + bottom hem",
        "            paint BODY (1)   = main skirt and wrap below pallu",
    ],
    "pose_03": [
        "BACK VIEW: paint PALLU (3) = NARROW LEFT STRIP only (pallu hanging)",
        "           paint BORDER (2) = both side strips + bottom hem",
        "           paint BODY (1)   = large central back section",
    ],
}

DISP_H = 750


class Painter:
    def __init__(self, pose_path, mask_path, out_path, prev_regions=None):
        self.out_path = str(out_path)

        bgr  = cv2.imread(str(pose_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = bgr.shape[:2]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        self.orig = bgr
        self.mask = mask
        self.H, self.W = H, W

        # Display scale
        self.scale  = DISP_H / H
        self.dW     = int(W * self.scale)
        self.d_orig = cv2.resize(bgr, (self.dW, DISP_H))
        self.d_mask = cv2.resize(mask, (self.dW, DISP_H), interpolation=cv2.INTER_NEAREST)

        # Label map at original resolution
        self.labels = np.zeros((H, W), dtype=np.uint8)
        if prev_regions and Path(prev_regions).exists():
            loaded = cv2.imread(str(prev_regions), cv2.IMREAD_GRAYSCALE)
            if loaded is not None and loaded.shape == (H, W):
                self.labels = loaded
        else:
            self.labels[mask > 127] = 1  # default: all body

        self.label     = 3   # start with PALLU — the most important to get right
        self.brush     = 28  # display pixels
        self.drawing   = False
        self.last_pos  = None
        self.undo_stack: list[np.ndarray] = []

    def _orig_brush(self):
        return max(2, int(self.brush / self.scale))

    def _paint(self, dx, dy):
        ox = int(dx / self.scale)
        oy = int(dy / self.scale)
        r  = self._orig_brush()
        tmp = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.circle(tmp, (ox, oy), r, 255, -1)
        region = (tmp > 0) & (self.mask > 127)
        self.labels[region] = self.label

    def stroke(self, x0, y0, x1, y1):
        dist  = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, dist // max(1, self.brush // 4))
        for i in range(steps + 1):
            t = i / steps
            self._paint(int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0)))

    def push_undo(self):
        if len(self.undo_stack) >= 30:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.labels.copy())

    def undo(self):
        if self.undo_stack:
            self.labels = self.undo_stack.pop()

    def reset(self):
        self.push_undo()
        self.labels = np.zeros((self.H, self.W), dtype=np.uint8)
        self.labels[self.mask > 127] = 1

    def render(self, cursor_xy=None):
        dl = cv2.resize(self.labels, (self.dW, DISP_H), interpolation=cv2.INTER_NEAREST)
        col_img = np.zeros((DISP_H, self.dW, 3), np.uint8)
        for lbl, col in LABEL_COLORS_BGR.items():
            col_img[dl == lbl] = col

        inside  = dl > 0
        display = self.d_orig.copy()
        display[inside] = (
            display[inside].astype(float) * 0.35
            + col_img[inside].astype(float) * 0.65
        ).astype(np.uint8)

        # Cursor circle
        if cursor_xy:
            cx, cy = cursor_xy
            cv2.circle(display, (cx, cy), self.brush, LABEL_COLORS_BGR[self.label], 2)

        # Status bar
        bar = np.zeros((72, self.dW, 3), np.uint8)
        c   = LABEL_COLORS_BGR[self.label]
        cv2.putText(bar,
            f"  Brush: {LABEL_NAMES[self.label]}  size={self.brush}px  "
            f"undo_stack={len(self.undo_stack)}",
            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, c, 1)
        cv2.putText(bar,
            "  1=body  2=border  3=pallu  +/-=size  Z=undo  R=reset  S=save  Q=skip"
            "   right-click=body",
            (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 148, 148), 1)
        return np.vstack([display, bar])

    def save(self):
        cv2.imwrite(self.out_path, self.labels)


cursor = [0, 0]


def mouse_cb(event, x, y, flags, param):
    p = param
    cursor[0], cursor[1] = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        p.push_undo()
        p.drawing = True
        p.last_pos = (x, y)
        p._paint(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and p.drawing:
        if p.last_pos:
            p.stroke(p.last_pos[0], p.last_pos[1], x, y)
        p.last_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        p.drawing = False
        p.last_pos = None

    # Right-click: instantly paint body (quick correction)
    elif event == cv2.EVENT_RBUTTONDOWN:
        p.push_undo()
        prev, p.label = p.label, 1
        p._paint(x, y)
        p.label = prev
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
        prev, p.label = p.label, 1
        if p.last_pos:
            p.stroke(p.last_pos[0], p.last_pos[1], x, y)
        p.last_pos = (x, y)
        p.label = prev


pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])
if not pose_files:
    print("No poses found. Run 01_pose_selector.py first.")
    sys.exit(1)

for pose_path in pose_files:
    name      = pose_path.stem
    mask_path = MASK_DIR / f"{name}_mask.png"
    out_path  = REG_DIR  / f"{name}_regions.png"

    if not mask_path.exists():
        print(f"[SKIP] {name}: mask not found. Run 02_mask_generator.py first.")
        continue

    print(f"\n{'='*60}")
    print(f"Region painter: {name}")
    for line in INSTRUCTIONS.get(name, ["Paint the three regions."]):
        print(line)
    print("  Workflow: press 1 → flood body → press 3 → paint pallu → press 2 → paint border")
    print("  Right-click anywhere = instant body (correction)")
    print(f"{'='*60}")

    prev = str(out_path) if out_path.exists() else None
    painter = Painter(str(pose_path), str(mask_path), out_path, prev)

    win = f"Region painter: {name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_cb, painter)

    while True:
        cv2.imshow(win, painter.render(cursor_xy=tuple(cursor)))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("1"):
            painter.label = 1
        elif key == ord("2"):
            painter.label = 2
        elif key == ord("3"):
            painter.label = 3
        elif key in (ord("+"), ord("=")):
            painter.brush = min(painter.brush + 5, 160)
        elif key == ord("-"):
            painter.brush = max(painter.brush - 5, 5)
        elif key == ord("z"):
            painter.undo()
            print("  Undone")
        elif key == ord("r"):
            painter.reset()
            print("  Reset to all-body")
        elif key == ord("s"):
            painter.save()
            print(f"  Saved: {out_path.name}")
            break
        elif key in (ord("q"), 27):
            print(f"  Skipped {name}")
            break

    cv2.destroyWindow(win)

print("\nDone. Next: python scripts/05_drape_mapper.py")
