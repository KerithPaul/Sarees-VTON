"""
Step 5: build the geometric mapping from flat saree → worn pose.

WHY THIS MATTERS:
  When you upload a flat saree photo, it is a rectangle. The model is wearing
  a draped, pleated garment. You need to geometrically WARP the flat fabric
  so it follows the curves of her body, the fall of the pleats, and the tuck
  of the pallu. This script lets you define that mapping by clicking.

BEFORE RUNNING:
  Photograph ANY saree laid completely flat on a white surface.
  Save it as:  saree_flat_reference.jpg  (in your main VTON/ folder)
  
  This reference image teaches the warper the geometry of a flat saree.
  The geometry is the same for all sarees, so you only need one photo.

HOW TO USE:
  A side-by-side window opens:
    LEFT panel  = flat saree reference
    RIGHT panel = worn pose photo

  Clicking alternates between panels:
    Click 1 → on a point on the FLAT saree  (yellow circle appears)
    Click 2 → on the EXACT SAME PHYSICAL LOCATION on the WORN pose photo
    Click 3 → next point on the FLAT saree
    ... and so on.

  Place 16 to 24 point pairs for good coverage. Good points to mark:
    - Top corners of the flat saree → shoulder/chest on pose
    - Where the border starts on each long edge → hem/pleat edge on pose
    - Bottom hem corners of flat saree → ankle/foot area on pose
    - Pallu corner → pallu end on shoulder on pose
    - A few mid-fabric points for interior warp accuracy

  Keys:
    D   undo the last point or pair
    C   clear everything and start over
    S   save the control points (need at least 8 complete pairs)
    Q   quit without saving

Output:
  poses/drape/NAME_pts_flat.npy   — (N, 2) array of flat saree points
  poses/drape/NAME_pts_pose.npy   — (N, 2) array of corresponding pose points

Run:
  python scripts/05_drape_mapper.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SEL_DIR   = Path("poses/selected")
DRAPE_DIR = Path("poses/drape")
DRAPE_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png"}

FLAT_PATH = next(
    (
        p
        for p in [
            Path("saree_flat_reference.jpg"),
            Path("saree_flat_reference.jpeg"),
            Path("saree_flat_reference.png"),
        ]
        if p.exists()
    ),
    None,
)

if FLAT_PATH is None:
    print("ERROR: saree_flat_reference.jpg not found in the VTON folder.")
    print("Photograph any saree laid flat on a white surface, save it as:")
    print("  saree_flat_reference.jpg")
    sys.exit(1)

PANEL_H = 680
DOT_COLORS = [
    (0, 220, 255), (0, 180, 100), (220, 80, 255),
    (255, 160, 0), (0, 80, 255),  (255, 60, 60),
    (60, 230, 60), (200, 200, 0), (0, 210, 210),
    (255, 120, 200), (140, 255, 0), (0, 140, 255),
]


def resize_to_h(img, target_h):
    h, w = img.shape[:2]
    s = target_h / h
    return cv2.resize(img, (int(w * s), target_h)), s


class DrapeMapper:
    def __init__(self, flat_bgr, pose_bgr, pose_name):
        self.name = pose_name
        self.flat_disp, self.fs = resize_to_h(flat_bgr, PANEL_H)
        self.pose_disp, self.ps = resize_to_h(pose_bgr, PANEL_H)
        self.wf = self.flat_disp.shape[1]

        # Points in ORIGINAL image space
        self.flat_pts: list[tuple[float, float]] = []
        self.pose_pts: list[tuple[float, float]] = []
        self.pending: tuple[float, float] | None = None  # flat pt awaiting its pair

        self.state = "flat"  # "flat" or "pose"

    def click(self, x, y):
        if self.state == "flat" and x < self.wf:
            self.pending = (x / self.fs, y / self.fs)
            self.state = "pose"
        elif self.state == "pose" and x >= self.wf:
            if self.pending is not None:
                px = (x - self.wf) / self.ps
                py = y / self.ps
                self.flat_pts.append(self.pending)
                self.pose_pts.append((px, py))
                self.pending = None
                self.state = "flat"

    def undo(self):
        if self.state == "pose" and self.pending is not None:
            self.pending = None
            self.state = "flat"
        elif self.flat_pts:
            self.flat_pts.pop()
            self.pose_pts.pop()

    def clear(self):
        self.flat_pts.clear()
        self.pose_pts.clear()
        self.pending = None
        self.state = "flat"

    def save(self):
        n = len(self.flat_pts)
        if n < 8:
            return False, f"Need at least 8 complete pairs, have {n}. Keep clicking."
        np.save(str(DRAPE_DIR / f"{self.name}_pts_flat.npy"), np.array(self.flat_pts, np.float32))
        np.save(str(DRAPE_DIR / f"{self.name}_pts_pose.npy"), np.array(self.pose_pts, np.float32))
        return True, f"Saved {n} control point pairs to poses/drape/"

    def render(self) -> np.ndarray:
        fd = self.flat_disp.copy()
        pd = self.pose_disp.copy()
        n = len(self.flat_pts)

        for i, (fp, pp) in enumerate(zip(self.flat_pts, self.pose_pts)):
            c = DOT_COLORS[i % len(DOT_COLORS)]
            fx, fy = int(fp[0] * self.fs), int(fp[1] * self.fs)
            px, py = int(pp[0] * self.ps), int(pp[1] * self.ps)
            for img, x, y in [(fd, fx, fy), (pd, px, py)]:
                cv2.circle(img, (x, y), 9, c, -1)
                cv2.circle(img, (x, y), 9, (0, 0, 0), 1)
                cv2.putText(img, str(i + 1), (x + 11, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 2)

        # Pending flat point
        if self.pending is not None:
            fx = int(self.pending[0] * self.fs)
            fy = int(self.pending[1] * self.fs)
            cv2.circle(fd, (fx, fy), 11, (0, 255, 80), 2)
            cv2.putText(fd, str(n + 1), (fx + 11, fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 80), 2)

        combined = np.hstack([fd, pd])
        cv2.line(combined, (self.wf, 0), (self.wf, PANEL_H), (70, 70, 70), 2)

        W = combined.shape[1]
        bar = np.zeros((68, W, 3), np.uint8)

        if self.state == "flat":
            msg = f"  Click on the FLAT SAREE (left panel) — point {n + 1}"
            mc = (0, 220, 255)
        else:
            msg = f"  Now click the SAME SPOT on the WORN POSE (right panel)"
            mc = (0, 255, 100)

        cv2.putText(bar, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 1)
        cv2.putText(
            bar,
            f"  Complete pairs: {n}  |  D=undo  C=clear  S=save  Q=quit  "
            f"(aim for 16-24 pairs)",
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1,
        )
        return np.vstack([combined, bar])


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        param.click(x, y)


flat_bgr = cv2.imread(str(FLAT_PATH))
if flat_bgr is None:
    print(f"Cannot read {FLAT_PATH}")
    sys.exit(1)

pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])
if not pose_files:
    print("No poses found. Run 01_pose_selector.py first.")
    sys.exit(1)

for pose_path in pose_files:
    name = pose_path.stem
    existing = DRAPE_DIR / f"{name}_pts_flat.npy"
    if existing.exists():
        resp = input(f"\n  {name} already has control points. Redo? (y/N): ").strip().lower()
        if resp != "y":
            continue

    pose_bgr = cv2.imread(str(pose_path))
    if pose_bgr is None:
        print(f"Cannot read {pose_path}")
        continue

    print(f"\nDrape mapper: {name}")
    print("Place 16-24 corresponding point pairs between flat saree and worn pose.")

    mapper = DrapeMapper(flat_bgr, pose_bgr, name)
    win = f"Drape Mapper: {name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse, mapper)

    while True:
        cv2.imshow(win, mapper.render())
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            ok, msg = mapper.save()
            print(f"  {msg}")
            if ok:
                break
        elif key == ord("d"):
            mapper.undo()
        elif key == ord("c"):
            mapper.clear()
            print("  Cleared all points.")
        elif key in (ord("q"), 27):
            print(f"  Skipped {name}")
            break

    cv2.destroyWindow(win)

print("\nDone. Next: python scripts/06_verify_drape.py")
