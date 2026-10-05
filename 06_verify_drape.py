"""
Step 6: CHECKPOINT — verify the drape mapping is correct.

This is the most important quality gate in Phase 3.

HOW IT WORKS:
  Takes the flat saree reference photo and warps it onto each pose
  using the control points from Step 5. Then blends the result with
  the original pose photo so you can compare.

WHAT TO LOOK FOR:
  GOOD — the warped fabric roughly follows her body:
    - The pleated area lines up with her pleat region
    - The pallu lands over her shoulder (may look stretched, that is okay)
    - The border runs along the right edges
    - No large black gaps inside the saree region

  BAD — rerun 05_drape_mapper.py for that pose and fix the points:
    - Large black holes inside the saree region (missing coverage)
    - The fabric is completely in the wrong area (landmark mismatch)
    - Extreme stretching or tearing (too few control points in that zone)
    - The border is placed on the body, or the pallu lands at the feet

  Perfect pixel accuracy is NOT required at this stage.
  The warp will be polished by the diffusion pass later.
  What you need is the general shape to be right.

Output:
  poses/drape/pose_XX_warp_check.jpg  — side-by-side comparison
  (Keep this for your records. It is proof Phase 3 worked.)

Run:
  python scripts/06_verify_drape.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

SEL_DIR   = Path("poses/selected")
MASK_DIR  = Path("poses/masks")
DRAPE_DIR = Path("poses/drape")

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
    print("saree_flat_reference.jpg not found.")
    sys.exit(1)

flat_bgr = cv2.imread(str(FLAT_PATH))
flat_bgr_rgb = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2RGB) / 255.0

pose_files = sorted([f for f in SEL_DIR.iterdir() if f.suffix.lower() in EXTS])
if not pose_files:
    print("No poses found.")
    sys.exit(1)

all_ok = True

for pose_path in pose_files:
    name = pose_path.stem
    pts_flat_path = DRAPE_DIR / f"{name}_pts_flat.npy"
    pts_pose_path = DRAPE_DIR / f"{name}_pts_pose.npy"

    if not pts_flat_path.exists():
        print(f"[SKIP] {name}: no control points. Run 05_drape_mapper.py first.")
        all_ok = False
        continue

    pts_flat = np.load(str(pts_flat_path))   # (N,2) source: flat saree coords
    pts_pose = np.load(str(pts_pose_path))   # (N,2) dest:   worn pose coords

    print(f"Verifying {name}  ({len(pts_flat)} pairs)...")

    pose_bgr = cv2.imread(str(pose_path))
    H, W = pose_bgr.shape[:2]

    # PiecewiseAffineTransform: maps destination → source (scikit-image convention)
    # We want: given a pixel in pose space, find where it came from in flat space.
    # So src=pts_pose, dst=pts_flat
    tform = PiecewiseAffineTransform()
    success = tform.estimate(pts_pose, pts_flat)
    if not success:
        print(f"  [WARN] Transform could not be estimated. Too few points or degenerate layout.")
        all_ok = False
        continue

    warped = warp(
        flat_bgr_rgb,
        tform,
        output_shape=(H, W),
        order=1,
        mode="constant",
        cval=0,
    )
    warped_bgr = (warped * 255).astype(np.uint8)
    warped_bgr = cv2.cvtColor(warped_bgr, cv2.COLOR_RGB2BGR)

    # Load mask if available (preferred) else use non-black warped pixels
    mask_path = MASK_DIR / f"{name}_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        inside = mask > 127
    else:
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        inside = gray > 8

    # Blend: 60% warped over original inside the saree region
    result = pose_bgr.copy()
    result[inside] = (
        warped_bgr[inside].astype(float) * 0.60
        + pose_bgr[inside].astype(float) * 0.40
    ).astype(np.uint8)

    # Build side-by-side comparison at a fixed display height
    DH = 700

    def rh(img):
        h, w = img.shape[:2]
        s = DH / h
        return cv2.resize(img, (int(w * s), DH))

    left  = rh(pose_bgr)
    right = rh(result)
    div   = np.full((DH, 3, 3), 60, np.uint8)
    panel = np.hstack([left, div, right])

    # Caption bar
    bar = np.zeros((28, panel.shape[1], 3), np.uint8)
    cv2.putText(bar, "Original", (left.shape[1] // 2 - 35, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    cv2.putText(bar, "Warp check (60% warped + 40% original)",
                (left.shape[1] + 12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
    panel = np.vstack([bar, panel])

    out_path = DRAPE_DIR / f"{name}_warp_check.jpg"
    cv2.imwrite(str(out_path), panel)
    print(f"  Saved: {out_path.name}")

    # Show in window — press any key to continue
    win = f"Warp check: {name}  (press any key to continue)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cw = min(panel.shape[1], 1400)
    ch = min(panel.shape[0], 800)
    cv2.resizeWindow(win, cw, ch)
    cv2.imshow(win, panel)
    print("  Press any key in the window to continue.")
    cv2.waitKey(0)
    cv2.destroyWindow(win)

print("\n" + "=" * 56)
if all_ok:
    print("All poses verified. Phase 3 is complete.")
    print()
    print("Files you now have per pose:")
    print("  poses/masks/NAME_mask.png              binary saree mask")
    print("  poses/masks/NAME_mask_feathered.png    soft-edge mask")
    print("  poses/shading/NAME_shading.npy         fold/lighting map")
    print("  poses/regions/NAME_regions.png         body/border/pallu labels")
    print("  poses/drape/NAME_pts_flat.npy          flat saree control points")
    print("  poses/drape/NAME_pts_pose.npy          corresponding pose points")
    print()
    print("Ready for Phase 4: runtime saree preprocessing pipeline.")
else:
    print("Some poses need fixing. Check the output above and rerun the relevant steps.")
