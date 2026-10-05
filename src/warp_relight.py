"""
Phase 5: warp the saree onto the model's body and transfer the fold lighting.

For each pose this does four things in sequence:

1. Build a flat composite of the saree (tiled body + border strips + pallu)
   on a single 2400 × 900 canvas, ready to be bent.

2. Bend it onto the model's body using the piecewise-affine drape map
   from Phase 3.  Each pixel in the model's garment area is mapped back
   to a pixel in the flat saree.

3. Transfer shading.  Replace the L channel of the warped result with
   (L_texture_mean × shading_map).  The a and b channels from the new
   saree are kept unchanged.  This step moves the original fold shadows
   and pleat highlights onto the new cloth while leaving its colour exact.

4. Alpha-composite the result onto the original photo using the feathered
   mask.  Everything outside the mask is untouched — face, hands, background
   stay pixel-for-pixel identical to the source photograph.

No GPU is used here.  Expect ~2 seconds per pose on CPU.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp as sk_warp

from .types import PipelineConfig, PoseData, SareeIntermediate


class WarpAndRelight:
    """No persistent state; construct once, call run() per pose."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------ public

    def run(self, intermediate: SareeIntermediate, pose: PoseData) -> np.ndarray:
        """
        Returns a full-size RGB image: the original photo with the new saree
        warped and relit onto the garment region.
        """
        t0 = time.perf_counter()

        # 1. Flat composite saree image
        flat = self._build_flat_composite(intermediate)

        # 2. Piecewise-affine warp into body space
        H, W = pose.photo_rgb.shape[:2]
        warped = _warp_to_body(flat, pose.src_points, pose.dst_points, (H, W))
        print(f"    warp:    {time.perf_counter()-t0:.2f}s")

        # 3. Transfer fold shading from original photo
        t1 = time.perf_counter()
        relighted = _apply_shading(warped, pose.shading_map, pose.mask)
        print(f"    relight: {time.perf_counter()-t1:.2f}s")

        # 4. Composite onto original photo
        result = _composite(pose.photo_rgb, relighted, pose.feather_mask)
        print(f"    [{pose.pose_id}] phase 5 total: {time.perf_counter()-t0:.2f}s")
        return result

    # ------------------------------------------------------------------ step 1

    def _build_flat_composite(self, s: SareeIntermediate) -> np.ndarray:
        """
        Lay out the saree as a single (canvas_height, canvas_width, 3) array:

          ┌─────────────────────────────────┬────────┐
          │ border_top (thin strip)         │        │
          ├─────────────────────────────────┤ pallu  │
          │ body (tiled texture)            │ (tall  │
          ├─────────────────────────────────┤  end)  │
          │ border_bottom (thin strip)      │        │
          └─────────────────────────────────┴────────┘

        The drape map will then bend this canvas onto the model's body.
        """
        H = self.config.canvas_height
        W = self.config.canvas_width

        border_h = max(1, int(H * s.border_fraction))
        pallu_w  = max(1, int(W * s.pallu_fraction))
        body_w   = W - pallu_w
        body_h   = H - 2 * border_h

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Body: tile the 512 × 512 seamless tile to fill the body area
        tile   = s.tile_rgb
        rh     = int(np.ceil(body_h / tile.shape[0])) + 1
        rw     = int(np.ceil(body_w / tile.shape[1])) + 1
        tiled  = np.tile(tile, (rh, rw, 1))[:body_h, :body_w]
        canvas[border_h : H - border_h, 0:body_w] = tiled

        # Top border strip
        top = cv2.resize(s.border_top_rgb, (body_w, border_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[0:border_h, 0:body_w] = top

        # Bottom border strip
        bot = cv2.resize(s.border_bottom_rgb, (body_w, border_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[H - border_h : H, 0:body_w] = bot

        # Pallu — the decorative end piece
        pallu = cv2.resize(s.pallu_rgb, (pallu_w, H), interpolation=cv2.INTER_LANCZOS4)
        canvas[0:H, body_w:W] = pallu

        return canvas


# ================================================================== helpers


def _warp_to_body(
    flat: np.ndarray,
    src_points: np.ndarray,   # (N, 2) float32 — (x, y) on the flat canvas
    dst_points: np.ndarray,   # (N, 2) float32 — (x, y) on the model photo
    output_shape: tuple,      # (H, W)
) -> np.ndarray:
    """
    Bend the flat saree into the model's garment space using a
    piecewise-affine transform built from the Phase 3 drape control points.

    How the coordinate mapping works
    ---------------------------------
    PiecewiseAffineTransform.estimate(A, B) defines a map A → B.
    skimage.warp() applies the transform as an *inverse* map
    (output coords → input coords to sample from).

    We want: for each pixel in the output (model-photo space) find which
    pixel of the flat saree to sample.  So we pass:
        A = dst_points  (model-photo coords)
        B = src_points  (flat-saree coords)

    skimage uses (row, col) = (y, x) order.  Our points are saved as (x, y),
    so we flip them with [:, ::-1] before passing to skimage.
    """
    # (x, y) → (row, col) = (y, x)
    dst_rc = dst_points[:, ::-1].astype(np.float64)
    src_rc = src_points[:, ::-1].astype(np.float64)

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_rc, src_rc)

    flat_f  = flat.astype(np.float64) / 255.0
    warped_f = sk_warp(
        flat_f,
        tform,
        output_shape=output_shape,
        order=1,            # bilinear — fast; good enough for fabric texture
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )
    return np.clip(warped_f * 255, 0, 255).astype(np.uint8)


def _apply_shading(
    warped: np.ndarray,
    shading_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Modulate the L channel of the warped cloth per pixel with
    (L_pixel × shading_map) inside the garment mask.  The a and b channels
    are kept from the warped saree, which preserves the new colour exactly.

    warped:      (H, W, 3) uint8 RGB — the new saree warped into body space
    shading_map: (H, W) float32 — relative brightness from the original photo
                 (~1.0 lit, ~0.6 shadow, ~1.4 highlight), zero outside mask
    mask:        (H, W) uint8 — 0 outside garment, 255 inside

    Returns (H, W, 3) uint8 RGB with the folds of the original drape
    transferred onto the new cloth.

    Multiplies each pixel's OWN L value, not a single mean across the whole
    garment — the body, border and pallu have different intrinsic
    brightness (e.g. cream body vs deep maroon border), and collapsing them
    to one mean before scaling washes out the border/pallu colour and
    erases their weave texture.
    """
    in_mask = mask > 127
    if not in_mask.any():
        return warped.copy()

    # OpenCV LAB: L is 0–255 (maps to 0–100); a and b are 0–255 (maps to -128–127)
    lab = cv2.cvtColor(warped, cv2.COLOR_RGB2LAB).astype(np.float32)
    L   = lab[:, :, 0]

    # New L: each pixel's own brightness scaled by the shading map
    # shading_map is ~1.0 at average; 0 outside mask (safe to multiply)
    L_new = np.clip(L * shading_map, 0, 255).astype(np.float32)

    out = lab.copy()
    out[:, :, 0][in_mask] = L_new[in_mask]

    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _composite(
    original: np.ndarray,
    warped: np.ndarray,
    feather_mask: np.ndarray,
) -> np.ndarray:
    """
    Alpha-blend the warped saree onto the original photograph.

    original:     (H, W, 3) uint8 — the studio photo, untouched
    warped:       (H, W, 3) uint8 — new saree, after shading
    feather_mask: (H, W) float32 0.0–1.0 — soft alpha

    Pixels outside the mask come entirely from original.
    Pixels inside the mask come entirely from warped.
    Pixels at the mask edge are blended to hide the boundary.
    """
    alpha = feather_mask[:, :, np.newaxis]  # (H, W, 1)
    blended = (
        original.astype(np.float32) * (1.0 - alpha)
        + warped.astype(np.float32)  * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
