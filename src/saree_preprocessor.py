"""
Phase 4: read and normalise one saree upload.

Takes a raw photo of a flat saree and returns a SareeIntermediate.
Runs entirely on CPU.  Expect ~8 seconds on first call; subsequent calls
with the same image return instantly from a disk cache.

Steps
-----
1. rembg strips the background.  You get the cloth on transparency.
2. OpenCV finds the largest quadrilateral and perspective-warps it to a
   clean 2400 × 900 rectangle.
3. The rectangle is split into border (top strip, bottom strip), pallu
   (right end), and body (everything else).
4. A 512 × 512 seamlessly-tiling tile is cut from the center of the body.
5. Average LAB colour per region is recorded for Quality Gate drift checks.
6. The whole result is pickled to disk so repeated uploads of the same
   image skip all work.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
from skimage.color import rgb2lab

from .types import PipelineConfig, SareeIntermediate


class SareePreprocessor:
    """
    Holds the rembg session in memory.
    Construct once per service; call process() for each upload.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        print("Loading rembg (isnet-general-use)…")
        self.rembg_session = new_session("isnet-general-use")
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public

    def process(
        self,
        image_path: str,
        border_fraction: Optional[float] = None,
        pallu_fraction: Optional[float] = None,
    ) -> SareeIntermediate:
        """
        Main entry point.

        border_fraction: fraction of canvas height that is border strip
                         (0.08–0.12 typical).  None → use config default.
        pallu_fraction:  fraction of canvas width that is pallu end
                         (0.20 typical).  None → use config default.

        Returns a SareeIntermediate, either freshly computed or from cache.
        """
        bf = border_fraction if border_fraction is not None else self.config.border_fraction
        pf = pallu_fraction  if pallu_fraction  is not None else self.config.pallu_fraction

        cache_path = self._cache_path(image_path, bf, pf)
        if cache_path.exists():
            print(f"  [Phase 4] Cache hit: {cache_path.name}")
            with open(cache_path, "rb") as fh:
                return pickle.load(fh)

        result = self._run(image_path, bf, pf)
        with open(cache_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    # ------------------------------------------------------------------ pipeline

    def _run(
        self,
        image_path: str,
        border_fraction: float,
        pallu_fraction: float,
    ) -> SareeIntermediate:
        t_all = time.perf_counter()

        # Step 1: strip background
        t = time.perf_counter()
        rgba = self._strip_background(image_path)
        print(f"  background removal:  {time.perf_counter()-t:.1f}s")

        # Step 2: perspective warp to fixed canvas
        t = time.perf_counter()
        canvas = self._perspective_warp(rgba)
        print(f"  perspective warp:    {time.perf_counter()-t:.1f}s")

        # Step 3: split into regions
        t = time.perf_counter()
        regions = self._split_regions(canvas, border_fraction, pallu_fraction)
        print(f"  region split:        {time.perf_counter()-t:.1f}s")

        # Step 4: seamless tile
        tile = _make_seamless_tile(regions["body"])

        # Step 5: LAB means
        body_lab   = _mean_lab(regions["body"])
        border_lab = _mean_lab(
            np.concatenate([regions["border_top"], regions["border_bottom"]], axis=0)
        )
        pallu_lab  = _mean_lab(regions["pallu"])

        print(f"  [Phase 4] total:     {time.perf_counter()-t_all:.1f}s")

        return SareeIntermediate(
            canvas_rgb        = canvas,
            border_top_rgb    = regions["border_top"],
            border_bottom_rgb = regions["border_bottom"],
            pallu_rgb         = regions["pallu"],
            body_rgb          = regions["body"],
            tile_rgb          = tile,
            body_lab_mean     = body_lab,
            border_lab_mean   = border_lab,
            pallu_lab_mean    = pallu_lab,
            border_fraction   = border_fraction,
            pallu_fraction    = pallu_fraction,
        )

    # ------------------------------------------------------------------ step 1

    def _strip_background(self, image_path: str) -> np.ndarray:
        """Return RGBA numpy array with the saree isolated on transparency."""
        raw = Image.open(image_path).convert("RGBA")
        result = remove(raw, session=self.rembg_session)
        return np.array(result.convert("RGBA"))

    # ------------------------------------------------------------------ step 2

    def _perspective_warp(self, rgba: np.ndarray) -> np.ndarray:
        """
        Find the saree quadrilateral, run perspective transform, return a
        clean (canvas_height, canvas_width, 3) RGB array.
        """
        alpha = rgba[:, :, 3]
        rgb   = rgba[:, :, :3]

        quad    = _find_quad(alpha)
        ordered = _order_quad(quad)

        W = self.config.canvas_width
        H = self.config.canvas_height

        dst = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(rgb, M, (W, H), flags=cv2.INTER_LANCZOS4)
        return warped  # (H, W, 3) uint8

    # ------------------------------------------------------------------ step 3

    def _split_regions(
        self,
        canvas: np.ndarray,
        border_fraction: float,
        pallu_fraction: float,
    ) -> dict[str, np.ndarray]:
        """
        canvas: (H, W, 3) uint8

        Regions:
          border_top    — top slice, height = border_h
          border_bottom — bottom slice, height = border_h
          body          — center rectangle (no pallu)
          pallu         — rightmost slice, width = pallu_w

        All four cover non-overlapping areas.
        """
        H, W = canvas.shape[:2]

        border_h = max(1, int(H * border_fraction))
        pallu_w  = max(1, int(W * pallu_fraction))
        body_w   = W - pallu_w
        body_top    = border_h
        body_bottom = H - border_h

        return {
            "border_top":    canvas[0:border_h,         0:body_w].copy(),
            "border_bottom": canvas[body_bottom:H,       0:body_w].copy(),
            "body":          canvas[body_top:body_bottom, 0:body_w].copy(),
            "pallu":         canvas[0:H,                 body_w:W].copy(),
        }

    # ------------------------------------------------------------------ cache

    def _cache_path(self, image_path: str, bf: float, pf: float) -> Path:
        with open(image_path, "rb") as fh:
            file_hash = hashlib.sha1(fh.read()).hexdigest()[:12]
        key = f"{file_hash}_{bf:.3f}_{pf:.3f}"
        return Path(self.config.cache_dir) / f"{key}.pkl"


# ================================================================== helpers


def _find_quad(alpha: np.ndarray) -> np.ndarray:
    """
    Find the largest 4-corner polygon in the alpha mask.
    Returns (4, 2) float32 in OpenCV (x, y) order.
    """
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # Close small holes and remove stray pixels
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "rembg produced an empty alpha channel.  "
            "Check that the saree image has a visible background to remove."
        )

    largest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)

    # Try progressively looser epsilon until we land on exactly 4 corners
    for eps in [0.02, 0.03, 0.05, 0.08, 0.12, 0.18]:
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Fall back to minimum-area rectangle
    rect = cv2.minAreaRect(largest)
    return cv2.boxPoints(rect).astype(np.float32)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    Return 4 points in TL, TR, BR, BL order.
    Works by summing and differencing coordinates.
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _make_seamless_tile(body: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Cut a seamlessly-tiling square from the center of the body region.

    Technique: hanning-window blend between a center crop and its
    (size/2, size/2)-shifted copy.  After blending, each edge of the tile
    matches the opposite edge so tiling produces no visible seam line.
    """
    h, w = body.shape[:2]
    half = size // 2

    # Try to take a center crop at native resolution
    cy, cx = h // 2, w // 2
    if h >= size and w >= size:
        patch = body[cy - half : cy + half, cx - half : cx + half]
    else:
        patch = body  # will be resized below

    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LANCZOS4)
    pf    = patch.astype(np.float32)

    # Roll so that the seam line (current edges) moves to the center
    rolled = np.roll(np.roll(pf, half, axis=0), half, axis=1)

    # 2-D Hanning window: 1 at center, 0 at all four edges
    h1d   = np.hanning(size).astype(np.float32)
    win2d = np.outer(h1d, h1d)[:, :, np.newaxis]  # (size, size, 1)

    # Blend: center region comes from the original patch (no seam there);
    # edges come from the rolled version (its seam is now in the center
    # and is covered by the patch).
    blended = pf * win2d + rolled * (1.0 - win2d)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _mean_lab(region_rgb: np.ndarray) -> np.ndarray:
    """Mean LAB colour of a region.  Returns (3,) float32."""
    lab = rgb2lab(region_rgb.astype(np.float32) / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
