"""
Phase 8: automated quality gates.

Run after every generation.  These checks catch problems before the client
does.  All checks log failures; none raise exceptions, so one bad pose
does not block the others from saving.

Checks
------
1. Colour check
   CIEDE2000 distance between the garment region of each output and the
   source saree's LAB mean.  If the delta is over 6, the colour drifted.
   correct_colour_drift() applies a global a/b shift to fix it.

2. Cross-view check
   Bhattacharyya distance between garment colour histograms of each pair
   of output views.  A large difference means one view went wrong.

3. Face check
   Pixel-level diff between the face region of each output and the original
   photo.  Any non-zero diff means the garment mask leaked into the face.
   This should never fire.

4. Latency check
   Not a pass/fail gate — just logs the timing dict so you notice when
   someone accidentally adds a model load into the request path.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from .types import PipelineConfig, SareeIntermediate


def run_all(
    outputs: dict[str, np.ndarray],              # pose_id → full-size RGB
    originals: dict[str, np.ndarray],            # pose_id → original photo
    masks: dict[str, np.ndarray],                # pose_id → binary mask
    face_masks: dict[str, Optional[np.ndarray]], # pose_id → face mask or None
    intermediate: SareeIntermediate,
    config: PipelineConfig,
    timing: Optional[dict] = None,
) -> dict:
    """
    Run all quality gates.  Returns a flat dict suitable for logging
    or returning to the caller.
    """
    results: dict = {}

    # ----- Check 1: colour drift -----
    colour_deltas = {}
    for pose_id, output in outputs.items():
        delta = _colour_drift(output, masks[pose_id], intermediate)
        colour_deltas[pose_id] = round(float(delta), 2)

    results["colour_delta"]  = colour_deltas
    results["colour_ok"]     = all(
        d <= config.colour_delta_threshold for d in colour_deltas.values()
    )

    if not results["colour_ok"]:
        bad = {k: v for k, v in colour_deltas.items() if v > config.colour_delta_threshold}
        print(f"  [QC] Colour drift too high: {bad}  (threshold: {config.colour_delta_threshold})")
        print("       Auto-correction will be applied after this check.")

    # ----- Check 2: cross-view consistency -----
    cross_delta = _cross_view_delta(outputs, masks)
    results["cross_view_delta"] = round(float(cross_delta), 4)
    results["cross_view_ok"]    = cross_delta <= config.cross_view_delta_threshold

    if not results["cross_view_ok"]:
        print(
            f"  [QC] Cross-view colour mismatch: {cross_delta:.4f}  "
            f"(threshold: {config.cross_view_delta_threshold})"
        )

    # ----- Check 3: face unchanged -----
    face_matches = {}
    for pose_id, output in outputs.items():
        fm  = face_masks.get(pose_id)
        ok  = _face_unchanged(output, originals[pose_id], fm)
        face_matches[pose_id] = ok

    results["face_match"]          = all(face_matches.values())
    results["face_match_per_pose"] = face_matches

    if not results["face_match"]:
        bad = [k for k, v in face_matches.items() if not v]
        print(
            f"  [QC] CRITICAL: Face pixels changed in poses: {bad}\n"
            "       The garment mask is leaking into the face region.\n"
            "       Open the mask in GIMP and close the gap at the blouse edge."
        )

    # ----- Check 4: latency -----
    if timing:
        results["timing_seconds"] = timing
        total = timing.get("total", 0)
        if total > 150:
            print(f"  [QC] Latency warning: {total:.0f}s total (target: under 150s)")

    return results


# ================================================================== individual checks


def _colour_drift(
    output: np.ndarray,
    mask: np.ndarray,
    intermediate: SareeIntermediate,
) -> float:
    """
    CIEDE2000 distance between the mean LAB colour of the garment region
    in the output and the body_lab_mean recorded from the source saree.

    A value under 6 is imperceptible.  Over 10 is obviously wrong.
    """
    in_mask = mask > 127
    if not in_mask.any():
        return 0.0

    # Compute mean LAB of the garment region in the output
    pixels_rgb = output[in_mask].astype(np.float32) / 255.0  # (N, 3)
    lab_pixels = rgb2lab(pixels_rgb.reshape(1, -1, 3)).reshape(-1, 3)
    mean_lab   = lab_pixels.mean(axis=0)  # (3,)

    # Target is the body region's mean from the source saree
    target_lab = intermediate.body_lab_mean  # (3,) float32

    delta = float(
        deltaE_ciede2000(mean_lab[np.newaxis], target_lab[np.newaxis])[0]
    )
    return delta


def _cross_view_delta(
    outputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> float:
    """
    Bhattacharyya distance between garment colour histograms of all output pairs.
    Returns the maximum pairwise distance.

    Bhattacharyya distance: 0 = identical histograms, 1 = no overlap at all.
    The threshold 0.05 catches cases where the diffusion polish drifted the
    colour of one view significantly relative to the others.
    """
    hists = {}
    for pose_id, output in outputs.items():
        garment = output[masks[pose_id] > 127]  # (N, 3)
        if garment.shape[0] == 0:
            continue
        h = cv2.calcHist(
            [garment], [0, 1, 2], None,
            [16, 16, 16],
            [0, 256, 0, 256, 0, 256],
        )
        cv2.normalize(h, h)
        hists[pose_id] = h.flatten()

    pose_ids = list(hists.keys())
    max_dist = 0.0
    for i in range(len(pose_ids)):
        for j in range(i + 1, len(pose_ids)):
            d = cv2.compareHist(
                hists[pose_ids[i]], hists[pose_ids[j]], cv2.HISTCMP_BHATTACHARYYA
            )
            max_dist = max(max_dist, d)
    return max_dist


def _face_unchanged(
    output: np.ndarray,
    original: np.ndarray,
    face_mask: Optional[np.ndarray],
) -> bool:
    """
    True if every pixel inside the face mask is byte-for-byte identical
    to the original photo.  Any change means the garment mask leaked.

    If face_mask is None (not provided in Phase 3), the check is skipped
    and True is returned.
    """
    if face_mask is None or not (face_mask > 127).any():
        return True

    in_face = face_mask > 127
    max_diff = np.abs(
        output[in_face].astype(np.int32) - original[in_face].astype(np.int32)
    ).max()
    return int(max_diff) == 0


# ================================================================== colour correction


def correct_colour_drift(
    output: np.ndarray,
    mask: np.ndarray,
    intermediate: SareeIntermediate,
) -> np.ndarray:
    """
    Shift the a and b channels of the garment region to match the source
    saree's LAB mean.

    Call this when colour_delta > threshold.
    The correction is a global additive shift in LAB space — it corrects
    a tint without touching contrast or sharpness.
    """
    in_mask = mask > 127
    if not in_mask.any():
        return output.copy()

    # OpenCV LAB: a and b channels are 0–255 (shifted by 128 from skimage's -128–127)
    lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB).astype(np.float32)

    current_a = float(lab[:, :, 1][in_mask].mean())
    current_b = float(lab[:, :, 2][in_mask].mean())

    # intermediate stores LAB in skimage scale; convert to OpenCV scale
    target_a = float(intermediate.body_lab_mean[1]) + 128.0
    target_b = float(intermediate.body_lab_mean[2]) + 128.0

    delta_a = target_a - current_a
    delta_b = target_b - current_b

    corrected = lab.copy()
    corrected[:, :, 1][in_mask] = np.clip(lab[:, :, 1][in_mask] + delta_a, 0, 255)
    corrected[:, :, 2][in_mask] = np.clip(lab[:, :, 2][in_mask] + delta_b, 0, 255)

    result = output.copy()
    rgb = cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2RGB)
    result[in_mask] = rgb[in_mask]
    return result
