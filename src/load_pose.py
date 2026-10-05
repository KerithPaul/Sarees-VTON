"""
Load pre-computed pose data from disk (written in Phase 3).

Every file this function reads was created once, by hand, during pose
preparation.  Nothing here touches the GPU.

Expected directory layout for each pose:

    poses/{pose_id}/
        photo.png         original model photo (RGB)
        mask.png          binary garment mask (0 / 255)
        feather_mask.png  Gaussian-blurred version of mask (0–255 uint8)
        shading.npy       float32 shading map, same size as photo
        regions.png       3-colour region label image (RGB)
        src_points.npy    (N, 2) float32 — (x, y) on flat saree canvas
        dst_points.npy    (N, 2) float32 — (x, y) on model photo
        face_mask.png     (optional) binary face mask for QC
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from .types import PoseData


def load_pose(pose_id: str, poses_dir: str = "poses") -> PoseData:
    base = Path(poses_dir) / pose_id
    if not base.is_dir():
        raise FileNotFoundError(
            f"Pose directory not found: {base}\n"
            "Phase 3 must be completed before running the runtime pipeline."
        )

    _check_required(base, [
        "photo.png", "mask.png", "feather_mask.png",
        "shading.npy", "regions.png", "src_points.npy", "dst_points.npy",
    ])

    photo        = _load_rgb(base / "photo.png")
    mask         = _load_binary_mask(base / "mask.png")
    feather      = _load_feather(base / "feather_mask.png")
    shading      = np.load(str(base / "shading.npy")).astype(np.float32)
    regions      = _load_rgb(base / "regions.png")
    src_points   = np.load(str(base / "src_points.npy")).astype(np.float32)
    dst_points   = np.load(str(base / "dst_points.npy")).astype(np.float32)

    # Resize shading map if it was saved at a different resolution than the photo
    if shading.shape[:2] != photo.shape[:2]:
        import cv2
        shading = cv2.resize(
            shading, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    face_mask = None
    face_path = base / "face_mask.png"
    if face_path.exists():
        face_mask = _load_binary_mask(face_path)

    return PoseData(
        pose_id=pose_id,
        photo_rgb=photo,
        mask=mask,
        feather_mask=feather,
        shading_map=shading,
        region_labels=regions,
        src_points=src_points,
        dst_points=dst_points,
        face_mask=face_mask,
    )


# ------------------------------------------------------------------ helpers


def _check_required(base: Path, files: list[str]) -> None:
    missing = [f for f in files if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Phase 3 files in {base}:\n  " + "\n  ".join(missing)
        )


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def _load_binary_mask(path: Path) -> np.ndarray:
    """Return a uint8 array with values 0 or 255."""
    grey = np.array(Image.open(str(path)).convert("L"))
    return (grey > 127).astype(np.uint8) * 255


def _load_feather(path: Path) -> np.ndarray:
    """Return float32 0.0–1.0."""
    grey = np.array(Image.open(str(path)).convert("L"))
    return grey.astype(np.float32) / 255.0
