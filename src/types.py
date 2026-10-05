"""
Shared data types for the saree generation pipeline.

Every phase reads and writes these structures. Nothing else is passed
between phases.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class SareeIntermediate:
    """
    Everything extracted from one saree upload in Phase 4.
    Saved to disk so Phase 5 can load it instantly without re-running rembg.
    """

    # The saree as a clean RGB rectangle (canvas_height × canvas_width).
    # Default: 900 × 2400.
    canvas_rgb: np.ndarray        # (H, W, 3) uint8

    # Per-region crops cut from the canvas
    border_top_rgb: np.ndarray    # (border_h, body_w, 3) uint8
    border_bottom_rgb: np.ndarray # (border_h, body_w, 3) uint8
    pallu_rgb: np.ndarray         # (H, pallu_w, 3) uint8
    body_rgb: np.ndarray          # (body_h, body_w, 3) uint8

    # 512 × 512 seamlessly-tiling tile cut from the body region.
    # Used to fill large body areas without visible repeats.
    tile_rgb: np.ndarray          # (512, 512, 3) uint8

    # Average LAB color per region.  Shape (3,) float32: [L, a, b].
    # Stored in skimage LAB scale: L 0–100, a/b roughly -128 to 127.
    # Quality Gate uses these to detect colour drift.
    body_lab_mean: np.ndarray
    border_lab_mean: np.ndarray
    pallu_lab_mean: np.ndarray

    # Region fractions actually used (admin may change these with sliders)
    border_fraction: float = 0.10  # 8–12% of the short edge
    pallu_fraction: float = 0.20   # 20% of the long edge, from one end


@dataclasses.dataclass
class PoseData:
    """
    All pre-computed files for one model pose from Phase 3.
    Loaded from disk once per request.
    """

    pose_id: str

    # Original studio photo, full resolution, RGB
    photo_rgb: np.ndarray   # (H, W, 3) uint8

    # Binary garment mask — 0 outside, 255 inside
    mask: np.ndarray        # (H, W) uint8

    # Same mask blurred by ~6 px.  Used for smooth alpha compositing.
    feather_mask: np.ndarray  # (H, W) float32, 0.0–1.0

    # Relative brightness of the original garment, derived from its L channel.
    # ~1.0 at average-lit areas, ~0.6 in fold shadows, ~1.4 on raised pleats.
    # Zero outside the mask.
    shading_map: np.ndarray   # (H, W) float32

    # Hand-painted region label image: three flat colours mark body/border/pallu
    region_labels: np.ndarray  # (H, W, 3) uint8

    # Control points for the piecewise-affine drape warp.
    # src_points: (x, y) pixel positions on the flat saree canvas
    # dst_points: matching (x, y) positions on this pose's photo
    src_points: np.ndarray    # (N, 2) float32
    dst_points: np.ndarray    # (N, 2) float32

    # Face mask for QC.  Optional: QC skips the face check if None.
    face_mask: Optional[np.ndarray] = None  # (H, W) uint8


@dataclasses.dataclass
class PipelineConfig:
    """
    Runtime configuration.  Change paths and numbers here; nothing else
    needs to know about them.
    """

    # ----- Model files -----
    base_model_path: str = "models/realisticVisionV51_v51VAE"
    tile_controlnet_path: str = "models/control_v11f1e_sd15_tile"
    # LoRA is optional until Phase 7 training is done
    lora_path: Optional[str] = "models/saree_fabric_lora.safetensors"

    # ----- Directories -----
    poses_dir: str = "poses"
    outputs_dir: str = "outputs"
    cache_dir: str = "saree_cache"

    # ----- Flat canvas dimensions after perspective warp -----
    canvas_width: int = 2400
    canvas_height: int = 900

    # ----- Default region fractions (admin can override per upload) -----
    border_fraction: float = 0.10   # 8–12% of short edge is border strip
    pallu_fraction: float = 0.20    # 20% of long edge is pallu end

    # ----- Diffusion (Phase 6) -----
    # Sweep denoise_strength once on your GPU; write down the best number;
    # hardcode it here.  Below 0.2 is too weak.  Above 0.35 invents patterns.
    denoise_strength: float = 0.25
    num_steps: int = 20
    guidance_scale: float = 5.0
    controlnet_scale: float = 0.6
    lora_weight: float = 0.6
    # Fix the seed for all poses so fabric grain stays consistent between views
    seed: int = 42

    prompt: str = (
        "sareestyle, photo of a woman wearing a handloom saree, "
        "soft studio light, natural fabric texture, sharp detail"
    )
    negative_prompt: str = (
        "deformed hands, extra fingers, blurry, plastic skin, watermark, text"
    )

    # ----- Quality gate thresholds -----
    # CIEDE2000 > 6 means visible colour drift
    colour_delta_threshold: float = 6.0
    # Bhattacharyya distance between garment histograms of different views
    cross_view_delta_threshold: float = 0.05


@dataclasses.dataclass
class GenerateViewsResult:
    """Return value of SareeService.generate_views()."""
    outputs: dict        # pose_id → absolute path of saved PNG
    qc: dict             # colour_delta, colour_ok, cross_view_ok, face_match …
    timing_seconds: dict # preprocess, warp, polish, total
