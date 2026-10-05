"""
Sweep the denoise_strength value for Phase 6.

Run this once.  Look at the outputs, pick the number that looks best,
and hardcode it as denoise_strength in PipelineConfig.

What to look for:
  0.15 — barely any improvement over Phase 5 warp
  0.20 — cloth starts reading as real fabric
  0.25 — usually the sweet spot
  0.30 — good but watch for pattern drift on complex prints
  0.35 — often invents new details; border motifs may shift slightly
  0.40 — too much; the model redesigns the saree

Usage:
    python scripts/sweep_denoise.py \
        --saree   uploads/test_saree.jpg \
        --pose    front \
        --values  0.15 0.20 0.25 0.30 0.35

The script saves one image per denoise value to outputs/sweep/.
Compare them side by side and write down the best number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.diffusion_polish import DiffusionPolish
from src.load_pose import load_pose
from src.saree_preprocessor import SareePreprocessor
from src.types import PipelineConfig
from src.warp_relight import WarpAndRelight


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep denoise_strength for Phase 6.")
    parser.add_argument("--saree",  required=True, help="Path to flat saree photo")
    parser.add_argument("--pose",   default="front", help="Pose ID to use (default: front)")
    parser.add_argument(
        "--values", nargs="+", type=float,
        default=[0.15, 0.20, 0.25, 0.30, 0.35],
        help="Denoise strength values to try",
    )
    args = parser.parse_args()

    out_dir = Path("outputs/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()

    # Load once; reuse for every denoise value
    print("Loading models (this takes 30–60 s; done only once)…")
    preprocessor = SareePreprocessor(config)
    warper       = WarpAndRelight(config)
    polisher     = DiffusionPolish(config)
    polisher.load()

    # Phase 4 + 5 once
    print(f"\nPreprocessing saree: {args.saree}")
    intermediate = preprocessor.process(args.saree)

    pose = load_pose(args.pose, config.poses_dir)
    print(f"Warping pose: {args.pose}")
    warped = warper.run(intermediate, pose)

    # Save Phase 5 output as the baseline
    baseline_path = out_dir / f"denoise_baseline_phase5.png"
    Image.fromarray(warped).save(str(baseline_path))
    print(f"Saved baseline (Phase 5 warp): {baseline_path}")

    # Phase 6 for each denoise value
    for strength in args.values:
        print(f"\nTrying denoise_strength={strength:.2f}…")
        config.denoise_strength = strength

        result = polisher.polish(
            composited=warped,
            original=pose.photo_rgb,
            mask=pose.mask,
            feather_mask=pose.feather_mask,
            pose_id=args.pose,
        )

        name = f"denoise_{strength:.2f}.png"
        path = out_dir / name
        Image.fromarray(result).save(str(path))
        print(f"  Saved: {path}")

    print(f"\nAll outputs saved to {out_dir}/")
    print("Open them side by side and pick the best strength.")
    print("Then set that value as denoise_strength in PipelineConfig.")


if __name__ == "__main__":
    main()
