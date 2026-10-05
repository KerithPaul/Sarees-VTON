"""
Quick end-to-end test that skips Phase 6 (diffusion).

Runs Phases 4 and 5 only.  Total time: ~15 seconds.
Use this to verify your Phase 3 drape map is correct before waiting
for the full 2-minute pipeline.

Good output: the saree sits naturally on the model, folds look right.
Bad output: the pattern is misaligned or warped in the wrong direction.
If bad, go back to Phase 3 and fix the control points.

Usage:
    python scripts/test_warp_only.py --saree path/to/saree.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate_views import SareeService
from src.types import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saree",   required=True, help="Path to flat saree photo")
    parser.add_argument("--poses",   nargs="+", default=["front", "side", "back"])
    parser.add_argument("--border",  type=float, default=None, help="Border fraction (0.08–0.12)")
    parser.add_argument("--pallu",   type=float, default=None, help="Pallu fraction (0.18–0.25)")
    args = parser.parse_args()

    service = SareeService(skip_diffusion=True)

    result = service.generate_views(
        saree_image_path=args.saree,
        pose_ids=args.poses,
        border_fraction=args.border,
        pallu_fraction=args.pallu,
    )

    print("\n[Result]")
    for pose_id, path in result["outputs"].items():
        print(f"  {pose_id}: {path}")

    print("\n[QC]")
    qc = result["qc"]
    for pose_id, delta in qc.get("colour_delta", {}).items():
        status = "OK" if delta <= 6.0 else "DRIFT"
        print(f"  colour_delta [{pose_id}]: {delta:.2f}  ({status})")
    print(f"  face_match: {qc.get('face_match', 'N/A')}")

    print("\n[Timing]")
    for key, val in result["timing_seconds"].items():
        print(f"  {key}: {val:.1f}s")


if __name__ == "__main__":
    main()
