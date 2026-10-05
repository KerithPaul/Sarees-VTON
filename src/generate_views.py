"""
Top-level pipeline: generate_views()

This is the single function you hand to the backend developer.

Usage
-----
The service is long-lived.  Load the models once at process start:

    from src.generate_views import SareeService
    service = SareeService()          # loads models into VRAM (~45 s)

Then call once per admin upload (as a background job, not inside a web request):

    result = service.generate_views("uploads/saree_123.jpg")
    # result["outputs"]         → {"front": "outputs/saree_123_front.png", …}
    # result["qc"]              → colour/face checks
    # result["timing_seconds"]  → per-phase latency

The function returns a plain dict so it serialises cleanly to JSON for
whatever job queue you use (Celery, RQ, etc.).

Skip-diffusion mode
-------------------
For rapid testing during development, pass skip_diffusion=True.
Phase 6 is omitted; you get Phase 5 warp output in ~15 seconds total.
The result still looks decent and is good enough to test QC and saving.

    service = SareeService(skip_diffusion=True)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .diffusion_polish import DiffusionPolish
from .load_pose import load_pose
from .quality_gates import correct_colour_drift, run_all
from .saree_preprocessor import SareePreprocessor
from .types import GenerateViewsResult, PipelineConfig
from .warp_relight import WarpAndRelight


class SareeService:
    """
    Long-lived service object.  Construct once per process.
    Never construct inside a request handler.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        skip_diffusion: bool = False,
    ):
        self.config = config or PipelineConfig()
        self.skip_diffusion = skip_diffusion

        Path(self.config.outputs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Phase 4 — holds rembg session
        self.preprocessor = SareePreprocessor(self.config)

        # Phase 5 — stateless; created once to hold the config
        self.warper = WarpAndRelight(self.config)

        # Phase 6 — loads base model + ControlNet + LoRA into VRAM
        self.polisher = DiffusionPolish(self.config)
        if not skip_diffusion:
            self.polisher.load()  # 30–60 s here, never per-request

    # ------------------------------------------------------------------ public

    def generate_views(
        self,
        saree_image_path: str,
        pose_ids: Optional[list[str]] = None,
        border_fraction: Optional[float] = None,
        pallu_fraction: Optional[float] = None,
    ) -> dict:
        """
        Run the full pipeline for one saree upload.

        saree_image_path  — path to the uploaded flat saree photo
        pose_ids          — which poses to generate; default ["front","side","back"]
        border_fraction   — override the border width (0.08–0.12)
        pallu_fraction    — override the pallu width (0.18–0.25)

        Returns a plain dict (JSON-serialisable) matching the contract in
        the implementation plan, section 12.
        """
        if pose_ids is None:
            pose_ids = ["front", "side", "back"]

        timing: dict[str, float] = {}
        t_total = time.perf_counter()

        # ----------------------------------------------------------------
        # Phase 4: preprocess saree
        # ----------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"[Phase 4] Preprocessing saree: {saree_image_path}")
        t = time.perf_counter()

        intermediate = self.preprocessor.process(
            saree_image_path,
            border_fraction=border_fraction,
            pallu_fraction=pallu_fraction,
        )
        timing["preprocess"] = round(time.perf_counter() - t, 2)
        print(f"[Phase 4] Done — {timing['preprocess']:.1f}s")

        # Load all pose data now (cheap; no GPU)
        poses = {pid: load_pose(pid, self.config.poses_dir) for pid in pose_ids}

        # ----------------------------------------------------------------
        # Phase 5: warp and relight
        # ----------------------------------------------------------------
        print(f"\n[Phase 5] Warp and relight — {len(pose_ids)} poses")
        t = time.perf_counter()

        warp_outputs: dict[str, np.ndarray] = {}
        for pose_id in pose_ids:
            print(f"  → {pose_id}")
            warp_outputs[pose_id] = self.warper.run(intermediate, poses[pose_id])

        timing["warp"] = round(time.perf_counter() - t, 2)
        print(f"[Phase 5] Done — {timing['warp']:.1f}s")

        # ----------------------------------------------------------------
        # Phase 6: diffusion polish
        # ----------------------------------------------------------------
        final_outputs: dict[str, np.ndarray] = {}

        if self.skip_diffusion:
            print("\n[Phase 6] Skipped (skip_diffusion=True)")
            final_outputs = warp_outputs
            timing["polish"] = 0.0
        else:
            print(f"\n[Phase 6] Diffusion polish — {len(pose_ids)} poses")
            t = time.perf_counter()

            for pose_id in pose_ids:
                print(f"  → {pose_id}")
                final_outputs[pose_id] = self.polisher.polish(
                    composited=warp_outputs[pose_id],
                    original=poses[pose_id].photo_rgb,
                    mask=poses[pose_id].mask,
                    feather_mask=poses[pose_id].feather_mask,
                    pose_id=pose_id,
                )
            timing["polish"] = round(time.perf_counter() - t, 2)
            print(f"[Phase 6] Done — {timing['polish']:.1f}s")

        # ----------------------------------------------------------------
        # Phase 8: quality gates
        # ----------------------------------------------------------------
        print("\n[QC] Running quality gates…")
        qc = run_all(
            outputs=final_outputs,
            originals={pid: poses[pid].photo_rgb for pid in pose_ids},
            masks={pid: poses[pid].mask for pid in pose_ids},
            face_masks={pid: poses[pid].face_mask for pid in pose_ids},
            intermediate=intermediate,
            config=self.config,
            timing=timing,
        )

        # Auto-correct colour drift where needed
        for pose_id in pose_ids:
            delta = qc["colour_delta"].get(pose_id, 0.0)
            if delta > self.config.colour_delta_threshold:
                print(f"  Correcting colour drift for {pose_id} (delta={delta:.1f})…")
                final_outputs[pose_id] = correct_colour_drift(
                    final_outputs[pose_id],
                    poses[pose_id].mask,
                    intermediate,
                )

        # ----------------------------------------------------------------
        # Save outputs
        # ----------------------------------------------------------------
        saree_stem = Path(saree_image_path).stem
        output_paths: dict[str, str] = {}
        for pose_id, image_arr in final_outputs.items():
            out_path = Path(self.config.outputs_dir) / f"{saree_stem}_{pose_id}.png"
            Image.fromarray(image_arr).save(str(out_path))
            output_paths[pose_id] = str(out_path.resolve())
            print(f"  Saved: {out_path}")

        timing["total"] = round(time.perf_counter() - t_total, 2)
        _print_timing_summary(timing)

        return GenerateViewsResult(
            outputs=output_paths,
            qc=qc,
            timing_seconds=timing,
        ).__dict__


# ================================================================== helpers


def _print_timing_summary(timing: dict) -> None:
    print("\n[Timing]")
    labels = {
        "preprocess": "Phase 4 (preprocess saree)",
        "warp":       "Phase 5 (warp + relight)  ",
        "polish":     "Phase 6 (diffusion polish)",
        "total":      "Total                     ",
    }
    for key, label in labels.items():
        if key in timing:
            t = timing[key]
            flag = "  ⚠ slow" if key == "total" and t > 150 else ""
            print(f"  {label}: {t:.1f}s{flag}")
