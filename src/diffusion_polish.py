"""
Phase 6: diffusion polish.

Runs a light img2img pass over the garment region.
  - Realistic Vision V5.1 at denoise 0.25 adds photo realism.
  - ControlNet Tile at weight 0.6 prevents the model from redesigning the saree.
  - The saree fabric LoRA (from Phase 7 training) makes cloth read correctly.
  - Fixed seed across all poses keeps fabric grain consistent between views.

Everything outside the feathered mask is copied from the original photo,
so the face never changes.

Expect ~30 seconds per pose on a 4 GB GPU.

IMPORTANT — load() once at service startup.  Never call it inside a
request handler.  Loading takes 30–60 s and destroys the 2-minute target.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image

from .types import PipelineConfig


class DiffusionPolish:
    """
    Holds all models in VRAM.
    Construct once per service process; call polish() per pose per request.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipe: Optional[StableDiffusionControlNetImg2ImgPipeline] = None

    # ------------------------------------------------------------------ loading

    def load(self) -> None:
        """
        Load ControlNet, base model, scheduler, and LoRA into VRAM.
        Call this once at service startup.
        """
        t0 = time.perf_counter()

        print("Loading ControlNet Tile…")
        controlnet = ControlNetModel.from_pretrained(
            self.config.tile_controlnet_path,
            torch_dtype=torch.float16,
        )

        print("Loading Realistic Vision V5.1…")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # DPM++ 2M Karras — fast convergence, good fabric texture at 20 steps
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            use_karras_sigmas=True,
        )

        # LoRA — optional until Phase 7 training is done on Colab
        self._load_lora()

        # VRAM settings for 4 GB.
        # Do NOT add enable_model_cpu_offload unless you run out of VRAM;
        # it costs 8 s per image in PCIe transfer time.
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.pipe.to("cuda", dtype=torch.float16)

        print(f"Models ready in {time.perf_counter()-t0:.1f}s")

    def _load_lora(self) -> None:
        """Load the saree fabric LoRA if the file exists."""
        lora_path = self.config.lora_path
        if not lora_path:
            return
        p = Path(lora_path)
        if not p.exists():
            print(
                f"  LoRA not found at {lora_path} — skipping.\n"
                "  Train it on Colab (Phase 7) and place the .safetensors file there."
            )
            return

        print(f"  Loading LoRA: {p.name}")
        try:
            # diffusers 0.24+: named adapter API — lets you set weight at runtime
            self.pipe.load_lora_weights(
                str(p.parent),
                weight_name=p.name,
                adapter_name="saree",
            )
            self.pipe.set_adapters(["saree"], adapter_weights=[self.config.lora_weight])
        except TypeError:
            # Older diffusers: fuse_lora path (weight baked in, cannot change at runtime)
            self.pipe.load_lora_weights(str(p))
            self.pipe.fuse_lora(lora_scale=self.config.lora_weight)

    # ------------------------------------------------------------------ public

    def polish(
        self,
        composited: np.ndarray,    # Phase 5 output, full-size RGB
        original: np.ndarray,      # Original model photo, full-size RGB
        mask: np.ndarray,          # Binary garment mask (0 / 255)
        feather_mask: np.ndarray,  # Float32 feathered mask (0.0–1.0)
        pose_id: str,
    ) -> np.ndarray:
        """
        Run the diffusion polish over the garment crop.
        Returns a full-size RGB image where only the garment changed.
        """
        assert self.pipe is not None, (
            "Call load() before polish().  "
            "Models must be in VRAM before processing any request."
        )

        t0 = time.perf_counter()

        # Crop to garment bounding box
        y1, y2, x1, x2 = _garment_bbox(mask, pad=24)
        crop          = composited[y1:y2, x1:x2]
        orig_h, orig_w = crop.shape[:2]

        # Resize so the long side is 768, aligned to 8 (VAE requirement)
        crop_768 = _resize_to_768(crop)
        crop_pil = Image.fromarray(crop_768)

        # img2img + ControlNet Tile.
        # The seed is fixed for all poses so fabric grain is consistent
        # across front / side / back views.
        generator = torch.Generator("cuda").manual_seed(self.config.seed)

        with torch.autocast("cuda", dtype=torch.float16):
            result_pil = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=crop_pil,         # img2img init
                control_image=crop_pil, # ControlNet Tile: conditions on itself
                strength=self.config.denoise_strength,
                num_inference_steps=self.config.num_steps,
                guidance_scale=self.config.guidance_scale,
                controlnet_conditioning_scale=self.config.controlnet_scale,
                generator=generator,
            ).images[0]

        # Scale result back to original crop dimensions
        result_np = cv2.resize(
            np.array(result_pil), (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
        )

        # Paste into full-size image.
        # We blend diffusion result with the Phase 5 composite (not the original)
        # because we want the new saree, not the plain saree underneath.
        full            = original.copy()
        crop_feather    = feather_mask[y1:y2, x1:x2, np.newaxis].astype(np.float32)
        composited_crop = composited[y1:y2, x1:x2].astype(np.float32)

        blended = (
            composited_crop           * (1.0 - crop_feather)
            + result_np.astype(np.float32) * crop_feather
        )
        full[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # Belt-and-suspenders: hard-stamp all pixels outside the mask with the
        # original photo.  This guarantees the face is pixel-for-pixel identical
        # even if the feather mask bleeds slightly.
        outside = mask < 127
        full[outside] = original[outside]

        print(f"    diffusion [{pose_id}]: {time.perf_counter()-t0:.1f}s")
        return full


# ================================================================== helpers


def _garment_bbox(mask: np.ndarray, pad: int = 24) -> tuple[int, int, int, int]:
    """Return (y1, y2, x1, x2) tight bounding box of the mask, with padding."""
    rows = np.any(mask > 127, axis=1)
    cols = np.any(mask > 127, axis=0)

    ys = np.where(rows)[0]
    xs = np.where(cols)[0]

    if ys.size == 0 or xs.size == 0:
        raise ValueError("Garment mask is empty — check pose mask file.")

    H, W = mask.shape
    return (
        max(0,     ys[0]  - pad),
        min(H,     ys[-1] + pad),
        max(0,     xs[0]  - pad),
        min(W,     xs[-1] + pad),
    )


def _resize_to_768(image: np.ndarray) -> np.ndarray:
    """
    Resize image so the long side is 768 pixels.
    Both dimensions are rounded to the nearest multiple of 8 (VAE requirement).
    """
    h, w = image.shape[:2]
    scale = 768.0 / max(h, w)

    new_h = round(h * scale / 8) * 8
    new_w = round(w * scale / 8) * 8

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
