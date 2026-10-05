"""
Phases 5 + 6: warp → relight → diffusion polish

For each of the three poses this script:
  Phase 5 — geometrically warps the flat saree onto the model using the
             drape control points, applies the fold / lighting structure
             from the shading map (LAB colour transfer), and feather-
             composites the result onto the original photograph.
  Phase 6 — crops the garment region and runs a light img2img pass with
             ControlNet Tile to make the fabric look photorealistic.
             The face, hands, hair and background are NEVER touched.

Prerequisites
-------------
  Phase 3 complete  → poses/  (masks, shading, regions, drape points)
  Phase 4 complete  → input/processed/  (meta.json, saree_normalized.png)

Usage
-----
  python scripts/generate.py

Outputs
-------
  outputs/<timestamp>/
    pose_XX_warp.jpg    intermediate after Phase 5 (useful for debugging)
    pose_XX_final.jpg   final after Phase 6 diffusion polish
    session_log.json    timings + quality check results
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
)
from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp as sk_warp

# ================================================================
# Paths
# ================================================================
PROC_DIR   = Path("input/processed")
POSES_DIR  = Path("poses")
OUT_DIR    = Path("outputs")

CKPT       = list(Path("models/realistic_vision").rglob("*.safetensors"))[0]
VAE_DIR    = Path("models/vae_ft_mse")
CTRL_DIR   = Path("models/controlnet_tile")
LORA_PATH  = Path("models/lora/saree_fabric.safetensors")

EXTS = {".jpg", ".jpeg", ".png"}

# ================================================================
# Generation settings
# ================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED      = 42       # fixed seed → consistent fabric grain across all 3 poses
STRENGTH  = 0.25     # low denoise = preserve the warp, just polish
STEPS     = 20
CFG       = 5.0
CTRL_W    = 0.60     # ControlNet Tile weight
LORA_W    = 0.60     # LoRA weight (ignored if no LoRA found)

PROMPT = (
    "sareestyle, photo of a woman wearing a traditional Indian handloom saree, "
    "real fabric texture, natural drape with folds, soft light, sharp detail, "
    "professional fashion photography"
)
NEGATIVE = (
    "deformed, blurry, watermark, text, low quality, cartoon, extra fingers, "
    "bad anatomy, plastic skin, overexposed, dull colour"
)


# ================================================================
# Phase 5 helpers: warp + relight
# ================================================================

def warp_flat_to_pose(flat_bgr, pts_flat, pts_pose, H, W):
    """
    Warp the normalised flat saree image into the pose canvas (H×W).
    PiecewiseAffineTransform: estimate maps pose coords → flat coords
    so warp() pulls the right pixel for each output location.
    Returns RGB uint8 at (H, W, 3).
    """
    flat_rgb = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2RGB) / 255.0
    tform    = PiecewiseAffineTransform()
    ok       = tform.estimate(pts_pose, pts_flat)
    if not ok:
        print("  [WARN] Transform could not be estimated. Check control points.")
        return np.zeros((H, W, 3), dtype=np.uint8)
    warped = sk_warp(flat_rgb, tform, output_shape=(H, W),
                     order=1, mode="constant", cval=0)
    return (warped * 255).astype(np.uint8)


def apply_shading(warped_rgb, shading, hard_mask):
    """
    Keep the new saree's actual colour (LAB a and b channels) but replace
    the brightness (L channel) with:
        L_saree_mean × shading_map
    which transfers the original draping folds onto the new fabric.
    Returns BGR uint8.
    """
    inside = hard_mask > 127
    if not inside.any():
        return cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)

    bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_mean = float(lab[:, :, 0][inside].mean())

    new_L = np.zeros_like(lab[:, :, 0])
    new_L[inside] = np.clip(L_mean * shading[inside], 0, 255)
    lab[:, :, 0]  = new_L
    for c in range(3):
        lab[:, :, c][~inside] = 0

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite(original_bgr, shaded_bgr, hard_mask, feather_mask):
    """
    Blend shaded saree onto original photo.
    Outside hard_mask → exact original pixels (face, hands, background untouched).
    Inside → feather blend so edges are seamless.
    """
    alpha  = feather_mask.astype(np.float32) / 255.0
    alpha3 = alpha[:, :, np.newaxis]
    blended = (shaded_bgr.astype(float) * alpha3
               + original_bgr.astype(float) * (1 - alpha3)).astype(np.uint8)
    # Hard guarantee: anything outside the mask is exactly the original
    blended[hard_mask < 10] = original_bgr[hard_mask < 10]
    return blended


def phase5(pose_name, flat_bgr):
    """Run Phase 5 for one pose. Returns (composite_bgr, original_bgr)."""
    sel_dir   = POSES_DIR / "selected"
    mask_dir  = POSES_DIR / "masks"
    shad_dir  = POSES_DIR / "shading"
    drape_dir = POSES_DIR / "drape"

    pose_path = next(
        (p for p in sel_dir.iterdir()
         if p.stem == pose_name and p.suffix.lower() in EXTS), None
    )
    if pose_path is None:
        raise FileNotFoundError(f"Pose photo not found for {pose_name}")

    pose_bgr = cv2.imread(str(pose_path))
    H, W     = pose_bgr.shape[:2]

    def load_mask(name):
        m = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {name}")
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m

    hard_mask    = load_mask(str(mask_dir / f"{pose_name}_mask.png"))
    feather_mask = load_mask(str(mask_dir / f"{pose_name}_mask_feathered.png"))

    shad_path = shad_dir / f"{pose_name}_shading.npy"
    if not shad_path.exists():
        raise FileNotFoundError(f"Shading map missing: {shad_path}  "
                                f"(run 03_shading_extractor.py)")
    shading = np.load(str(shad_path))
    if shading.shape != (H, W):
        shading = cv2.resize(shading, (W, H))

    pts_flat = np.load(str(drape_dir / f"{pose_name}_pts_flat.npy"))
    pts_pose = np.load(str(drape_dir / f"{pose_name}_pts_pose.npy"))

    print(f"  Warping saree → {pose_name}...")
    warped_rgb = warp_flat_to_pose(flat_bgr, pts_flat, pts_pose, H, W)

    print(f"  Applying shading map...")
    shaded_bgr = apply_shading(warped_rgb, shading, hard_mask)

    print(f"  Compositing onto original...")
    result = composite(pose_bgr, shaded_bgr, hard_mask, feather_mask)

    return result, pose_bgr, hard_mask


# ================================================================
# Phase 6 helpers: diffusion polish
# ================================================================

def load_pipeline():
    """Load and return the ControlNet img2img pipeline. Call once."""
    print("Loading diffusion pipeline...")

    vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=torch.float16)

    # 1. Load the custom checkpoint as a plain img2img pipeline
    base = StableDiffusionImg2ImgPipeline.from_single_file(
        str(CKPT),
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    base.scheduler = DPMSolverMultistepScheduler.from_config(
        base.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    # 2. Load ControlNet Tile
    controlnet = ControlNetModel.from_pretrained(
        str(CTRL_DIR), torch_dtype=torch.float16
    )

    # 3. Promote to ControlNet img2img pipeline by swapping in the ControlNet
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        unet=base.unet,
        controlnet=controlnet,
        scheduler=base.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")

    # 4. Load LoRA if trained (optional)
    if LORA_PATH.exists():
        print(f"  Loading saree LoRA from {LORA_PATH.name}...")
        pipe.load_lora_weights(
            str(LORA_PATH.parent), weight_name=LORA_PATH.name
        )
        pipe.fuse_lora(lora_scale=LORA_W)
        print(f"  LoRA fused at weight {LORA_W}")
    else:
        print(f"  No LoRA at {LORA_PATH}  — running without.")
        print(f"  (Train on Colab via Phase 7 for sharper fabric detail.)")

    print("  Pipeline ready.\n")
    return pipe


def _resize_for_diff(bgr, max_side=768):
    """Resize so the longer side ≤ max_side and both dims are div-by-8."""
    H, W   = bgr.shape[:2]
    scale  = min(1.0, max_side / max(H, W))
    nH     = max(8, (int(H * scale) // 8) * 8)
    nW     = max(8, (int(W * scale) // 8) * 8)
    return cv2.resize(bgr, (nW, nH)), H, W


def phase6(pipe, composite_bgr, original_bgr, hard_mask, pose_name):
    """
    Run Phase 6 for one pose.
    Crops the garment region, runs img2img, pastes back.
    Face, hands and background are never sent through diffusion.
    Returns final BGR at original resolution.
    """
    H, W = original_bgr.shape[:2]
    ys, xs = np.where(hard_mask > 127)
    if len(ys) == 0:
        print(f"  [WARN] Empty mask for {pose_name}, skipping polish.")
        return composite_bgr

    pad = 24
    y1 = max(0, ys.min() - pad);  y2 = min(H, ys.max() + pad)
    x1 = max(0, xs.min() - pad);  x2 = min(W, xs.max() + pad)

    crop_comp = composite_bgr[y1:y2, x1:x2]
    crop_mask = hard_mask[y1:y2, x1:x2]

    crop_rs, orig_ch, orig_cw = _resize_for_diff(crop_comp)
    pil_img  = Image.fromarray(cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB))
    pil_ctrl = pil_img   # ControlNet Tile: control = source image

    print(f"  Diffusion polish ({crop_rs.shape[1]}×{crop_rs.shape[0]}, "
          f"strength={STRENGTH}, steps={STEPS})...")
    gen = torch.Generator("cuda").manual_seed(SEED)
    with torch.inference_mode():
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=pil_img,
            control_image=pil_ctrl,
            strength=STRENGTH,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            controlnet_conditioning_scale=CTRL_W,
            generator=gen,
        ).images[0]

    # Resize diffusion output back to crop size
    out_bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    out_bgr = cv2.resize(out_bgr, (orig_cw, orig_ch))

    # Feather-blend diffusion result with original crop (inside mask only)
    mask_rs  = cv2.resize(crop_mask, (orig_cw, orig_ch)).astype(float) / 255.0
    mask_rs3 = mask_rs[:, :, np.newaxis]
    orig_crop = original_bgr[y1:y2, x1:x2]
    blended   = (out_bgr.astype(float) * mask_rs3
                 + orig_crop.astype(float) * (1 - mask_rs3)).astype(np.uint8)

    # Paste back into full image (pixels outside box untouched)
    final = original_bgr.copy()
    final[y1:y2, x1:x2] = blended
    return final


# ================================================================
# Quality checks
# ================================================================

def qc_colour(final_bgr, mask, src_lab_mean, pose_name):
    inside = mask > 127
    if not inside.any():
        return True
    lab    = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2LAB).astype(float)
    out_L  = float(lab[:, :, 0][inside].mean())
    src_L  = float(src_lab_mean[0])
    delta  = abs(out_L - src_L)
    status = "PASS" if delta < 35 else "WARN (colour drifted — try lower strength)"
    print(f"  QC colour [{pose_name}]: ΔL={delta:.1f}  {status}")
    return delta < 35


def qc_face(final_bgr, original_bgr, mask, pose_name):
    H         = final_bgr.shape[0]
    face_zone = np.zeros(final_bgr.shape[:2], dtype=bool)
    face_zone[:H // 5, :] = True
    face_zone &= (mask < 10)
    if not face_zone.any():
        return True
    diff = np.abs(
        final_bgr[face_zone].astype(float)
        - original_bgr[face_zone].astype(float)
    ).mean()
    status = "PASS" if diff < 5.0 else "WARN (face pixels changed — check mask)"
    print(f"  QC face   [{pose_name}]: mean_diff={diff:.2f}  {status}")
    return diff < 5.0


# ================================================================
# Main
# ================================================================

def main():
    # ---- Validate prerequisites
    if not PROC_DIR.exists() or not (PROC_DIR / "meta.json").exists():
        print("ERROR: input/processed/ not found.")
        print("Run first: python scripts/phase4_preprocess.py input/your_saree.jpg")
        return

    flat_path = PROC_DIR / "saree_normalized.png"
    if not flat_path.exists():
        print("ERROR: saree_normalized.png missing from input/processed/")
        return

    with open(PROC_DIR / "meta.json") as f:
        meta = json.load(f)

    flat_bgr = cv2.imread(str(flat_path))
    print(f"Flat saree: {flat_bgr.shape[1]}×{flat_bgr.shape[0]}")

    # ---- Discover poses
    sel_dir    = POSES_DIR / "selected"
    pose_files = sorted(
        p for p in sel_dir.iterdir() if p.suffix.lower() in EXTS
    )
    pose_names = [p.stem for p in pose_files]
    if not pose_names:
        print("ERROR: No poses found in poses/selected/")
        return
    print(f"Poses: {pose_names}\n")

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- Output directory
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs -> {out_dir}/\n")

    # ---- Load diffusion pipeline once
    pipe = load_pipeline()

    results  = {}
    timings  = {}
    qc_log   = {}
    lab_body = meta.get("lab_means", {}).get("body", [128, 128, 128])

    for pose_name in pose_names:
        print(f"{'='*54}")
        print(f"  {pose_name}")
        print(f"{'='*54}")
        t0 = time.time()

        # Phase 5
        t5 = time.time()
        comp_bgr, orig_bgr, hard_mask = phase5(pose_name, flat_bgr)
        t5 = time.time() - t5

        warp_path = out_dir / f"{pose_name}_warp.jpg"
        cv2.imwrite(str(warp_path), comp_bgr)
        print(f"  Warp saved: {warp_path.name}")

        # Phase 6
        t6 = time.time()
        final_bgr = phase6(pipe, comp_bgr, orig_bgr, hard_mask, pose_name)
        t6 = time.time() - t6

        final_path = out_dir / f"{pose_name}_final.jpg"
        cv2.imwrite(str(final_path), final_bgr)
        print(f"  Final saved: {final_path.name}")

        # QC
        c_ok = qc_colour(final_bgr, hard_mask, lab_body, pose_name)
        f_ok = qc_face(final_bgr, orig_bgr, hard_mask, pose_name)

        total = time.time() - t0
        timings[pose_name] = {
            "warp_s": round(t5, 1),
            "polish_s": round(t6, 1),
            "total_s": round(total, 1),
        }
        qc_log[pose_name] = {"colour": bool(c_ok), "face": bool(f_ok)}
        results[pose_name] = str(final_path)
        print(f"  Time: warp={t5:.1f}s  polish={t6:.1f}s  total={total:.1f}s\n")

    # ---- Session log
    log = {
        "timestamp": ts,
        "source_saree": meta.get("source", ""),
        "timings": timings,
        "qc": qc_log,
        "outputs": results,
    }
    with open(out_dir / "session_log.json", "w") as f:
        json.dump(log, f, indent=2)

    total_all = sum(t["total_s"] for t in timings.values())
    print(f"{'='*54}")
    print(f"Done in {total_all:.0f}s ({total_all/60:.1f} min)")
    print(f"All outputs in: {out_dir}/")
    for pose, path in results.items():
        print(f"  {pose}: {Path(path).name}")
    print()
    print("Next: share output images with your backend developer.")
    print("      Train the saree LoRA on Colab (Phase 7) for better fabric detail.")


if __name__ == "__main__":
    main()
