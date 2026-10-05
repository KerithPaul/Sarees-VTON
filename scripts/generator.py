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
from scipy.ndimage import distance_transform_edt
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

# The drape mapper stores control points in the pixel space of the
# reference image that was open during draping. The warp derives the
# correct canvas size from the control points themselves, not from any file.

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

def _get_flat_ref_dims(pts_flat: np.ndarray, pose_name: str):
    """
    Return (H, W) — the canvas size to resize the new saree into before warping.

    WHY NOT USE saree_flat_reference.jpg dimensions:
    The reference file on disk may be a compressed/resized copy at a different
    resolution than the original image that was open in the drape mapper when
    the user clicked the control points. Using the file dims can silently
    cause the warp to sample outside the image bounds, producing black output.

    CORRECT APPROACH — bounding box of pts_flat:
    The control points ARE their own coordinate system. max(pts_flat_x) and
    max(pts_flat_y) tell us exactly how large the canvas needs to be.
    A 12 % margin accounts for the fact that outermost clicks rarely land
    right at the pixel edge of the image.
    """
    max_x = float(pts_flat[:, 0].max())
    max_y = float(pts_flat[:, 1].max())
    W = int(max_x * 1.12)
    H = int(max_y * 1.12)
    return H, W


def warp_flat_to_pose(flat_bgr, pts_flat, pts_pose, H, W, pose_name=""):
    """
    Warp the normalised flat saree image into the pose canvas (H×W).
    Resizes the new saree to fill the coordinate space the control points
    were placed in (derived from pts_flat bounding box), then applies the
    PiecewiseAffineTransform.
    """
    ref_H, ref_W = _get_flat_ref_dims(pts_flat, pose_name)
    flat_to_warp  = cv2.resize(flat_bgr, (ref_W, ref_H))
    print(f"  Warp space: saree→{ref_W}×{ref_H}  canvas→{W}×{H}")

    flat_rgb = cv2.cvtColor(flat_to_warp, cv2.COLOR_BGR2RGB) / 255.0
    tform    = PiecewiseAffineTransform()
    ok       = tform.estimate(pts_pose, pts_flat)
    if not ok:
        print("  [FAIL] PiecewiseAffineTransform could not be estimated.")
        return np.zeros((H, W, 3), dtype=np.uint8)

    warped = sk_warp(flat_rgb, tform, output_shape=(H, W),
                     order=1, mode="constant", cval=0)
    result = (warped * 255).astype(np.uint8)

    # Coverage check — prints % of canvas pixels that received fabric
    covered = int((result.sum(axis=2) > 10).sum())
    pct     = 100.0 * covered / (H * W)
    print(f"  Warp coverage: {pct:.1f}%  ({covered}/{H*W} px)")
    if pct < 5.0:
        print("  [WARN] Coverage <5%  — control points need more spread across the saree")
        print("         Redo 05_drape_mapper.py for this pose, adding points at all 4 corners.")

    return result


def fill_warp_gaps(warped_rgb, feather_mask, pose_name=""):
    """
    PiecewiseAffineTransform only fills the *triangulated hull* of the drape
    control points. When those points don't reach all the way to the true
    garment silhouette (most commonly along the outer/border edge — see
    05_drape_mapper.py), a real hole is left between the warped fabric and
    the mask boundary: not anti-aliasing noise, but a solid black patch
    covering a meaningful fraction of the garment, often the visible border.

    Uses feather_mask (not hard_mask) to define "needs fabric colour": a
    Gaussian-blurred mask's non-zero footprint is always a few px wider than
    the binary mask it was blurred from, and composite() below blends using
    feather_mask. Filling only up to hard_mask left that extra band black,
    which composite() then partially blended in as a thin dark outline.

    The correct fix is denser/better-placed control points in
    05_drape_mapper.py. Until that's redone, extend the surrounding warped
    fabric into the hole so previews show colour and texture instead of
    solid black — this does not invent new fabric detail, it's a stopgap
    for judging overall look before Phase 7 (LoRA) fine-tuning.

    Uses nearest-valid-pixel fill (via a distance transform), not
    cv2.inpaint: this gap is a single connected region tens of thousands of
    pixels large spanning most of the mask's bounding box (control points
    only trace part of the silhouette), and cv2.inpaint's Telea/NS
    fast-marching does not reliably reach every pixel of a hole that big —
    testing showed pixels deep inside it were left exactly [0, 0, 0].
    Nearest-valid-pixel fill has no such limit: every hole pixel is
    guaranteed to get a real colour, copied from whichever warped fabric
    pixel is geometrically closest to it.

    IMPORTANT: the distance transform's "background" set (the landmarks a
    hole pixel can be matched to) must be exactly the actually-covered
    fabric pixels (`covered`) — NOT "not hole" (`~hole`). `~hole` also
    includes every pixel outside the feather mask entirely, which is just
    as black as the hole itself (nothing was ever warped there — it's
    simply outside the canvas region), for a completely different reason.
    Using `~hole` as the background let hole pixels near the mask's edge
    snap to one of *those* irrelevant black points instead of real fabric
    colour, since Euclidean nearest-neighbour has no notion of which zero
    pixels are "real" vs "coincidentally black" — it silently reproduced
    the exact black-outline bug this function exists to fix.
    """
    inside  = feather_mask > 0
    covered = warped_rgb.sum(axis=2) > 10
    hole    = inside & ~covered

    hole_px = int(hole.sum())
    if hole_px == 0:
        return warped_rgb

    mask_px = int(inside.sum())
    print(f"  [WARN] Warp gap: {100.0*hole_px/mask_px:.1f}% of {pose_name or 'pose'}'s "
          f"garment mask has no warp coverage (control points don't reach the edge) "
          f"— extending nearest fabric colour into the gap.")

    _, (rows, cols) = distance_transform_edt(~covered, return_distances=True, return_indices=True)
    filled = warped_rgb.copy()
    filled[hole] = warped_rgb[rows[hole], cols[hole]]
    return filled


def apply_shading(warped_rgb, shading):
    """
    Keep the new saree's actual colour (LAB a and b channels) but modulate
    the brightness (L channel) per pixel with:
        L_pixel × shading_map
    which transfers the original draping folds onto the new fabric.

    IMPORTANT: shading multiplies each pixel's OWN L value from the warped
    saree, not a single flat mean of the whole garment. The body, border and
    pallu each have a different intrinsic brightness (cream vs deep maroon);
    collapsing them to one global mean before scaling washes the border and
    pallu out to a flat, desaturated colour and erases their weave texture.
    03_shading_extractor.py's own docstring says it plainly:
    "new_texture × shading_map" — texture, not texture's average.

    Uses shading > 0 (not hard_mask) to decide where fold-shading actually
    applies: shading.npy is only meaningfully non-zero where
    03_shading_extractor.py's hard mask was > 127, everywhere else it's
    exactly 0. Gating on hard_mask here (a stricter/differently-shaped
    threshold than feather_mask, which composite() actually blends with)
    left a thin band right outside hard_mask multiplied by shading==0,
    i.e. forced to black — which then bled into the final feathered edge
    as a dark outline. Gating on shading>0 instead means every pixel this
    function doesn't touch keeps its own natural (already gap-filled)
    colour, so there's nothing black left for composite() to blend in.
    Returns BGR uint8.
    """
    inside = shading > 0
    if not inside.any():
        return cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)

    bgr = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0]
    new_L = L.copy()
    new_L[inside] = np.clip(L[inside] * shading[inside], 0, 255)
    lab[:, :, 0]  = new_L

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
    warped_rgb = warp_flat_to_pose(flat_bgr, pts_flat, pts_pose, H, W, pose_name)
    warped_rgb = fill_warp_gaps(warped_rgb, feather_mask, pose_name)

    print(f"  Applying shading map...")
    shaded_bgr = apply_shading(warped_rgb, shading)

    print(f"  Compositing onto original...")
    result = composite(pose_bgr, shaded_bgr, hard_mask, feather_mask)

    return result, pose_bgr, hard_mask


# ================================================================
# Phase 6 helpers: diffusion polish
# ================================================================

def load_pipeline():
    """Load and return the ControlNet img2img pipeline. Call once."""
    print("Loading diffusion pipeline...")

    # fp16 throughout: ~30s/pose vs ~270s/pose in fp32. The T1200 (a Turing
    # chip with no real fp16 tensor cores, same family as the GTX16xx
    # black-image bug) occasionally overflows to NaN in the UNet in fp16,
    # which decodes to a solid black crop. fp32 avoids it but is too slow to
    # be worth it for a 0.25-strength polish pass. Instead phase6() below
    # detects a NaN/black diffusion result and falls back to the (already
    # correctly coloured) warp composite for that pose rather than pasting
    # in garbage.
    DTYPE = torch.float16
    vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=DTYPE)

    # 1. Load the custom checkpoint as a plain img2img pipeline
    base = StableDiffusionImg2ImgPipeline.from_single_file(
        str(CKPT),
        vae=vae,
        torch_dtype=DTYPE,
        safety_checker=None,
        requires_safety_checker=False,
    )
    base.scheduler = DPMSolverMultistepScheduler.from_config(
        base.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )

    # 2. Load ControlNet Tile — handle both .safetensors and .bin
    has_safetensors = (CTRL_DIR / "diffusion_pytorch_model.safetensors").exists()
    has_bin         = (CTRL_DIR / "diffusion_pytorch_model.bin").exists()
    if not has_safetensors and not has_bin:
        raise FileNotFoundError(
            f"No ControlNet weights found in {CTRL_DIR}\n"
            f"Run: python fix_controlnet.py"
        )
    controlnet = ControlNetModel.from_pretrained(
        str(CTRL_DIR),
        torch_dtype=DTYPE,
        use_safetensors=has_safetensors,   # False → loads .bin silently
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

    # 4. Load LoRA if trained — completely optional, system works without it
    if LORA_PATH.exists():
        print(f"  Loading saree LoRA: {LORA_PATH.name}")
        pipe.load_lora_weights(
            str(LORA_PATH.parent), weight_name=LORA_PATH.name
        )
        pipe.fuse_lora(lora_scale=LORA_W)
        print(f"  LoRA fused at weight {LORA_W}")
    else:
        print(f"  LoRA not found — running without (this is fine for now).")

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

    # Guard against the fp16 NaN-decode bug (see load_pipeline): on this GPU
    # the UNet occasionally overflows to NaN in fp16, which decodes to a
    # solid black crop. Detect it by comparing mean brightness inside the
    # mask against the pre-diffusion crop, and fall back to the warp
    # composite — already correctly coloured — instead of pasting black.
    mask_bool = crop_mask > 127
    if mask_bool.any():
        out_mean  = float(out_bgr[mask_bool].mean())
        comp_mean = float(crop_comp[mask_bool].mean())
        if out_mean < comp_mean * 0.25:
            print(f"  [WARN] Diffusion polish looks corrupted (mean {out_mean:.0f} vs "
                  f"input {comp_mean:.0f}) — using warp composite for {pose_name} instead.")
            return composite_bgr

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
    if delta > 150:
        status = "FAIL — warp output is black. saree_flat_reference.jpg missing or wrong size."
    elif delta > 35:
        status = "WARN — colour drifted. Check warp output (*_warp.jpg) first."
    else:
        status = "PASS"
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

    # ---- Output directory
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {out_dir}/\n")

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

        # QC — convert numpy.bool_ → Python bool so json.dump works
        c_ok = bool(qc_colour(final_bgr, hard_mask, lab_body, pose_name))
        f_ok = bool(qc_face(final_bgr, orig_bgr, hard_mask, pose_name))

        total = time.time() - t0
        timings[pose_name] = {
            "warp_s": round(t5, 1),
            "polish_s": round(t6, 1),
            "total_s": round(total, 1),
        }
        qc_log[pose_name] = {"colour": c_ok, "face": f_ok}
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
