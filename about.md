# About this project

This repo is a saree virtual try-on system. You give it a photo of a saree laid flat. It dresses a fixed studio model in that saree, in three already chosen poses (side, front, back), and tries to keep her face, hands, hair, and background identical to the original photos.

The folder is named VTON. The code itself calls the work a "saree generation system." There is no separate product name in the repo.

It was written for a Windows machine with Python 3.10 and an NVIDIA T1200 (4 GB, Turing). The design target in the comments is a full 3-pose run in under about 150 seconds after the models are already loaded.

---

## What problem it solves

A saree is not a T-shirt. The same length of cloth is wrapped, pleated, and thrown over the shoulder. A flat product photo and a worn photo do not share a simple 1:1 overlay.

The system splits the job into two timescales:

1. One-time pose work (Phase 3). A human picks 3 model photos, paints masks and regions, and clicks corresponding points between a flat saree and each worn pose. That work is already done in this checkout.
2. Per-saree work (Phases 4, 5, 6). For each new saree photo, the code flattens the fabric, warps it onto those 3 poses using the saved points, copies the original fold lighting, then runs a light Stable Diffusion polish on the garment crop only.

The face is never sent through diffusion in the local pipeline. Pixels outside the garment mask are copied back from the original photo.

---

## How a local run works

### Phase 3: prepare the model poses (done once)

Scripts live in the project root, even though their docstrings say `python scripts/01_….py`.

1. `01_pose_selector.py` — pick 3 photos from `poses/raw/` (45 JPEGs in this checkout). Copies them to `poses/selected/` as `pose_01.jpeg`, `pose_02.jpeg`, `pose_03.jpeg`.
2. `02_mask_generator.py` — SegFormer B2 clothes (`models/segformer_clothes`) finds upper-clothes, skirt, and dress pixels. You then clean `*_mask_auto.png` in GIMP or Photopea and save as `poses/masks/pose_XX_mask.png`.
3. `03_shading_extractor.py` — builds a per-pixel brightness ratio inside the mask (LAB L channel vs the mask mean). Saved as `poses/shading/pose_XX_shading.npy`. At runtime, new fabric colour is kept and this map supplies the folds.
4. `04_region_labeler.py` — automatic split into body / border / pallu. `04b_region_painter.py` is the replacement for accurate painting. The painter comments treat `pose_01` as side, `pose_02` as front, `pose_03` as back, for a Nivi drape.
5. `05_drape_mapper.py` — click matching points on `saree_flat_reference.jpg` and each worn pose. Needs at least 8 pairs; the UI asks for 16–24. Saves `poses/drape/pose_XX_pts_flat.npy` and `pose_XX_pts_pose.npy`.
6. `06_verify_drape.py` — warps the reference saree onto each pose and shows a 60/40 blend so you can check coverage.

This checkout already has selected poses, cleaned masks, shading maps, region maps, and drape points.

### Phase 4: preprocess one saree upload

`scripts/phase4_preprocess.py`. Press F in the window to flip the image if the pallu is on the wrong side.

It strips the background with rembg (`isnet-general-use`), finds the fabric rectangle, warps it to landscape, lets you set border % and pallu % with trackbars, then writes `input/processed/`:

1. `saree_normalized.png` — the flattened rectangle
2. `body.png`, `body_tile.png`, `border_h.png`, `pallu.png` — zone crops
3. `meta.json` — sizes, fractions, mean LAB per zone

The last processed upload in this checkout is `input/your_saree.jpg` (canvas 712×641, border 8%, pallu 22%).

### Phases 5 and 6: warp, relight, polish

The docstring entry point is `python scripts/generate.py`. That file loads:

1. The normalised saree from `input/processed/`
2. Each pose from `poses/selected/`, plus mask, feathered mask, shading, and drape points
3. Realistic Vision V5.1 from `models/realistic_vision/*.safetensors`
4. VAE from `models/vae_ft_mse`
5. ControlNet Tile from `models/controlnet_tile`
6. Optional LoRA from `models/lora/saree_fabric.safetensors` (this file is not present)

For each pose it warps the flat saree with a piecewise affine transform (scikit-image), multiplies brightness by the shading map, composites with the feathered mask, then runs ControlNet Tile img2img on the garment crop (strength 0.25, 20 steps, CFG 5.0, ControlNet weight 0.6, seed 42). Output goes to `outputs/<YYYYMMDD_HHMMSS>/` as `pose_XX_warp.jpg`, `pose_XX_final.jpg`, and `session_log.json`.

There is a more complete rewrite of the same job in `scripts/generator.py` (warp-gap fill, per-pixel shading instead of a single L mean, fp16 black-image fallback, ControlNet `.bin` loading). See README for which command to run.

### Phase 7: fabric LoRA (not finished in this repo)

`models/lora/` is empty. Generation still runs without the LoRA. The Colab training file `phase7_colab.py` is not in the repo.

### Phase 8: quality checks

In `scripts/generate.py` these are simple:

1. Colour: mean LAB L of the output garment vs the source body mean. Pass if ΔL < 35.
2. Face: mean pixel diff in the top fifth of the image, outside the garment mask. Pass if mean diff < 5.

`src/quality_gates.py` is a stricter set (CIEDE2000, Bhattacharyya cross-view distance, byte-exact face mask) meant for the service API. It is not what `scripts/generate.py` runs.

---

## The second codebase (`src/`)

`src/` is a library-shaped service meant to be constructed once and called per upload:

```
from src.generate_views import SareeService
service = SareeService()   # loads GPU models once
result = service.generate_views("uploads/saree_123.jpg")
```

`SareeService.generate_views()` runs Phase 4 (`SareePreprocessor`), Phase 5 (`WarpAndRelight`), Phase 6 (`DiffusionPolish`), then `quality_gates.run_all()`. It can skip diffusion with `SareeService(skip_diffusion=True)`.

This path does **not** match the pose files that exist today.

`src/load_pose.py` expects:

```
poses/{front|side|back}/
  photo.png
  mask.png
  feather_mask.png
  shading.npy
  regions.png
  src_points.npy
  dst_points.npy
  face_mask.png   (optional)
```

What Phase 3 actually wrote is:

```
poses/selected/pose_01.jpeg
poses/masks/pose_01_mask.png
poses/masks/pose_01_mask_feathered.png
poses/shading/pose_01_shading.npy
poses/regions/pose_01_regions.png
poses/drape/pose_01_pts_flat.npy
poses/drape/pose_01_pts_pose.npy
```

There is no `poses/front`, `poses/side`, or `poses/back` directory. `scripts/verify_setup.py` and `scripts/test_warp_only.py` check the `src/` layout, so they will fail on this checkout until someone copies and renames the Phase 3 files.

Model paths also differ. `src/types.py` `PipelineConfig` points at `models/realisticVisionV51_v51VAE` and `models/control_v11f1e_sd15_tile`. `scripts/generate.py` points at `models/realistic_vision` and `models/controlnet_tile`. Both copies exist on disk.

Treat `src/` as a planned service API. Treat `scripts/phase4_preprocess.py` + `scripts/generate.py` (or `generator.py`) as the path that actually runs on the pose files in this checkout.

---

## Hardware notes that are in the code

`requirements.txt` and `check_env.py` assume Python 3.10, PyTorch 2.4.1 + CUDA 12.1, numpy 1.x, and at least about 3.5–4 GB of VRAM.

`scripts/generator.py` documents a Turing fp16 issue: the T1200 can overflow to NaN in the UNet, which decodes as a solid black crop. That file detects a suspiciously dark polish result and falls back to the warp composite. `scripts/generate.py` does not have that guard.

Real polish is img2img at strength 0.25–0.30, which only runs `ceil(steps × strength)` denoising steps (5 steps at 0.25 × 20), not a full 20-step txt2img. TF32 is enabled in the generate scripts.

`watch_gpu.bat` runs `nvidia-smi dmon` so you can see compute (`sm`) use, which Task Manager hides.

---

## What is present vs missing

Present in this checkout:

1. Phase 3 pose pack for 3 poses
2. One processed saree under `input/processed/`
3. Local model weights (Realistic Vision, VAE, ControlNet Tile, SegFormer; rembg cache is external under `~/.u2net`)
4. A Python 3.10.8 venv

Missing or unfinished:

1. `phase7_colab.py` and `LORA_DATASET_GUIDE.md`
2. Trained LoRA weights
3. `poses/front`, `poses/side`, `poses/back` for the `src/` API
4. Face masks (`face_mask.png`) for the `src/` QC face check
5. A git remote
6. A filled `.env` (the file exists and is empty)

---

## Glossary used in the code

**Pallu.** The decorated end of the saree, draped over the shoulder.

**Border.** The decorative edge strip (pleat edge and hem).

**Body.** The main field of the cloth, used as a tileable texture.

**Nivi drape.** The drape style the region painter comments assume.

**Piecewise affine warp.** A triangle mesh warp from clicked control points. Coverage only fills the convex hull of those points. Sparse clicks leave black holes inside the mask; `scripts/generator.py` fills those with nearest fabric colour as a stopgap.

**ControlNet Tile.** A ControlNet that conditions img2img on the image itself, so the polish pass should add fabric realism without redrawing the pattern.

**Denoise strength.** How far img2img is allowed to move from the warp. Comments treat 0.25 as the usual setting; above about 0.35 the model starts inventing pattern.
