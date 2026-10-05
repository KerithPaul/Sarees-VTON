# Saree virtual try-on

Upload a photo of a saree laid flat. The system puts that saree on a fixed model in 3 poses and writes the images under `outputs/`.

Read `about.md` for how the phases work, what is finished, and how `src/` differs from the scripts that have actually been run.

This README is the install and run guide for the machine this repo was built on: Windows, Python 3.10, NVIDIA GPU with about 4 GB of VRAM.

---

## What you need

1. Windows (the Phase 3 and Phase 4 scripts open OpenCV windows)
2. Python 3.10 (the existing venv is 3.10.8)
3. An NVIDIA GPU with CUDA. Comments and checks assume a T1200-class 4 GB card. Phase 6 will not run on CPU.
4. About 15 GB free disk for a fresh model download. This checkout already has the weights (about 9 GB under `models/`).
5. A display for the Phase 3 and Phase 4 OpenCV UIs

A Hugging Face account is optional.

---

## If this checkout is already set up

A `venv/` is already here. Phase 3 pose files are already here. One saree is already processed at `input/processed/`.

From the project root:

```
venv\Scripts\activate
python scripts/phase4_preprocess.py input/your_saree.jpg
python scripts/generate.py
```

Outputs land in `outputs/<timestamp>/`:

1. `pose_XX_warp.jpg` — after warp and relight
2. `pose_XX_final.jpg` — after diffusion polish
3. `session_log.json` — timings and QC

Replace `your_saree.jpg` with any flat saree photo in `input/`.

`scripts/generate.py` is the command every docstring names. `scripts/generator.py` is a later rewrite of the same job with warp-gap fill, per-pixel shading, and a fallback when fp16 polish goes black. If you are unsure which to run, start with `generate.py`, then try `generator.py` if the warp has black holes or the polish comes out black.

Do not run `src.generate_views` on this checkout until pose files exist at `poses/front`, `poses/side`, and `poses/back`. They do not exist today. See "The src/ service" below.

---

## Fresh install

Run everything from the `VTON` folder.

### 1. Python venv

```
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. PyTorch (do this before requirements.txt)

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. The rest of the packages

```
pip install -r requirements.txt
```

Pinned stack: numpy 1.26.4 (must stay on 1.x), diffusers 0.31.0, transformers 4.46.3, opencv-python 4.10, rembg 2.0.59, onnxruntime 1.19.2 (CPU). xformers and bitsandbytes are left out on purpose.

### 4. Download models

```
python download_models.py
```

If ControlNet Tile is still missing after that, run:

```
python fix_controlnet.py
```

That writes `models/controlnet_tile/diffusion_pytorch_model.bin` (about 1.4 GB) plus `config.json`. `scripts/generator.py` loads `.bin` with `use_safetensors=False`. `scripts/generate.py` calls `from_pretrained` without that flag.

`check_env.py` accepts `.safetensors` or `.bin` weights.

### 5. Confirm the GPU path

```
python check_env.py
```

It checks Python 3.10, CUDA, VRAM, packages, numpy 1.x, and the four local weight folders, then generates one 512 px txt2img image to `check_output.png` and prints load time, generation time, and peak VRAM.

Close the browser and other GPU apps if free VRAM is under 3 GB.

Optional: in a second terminal, `watch_gpu.bat` prints `nvidia-smi dmon` so you can see real compute use (`sm` column).

---

## One-time pose setup (Phase 3)

Skip this if `poses/selected/`, `poses/masks/pose_XX_mask.png`, `poses/shading/`, `poses/regions/`, and `poses/drape/` already exist. They do in this checkout.

The numbered scripts are in the **project root**, not in `scripts/`.

1. Put model photos in `poses/raw/`.
2. `python 01_pose_selector.py` — click 3 photos, press Q. Files become `poses/selected/pose_01…03`.
3. `python 02_mask_generator.py` — then clean each `poses/masks/pose_XX_mask_auto.png` and save as `pose_XX_mask.png`.
4. `python 03_shading_extractor.py`
5. `python 04b_region_painter.py` (use this, not `04_region_labeler.py`, for accurate pallu/border). Keys: 1 body, 2 border, 3 pallu, S save.
6. Photograph any saree flat on a white surface and save it as `saree_flat_reference.jpg` in the project root. This file is already here.
7. `python 05_drape_mapper.py` — click 16–24 matching pairs, S to save (minimum 8).
8. `python 06_verify_drape.py` — look at `poses/drape/pose_XX_warp_check.jpg`. If the fabric is in the wrong place, redo step 7 for that pose.

Painter comments: `pose_01` side, `pose_02` front, `pose_03` back.

---

## Run for a new saree

### Preprocess

```
python scripts/phase4_preprocess.py input/your_saree.jpg
```

If you omit the path, the script takes the first jpg/png in `input/` that is not under `processed/`.

A window opens with Border % and Pallu % trackbars. Put the pallu on the **right**. Press F to flip if it is on the left. Press S to save, Q to keep defaults (border 8%, pallu 28%).

Writes `input/processed/saree_normalized.png` and `meta.json`.

### Generate

```
python scripts/generate.py
```

First load of Realistic Vision + ControlNet takes tens of seconds. After that, a 3-pose warp+polish run on this machine has been around 100 seconds.

Settings currently hardcoded in `scripts/generate.py`:

1. Seed 42
2. Strength 0.25
3. 20 steps (actual denoising steps are `ceil(20 × 0.25)` = 5)
4. CFG 5.0
5. ControlNet weight 0.60
6. TF32 on

Optional LoRA path: `models/lora/saree_fabric.safetensors`. Generation continues if that file is missing.

---

## The src/ service (not wired to current pose files)

Intended use, after models are in VRAM:

```
from src.generate_views import SareeService
service = SareeService()
result = service.generate_views("input/your_saree.jpg")
```

`scripts/test_warp_only.py --saree input/your_saree.jpg` runs this with `skip_diffusion=True`.

It will raise `FileNotFoundError` until you create `poses/front`, `poses/side`, and `poses/back` with the filenames in `src/load_pose.py` (`photo.png`, `mask.png`, `feather_mask.png`, `shading.npy`, `regions.png`, `src_points.npy`, `dst_points.npy`).

`scripts/verify_setup.py` checks that layout plus `models/realisticVisionV51_v51VAE` and `models/control_v11f1e_sd15_tile`. Those model folders exist. The pose folders do not.

`scripts/sweep_denoise.py` also uses `src/` and the same pose IDs.

---

## Folder map

| Path | What it is |
| --- | --- |
| `01_*.py` … `06_*.py` | Phase 3 tools (project root) |
| `scripts/phase4_preprocess.py` | Flatten one saree |
| `scripts/generate.py` | Warp + polish (named entry point) |
| `scripts/generator.py` | Same job, extra warp and fp16 guards |
| `src/` | Planned `SareeService` API |
| `poses/` | Phase 3 artifacts |
| `input/` | Saree uploads + `processed/` |
| `models/` | Local weights |
| `outputs/` | Run folders (created on each generate) |
| `requirements.txt` | pip pins |
| `download_models.py` | First-time weight fetch |
| `fix_controlnet.py` | ControlNet Tile when fp16 is missing |
| `check_env.py` | GPU and weight check |

---

## Troubleshooting

**`No ControlNet weights found`.** Run `python fix_controlnet.py`. If you then use `scripts/generate.py` and load fails on a `.bin` file, use `scripts/generator.py` or pass `use_safetensors=False` the way that file does.

**Black garment after polish.** Turing fp16 NaN. `scripts/generator.py` detects this and keeps the warp image. `scripts/generate.py` does not.

**Black holes in the warp.** Control points do not cover the mask. Redo `05_drape_mapper.py` with points on all four corners and the outer edge, or run `scripts/generator.py` which fills gaps with nearest fabric colour.

**`saree_flat_reference.jpg not found`.** Needed for Phase 3 mapping and for `06_verify_drape.py`. Put it in the project root.

**Colour QC fail.** `scripts/generate.py` flags ΔL > 35. Check `*_warp.jpg` before blaming diffusion.

**Face pixels changed.** The garment mask leaked into the face. Edit `poses/masks/pose_XX_mask.png` and close the gap at the blouse.

**`src` FileNotFoundError for `poses/front`.** Expected on this checkout. Use the `scripts/` path, or copy Phase 3 files into the layout `src/load_pose.py` wants.

**VRAM errors.** Enable `pipe.enable_model_cpu_offload()` only as a fallback. Comments say it costs about 8 s per image in PCIe copies.

**numpy 2.x.** `requirements.txt` pins 1.26.4 because several deps still break on numpy 2.

---

## What this README does not claim

1. There is no production web server in the repo. `SareeService` is a Python object, not an HTTP app.
2. There is no trained saree LoRA. `models/lora/` is empty. Generation still runs without it.
3. There is no `phase7_colab.py` in the tree.
4. Git is not initialised here.
