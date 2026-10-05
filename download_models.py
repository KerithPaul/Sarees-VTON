"""
Downloads every model the saree pipeline needs.
Run this once, right after installing the packages.

    python download_models.py

Total download: about 3.5 GB. Everything lands in ./models
Nothing here needs a paid account. A free Hugging Face login is optional
but makes downloads faster and more reliable.
"""

import os
import sys
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

api = HfApi()


def free_space_gb(path="."):
    return shutil.disk_usage(path).free / (1024 ** 3)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ---------------------------------------------------------------- 0. disk check
free = free_space_gb()
print(f"Free disk space: {free:.1f} GB")
if free < 15:
    print("Stop. You need at least 15 GB free. Clear space and run again.")
    sys.exit(1)


# ---------------------------------------- 1. base image model, Realistic Vision
banner("1 of 5  Realistic Vision V5.1  (about 2.0 GB)")

REPO_RV = "SG161222/Realistic_Vision_V5.1_noVAE"
files = api.list_repo_files(REPO_RV)

# pick the fp16 no-ema single file checkpoint, whatever it is called today
candidates = [
    f for f in files
    if f.endswith(".safetensors")
    and "fp16" in f.lower()
    and "noema" in f.lower().replace("-", "").replace("_", "")
]
if not candidates:
    candidates = [f for f in files if f.endswith(".safetensors") and "/" not in f]

if not candidates:
    print("Could not find a checkpoint file. Files in the repo:")
    for f in files:
        print("   ", f)
    sys.exit(1)

ckpt_name = sorted(candidates, key=len)[0]
print(f"Downloading: {ckpt_name}")
rv_path = hf_hub_download(
    repo_id=REPO_RV,
    filename=ckpt_name,
    local_dir=MODELS / "realistic_vision",
)
print("Saved to:", rv_path)


# ------------------------------------------------------------------- 2. the VAE
banner("2 of 5  sd-vae-ft-mse  (about 335 MB)")

snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir=MODELS / "vae_ft_mse",
    allow_patterns=["*.json", "*.safetensors"],
)
print("Done.")


# ------------------------------------------------------- 3. ControlNet Tile fp16
banner("3 of 5  ControlNet Tile  (about 1.45 GB)")

# The tile ControlNet is the odd one out. Every other control_v11 repo ships
# an fp16 variant, this one only has the full precision file. So pick whatever
# is actually there instead of assuming a filename. We cast to fp16 at load.
REPO_TILE = "lllyasviel/control_v11f1e_sd15_tile"
tile_files = api.list_repo_files(REPO_TILE)

preferred = [
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
]
weight_file = next((p for p in preferred if p in tile_files), None)

if weight_file is None:
    print("No usable ControlNet weight file found. Files in the repo:")
    for f in sorted(tile_files):
        print("   ", f)
    sys.exit(1)

print(f"Downloading: {weight_file}")
for name in ["config.json", weight_file]:
    hf_hub_download(repo_id=REPO_TILE, filename=name, local_dir=MODELS / "controlnet_tile")
print("Done.")


# ------------------------------------------- 4. clothes segmentation, SegFormer
banner("4 of 5  SegFormer B2 clothes  (about 110 MB)")

snapshot_download(
    repo_id="mattmdjaga/segformer_b2_clothes",
    local_dir=MODELS / "segformer_clothes",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
print("Done.")


# ------------------------------------------------- 5. rembg background removal
banner("5 of 5  rembg isnet-general-use  (about 180 MB)")

# rembg downloads its own weights on first use, into ~/.u2net
# trigger that now so demo day is not the first time it hits the network
try:
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove

    session = new_session("isnet-general-use")
    dummy = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
    remove(dummy, session=session)
    print("Done. Weights cached in", os.path.expanduser("~/.u2net"))
except Exception as e:
    print("rembg warm up failed:", e)
    print("Not fatal. It will download on first real use.")


banner("All downloads finished")
print(f"Free disk space now: {free_space_gb():.1f} GB")
print("\nNext step:  python check_env.py")
