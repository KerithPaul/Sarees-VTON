"""
Fixes the missing ControlNet Tile download.

The tile ControlNet is the one repo in lllyasviel's set that never got fp16
variants uploaded. Only the full precision file exists, so the original
filter matched nothing. This script lists what is actually in the repo,
picks the best available weight file, and downloads it.

    python fix_controlnet.py

About 1.45 GB. You cast it to fp16 at load time instead, which costs nothing.
"""

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "lllyasviel/control_v11f1e_sd15_tile"
DEST = Path("models/controlnet_tile")
DEST.mkdir(parents=True, exist_ok=True)

api = HfApi()
files = api.list_repo_files(REPO)

print("Files in the repo:")
for f in sorted(files):
    print("   ", f)

# best to worst. diffusers can load any of these
preferred = [
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
]

weight_file = next((p for p in preferred if p in files), None)

if weight_file is None:
    print("\nNo usable weight file found. Something changed upstream.")
    raise SystemExit(1)

print(f"\nPicked: {weight_file}")
if not weight_file.endswith(".safetensors"):
    print("Note: this is a .bin file, so pass use_safetensors=False when loading.")

for name in ["config.json", weight_file]:
    path = hf_hub_download(repo_id=REPO, filename=name, local_dir=DEST)
    size = Path(path).stat().st_size / (1024 ** 2)
    print(f"Saved {name}  ({size:.0f} MB)")

print("\nDone. Run check_env.py again.")
