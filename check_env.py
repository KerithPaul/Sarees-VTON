"""
Checks that the environment is actually usable before you write any real code.
Run after download_models.py

    python check_env.py

It prints a pass or fail line for each item, then generates one small test
image and times it. Write that number down. It is your speed baseline.
"""

import sys
import time
from pathlib import Path

PASS = "[ ok ]"
FAIL = "[FAIL]"
WARN = "[warn]"

problems = []


def check(label, ok, detail="", fatal=True):
    tag = PASS if ok else (FAIL if fatal else WARN)
    print(f"{tag} {label}" + (f"   {detail}" if detail else ""))
    if not ok and fatal:
        problems.append(label)


print("=" * 60)
print("Environment check")
print("=" * 60)

# ------------------------------------------------------------------- python
v = sys.version_info
check("Python 3.10", v.major == 3 and v.minor == 10, f"found {v.major}.{v.minor}.{v.micro}")

# -------------------------------------------------------------------- torch
try:
    import torch
    check("PyTorch installed", True, torch.__version__)
    check("CUDA available", torch.cuda.is_available())

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability(0)
        check("GPU detected", True, name)
        check("VRAM", vram >= 3.5, f"{vram:.1f} GB")
        check("Compute capability 7.0 or higher", cap[0] >= 7, f"sm_{cap[0]}{cap[1]}")

        free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        check(
            "Free VRAM right now",
            free_vram >= 3.0,
            f"{free_vram:.2f} GB free. Close Chrome and other GPU apps if this is low.",
            fatal=False,
        )
except Exception as e:
    check("PyTorch installed", False, str(e))

# ----------------------------------------------------------------- libraries
for mod, label in [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("cv2", "opencv"),
    ("skimage", "scikit-image"),
    ("rembg", "rembg"),
    ("onnxruntime", "onnxruntime"),
    ("PIL", "pillow"),
]:
    try:
        m = __import__(mod)
        check(label, True, getattr(m, "__version__", ""))
    except Exception as e:
        check(label, False, str(e))

# --------------------------------------------------------------- numpy pin
try:
    import numpy as np
    check("numpy below 2.0", np.__version__.startswith("1."), np.__version__)
except Exception as e:
    check("numpy", False, str(e))

# ----------------------------------------------------------- model files
def weights(folder):
    """Any real weight file counts. Not every repo ships safetensors."""
    root = Path(folder)
    found = list(root.rglob("*.safetensors")) + list(root.rglob("*.bin"))
    # ignore tiny stub files
    return [f for f in found if f.stat().st_size > 1_000_000]


paths = {
    "Realistic Vision checkpoint": weights("models/realistic_vision"),
    "VAE": weights("models/vae_ft_mse"),
    "ControlNet Tile": weights("models/controlnet_tile"),
    "SegFormer clothes": weights("models/segformer_clothes"),
}
for label, found in paths.items():
    check(label, len(found) > 0, found[0].name if found else "not found")

if problems:
    print("\nFix these before continuing:")
    for p in problems:
        print("   ", p)
    sys.exit(1)

# --------------------------------------------------- real generation test
print("\n" + "=" * 60)
print("Speed test: one 512 px image, 20 steps")
print("=" * 60)

import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler

ckpt = list(Path("models/realistic_vision").rglob("*.safetensors"))[0]

t0 = time.time()
vae = AutoencoderKL.from_pretrained("models/vae_ft_mse", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_single_file(
    str(ckpt),
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
load_time = time.time() - t0
print(f"Model load: {load_time:.1f} s   (this happens once at service start, not per request)")

# warm up run, first call is always slower
_ = pipe("test", num_inference_steps=2, height=512, width=512)

t0 = time.time()
img = pipe(
    "photo of a woman wearing a handloom cotton saree, soft studio light",
    negative_prompt="blurry, deformed, watermark, text",
    num_inference_steps=20,
    guidance_scale=5.0,
    height=512,
    width=512,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
gen_time = time.time() - t0

img.save("check_output.png")
peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"Generation: {gen_time:.1f} s")
print(f"Peak VRAM:  {peak:.2f} GB")
print("Saved: check_output.png")

print("\n" + "=" * 60)
if gen_time < 20:
    print("Good. You are on track for a 2 to 3 minute full run.")
elif gen_time < 35:
    print("Workable, but slow. Close every other GPU app and try again.")
else:
    print("Too slow. Check that the NVIDIA GPU is actually being used,")
    print("and that your display is running off the Intel chip, not the T1200.")
print("=" * 60)
