"""
Verify that all Phase 4–6 dependencies are in place.

Run this before the first real saree upload.  It checks:
  - Python packages are importable
  - CUDA is available and has enough VRAM
  - Model files are present and roughly the right size
  - Pose files are present for each pose ID
  - Outputs and cache directories are writable

Usage:
    python scripts/verify_setup.py

Exit code 0 = all clear.
Exit code 1 = something is missing; fix the listed items and re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_POSE_IDS  = ["front", "side", "back"]
POSES_DIR          = Path("poses")
OUTPUTS_DIR        = Path("outputs")
CACHE_DIR          = Path("saree_cache")

# Model file → minimum expected size in MB
MODEL_FILES: dict[str, int] = {
    "models/realisticVisionV51_v51VAE": 2_000,   # SD 1.5 checkpoint
    "models/control_v11f1e_sd15_tile":  600,     # ControlNet Tile fp16
}
# LoRA is optional — warn if missing but don't fail
OPTIONAL_MODEL_FILES: dict[str, int] = {
    "models/saree_fabric_lora.safetensors": 30,
}

REQUIRED_POSE_FILES = [
    "photo.png",
    "mask.png",
    "feather_mask.png",
    "shading.npy",
    "regions.png",
    "src_points.npy",
    "dst_points.npy",
]

errors:   list[str] = []
warnings: list[str] = []


# ------------------------------------------------------------------ packages

def check_packages() -> None:
    packages = [
        ("cv2",           "opencv-python"),
        ("numpy",         "numpy"),
        ("PIL",           "Pillow"),
        ("skimage",       "scikit-image"),
        ("rembg",         "rembg"),
        ("diffusers",     "diffusers"),
        ("torch",         "torch"),
        ("transformers",  "transformers"),
        ("accelerate",    "accelerate"),
        ("safetensors",   "safetensors"),
    ]
    print("Checking packages…")
    for module, pip_name in packages:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            errors.append(f"Missing package: pip install {pip_name}")
            print(f"  ✗ {module}")


# ------------------------------------------------------------------ CUDA

def check_cuda() -> None:
    print("\nChecking CUDA…")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append(
                "CUDA is not available.  Phase 6 needs a CUDA GPU.\n"
                "  Check your PyTorch install: https://pytorch.org/get-started/locally/"
            )
            print("  ✗ CUDA not available")
            return

        device = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA: {device}")
        print(f"  ✓ VRAM: {vram:.1f} GB")

        if vram < 3.8:
            errors.append(
                f"VRAM is {vram:.1f} GB — Phase 6 needs at least 4 GB.\n"
                "  Enable pipe.enable_model_cpu_offload() as a fallback "
                "(costs ~8 s per image)."
            )

    except Exception as e:
        errors.append(f"CUDA check failed: {e}")


# ------------------------------------------------------------------ model files

def check_models() -> None:
    print("\nChecking model files…")
    for path_str, min_mb in MODEL_FILES.items():
        p = Path(path_str)
        if p.is_dir():
            # HuggingFace snapshot directory
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
            if size_mb < min_mb:
                errors.append(
                    f"Model directory too small: {p}  ({size_mb:.0f} MB, expected ≥{min_mb} MB)\n"
                    "  Re-download: huggingface_hub.snapshot_download(…)"
                )
                print(f"  ✗ {p}  ({size_mb:.0f} MB)")
            else:
                print(f"  ✓ {p}  ({size_mb:.0f} MB)")
        elif p.is_file():
            size_mb = p.stat().st_size / 1e6
            if size_mb < min_mb:
                errors.append(f"Model file too small: {p}  ({size_mb:.0f} MB)")
                print(f"  ✗ {p}  ({size_mb:.0f} MB)")
            else:
                print(f"  ✓ {p}  ({size_mb:.0f} MB)")
        else:
            errors.append(
                f"Model not found: {p}\n"
                "  Run scripts/download_models.py to download all models."
            )
            print(f"  ✗ {p}  (missing)")

    for path_str, min_mb in OPTIONAL_MODEL_FILES.items():
        p = Path(path_str)
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  ✓ {p}  ({size_mb:.0f} MB)  [optional]")
        else:
            warnings.append(
                f"LoRA not found: {p}\n"
                "  Train it on Colab (Phase 7) and copy the .safetensors here.\n"
                "  Phase 6 will still run without it — cloth texture will be less accurate."
            )
            print(f"  ⚠ {p}  (not found — Phase 7 training needed)")


# ------------------------------------------------------------------ pose files

def check_poses() -> None:
    print("\nChecking pose files…")
    for pose_id in REQUIRED_POSE_IDS:
        pose_dir = POSES_DIR / pose_id
        if not pose_dir.is_dir():
            errors.append(
                f"Pose directory missing: {pose_dir}\n"
                "  Complete Phase 3 for pose '{pose_id}'."
            )
            print(f"  ✗ {pose_dir}/  (directory missing)")
            continue

        missing = [f for f in REQUIRED_POSE_FILES if not (pose_dir / f).exists()]
        if missing:
            for f in missing:
                errors.append(f"Missing pose file: {pose_dir / f}")
            print(f"  ✗ {pose_dir}/  missing: {missing}")
        else:
            # Quick sanity check on point arrays
            try:
                import numpy as np
                src = np.load(str(pose_dir / "src_points.npy"))
                dst = np.load(str(pose_dir / "dst_points.npy"))
                if src.shape != dst.shape:
                    errors.append(
                        f"Point array shape mismatch in {pose_dir}:\n"
                        f"  src_points {src.shape} != dst_points {dst.shape}"
                    )
                    print(f"  ✗ {pose_dir}/  point shape mismatch")
                elif src.shape[0] < 8:
                    warnings.append(
                        f"Only {src.shape[0]} control points in {pose_dir}.\n"
                        "  16–24 points give better warp quality."
                    )
                    print(f"  ⚠ {pose_dir}/  only {src.shape[0]} control points (need ≥16)")
                else:
                    print(f"  ✓ {pose_dir}/  ({src.shape[0]} control points)")
            except Exception as e:
                errors.append(f"Could not load points from {pose_dir}: {e}")
                print(f"  ✗ {pose_dir}/  point load error: {e}")

        # Warn if face mask is absent
        if not (pose_dir / "face_mask.png").exists():
            warnings.append(
                f"No face_mask.png in {pose_dir}.\n"
                "  QC face check will be skipped for this pose.\n"
                "  Extract face pixels from the SegFormer output and save them there."
            )
            print(f"  ⚠ {pose_dir}/face_mask.png  (not found — QC face check disabled)")


# ------------------------------------------------------------------ directories

def check_directories() -> None:
    print("\nChecking writable directories…")
    for d in [OUTPUTS_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            print(f"  ✓ {d}/")
        except OSError as e:
            errors.append(f"Directory not writable: {d}  ({e})")
            print(f"  ✗ {d}/  not writable")


# ------------------------------------------------------------------ main

def main() -> int:
    print("=" * 56)
    print("Saree pipeline setup check — Phases 4–6")
    print("=" * 56)

    check_packages()
    check_cuda()
    check_models()
    check_poses()
    check_directories()

    print("\n" + "=" * 56)
    if warnings:
        print("WARNINGS (non-blocking):")
        for w in warnings:
            for line in w.split("\n"):
                print(f"  {line}")
    if errors:
        print("\nERRORS (fix these before running):")
        for e in errors:
            for line in e.split("\n"):
                print(f"  {line}")
        print("\nSetup FAILED — see errors above.")
        return 1
    else:
        print("All checks passed.  Ready to run.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
