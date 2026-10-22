"""
Saree & Garment Try-On — Gradio web app
Powered by FASHN API  |  product-to-model  |  1 credit per image

Strategy:
  • Primary mode  : product_to_model (auto-generate) — best for sarees
  • Fallback mode : tryon-v1.6 with pose photos — NOTE: does NOT preserve the
                    model's face at 1 credit (face_reference = +3 credits extra)

Usage:
    python app.py          →  http://localhost:7860
"""

import colorsys
import io
import os
import random
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from fashn_client import tryon, product_to_model

load_dotenv()

OUTPUTS_DIR     = Path("outputs")
MODEL_POSES_DIR = Path("model_poses")
OUTPUTS_DIR.mkdir(exist_ok=True)
MODEL_POSES_DIR.mkdir(exist_ok=True)

POSE_KEYS   = ["front", "side", "back"]
POSE_LABELS = ["Front View", "Side View  (pallu visible)", "Back View"]

# ─────────────────────────────────────────────────────────────── colour helper

def extract_dominant_color(img: Image.Image) -> str:
    """Return a simple English colour name for the garment dominant hue."""
    try:
        small = img.copy()
        small.thumbnail((80, 80))
        arr = np.array(small.convert("RGB"), dtype=float)
        # Ignore near-white background pixels (value > 220 on all channels)
        mask = np.any(arr < 215, axis=2)
        if mask.sum() < 30:
            return "silk"
        pixels = arr[mask]
        avg    = pixels.mean(axis=0)
        r, g, b = avg[0] / 255, avg[1] / 255, avg[2] / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if v < 0.22:
            return "deep dark"
        if s < 0.15:
            return "cream" if v > 0.85 else "silver grey"
        h360 = h * 360
        if   h360 < 15 or h360 >= 345: return "deep red"
        elif h360 < 40:                 return "orange"
        elif h360 < 75:                 return "golden yellow"
        elif h360 < 150:                return "green"
        elif h360 < 195:                return "teal"
        elif h360 < 255:                return "blue"
        elif h360 < 285:                return "indigo"
        elif h360 < 315:                return "purple"
        else:                           return "pink"
    except Exception:
        return "silk"


# ───────────────────────────────────────────────────────── prompt builder

# Shared descriptors for each pose — these are the key fixes:
#   BACK:  "NOT turned" + "straight forward" (fixes head-turn issue)
#   ALL:   blouse always explicitly named (fixes missing blouse)
#   ALL:   anchor on {desc} so garment colour/type goes into prompt

_POSE_BLOCKS = {
    "front": (
        "standing directly facing the camera, "
        "body square to camera, head upright, face looking straight at the lens, "
        "pallu of the saree draped diagonally from left shoulder falling to mid-thigh, "
        "neat fan-pleats visible at the front waist, arms relaxed at the sides, "
    ),
    "side": (
        "standing in right three-quarter profile, "
        "head turned gently toward camera (no more than 45 degrees), "
        "pallu clearly visible draped over the left shoulder and falling to the floor, "
        "natural fabric folds showing depth and weight of the saree from the side, "
        "arms relaxed, one hand loosely at the waist, "
    ),
    "back": (
        # ← explicit double emphasis: fixes the 'head turning too much' bug
        "standing with BACK squarely facing the camera, "
        "head facing STRAIGHT FORWARD — NOT turned, NOT tilted sideways, "
        "neck visible, gaze directed away from camera toward the horizon, "
        "pallu flowing straight down the spine from the left shoulder to the floor, "
        "traditional blouse back-neckline fully exposed and visible, "
        "arms relaxed at sides, "
    ),
}

_COMMON_BASE = (
    "Professional fashion catalogue photography, "
    "beautiful South Indian female model, full body portrait head to feet, "
    "wearing a {desc} saree in traditional six-yard Nivi drape, "
    # ← always include blouse (fixes missing blouse bug)
    "matching traditional saree blouse with elbow-length sleeves clearly visible "
    "at the torso, midriff visible between blouse hem and saree waistband, "
    "woven decorative border clearly visible along the saree edge, "
)

_COMMON_TAIL = (
    "soft diffused even studio lighting, "
    "seamless light grey background, no props, no text, "
    "sharp focus on the entire garment from blouse to hem, "
    "photorealistic 4k fashion catalogue, high detail"
)


def build_prompt(pose_key: str, garment_desc: str, color_hint: str) -> str:
    """
    Assemble a complete prompt for product_to_model.

    garment_desc : user-supplied free text, e.g. "Kanjivaram red with gold zari border"
    color_hint   : auto-extracted colour word from the garment image
    """
    desc = garment_desc.strip() if garment_desc.strip() else f"{color_hint} silk"
    base = _COMMON_BASE.format(desc=desc)
    return base + _POSE_BLOCKS[pose_key] + _COMMON_TAIL


# ─────────────────────────────────────────────────── dropdown maps

CATEGORY_MAP = {
    "One-piece / Full-body  (saree, lehenga, dress)  ← best for sarees": "one-pieces",
    "Auto-detect":                                                          "auto",
    "Tops  (kurta top, blouse, t-shirt)":                                  "tops",
    "Bottoms  (skirt, pants, dhoti)":                                       "bottoms",
}
MODE_MAP = {
    "Balanced  — fast, good quality  (recommended)": "balanced",
    "Quality   — slower, best accuracy":             "quality",
    "Performance  — fastest preview":                "performance",
}
PHOTO_TYPE_MAP = {
    "Flat-lay  (saree/fabric spread on surface)": "flat-lay",
    "Auto-detect":                                "auto",
    "On model  (garment already worn in photo)":  "model",
}


# ────────────────────────────────────────────────────────────────── helpers

def load_pose(name: str) -> Image.Image | None:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = MODEL_POSES_DIR / f"{name}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def pick_best_garment(files) -> Image.Image:
    """
    Pick the highest-resolution uploaded garment image.
    Higher pixel count = more detail for FASHN to work with.
    """
    if not files:
        raise ValueError("No garment images uploaded.")
    best_px, best_img = -1, None
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        try:
            img = Image.open(path).convert("RGB")
            px  = img.size[0] * img.size[1]
            if px > best_px:
                best_px, best_img = px, img
        except Exception:
            continue
    if best_img is None:
        raise ValueError("Could not read any uploaded garment image.")
    return best_img


def save_result(img: Image.Image, key: str, ts: str) -> str:
    path = OUTPUTS_DIR / f"{ts}_{key}.jpg"
    img.save(str(path), "JPEG", quality=95)
    return str(path)


def make_zip(front, side, back) -> str | None:
    imgs = {k: v for k, v in zip(POSE_KEYS, [front, side, back]) if v is not None}
    if not imgs:
        return None
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"tryon_{ts}.zip"
    with zipfile.ZipFile(str(path), "w") as zf:
        for name, img in imgs.items():
            if isinstance(img, Image.Image):
                pil = img
            else:
                try:
                    pil = Image.fromarray(img).convert("RGB")
                except Exception:
                    continue
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=95)
            zf.writestr(f"{name}.jpg", buf.getvalue())
    return str(path)


# ──────────────────────────────────────────────────────────── generation

def generate(
    garment_files,
    front_img, side_img, back_img,
    category_label, mode_label, photo_type_label,
    garment_desc,
    use_seed, custom_seed,
    api_key,
    progress=gr.Progress(track_tqdm=False),
):
    api_key = (api_key or "").strip()
    if not api_key:
        raise gr.Error("Enter your FASHN API key in the field at the top.")
    if not garment_files:
        raise gr.Error("Upload at least one garment image.")

    poses = {"front": front_img, "side": side_img, "back": back_img}

    category   = CATEGORY_MAP.get(category_label, "one-pieces")
    mode       = MODE_MAP.get(mode_label, "balanced")
    photo_type = PHOTO_TYPE_MAP.get(photo_type_label, "flat-lay")

    garment    = pick_best_garment(garment_files)
    color_hint = extract_dominant_color(garment)

    # Seed: same seed across all 3 poses encourages a consistent-looking model.
    seed: int | None = None
    if use_seed:
        seed = int(custom_seed) if str(custom_seed).strip().isdigit() else random.randint(1, 99999)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"front": None, "side": None, "back": None}

    for idx, key in enumerate(POSE_KEYS):
        label    = POSE_LABELS[idx]
        pose_raw = poses[key]
        progress(idx / 3, desc=f"Generating {label}  ({idx+1}/3)…")

        pose_img = None
        if pose_raw is not None:
            if isinstance(pose_raw, Image.Image):
                pose_img = pose_raw
            else:
                try:
                    pose_img = Image.fromarray(pose_raw).convert("RGB")
                except Exception:
                    gr.Warning(f"{label}: could not read pose photo — using auto-generate.")

        try:
            if pose_img is not None:
                # tryon mode — pose guides body ONLY, face not preserved
                result = tryon(
                    api_key            = api_key,
                    model_image        = pose_img,
                    garment_image      = garment,
                    category           = category,
                    mode               = mode,
                    garment_photo_type = photo_type,
                    seed               = seed,
                )
            else:
                # ← primary recommended path for sarees
                prompt = build_prompt(key, garment_desc, color_hint)
                result = product_to_model(
                    api_key       = api_key,
                    product_image = garment,
                    prompt        = prompt,
                    seed          = seed,
                )
            save_result(result, key, ts)
            results[key] = result
        except Exception as e:
            gr.Warning(f"{label} failed: {e}")

    progress(1.0, desc="Done!")
    return results["front"], results["side"], results["back"]


# ──────────────────────────────────────────────────────────── Gradio UI

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate"),
    title="Saree Try-On",
    css="""
        footer { display: none !important; }
        .hero  { text-align: center; padding: 18px 0 10px; }
        .credit-badge {
            display: inline-block; background: #e8f5e9;
            color: #2e7d32; border-radius: 20px;
            padding: 3px 12px; font-size: 0.85em; font-weight: 600;
        }
        .tip  { background: #f8f4ff; border-radius: 10px;
                padding: 10px 14px; font-size: 0.86em; color: #555; }
        .warn { background: #fff3e0; border-radius: 10px;
                padding: 10px 14px; font-size: 0.85em; color: #bf360c;
                border-left: 4px solid #ff6d00; }
        .info { background: #e3f2fd; border-radius: 10px;
                padding: 10px 14px; font-size: 0.85em; color: #0d47a1; }
    """,
) as demo:

    # ── header
    gr.HTML("""
    <div class="hero">
      <h1 style="margin:0 0 6px">👗  Saree &amp; Garment Try-On</h1>
      <p style="color:#777;margin:0 0 8px">
        Upload garment flat-lay &rarr; AI model wears it in 3 poses &rarr; download
      </p>
      <span class="credit-badge">⚡ 1 credit per image &nbsp;·&nbsp; 3 credits per saree</span>
    </div>
    """)

    # ── API key + dropdowns
    with gr.Row():
        api_key_box = gr.Textbox(
            label="FASHN API Key",
            value=os.environ.get("FASHN_API_KEY", ""),
            placeholder="fas_xxxxxxxxxxxxxxxx",
            type="password",
            info="fashn.ai — 15 free credits on signup",
            scale=3,
        )
        with gr.Column(scale=1):
            category_dd = gr.Dropdown(
                label="Garment category",
                choices=list(CATEGORY_MAP.keys()),
                value=list(CATEGORY_MAP.keys())[0],
            )
            mode_dd = gr.Dropdown(
                label="Quality mode",
                choices=list(MODE_MAP.keys()),
                value=list(MODE_MAP.keys())[0],
            )

    gr.Markdown("---")

    # ── main layout
    with gr.Row():

        # ── left column: garment
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📸 Garment Image")
            garment_upload = gr.File(
                label="Upload 1–3 photos of the saree / garment",
                file_count="multiple",
                file_types=["image"],
            )
            garment_gallery = gr.Gallery(
                label="Preview",
                columns=3, height=140,
                show_label=False,
                object_fit="contain",
            )
            garment_upload.change(
                fn=lambda fs: ([f.name if hasattr(f, "name") else f for f in fs] if fs else []),
                inputs=[garment_upload],
                outputs=[garment_gallery],
            )

            garment_desc_box = gr.Textbox(
                label="Saree description  (recommended)",
                placeholder="e.g.  deep red Kanjivaram silk with wide gold zari border and small butta motifs",
                info="Used in the generation prompt. Colour is also auto-detected from the image. "
                     "The more specific, the better.",
                lines=2,
            )

            photo_type_dd = gr.Dropdown(
                label="Garment photo type",
                choices=list(PHOTO_TYPE_MAP.keys()),
                value=list(PHOTO_TYPE_MAP.keys())[0],
                info="Only matters when pose photos are loaded (tryon mode).",
            )

            gr.HTML("""
            <div class="tip">
              <b>📌 Garment flat-lay tips for best results:</b><br>
              · Plain white or seamless background<br>
              · Spread the FULL saree — show border, body, and pallu<br>
              · Do <b>not</b> include blouse in the same frame<br>
              · Even diffused lighting, no harsh shadows<br>
              · Landscape / wide orientation preferred
            </div>
            """)

            with gr.Accordion("🎲 Consistency seed  (optional)", open=False):
                gr.HTML("""
                <div class="info">
                  Using the <b>same seed</b> for all 3 poses encourages FASHN to pick
                  a similar-looking model across front, side, and back.
                  Try a few different seeds until you find one you like, then lock it.
                </div>
                """)
                use_seed    = gr.Checkbox(label="Use a fixed seed", value=False)
                custom_seed = gr.Number(
                    label="Seed value", value=42, precision=0,
                    info="Any positive integer. Change to try a different model appearance.",
                )

        # ── right column: pose photos (tryon mode)
        with gr.Column(scale=2):
            gr.Markdown("### 🧍 Pose Photos  *(leave blank for auto-generate — recommended for sarees)*")

            gr.HTML("""
            <div class="warn">
              <b>⚠️ Why pose photos often give BAD results for sarees:</b><br>
              1. FASHN's tryon model sees a 6-meter unstitched flat-lay and can't figure out
                 how to drape it — it treats the fabric like a regular stitched garment.<br>
              2. Without Face Reference (+3 credits), the model's face is <b>not preserved</b>
                 anyway — you get a different person in each pose.<br><br>
              <b>Recommendation:</b> keep these boxes empty and use <i>Auto-Generate</i> mode
              (product_to_model). It produces correct Nivi draping and proper blouse,
              costs the same 1 credit, and looks better. See the right outputs for proof!
            </div>
            """)

            with gr.Row():
                pose_front = gr.Image(
                    label=POSE_LABELS[0],
                    value=load_pose("front"),
                    height=280, sources=["upload"], type="pil",
                )
                pose_side = gr.Image(
                    label=POSE_LABELS[1],
                    value=load_pose("side"),
                    height=280, sources=["upload"], type="pil",
                )
                pose_back = gr.Image(
                    label=POSE_LABELS[2],
                    value=load_pose("back"),
                    height=280, sources=["upload"], type="pil",
                )

            loaded = [POSE_LABELS[i] for i, k in enumerate(POSE_KEYS) if load_pose(k) is not None]
            if loaded:
                gr.HTML(
                    "<div class='info'>ℹ️ Pose photos loaded from <code>model_poses/</code>: "
                    f"<b>{', '.join(loaded)}</b>. These will use tryon mode. "
                    "Clear the boxes above to switch to auto-generate.</div>"
                )

    # ── generate
    generate_btn = gr.Button(
        "⚡  Generate — 3 poses  (3 credits total)",
        variant="primary", size="lg",
    )

    gr.Markdown("---")

    # ── results
    gr.Markdown("### ✨ Results")
    with gr.Row():
        out_front = gr.Image(label=POSE_LABELS[0], height=460)
        out_side  = gr.Image(label=POSE_LABELS[1], height=460)
        out_back  = gr.Image(label=POSE_LABELS[2], height=460)

    with gr.Row():
        zip_btn  = gr.Button("📦 Download all 3 as ZIP", scale=0)
        zip_file = gr.File(label="ZIP", scale=1)

    generate_btn.click(
        fn=lambda gf, pf, ps, pb, cat, mod, pt, desc, us, cs, ak: generate(
            gf, pf, ps, pb, cat, mod, pt, desc, us, cs, ak
        ),
        inputs=[
            garment_upload, pose_front, pose_side, pose_back,
            category_dd, mode_dd, photo_type_dd,
            garment_desc_box,
            use_seed, custom_seed,
            api_key_box,
        ],
        outputs=[out_front, out_side, out_back],
    )

    zip_btn.click(
        fn=make_zip,
        inputs=[out_front, out_side, out_back],
        outputs=[zip_file],
    )

    gr.HTML("""
    <div style="text-align:center;color:#bbb;font-size:0.8em;padding:14px 0 2px">
      All images saved to <code>outputs/</code> automatically &nbsp;·&nbsp;
      1 credit per image guaranteed — no face reference used
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )