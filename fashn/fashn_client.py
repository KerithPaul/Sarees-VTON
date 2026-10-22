"""
FASHN API client — Try-On + Product-to-Model, no face reference.

tryon-v1.6:       model photo + garment → dressed model  (1 credit/image)
product-to-model: garment + prompt     → AI model + garment (1 credit/image)

face_reference is intentionally excluded — it costs +3 credits per image.

POST https://api.fashn.ai/v1/run         — submit
GET  https://api.fashn.ai/v1/status/{id} — poll
"""

import base64
import io
import time
from pathlib import Path
from typing import Union

import requests
from PIL import Image

API_BASE      = "https://api.fashn.ai/v1"
POLL_INTERVAL = 3    # seconds between polls
MAX_WAIT      = 300  # 5 min — quality mode can be slow


def _encode(source: Union[str, Path, Image.Image, bytes]) -> str:
    """Convert image to base64 data URI."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        raw  = path.read_bytes()
    elif isinstance(source, Image.Image):
        buf = io.BytesIO()
        source.convert("RGB").save(buf, format="JPEG", quality=95)
        raw, mime = buf.getvalue(), "image/jpeg"
    elif isinstance(source, bytes):
        raw, mime = source, "image/jpeg"
    else:
        raise TypeError(f"Unsupported type: {type(source)}")
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _resize(img: Image.Image, max_px: int = 1200) -> Image.Image:
    """Limit longest edge so we don't send huge payloads."""
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _submit(api_key: str, body: dict) -> str:
    """POST to /v1/run, return job ID."""
    r = requests.post(
        f"{API_BASE}/run",
        json=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type":  "application/json"},
        timeout=40,
    )
    if not r.ok:
        raise RuntimeError(f"FASHN error {r.status_code}: {r.text}")
    job_id = r.json().get("id")
    if not job_id:
        raise RuntimeError(f"No job ID returned: {r.json()}")
    return job_id


def _poll(api_key: str, job_id: str) -> str:
    """Poll until completed. Returns output image URL."""
    headers = {"Authorization": f"Bearer {api_key}"}
    waited  = 0
    while waited < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        r = requests.get(f"{API_BASE}/status/{job_id}",
                         headers=headers, timeout=20)
        r.raise_for_status()
        d      = r.json()
        status = d.get("status", "")
        if status == "completed":
            out = d.get("output", [])
            if out:
                return out[0]
            raise RuntimeError("Job completed but output is empty")
        if status in ("failed", "error"):
            raise RuntimeError(f"FASHN job failed: {d.get('error', 'unknown')}")
    raise TimeoutError(f"Job {job_id} timed out after {MAX_WAIT}s")


def _download(url: str) -> Image.Image:
    """Download result image from URL."""
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ------------------------------------------------------------------ public

def tryon(
    api_key:            str,
    model_image:        Union[str, Path, Image.Image, bytes],
    garment_image:      Union[str, Path, Image.Image, bytes],
    category:           str = "one-pieces",
    mode:               str = "balanced",
    garment_photo_type: str = "flat-lay",
    seed:               int | None = None,
) -> Image.Image:
    """
    Virtual Try-On using tryon-v1.6.
    Cost: 1 credit per image. No face_reference — no extra charge.

    ⚠ For sarees: tryon-v1.6 struggles with flat-lay fabric because a 6-meter
    unstitched saree looks nothing like a stitched garment. Use product_to_model
    (auto-generate) instead for better draping accuracy.

    Without face_reference the model's face is NOT preserved — pose photo
    guides body position only.

    seed : Same seed across all 3 poses may improve visual consistency.
    """
    if isinstance(model_image, Image.Image):
        model_image = _resize(model_image)
    if isinstance(garment_image, Image.Image):
        garment_image = _resize(garment_image)

    inputs = {
        "model_image":        _encode(model_image),
        "garment_image":      _encode(garment_image),
        "category":           category,
        "mode":               mode,
        "garment_photo_type": garment_photo_type,
    }
    if seed is not None:
        inputs["seed"] = seed

    body = {"model_name": "tryon-v1.6", "inputs": inputs}
    job_id = _submit(api_key, body)
    url    = _poll(api_key, job_id)
    return _download(url)


def product_to_model(
    api_key:       str,
    product_image: Union[str, Path, Image.Image, bytes],
    prompt:        str,
    seed:          int | None = None,
) -> Image.Image:
    """
    Product to Model — generates an AI Indian female model wearing the garment.
    Recommended primary mode for sarees. Cost: 1 credit per image.

    Better than tryon for sarees because:
    - Understands full Nivi drape from a flat-lay image
    - Generates proper pleats, pallu flow, and blouse
    - No confusion from a pre-existing garment on the pose photo

    seed : Use the SAME integer (e.g. 42) for front/side/back to get a
           consistent-looking model across all three poses.
    """
    if isinstance(product_image, Image.Image):
        product_image = _resize(product_image)

    inputs: dict = {
        "product_image": _encode(product_image),
        "prompt":        prompt,
    }
    if seed is not None:
        inputs["seed"] = seed

    body = {"model_name": "product-to-model", "inputs": inputs}
    job_id = _submit(api_key, body)
    url    = _poll(api_key, job_id)
    return _download(url)