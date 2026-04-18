"""
main.py — FastAPI serving layer for LightUNet polyp segmentation
================================================================
Endpoints:
  GET  /health   → liveness + model readiness check
  POST /predict  → upload an image, receive a PNG mask + JSON metrics

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Test with curl:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
         -F "file=@colonoscopy.jpg" \
         --output mask.png
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from model import LightUNet

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH    = Path("model.pt")
IMG_SIZE      = 256                          # must match training resolution
MAX_UPLOAD_MB = 10
MAX_BYTES     = MAX_UPLOAD_MB * 1_024 * 1_024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation — same values used during training
_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])

# ── Shared state loaded once at startup ───────────────────────────────────────

class _ModelState:
    model:  LightUNet | None = None
    ready:  bool             = False
    error:  str | None       = None
    loaded_at: float | None  = None

_state = _ModelState()


def _load_model() -> None:
    """Instantiate LightUNet and load weights from disk."""
    if not MODEL_PATH.exists():
        _state.error = f"Weights file not found: {MODEL_PATH.resolve()}"
        log.warning(_state.error)
        return

    try:
        net = LightUNet(in_channels=3, out_channels=1)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(state_dict)
        net.to(DEVICE).eval()

        # Warm-up forward pass — catches shape or weight issues immediately
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        with torch.no_grad():
            out = net(dummy)
        assert out.shape == (1, 1, IMG_SIZE, IMG_SIZE), f"Unexpected output shape: {out.shape}"

        _state.model     = net
        _state.ready     = True
        _state.loaded_at = time.time()
        log.info("Model loaded successfully from %s (device=%s)", MODEL_PATH, DEVICE)

    except Exception as exc:
        _state.error = str(exc)
        log.exception("Failed to load model: %s", exc)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading model…")
    _load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title       = "Polyp Segmentation API",
    description = "Upload a colonoscopy frame → receive a binary segmentation mask.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Helper: preprocess ────────────────────────────────────────────────────────

def _preprocess(image: Image.Image) -> torch.Tensor:
    """
    PIL Image → normalised float tensor (1, 3, H, W) on DEVICE.
    Handles any input size; albumentations resizes to IMG_SIZE.
    """
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)   # (H, W, 3)
    transformed = _TRANSFORM(image=rgb)
    tensor = transformed["image"].unsqueeze(0)             # (1, 3, H, W)
    return tensor.to(DEVICE)


def _postprocess(logits: torch.Tensor, orig_w: int, orig_h: int) -> np.ndarray:
    """
    Raw logit tensor (1, 1, H, W) → uint8 mask (orig_h, orig_w).

    The mask pixel values are:
        0   = background
        255 = polyp
    """
    prob = torch.sigmoid(logits)                               # (1, 1, H, W)  0–1
    # Resize back to original image dimensions
    prob = F.interpolate(
        prob,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )
    binary = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return binary  # (orig_h, orig_w)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Liveness + model readiness check",
    tags=["ops"],
)
def health():
    """
    Returns 200 when the server is alive and the model is loaded.
    Returns 503 when the model failed to load or weights file is missing.

    Use this endpoint for:
      - Docker HEALTHCHECK
      - Kubernetes readiness probes
      - Load-balancer health gates
    """
    uptime = round(time.time() - _state.loaded_at, 1) if _state.loaded_at else None

    if _state.ready:
        return {
            "status":    "ok",
            "model":     "LightUNet",
            "device":    str(DEVICE),
            "img_size":  IMG_SIZE,
            "uptime_s":  uptime,
        }

    # Model not ready — return 503 so orchestrators stop routing traffic here
    raise HTTPException(
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
        detail      = {
            "status": "model_not_ready",
            "reason": _state.error or "Model not yet loaded",
        },
    )


@app.post(
    "/predict",
    summary="Segment polyps in a colonoscopy image",
    tags=["inference"],
    responses={
        200: {
            "content":     {"image/png": {}},
            "description": "Binary segmentation mask as PNG (0=background, 255=polyp) "
                           "with JSON metrics in response headers.",
        },
        400: {"description": "Invalid file — wrong type, corrupt, or too large."},
        503: {"description": "Model not loaded."},
    },
)
async def predict(
    file: Annotated[UploadFile, File(description="Colonoscopy image (.jpg or .png, max 10 MB)")],
):
    """
    Upload a colonoscopy frame and receive:
    - **Body**: PNG mask (same dimensions as input; 255 = polyp, 0 = background)
    - **X-Polyp-Coverage**: fraction of pixels predicted as polyp (0.0 – 1.0)
    - **X-Inference-Ms**: server-side inference time in milliseconds
    - **X-Input-Size**: original WxH of the uploaded image

    Example:
        curl -X POST http://localhost:8000/predict \\
             -F "file=@colon.jpg" --output mask.png
    """
    # ── 1. Guard: model readiness ─────────────────────────────────────────────
    if not _state.ready:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = _state.error or "Model is not loaded",
        )

    # ── 2. Validate content type ──────────────────────────────────────────────
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"Unsupported file type '{content_type}'. "
                          f"Allowed: {sorted(ALLOWED_TYPES)}",
        )

    # ── 3. Read & validate size ───────────────────────────────────────────────
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"File too large ({len(raw) / 1_024 / 1_024:.1f} MB). "
                          f"Maximum allowed: {MAX_UPLOAD_MB} MB",
        )

    # ── 4. Decode image ───────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()                  # detect truncated / corrupt files
        image = Image.open(io.BytesIO(raw))   # re-open after verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Cannot decode image. File may be corrupt or not a valid image.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"Image decoding error: {exc}",
        )

    orig_w, orig_h = image.size   # PIL uses (width, height)

    # ── 5. Preprocess → inference ─────────────────────────────────────────────
    tensor = _preprocess(image)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _state.model(tensor)           # (1, 1, IMG_SIZE, IMG_SIZE)
    inference_ms = round((time.perf_counter() - t0) * 1_000, 1)

    log.info(
        "predict | file=%s size=%dx%d inference=%.1fms",
        file.filename, orig_w, orig_h, inference_ms,
    )

    # ── 6. Postprocess ────────────────────────────────────────────────────────
    mask_arr  = _postprocess(logits, orig_w, orig_h)     # uint8 (H, W)
    coverage  = round(float((mask_arr > 0).mean()), 4)   # fraction 0–1

    # ── 7. Encode mask as PNG ─────────────────────────────────────────────────
    mask_img = Image.fromarray(mask_arr, mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    # ── 8. Return mask with metadata headers ──────────────────────────────────
    return Response(
        content      = png_bytes,
        media_type   = "image/png",
        headers      = {
            "X-Polyp-Coverage":  str(coverage),
            "X-Inference-Ms":    str(inference_ms),
            "X-Input-Size":      f"{orig_w}x{orig_h}",
            "X-Model":           "LightUNet",
        },
    )
