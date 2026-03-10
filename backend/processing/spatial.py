"""
Spatial Domain Processing — Gonzalez & Woods, Ch. 3
Intensity Transformations & Spatial Filtering

Endpoints:
    POST /api/spatial/histogram-eq      Histogram equalization (Ch. 3.3)
    POST /api/spatial/clahe             CLAHE (Ch. 3.3)
    POST /api/spatial/contrast-stretch  Linear contrast stretch (Ch. 3.2)
    POST /api/spatial/gamma             Gamma (power-law) correction (Ch. 3.2)
    POST /api/spatial/log-transform     Log transformation (Ch. 3.2)
    POST /api/spatial/filter            Spatial filtering — mean/gaussian/median/laplacian/sobel (Ch. 3.5–3.6)
    POST /api/spatial/unsharp-mask      Unsharp masking (Ch. 3.6)
"""

import base64
import logging
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PilImage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spatial", tags=["Spatial Processing"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _read_image(file: UploadFile) -> np.ndarray:
    """
    Read uploaded file to numpy array in BGR format (OpenCV convention).
    Supports JPEG, PNG, BMP, WebP (via cv2.imdecode) and
    TIFF, multi-page, 16-bit (via Pillow fallback).
    """
    contents = await file.read()

    # Fast path — OpenCV handles JPEG / PNG / BMP / WebP
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # Fallback — Pillow handles TIFF and other formats cv2 misses
    try:
        pil_img = PilImage.open(BytesIO(contents)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        pass

    raise ValueError(
        "Could not decode image. "
        "Supported formats: JPEG, PNG, BMP, WebP, TIFF."
    )


def _make_response(img: np.ndarray, metadata: dict) -> JSONResponse:
    """Convert processed numpy array to base64 JSON response."""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse({
        "image": img_base64,
        "metadata": metadata,
    })


def _is_grayscale(img: np.ndarray) -> bool:
    """Check if an image is grayscale (single channel or all channels equal)."""
    if len(img.shape) == 2:
        return True
    if img.shape[2] == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# 1. POST /histogram-eq — Histogram Equalization
# ---------------------------------------------------------------------------

def _apply_histogram_eq(img: np.ndarray) -> np.ndarray:
    """
    Apply standard histogram equalization.
    For color images, convert to YCrCb and equalize the Y (luminance) channel
    to avoid color distortion.
    Reference: Gonzalez & Woods, Ch. 3.3
    """
    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result = cv2.equalizeHist(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Color image: equalize luminance channel only
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


@router.post("/histogram-eq")
async def histogram_equalization(file: UploadFile = File(...)):
    """
    Standard histogram equalization for contrast enhancement.
    Reference: Gonzalez & Woods, Ch. 3.3
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_histogram_eq(img)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "histogram_equalization",
            "chapter": "Ch. 3.3",
            "params": {},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("histogram_equalization failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /clahe — Contrast Limited Adaptive Histogram Equalization
# ---------------------------------------------------------------------------

def _apply_clahe(img: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel.
    Reference: Gonzalez & Woods, Ch. 3.3
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result = clahe.apply(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


@router.post("/clahe")
async def clahe(
    file: UploadFile = File(...),
    clip_limit: float = Form(2.0),
    tile_size: int = Form(8),
):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Params: clip_limit (float, default 2.0), tile_size (int, default 8).
    Reference: Gonzalez & Woods, Ch. 3.3
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_clahe(img, clip_limit, tile_size)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "clahe",
            "chapter": "Ch. 3.3",
            "params": {"clip_limit": clip_limit, "tile_size": tile_size},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("clahe failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /contrast-stretch — Linear Contrast Stretching
# ---------------------------------------------------------------------------

def _apply_contrast_stretch(
    img: np.ndarray, low_percentile: float, high_percentile: float
) -> np.ndarray:
    """
    Linear contrast stretch using percentile-based min/max.
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    # Work on each channel independently to preserve color balance
    result = np.zeros_like(img)
    channels = 1 if _is_grayscale(img) else img.shape[2]

    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        p_low = np.percentile(gray, low_percentile)
        p_high = np.percentile(gray, high_percentile)
        if p_high - p_low == 0:
            stretched = gray.copy()
        else:
            stretched = (gray.astype(np.float64) - p_low) / (p_high - p_low) * 255.0
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

    for c in range(channels):
        channel = img[:, :, c]
        p_low = np.percentile(channel, low_percentile)
        p_high = np.percentile(channel, high_percentile)
        if p_high - p_low == 0:
            result[:, :, c] = channel
        else:
            stretched = (channel.astype(np.float64) - p_low) / (p_high - p_low) * 255.0
            result[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
    return result


@router.post("/contrast-stretch")
async def contrast_stretch(
    file: UploadFile = File(...),
    low_percentile: float = Form(2.0),
    high_percentile: float = Form(98.0),
):
    """
    Linear contrast stretch using percentile-based intensity limits.
    Params: low_percentile (float, default 2.0), high_percentile (float, default 98.0).
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_contrast_stretch(img, low_percentile, high_percentile)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "contrast_stretch",
            "chapter": "Ch. 3.2",
            "params": {
                "low_percentile": low_percentile,
                "high_percentile": high_percentile,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("contrast_stretch failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /gamma — Gamma (Power-Law) Correction
# ---------------------------------------------------------------------------

def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Power-law (gamma) transformation:  s = c * r^γ
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    # Normalize to [0, 1], apply gamma, scale back to [0, 255]
    normalized = img.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


@router.post("/gamma")
async def gamma_correction(
    file: UploadFile = File(...),
    gamma: float = Form(1.0),
):
    """
    Power-law (gamma) intensity transformation.
    Param: gamma (float, default 1.0). Values < 1 brighten, > 1 darken.
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    try:
        if gamma <= 0:
            raise ValueError("Gamma must be a positive number.")
        start = time.time()
        img = await _read_image(file)
        result = _apply_gamma(img, gamma)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "gamma_correction",
            "chapter": "Ch. 3.2",
            "params": {"gamma": gamma},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("gamma_correction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. POST /log-transform — Logarithmic Transformation
# ---------------------------------------------------------------------------

def _apply_log_transform(img: np.ndarray) -> np.ndarray:
    """
    Log transformation:  s = c * log(1 + r)
    Expands dark pixel values, compresses bright ones.
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    img_float = img.astype(np.float64)
    # c chosen so that max output = 255
    c = 255.0 / np.log(1.0 + img_float.max()) if img_float.max() > 0 else 1.0
    result = c * np.log(1.0 + img_float)
    return np.clip(result, 0, 255).astype(np.uint8)


@router.post("/log-transform")
async def log_transform(file: UploadFile = File(...)):
    """
    Logarithmic intensity transformation: s = c * log(1 + r).
    Reference: Gonzalez & Woods, Ch. 3.2
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_log_transform(img)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "log_transform",
            "chapter": "Ch. 3.2",
            "params": {},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("log_transform failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 6. POST /filter — Spatial Filtering (multiple types)
# ---------------------------------------------------------------------------

_VALID_FILTER_TYPES = {"mean", "gaussian", "median", "laplacian", "sobel"}


def _apply_spatial_filter(
    img: np.ndarray, filter_type: str, kernel_size: int
) -> np.ndarray:
    """
    Apply a spatial filter.
    Supported: mean, gaussian, median, laplacian, sobel.
    Reference: Gonzalez & Woods, Ch. 3.5 (smoothing), Ch. 3.6 (sharpening)
    """
    # Kernel size must be odd and positive
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")
    if kernel_size % 2 == 0:
        kernel_size += 1  # Force odd

    if filter_type == "mean":
        return cv2.blur(img, (kernel_size, kernel_size))

    if filter_type == "gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    if filter_type == "median":
        return cv2.medianBlur(img, kernel_size)

    # --- Edge / sharpening filters operate on grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if not _is_grayscale(img) else (
        img if len(img.shape) == 2 else img[:, :, 0]
    )

    if filter_type == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        result = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if filter_type == "sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        result = np.clip(magnitude, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unknown filter type: {filter_type}")


@router.post("/filter")
async def spatial_filter(
    file: UploadFile = File(...),
    filter_type: str = Form("mean"),
    kernel_size: int = Form(3),
):
    """
    Apply a spatial filter. Supported types: mean, gaussian, median, laplacian, sobel.
    Params: filter_type (str), kernel_size (int, default 3, must be odd).
    Reference: Gonzalez & Woods, Ch. 3.5 (smoothing), Ch. 3.6 (sharpening)
    """
    try:
        if filter_type not in _VALID_FILTER_TYPES:
            raise ValueError(
                f"Invalid filter_type '{filter_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_FILTER_TYPES))}"
            )
        start = time.time()
        img = await _read_image(file)
        result = _apply_spatial_filter(img, filter_type, kernel_size)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "spatial_filter",
            "chapter": "Ch. 3.5 / 3.6",
            "params": {"filter_type": filter_type, "kernel_size": kernel_size},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("spatial_filter failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 7. POST /unsharp-mask — Unsharp Masking
# ---------------------------------------------------------------------------

def _apply_unsharp_mask(
    img: np.ndarray, radius: float, amount: float
) -> np.ndarray:
    """
    Unsharp masking: sharpened = original + amount * (original – blurred).
    Reference: Gonzalez & Woods, Ch. 3.6
    """
    # Radius → kernel size (must be odd)
    ksize = int(round(radius)) * 2 + 1
    if ksize < 3:
        ksize = 3

    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    # Apply unsharp formula in float to avoid overflow
    sharpened = img.astype(np.float64) + amount * (img.astype(np.float64) - blurred.astype(np.float64))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


@router.post("/unsharp-mask")
async def unsharp_mask(
    file: UploadFile = File(...),
    radius: float = Form(2.0),
    amount: float = Form(1.5),
):
    """
    Unsharp masking for image sharpening.
    Params: radius (float, default 2.0), amount (float, default 1.5).
    Reference: Gonzalez & Woods, Ch. 3.6
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_unsharp_mask(img, radius, amount)
        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "unsharp_mask",
            "chapter": "Ch. 3.6",
            "params": {"radius": radius, "amount": amount},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("unsharp_mask failed")
        raise HTTPException(status_code=500, detail=str(e))
