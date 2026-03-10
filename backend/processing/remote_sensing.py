"""
Remote Sensing Processing
NDVI computation, band compositing, satellite image enhancement,
and Hyperspectral Imaging (HSI) utilities.

Endpoints:
    POST /api/remote/ndvi              NDVI from NIR + Red bands
    POST /api/remote/band-composite    False-color composite from multi-band TIFF
    POST /api/remote/band-stats        Per-band statistics
    POST /api/remote/band-viewer       Extract and display a single band
    POST /api/remote/enhance-satellite CLAHE enhancement for satellite imagery
    POST /api/remote/spectral-profile  Spectral profile at a pixel location
"""

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import tifffile
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PilImage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/remote", tags=["Remote Sensing"])


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


async def _read_multiband(file: UploadFile) -> np.ndarray:
    """
    Read a multi-band image (e.g. GeoTIFF) preserving all bands.
    Returns array with shape (H, W, bands) or (H, W) for single-band.
    Uses tifffile for multi-band TIFF, falls back to Pillow / OpenCV.
    """
    contents = await file.read()
    buf = BytesIO(contents)

    # Try tifffile first (best for multi-band / GeoTIFF)
    try:
        img = tifffile.imread(buf)
        # tifffile may return (bands, H, W) for multi-band — transpose to (H, W, bands)
        if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        return img
    except Exception:
        pass

    # Fallback: OpenCV (reads up to 3 channels)
    buf.seek(0)
    nparr = np.frombuffer(buf.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    # Fallback: Pillow
    buf.seek(0)
    try:
        pil_img = PilImage.open(buf)
        return np.array(pil_img)
    except Exception:
        pass

    raise ValueError(
        "Could not decode multi-band image. "
        "Supported: GeoTIFF, TIFF, PNG, JPEG."
    )


def _make_response(img: np.ndarray, metadata: dict) -> JSONResponse:
    """Convert processed numpy array to base64 JSON response."""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse({
        "image": img_base64,
        "metadata": metadata,
    })


def _normalize_band_to_uint8(band: np.ndarray) -> np.ndarray:
    """Normalize any-dtype single band to 0–255 uint8."""
    band_f = band.astype(np.float64)
    bmin, bmax = band_f.min(), band_f.max()
    if bmax - bmin == 0:
        return np.zeros_like(band, dtype=np.uint8)
    normalized = (band_f - bmin) / (bmax - bmin) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _get_num_bands(img: np.ndarray) -> int:
    """Return the number of bands in the image."""
    if img.ndim == 2:
        return 1
    return img.shape[2]


def _get_band(img: np.ndarray, index: int) -> np.ndarray:
    """Extract a single band (0-indexed)."""
    num_bands = _get_num_bands(img)
    if index < 0 or index >= num_bands:
        raise ValueError(
            f"band_index {index} out of range. "
            f"Image has {num_bands} band(s) (0-indexed)."
        )
    if img.ndim == 2:
        return img
    return img[:, :, index]


# ---------------------------------------------------------------------------
# 1. POST /ndvi — NDVI Computation
# ---------------------------------------------------------------------------

@router.post("/ndvi")
async def compute_ndvi(
    file: UploadFile = File(...),
    nir_band: int = Form(0),
    red_band: int = Form(1),
):
    """
    Compute NDVI (Normalized Difference Vegetation Index).
    Formula: NDVI = (NIR − Red) / (NIR + Red)

    Input: Multi-band image (e.g. 2-band TIFF with NIR and Red).
    Params:
        nir_band — Band index for NIR (default 0).
        red_band — Band index for Red (default 1).

    Returns:
        Colorized NDVI map (RdYlGn colormap) + statistics
        (min, max, mean, std of NDVI values).
    """
    try:
        start = time.time()
        img = await _read_multiband(file)

        nir = _get_band(img, nir_band).astype(np.float64)
        red = _get_band(img, red_band).astype(np.float64)

        # NDVI = (NIR - Red) / (NIR + Red), handle division by zero
        denominator = nir + red
        ndvi = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0.0,
        )
        # NDVI range is [-1, 1]
        ndvi = np.clip(ndvi, -1.0, 1.0)

        # Statistics
        ndvi_stats = {
            "min": round(float(ndvi.min()), 4),
            "max": round(float(ndvi.max()), 4),
            "mean": round(float(ndvi.mean()), 4),
            "std": round(float(ndvi.std()), 4),
        }

        # Normalize to 0–255 for colormap application
        ndvi_norm = ((ndvi + 1.0) / 2.0 * 255.0).astype(np.uint8)

        # Apply RdYlGn-like colormap
        # OpenCV doesn't have RdYlGn directly — build a custom LUT
        ndvi_colored = _apply_rdylgn_colormap(ndvi_norm)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(ndvi_colored, {
            "operation": "ndvi",
            "params": {"nir_band": nir_band, "red_band": red_band},
            "ndvi_statistics": ndvi_stats,
            "num_bands": _get_num_bands(img),
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("compute_ndvi failed")
        raise HTTPException(status_code=500, detail=str(e))


def _apply_rdylgn_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Apply an RdYlGn-style colormap via lookup table.
    0 = dark red (low NDVI), 128 = yellow, 255 = dark green (high NDVI).
    """
    # Build 256-entry BGR lookup table
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            # Red → Yellow
            s = t / 0.5
            r = int(215 - s * 45)       # 215 → 170
            g = int(48 + s * 197)        # 48 → 245
            b = int(39 + s * 0)          # 39 → 39
        else:
            # Yellow → Green
            s = (t - 0.5) / 0.5
            r = int(170 - s * 170)       # 170 → 0
            g = int(245 - s * 95)        # 245 → 150
            b = int(39 + s * 50)         # 39 → 89

        # BGR order for OpenCV
        lut[i] = [b, g, r]

    # Ensure input is 2D uint8
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    if gray_u8.ndim != 2:
        gray_u8 = gray_u8.reshape(gray_u8.shape[0], gray_u8.shape[1])

    # Direct index lookup — maps each pixel value to its BGR color
    colored = lut[gray_u8]  # shape: (H, W, 3)
    return colored


# ---------------------------------------------------------------------------
# 2. POST /band-composite — False Color Composite
# ---------------------------------------------------------------------------

@router.post("/band-composite")
async def band_composite(
    file: UploadFile = File(...),
    red_band: int = Form(0),
    green_band: int = Form(1),
    blue_band: int = Form(2),
):
    """
    Create a false-color (or true-color) composite from a multi-band image.
    Maps selected bands to the Red, Green, Blue display channels.

    Params:
        red_band   — Band index for the Red display channel (default 0).
        green_band — Band index for the Green display channel (default 1).
        blue_band  — Band index for the Blue display channel (default 2).

    Common presets:
        True color:   red_band=2, green_band=1, blue_band=0 (for Landsat)
        CIR:          red_band=3, green_band=2, blue_band=1
        SWIR:         red_band=5, green_band=4, blue_band=3
    """
    try:
        start = time.time()
        img = await _read_multiband(file)
        num_bands = _get_num_bands(img)

        r = _normalize_band_to_uint8(_get_band(img, red_band))
        g = _normalize_band_to_uint8(_get_band(img, green_band))
        b = _normalize_band_to_uint8(_get_band(img, blue_band))

        # Stack as BGR (OpenCV convention) for encoding
        composite = cv2.merge([b, g, r])

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(composite, {
            "operation": "band_composite",
            "params": {
                "red_band": red_band,
                "green_band": green_band,
                "blue_band": blue_band,
            },
            "num_bands": num_bands,
            "image_shape": list(img.shape),
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("band_composite failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /band-stats — Per-Band Statistics
# ---------------------------------------------------------------------------

@router.post("/band-stats")
async def band_stats(file: UploadFile = File(...)):
    """
    Compute per-band statistics for a multi-band image.

    Returns for each band:
        mean, std, min, max, and a 256-bin histogram.
    """
    try:
        start = time.time()
        img = await _read_multiband(file)
        num_bands = _get_num_bands(img)

        stats = []
        for i in range(num_bands):
            band = _get_band(img, i).astype(np.float64)

            # Compute histogram (normalise band to 0–255 range for histogram)
            band_u8 = _normalize_band_to_uint8(band)
            hist, _ = np.histogram(band_u8, bins=256, range=(0, 256))

            stats.append({
                "band": i,
                "mean": round(float(band.mean()), 4),
                "std": round(float(band.std()), 4),
                "min": round(float(band.min()), 4),
                "max": round(float(band.max()), 4),
                "histogram": hist.tolist(),
            })

        # Create a visual: side-by-side normalised bands (up to 6 bands)
        display_bands = min(num_bands, 6)
        band_images = []
        for i in range(display_bands):
            band_u8 = _normalize_band_to_uint8(_get_band(img, i))
            band_bgr = cv2.cvtColor(band_u8, cv2.COLOR_GRAY2BGR)
            # Add band label
            cv2.putText(
                band_bgr, f"B{i}", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            band_images.append(band_bgr)

        if band_images:
            display = np.hstack(band_images)
        else:
            display = np.zeros((100, 200, 3), dtype=np.uint8)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(display, {
            "operation": "band_stats",
            "num_bands": num_bands,
            "image_shape": list(img.shape),
            "band_statistics": stats,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("band_stats failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /band-viewer — Display a Single Band
# ---------------------------------------------------------------------------

@router.post("/band-viewer")
async def band_viewer(
    file: UploadFile = File(...),
    band_index: int = Form(0),
):
    """
    Extract and display a single band from a multi-band image.
    The band is normalized to 0–255 for display.

    Params:
        band_index — Band to display (0-indexed, default 0).
    """
    try:
        start = time.time()
        img = await _read_multiband(file)
        num_bands = _get_num_bands(img)

        band = _get_band(img, band_index)
        band_u8 = _normalize_band_to_uint8(band)
        result = cv2.cvtColor(band_u8, cv2.COLOR_GRAY2BGR)

        # Compute basic stats for this band
        band_f = band.astype(np.float64)
        band_meta = {
            "mean": round(float(band_f.mean()), 4),
            "std": round(float(band_f.std()), 4),
            "min": round(float(band_f.min()), 4),
            "max": round(float(band_f.max()), 4),
            "dtype": str(band.dtype),
        }

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "band_viewer",
            "params": {"band_index": band_index},
            "num_bands": num_bands,
            "band_info": band_meta,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("band_viewer failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. POST /enhance-satellite — CLAHE for Satellite Imagery
# ---------------------------------------------------------------------------

@router.post("/enhance-satellite")
async def enhance_satellite(
    file: UploadFile = File(...),
    clip_limit: float = Form(3.0),
    tile_size: int = Form(16),
):
    """
    Apply CLAHE enhancement optimized for satellite / aerial imagery.
    Uses larger tile sizes and higher clip limits than typical CLAHE
    to handle large, slowly-varying illumination gradients.

    Params:
        clip_limit — CLAHE clip limit (default 3.0, good for satellite).
        tile_size  — Tile grid size (default 16, larger = less local contrast).

    Works on multi-band images by enhancing each band independently
    after normalizing to uint8.
    """
    try:
        start = time.time()
        img = await _read_multiband(file)
        num_bands = _get_num_bands(img)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size),
        )

        if num_bands == 1:
            band_u8 = _normalize_band_to_uint8(
                _get_band(img, 0)
            )
            enhanced = clahe.apply(band_u8)
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif num_bands == 3:
            # Treat as BGR: convert to YCrCb, enhance Y
            bgr = np.zeros(
                (img.shape[0], img.shape[1], 3), dtype=np.uint8
            )
            for c in range(3):
                bgr[:, :, c] = _normalize_band_to_uint8(_get_band(img, c))
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            # Multi-band: enhance each separately, display first 3 as composite
            enhanced_bands = []
            for i in range(num_bands):
                band_u8 = _normalize_band_to_uint8(_get_band(img, i))
                enhanced_bands.append(clahe.apply(band_u8))

            # Display composite of first 3 bands (or repeat if fewer)
            r = enhanced_bands[0] if len(enhanced_bands) > 0 else np.zeros_like(enhanced_bands[0])
            g = enhanced_bands[1] if len(enhanced_bands) > 1 else r
            b = enhanced_bands[2] if len(enhanced_bands) > 2 else g
            result = cv2.merge([b, g, r])

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "enhance_satellite",
            "params": {"clip_limit": clip_limit, "tile_size": tile_size},
            "num_bands": num_bands,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("enhance_satellite failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 6. POST /spectral-profile — Spectral Profile at a Pixel
# ---------------------------------------------------------------------------

@router.post("/spectral-profile")
async def spectral_profile(
    file: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
):
    """
    Extract the spectral profile (values across all bands) at pixel (x, y).
    Returns array data suitable for plotting a spectral signature chart.

    Params:
        x — Column (pixel x-coordinate, 0-indexed).
        y — Row (pixel y-coordinate, 0-indexed).

    Response includes:
        profile — list of band values at (x, y).
        chart data suitable for frontend plotting.
    """
    try:
        start = time.time()
        img = await _read_multiband(file)
        rows, cols = img.shape[0], img.shape[1]
        num_bands = _get_num_bands(img)

        # Clamp coordinates
        x_clamped = max(0, min(x, cols - 1))
        y_clamped = max(0, min(y, rows - 1))

        # Extract values across all bands
        if img.ndim == 2:
            profile = [float(img[y_clamped, x_clamped])]
        else:
            profile = [float(img[y_clamped, x_clamped, b]) for b in range(num_bands)]

        # Build chart data
        chart_data = {
            "labels": [f"Band {i}" for i in range(num_bands)],
            "values": profile,
        }

        # Create a simple spectral profile visualization image
        profile_img = _draw_spectral_chart(profile, num_bands)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(profile_img, {
            "operation": "spectral_profile",
            "params": {"x": x_clamped, "y": y_clamped},
            "num_bands": num_bands,
            "profile": profile,
            "chart_data": chart_data,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("spectral_profile failed")
        raise HTTPException(status_code=500, detail=str(e))


def _draw_spectral_chart(
    profile: list[float], num_bands: int,
    width: int = 640, height: int = 400,
) -> np.ndarray:
    """
    Render a simple spectral profile bar/line chart as an image.
    Returns a BGR numpy array.
    """
    chart = np.ones((height, width, 3), dtype=np.uint8) * 30  # dark background

    if num_bands == 0 or not profile:
        return chart

    margin_left = 60
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Draw axes
    cv2.line(
        chart,
        (margin_left, margin_top),
        (margin_left, height - margin_bottom),
        (200, 200, 200), 1,
    )
    cv2.line(
        chart,
        (margin_left, height - margin_bottom),
        (width - margin_right, height - margin_bottom),
        (200, 200, 200), 1,
    )

    # Title
    cv2.putText(
        chart, "Spectral Profile", (margin_left, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
    )

    # Normalize values for plotting
    max_val = max(profile) if max(profile) > 0 else 1.0
    min_val = min(profile)
    val_range = max_val - min_val if max_val != min_val else 1.0

    # Plot bars and connecting line
    points = []
    bar_width = max(plot_w // (num_bands * 2), 4)
    for i, val in enumerate(profile):
        x_pos = margin_left + int((i + 0.5) / num_bands * plot_w)
        normalized = (val - min_val) / val_range
        bar_h = int(normalized * plot_h)
        y_top = height - margin_bottom - bar_h
        y_bot = height - margin_bottom

        # Draw bar
        cv2.rectangle(
            chart,
            (x_pos - bar_width // 2, y_top),
            (x_pos + bar_width // 2, y_bot),
            (50, 200, 50), -1,
        )
        points.append((x_pos, y_top))

        # Band label
        cv2.putText(
            chart, f"B{i}", (x_pos - 8, height - margin_bottom + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
        )

        # Value label
        cv2.putText(
            chart, f"{val:.0f}", (x_pos - 12, y_top - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
        )

    # Connect points with a line
    for i in range(len(points) - 1):
        cv2.line(chart, points[i], points[i + 1], (100, 255, 100), 2)

    # Y-axis labels
    cv2.putText(
        chart, f"{max_val:.0f}",
        (5, margin_top + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
    )
    cv2.putText(
        chart, f"{min_val:.0f}",
        (5, height - margin_bottom),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
    )

    return chart
