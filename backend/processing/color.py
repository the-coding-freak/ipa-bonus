"""
Color Image Processing — Gonzalez & Woods, Ch. 6
Color Space Conversion, Pseudo-coloring, and Color Filtering

Endpoints:
    POST /api/color/convert            Color space conversion (Ch. 6.2)
    POST /api/color/false-color        Pseudocolor / false color mapping (Ch. 6.3)
    POST /api/color/histogram-eq-color Per-channel histogram equalization (Ch. 6.5)
    POST /api/color/color-segment      HSV-based color segmentation (Ch. 6.6)
    POST /api/color/channel-split      Split and return individual channels (Ch. 6.1)
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

router = APIRouter(prefix="/api/color", tags=["Color Processing"])


# ---------------------------------------------------------------------------
# Valid parameter values
# ---------------------------------------------------------------------------

_VALID_COLOR_SPACES = {"HSV", "HSI", "LAB", "YCbCr", "GRAY", "RGB"}

_VALID_COLORMAPS = {
    "jet":     cv2.COLORMAP_JET,
    "hot":     cv2.COLORMAP_HOT,
    "cool":    cv2.COLORMAP_COOL,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "plasma":  cv2.COLORMAP_PLASMA,
}


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


def _encode_image_base64(img: np.ndarray) -> str:
    """Encode a single image (grayscale or BGR) to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def _is_grayscale(img: np.ndarray) -> bool:
    """Check if an image is grayscale (single channel or all channels equal)."""
    if len(img.shape) == 2:
        return True
    if img.shape[2] == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Color space conversion helpers
# ---------------------------------------------------------------------------

def _bgr_to_hsi(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSI color space (custom implementation).
    OpenCV does not have a built-in BGR→HSI conversion.
    Reference: Gonzalez & Woods, Ch. 6.2
    """
    bgr = img.astype(np.float64) / 255.0
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1.0 - (min_rgb / (I + 1e-10)), 0.0)

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B) + 1e-10)
    theta = np.arccos(np.clip(num / (den + 1e-10), -1.0, 1.0))
    H = np.where(B <= G, theta, 2.0 * np.pi - theta)
    H = H / (2.0 * np.pi)  # normalize to [0, 1]

    # Stack and scale to uint8 for display
    hsi = np.stack([
        np.clip(H * 255, 0, 255),
        np.clip(S * 255, 0, 255),
        np.clip(I * 255, 0, 255),
    ], axis=-1).astype(np.uint8)
    return hsi


def _convert_color_space(img: np.ndarray, target: str) -> np.ndarray:
    """
    Convert a BGR image to the requested color space.
    Returns a 3-channel uint8 image suitable for PNG encoding.
    """
    if target == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if target == "HSI":
        return _bgr_to_hsi(img)
    if target == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if target == "YCbCr":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if target == "GRAY":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 3-ch for consistency
    if target == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unsupported color space: {target}")


# ---------------------------------------------------------------------------
# 1. POST /convert — Color Space Conversion
# ---------------------------------------------------------------------------

@router.post("/convert")
async def color_convert(
    file: UploadFile = File(...),
    target: str = Form("HSV"),
):
    """
    Convert an image to a different color space.

    Params:
        target — "HSV" | "HSI" | "LAB" | "YCbCr" | "GRAY" | "RGB"

    Note: Input is always assumed to be BGR (OpenCV default from upload).
    The output is the raw channel values of that color space encoded as
    a 3-channel PNG for visualization.
    Reference: Gonzalez & Woods, Ch. 6.2
    """
    try:
        # Case-insensitive lookup → canonical name from the valid set
        _target_map = {k.upper(): k for k in _VALID_COLOR_SPACES}
        target_key = target.upper()
        if target_key not in _target_map:
            raise ValueError(
                f"Invalid target '{target}'. "
                f"Must be one of: {', '.join(sorted(_VALID_COLOR_SPACES))}"
            )
        target = _target_map[target_key]

        start = time.time()
        img = await _read_image(file)
        result = _convert_color_space(img, target)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "color_convert",
            "chapter": "Ch. 6.2",
            "params": {"target": target},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("color_convert failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /false-color — Pseudocolor / False Color
# ---------------------------------------------------------------------------

@router.post("/false-color")
async def false_color(
    file: UploadFile = File(...),
    colormap: str = Form("jet"),
):
    """
    Apply a pseudocolor (false color) mapping to a grayscale image.
    If input is color, it is first converted to grayscale.

    Params:
        colormap — "jet" | "hot" | "cool" | "viridis" | "rainbow" | "plasma"

    Reference: Gonzalez & Woods, Ch. 6.3
    """
    try:
        colormap_lower = colormap.lower()
        if colormap_lower not in _VALID_COLORMAPS:
            raise ValueError(
                f"Invalid colormap '{colormap}'. "
                f"Must be one of: {', '.join(sorted(_VALID_COLORMAPS.keys()))}"
            )

        start = time.time()
        img = await _read_image(file)

        # Convert to grayscale if needed
        if _is_grayscale(img):
            gray = img if len(img.shape) == 2 else img[:, :, 0]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply colormap
        result = cv2.applyColorMap(gray, _VALID_COLORMAPS[colormap_lower])

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "false_color",
            "chapter": "Ch. 6.3",
            "params": {"colormap": colormap_lower},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("false_color failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /histogram-eq-color — Per-Channel Histogram Equalization
# ---------------------------------------------------------------------------

def _apply_histogram_eq_color(img: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization independently to each channel.
    For grayscale images, applies standard equalization.
    Reference: Gonzalez & Woods, Ch. 6.5
    """
    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result = cv2.equalizeHist(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Equalize each channel independently
    channels = cv2.split(img)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    return cv2.merge(eq_channels)


@router.post("/histogram-eq-color")
async def histogram_eq_color(file: UploadFile = File(...)):
    """
    Per-channel histogram equalization.
    Equalizes each B, G, R channel independently for maximum contrast.
    Reference: Gonzalez & Woods, Ch. 6.5
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result = _apply_histogram_eq_color(img)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "histogram_eq_color",
            "chapter": "Ch. 6.5",
            "params": {},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("histogram_eq_color failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /color-segment — HSV-Based Color Segmentation
# ---------------------------------------------------------------------------

def _apply_color_segmentation(
    img: np.ndarray,
    hue_min: float, hue_max: float,
    sat_min: float, sat_max: float,
) -> np.ndarray:
    """
    Segment an image by HSV range.
    All params are normalized 0–1; mapped to OpenCV's H:0-179, S:0-255 ranges.
    Returns the masked (segmented) region on a black background.
    Reference: Gonzalez & Woods, Ch. 6.6
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Map normalized values to OpenCV HSV range
    # Hue: 0–179 in OpenCV (0–360° mapped to 0–180)
    h_lo = int(hue_min * 179)
    h_hi = int(hue_max * 179)
    # Saturation: 0–255
    s_lo = int(sat_min * 255)
    s_hi = int(sat_max * 255)

    lower = np.array([h_lo, s_lo, 0], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, 255], dtype=np.uint8)

    # Handle hue wraparound (e.g. red: hue_min=0.9, hue_max=0.1)
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Split into two ranges: [h_lo, 179] and [0, h_hi]
        lower1 = np.array([h_lo, s_lo, 0], dtype=np.uint8)
        upper1 = np.array([179, s_hi, 255], dtype=np.uint8)
        lower2 = np.array([0, s_lo, 0], dtype=np.uint8)
        upper2 = np.array([h_hi, s_hi, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    # Apply mask to original BGR image
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


@router.post("/color-segment")
async def color_segment(
    file: UploadFile = File(...),
    hue_min: float = Form(0.0),
    hue_max: float = Form(1.0),
    sat_min: float = Form(0.0),
    sat_max: float = Form(1.0),
):
    """
    Segment an image based on HSV color range.
    All parameters are normalized to 0–1.

    Params:
        hue_min / hue_max — Hue range (0–1). Supports wraparound
                             (e.g. hue_min=0.9, hue_max=0.1 for red).
        sat_min / sat_max — Saturation range (0–1).

    Returns the segmented region (pixels matching the range) on a black
    background.
    Reference: Gonzalez & Woods, Ch. 6.6
    """
    try:
        for name, val in [("hue_min", hue_min), ("hue_max", hue_max),
                          ("sat_min", sat_min), ("sat_max", sat_max)]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]. Got {val}.")

        start = time.time()
        img = await _read_image(file)
        result = _apply_color_segmentation(
            img, hue_min, hue_max, sat_min, sat_max
        )

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "color_segment",
            "chapter": "Ch. 6.6",
            "params": {
                "hue_min": hue_min,
                "hue_max": hue_max,
                "sat_min": sat_min,
                "sat_max": sat_max,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("color_segment failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. POST /channel-split — Split Into Individual Channels
# ---------------------------------------------------------------------------

@router.post("/channel-split")
async def channel_split(file: UploadFile = File(...)):
    """
    Split an image into its individual channels (B, G, R for BGR input)
    and return each as a separate base64-encoded grayscale image.

    Response includes an 'image' key (composite side-by-side of all channels)
    and a 'channels' dict with individual channel images.
    Reference: Gonzalez & Woods, Ch. 6.1
    """
    try:
        start = time.time()
        img = await _read_image(file)

        if _is_grayscale(img):
            gray = img if len(img.shape) == 2 else img[:, :, 0]
            channels_dict = {
                "gray": _encode_image_base64(gray),
            }
            # For display, just return the grayscale as the main image
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            b, g, r = cv2.split(img)
            channels_dict = {
                "blue": _encode_image_base64(b),
                "green": _encode_image_base64(g),
                "red": _encode_image_base64(r),
            }
            # Create a side-by-side composite for the main display image
            # Color-tint each channel for visual clarity
            h, w = img.shape[:2]
            blue_vis = np.zeros((h, w, 3), dtype=np.uint8)
            blue_vis[:, :, 0] = b  # Blue channel in blue
            green_vis = np.zeros((h, w, 3), dtype=np.uint8)
            green_vis[:, :, 1] = g  # Green channel in green
            red_vis = np.zeros((h, w, 3), dtype=np.uint8)
            red_vis[:, :, 2] = r  # Red channel in red

            display = np.hstack([blue_vis, green_vis, red_vis])

        elapsed = round((time.time() - start) * 1000, 2)

        # Encode composite display image
        _, buffer = cv2.imencode('.png', display)
        display_b64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "image": display_b64,
            "channels": channels_dict,
            "metadata": {
                "operation": "channel_split",
                "chapter": "Ch. 6.1",
                "params": {},
                "num_channels": len(channels_dict),
                "time_ms": elapsed,
            },
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("channel_split failed")
        raise HTTPException(status_code=500, detail=str(e))
