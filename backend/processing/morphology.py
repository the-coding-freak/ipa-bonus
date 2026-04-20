"""
Morphological Image Processing — Gonzalez & Woods, Ch. 9
Binary and Grayscale Morphological Operations

Endpoints:
    POST /api/morphology/erode      Erosion (Ch. 9.1)
    POST /api/morphology/dilate     Dilation (Ch. 9.1)
    POST /api/morphology/open       Opening — erosion then dilation (Ch. 9.2)
    POST /api/morphology/close      Closing — dilation then erosion (Ch. 9.2)
    POST /api/morphology/gradient   Morphological gradient (Ch. 9.4)
    POST /api/morphology/tophat     Top-hat (white) transform (Ch. 9.4)
    POST /api/morphology/blackhat   Black-hat transform (Ch. 9.4)
    POST /api/morphology/skeleton   Skeletonization via thinning (Ch. 9.5)
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

router = APIRouter(prefix="/api/morphology", tags=["Morphological Processing"])


# ---------------------------------------------------------------------------
# Valid parameter values
# ---------------------------------------------------------------------------

_VALID_KERNEL_SHAPES = {
    "rect":    cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross":   cv2.MORPH_CROSS,
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


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or grayscale image to a single-channel uint8 grayscale.
    Morphological operations are applied on grayscale directly.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return img  # already single channel


def _to_binary(img: np.ndarray) -> np.ndarray:
    """
    Convert to binary via Otsu's thresholding.
    Used only for skeletonization which requires a strict binary input.
    """
    gray = _to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _get_kernel(kernel_shape: str, kernel_size: int) -> np.ndarray:
    """Build a structuring element of the given shape and size."""
    shape_enum = _VALID_KERNEL_SHAPES.get(kernel_shape)
    if shape_enum is None:
        raise ValueError(
            f"Invalid kernel_shape '{kernel_shape}'. "
            f"Must be one of: {', '.join(sorted(_VALID_KERNEL_SHAPES.keys()))}"
        )
    return cv2.getStructuringElement(shape_enum, (kernel_size, kernel_size))


def _validate_common_params(kernel_shape: str, kernel_size: int) -> int:
    """Validate and normalise shared morph params. Returns corrected kernel_size."""
    if kernel_shape not in _VALID_KERNEL_SHAPES:
        raise ValueError(
            f"Invalid kernel_shape '{kernel_shape}'. "
            f"Must be one of: {', '.join(sorted(_VALID_KERNEL_SHAPES.keys()))}"
        )
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")
    # Force odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


# ---------------------------------------------------------------------------
# 1. POST /erode — Morphological Erosion
# ---------------------------------------------------------------------------

@router.post("/erode")
async def erode(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    kernel_shape: str = Form("rect"),
    iterations: int = Form(1),
):
    """
    Morphological erosion — shrinks bright regions / expands dark ones.
    Auto-thresholds color images to binary via Otsu's method.

    Params:
        kernel_size  — Structuring element size (odd int, default 3).
        kernel_shape — "rect" | "ellipse" | "cross"
        iterations   — Number of times to apply (default 1).

    Reference: Gonzalez & Woods, Ch. 9.1
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.erode(gray, kernel, iterations=iterations)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "erosion",
            "chapter": "Ch. 9.1",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
                "iterations": iterations,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("erode failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /dilate — Morphological Dilation
# ---------------------------------------------------------------------------

@router.post("/dilate")
async def dilate(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    kernel_shape: str = Form("rect"),
    iterations: int = Form(1),
):
    """
    Morphological dilation — expands bright regions / shrinks dark ones.

    Params:
        kernel_size  — Structuring element size (odd int, default 3).
        kernel_shape — "rect" | "ellipse" | "cross"
        iterations   — Number of times to apply (default 1).

    Reference: Gonzalez & Woods, Ch. 9.1
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.dilate(gray, kernel, iterations=iterations)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "dilation",
            "chapter": "Ch. 9.1",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
                "iterations": iterations,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("dilate failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /open — Morphological Opening
# ---------------------------------------------------------------------------

@router.post("/open")
async def morph_open(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    kernel_shape: str = Form("rect"),
    iterations: int = Form(1),
):
    """
    Morphological opening = erosion → dilation.
    Removes small bright spots / noise while preserving shape.

    Params:
        kernel_size  — Structuring element size (odd int, default 3).
        kernel_shape — "rect" | "ellipse" | "cross"
        iterations   — Number of times to apply (default 1).

    Reference: Gonzalez & Woods, Ch. 9.2
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "opening",
            "chapter": "Ch. 9.2",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
                "iterations": iterations,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("morph_open failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /close — Morphological Closing
# ---------------------------------------------------------------------------

@router.post("/close")
async def morph_close(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    kernel_shape: str = Form("rect"),
    iterations: int = Form(1),
):
    """
    Morphological closing = dilation → erosion.
    Fills small dark holes while preserving shape.

    Params:
        kernel_size  — Structuring element size (odd int, default 3).
        kernel_shape — "rect" | "ellipse" | "cross"
        iterations   — Number of times to apply (default 1).

    Reference: Gonzalez & Woods, Ch. 9.2
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "closing",
            "chapter": "Ch. 9.2",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
                "iterations": iterations,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("morph_close failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. POST /gradient — Morphological Gradient
# ---------------------------------------------------------------------------

@router.post("/gradient")
async def morph_gradient(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    kernel_shape: str = Form("rect"),
):
    """
    Morphological gradient = dilation − erosion.
    Produces an edge / outline of objects.

    Params:
        kernel_size  — Structuring element size (odd int, default 3).
        kernel_shape — "rect" | "ellipse" | "cross"

    Reference: Gonzalez & Woods, Ch. 9.4
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "morphological_gradient",
            "chapter": "Ch. 9.4",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("morph_gradient failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 6. POST /tophat — Top-Hat (White) Transform
# ---------------------------------------------------------------------------

@router.post("/tophat")
async def top_hat(
    file: UploadFile = File(...),
    kernel_size: int = Form(9),
    kernel_shape: str = Form("rect"),
):
    """
    Top-hat (white) transform = original − opening.
    Extracts small bright features on a dark background.

    Params:
        kernel_size  — Structuring element size (odd int, default 9).
        kernel_shape — "rect" | "ellipse" | "cross"

    Reference: Gonzalez & Woods, Ch. 9.4
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "top_hat",
            "chapter": "Ch. 9.4",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("top_hat failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 7. POST /blackhat — Black-Hat Transform
# ---------------------------------------------------------------------------

@router.post("/blackhat")
async def black_hat(
    file: UploadFile = File(...),
    kernel_size: int = Form(9),
    kernel_shape: str = Form("rect"),
):
    """
    Black-hat transform = closing − original.
    Extracts small dark features on a bright background.

    Params:
        kernel_size  — Structuring element size (odd int, default 9).
        kernel_shape — "rect" | "ellipse" | "cross"

    Reference: Gonzalez & Woods, Ch. 9.4
    """
    try:
        kernel_size = _validate_common_params(kernel_shape, kernel_size)

        start = time.time()
        img = await _read_image(file)
        gray = _to_grayscale(img)
        kernel = _get_kernel(kernel_shape, kernel_size)

        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "black_hat",
            "chapter": "Ch. 9.4",
            "params": {
                "kernel_size": kernel_size,
                "kernel_shape": kernel_shape,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("black_hat failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 8. POST /skeleton — Skeletonization (Thinning)
# ---------------------------------------------------------------------------

def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization using iterative erosion and opening.
    Algorithm:
        skeleton = ∅
        while img is not empty:
            eroded  = erode(img)
            opened  = open(eroded)     → dilate(eroded)
            sub     = eroded − opened  → skeleton branches at this scale
            skeleton = skeleton ∪ sub
            img     = eroded
    Reference: Gonzalez & Woods, Ch. 9.5
    """
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(binary)
    temp = binary.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        sub = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, sub)
        temp = eroded.copy()

        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


@router.post("/skeleton")
async def skeleton(
    file: UploadFile = File(...),
    kernel_shape: str = Form("cross"),
):
    """
    Skeletonization (morphological thinning) — reduces binary shapes
    to their 1-pixel-wide skeleton / medial axis.

    Params:
        kernel_shape — "rect" | "ellipse" | "cross" (default "cross",
                        used only for the initial threshold; skeleton
                        always uses a 3×3 cross element internally).

    Reference: Gonzalez & Woods, Ch. 9.5
    """
    try:
        if kernel_shape not in _VALID_KERNEL_SHAPES:
            raise ValueError(
                f"Invalid kernel_shape '{kernel_shape}'. "
                f"Must be one of: {', '.join(sorted(_VALID_KERNEL_SHAPES.keys()))}"
            )

        start = time.time()
        img = await _read_image(file)
        binary = _to_binary(img)

        result = _skeletonize(binary)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "skeletonization",
            "chapter": "Ch. 9.5",
            "params": {"kernel_shape": kernel_shape},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("skeleton failed")
        raise HTTPException(status_code=500, detail=str(e))
