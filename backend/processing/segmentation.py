"""
Image Segmentation — Gonzalez & Woods, Ch. 10
Thresholding, Edge Detection, and Region-based Segmentation

Endpoints:
    POST /api/segmentation/threshold            Thresholding (Ch. 10.3)
    POST /api/segmentation/edge-detect          Edge detection (Ch. 10.2)
    POST /api/segmentation/hough-lines          Hough line detection (Ch. 10.2)
    POST /api/segmentation/hough-circles        Hough circle detection (Ch. 10.2)
    POST /api/segmentation/region-grow          Region growing (Ch. 10.4)
    POST /api/segmentation/watershed            Watershed segmentation (Ch. 10.5)
    POST /api/segmentation/connected-components Connected component labeling (Ch. 10.4)
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

router = APIRouter(prefix="/api/segmentation", tags=["Image Segmentation"])


# ---------------------------------------------------------------------------
# Valid parameter values
# ---------------------------------------------------------------------------

_VALID_THRESHOLD_TYPES = {"global", "otsu", "adaptive_mean", "adaptive_gaussian"}
_VALID_EDGE_TYPES = {"sobel", "prewitt", "roberts", "canny", "laplacian"}


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


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR to single-channel grayscale uint8."""
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 1. POST /threshold — Thresholding
# ---------------------------------------------------------------------------

def _apply_threshold(
    img: np.ndarray,
    thresh_type: str,
    threshold_value: int,
    block_size: int,
    c_value: int,
) -> np.ndarray:
    """
    Apply various thresholding methods.
    Reference: Gonzalez & Woods, Ch. 10.3
    """
    gray = _to_gray(img)

    if thresh_type == "global":
        _, result = cv2.threshold(
            gray, threshold_value, 255, cv2.THRESH_BINARY
        )
    elif thresh_type == "otsu":
        _, result = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif thresh_type == "adaptive_mean":
        # block_size must be odd and > 1
        bs = max(block_size, 3)
        if bs % 2 == 0:
            bs += 1
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, bs, c_value,
        )
    elif thresh_type == "adaptive_gaussian":
        bs = max(block_size, 3)
        if bs % 2 == 0:
            bs += 1
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, bs, c_value,
        )
    else:
        raise ValueError(f"Unknown threshold type: {thresh_type}")

    return result


@router.post("/threshold")
async def threshold(
    file: UploadFile = File(...),
    threshold_type: str = Form("otsu"),
    threshold_value: int = Form(128),
    block_size: int = Form(11),
    c_value: int = Form(2),
):
    """
    Apply thresholding to segment an image into foreground/background.

    Params:
        threshold_type  — "global" | "otsu" | "adaptive_mean" | "adaptive_gaussian"
        threshold_value — Threshold (0–255, used only for "global").
        block_size      — Neighbourhood size for adaptive methods (odd int, default 11).
        c_value         — Constant subtracted from mean in adaptive methods (default 2).

    Reference: Gonzalez & Woods, Ch. 10.3
    """
    try:
        if threshold_type not in _VALID_THRESHOLD_TYPES:
            raise ValueError(
                f"Invalid threshold_type '{threshold_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_THRESHOLD_TYPES))}"
            )

        start = time.time()
        img = await _read_image(file)
        result = _apply_threshold(
            img, threshold_type, threshold_value, block_size, c_value
        )
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "threshold",
            "chapter": "Ch. 10.3",
            "params": {
                "threshold_type": threshold_type,
                "threshold_value": threshold_value
                    if threshold_type == "global" else "auto",
                "block_size": block_size
                    if threshold_type.startswith("adaptive") else None,
                "c_value": c_value
                    if threshold_type.startswith("adaptive") else None,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("threshold failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /edge-detect — Edge Detection
# ---------------------------------------------------------------------------

def _apply_edge_detection(
    img: np.ndarray,
    edge_type: str,
    low_threshold: float,
    high_threshold: float,
    kernel_size: int,
) -> np.ndarray:
    """
    Apply edge detection.
    Reference: Gonzalez & Woods, Ch. 10.2
    """
    gray = _to_gray(img)

    if edge_type == "sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    if edge_type == "prewitt":
        # Prewitt kernels
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kx)
        gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, ky)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    if edge_type == "roberts":
        # Roberts cross kernels
        kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kx)
        gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, ky)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    if edge_type == "canny":
        return cv2.Canny(gray, low_threshold, high_threshold)

    if edge_type == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        return np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown edge type: {edge_type}")


@router.post("/edge-detect")
async def edge_detect(
    file: UploadFile = File(...),
    edge_type: str = Form("canny"),
    low_threshold: float = Form(50.0),
    high_threshold: float = Form(150.0),
    kernel_size: int = Form(3),
):
    """
    Apply edge detection.

    Params:
        edge_type      — "sobel" | "prewitt" | "roberts" | "canny" | "laplacian"
        low_threshold  — Lower threshold for Canny (default 50).
        high_threshold — Upper threshold for Canny (default 150).
        kernel_size    — Kernel size for Sobel / Laplacian (odd int, default 3).

    Reference: Gonzalez & Woods, Ch. 10.2
    """
    try:
        if edge_type not in _VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type '{edge_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_EDGE_TYPES))}"
            )
        if kernel_size % 2 == 0:
            kernel_size += 1

        start = time.time()
        img = await _read_image(file)
        result = _apply_edge_detection(
            img, edge_type, low_threshold, high_threshold, kernel_size
        )
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "edge_detect",
            "chapter": "Ch. 10.2",
            "params": {
                "edge_type": edge_type,
                "low_threshold": low_threshold if edge_type == "canny" else None,
                "high_threshold": high_threshold if edge_type == "canny" else None,
                "kernel_size": kernel_size,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("edge_detect failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /hough-lines — Hough Line Detection
# ---------------------------------------------------------------------------

@router.post("/hough-lines")
async def hough_lines(
    file: UploadFile = File(...),
    canny_low: float = Form(50.0),
    canny_high: float = Form(150.0),
    threshold: int = Form(100),
    min_line_length: int = Form(50),
    max_line_gap: int = Form(10),
):
    """
    Detect lines using the probabilistic Hough transform.
    Draws detected lines (red) over the original image.

    Params:
        canny_low / canny_high — Canny edge thresholds (for preprocessing).
        threshold              — Hough accumulator threshold (default 100).
        min_line_length        — Minimum line length in pixels (default 50).
        max_line_gap           — Maximum gap between line segments (default 10).

    Reference: Gonzalez & Woods, Ch. 10.2 (Hough Transform)
    """
    try:
        start = time.time()
        img = await _read_image(file)
        gray = _to_gray(img)

        # Edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        # Draw lines on a copy of the original
        output = img.copy()
        num_lines = 0
        if lines is not None:
            num_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(output, {
            "operation": "hough_lines",
            "chapter": "Ch. 10.2",
            "params": {
                "canny_low": canny_low,
                "canny_high": canny_high,
                "threshold": threshold,
                "min_line_length": min_line_length,
                "max_line_gap": max_line_gap,
            },
            "lines_detected": num_lines,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("hough_lines failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /hough-circles — Hough Circle Detection
# ---------------------------------------------------------------------------

@router.post("/hough-circles")
async def hough_circles(
    file: UploadFile = File(...),
    dp: float = Form(1.2),
    min_dist: int = Form(50),
    param1: float = Form(100.0),
    param2: float = Form(30.0),
    min_radius: int = Form(10),
    max_radius: int = Form(200),
):
    """
    Detect circles using the Hough Circle Transform.
    Draws detected circles (green) and centers (red) on the original.

    Params:
        dp         — Inverse ratio of accumulator resolution (default 1.2).
        min_dist   — Minimum distance between circle centers (default 50).
        param1     — Higher Canny threshold (default 100).
        param2     — Accumulator threshold (default 30; lower = more circles).
        min_radius — Minimum circle radius (default 10).
        max_radius — Maximum circle radius (default 200).

    Reference: Gonzalez & Woods, Ch. 10.2 (Hough Transform)
    """
    try:
        start = time.time()
        img = await _read_image(file)
        gray = _to_gray(img)

        # Slight blur to reduce noise
        gray_blur = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        output = img.copy()
        num_circles = 0
        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            num_circles = len(circles_rounded[0])
            for cx, cy, r in circles_rounded[0, :]:
                # Draw circle outline (green)
                cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                # Draw center (red)
                cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(output, {
            "operation": "hough_circles",
            "chapter": "Ch. 10.2",
            "params": {
                "dp": dp,
                "min_dist": min_dist,
                "param1": param1,
                "param2": param2,
                "min_radius": min_radius,
                "max_radius": max_radius,
            },
            "circles_detected": num_circles,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("hough_circles failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 5. POST /region-grow — Region Growing Segmentation
# ---------------------------------------------------------------------------

def _region_grow(
    img: np.ndarray, seed_x: int, seed_y: int, tolerance: float
) -> np.ndarray:
    """
    Simple region growing from a seed point.
    Grows outward from (seed_x, seed_y) adding neighbours whose intensity
    is within ±tolerance of the seed intensity.
    Reference: Gonzalez & Woods, Ch. 10.4
    """
    gray = _to_gray(img)
    rows, cols = gray.shape
    visited = np.zeros((rows, cols), dtype=np.uint8)
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Clamp seed to image bounds
    seed_y = max(0, min(seed_y, rows - 1))
    seed_x = max(0, min(seed_x, cols - 1))

    seed_val = float(gray[seed_y, seed_x])
    stack = [(seed_y, seed_x)]

    while stack:
        y, x = stack.pop()
        if y < 0 or y >= rows or x < 0 or x >= cols:
            continue
        if visited[y, x]:
            continue
        visited[y, x] = 1

        if abs(float(gray[y, x]) - seed_val) <= tolerance:
            mask[y, x] = 255
            # 4-connected neighbours
            stack.append((y - 1, x))
            stack.append((y + 1, x))
            stack.append((y, x - 1))
            stack.append((y, x + 1))

    return mask


@router.post("/region-grow")
async def region_grow(
    file: UploadFile = File(...),
    seed_x: int = Form(0),
    seed_y: int = Form(0),
    tolerance: float = Form(15.0),
):
    """
    Region growing segmentation from a seed point.

    Params:
        seed_x / seed_y — Seed pixel coordinates.
        tolerance       — Max intensity difference from seed to include
                           a pixel (default 15).

    Returns a binary mask highlighting the grown region, and an overlay
    showing the region on the original image.
    Reference: Gonzalez & Woods, Ch. 10.4
    """
    try:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative.")

        start = time.time()
        img = await _read_image(file)

        mask = _region_grow(img, seed_x, seed_y, tolerance)

        # Create overlay: original with green tint on segmented region
        overlay = img.copy()
        overlay[mask == 255] = (
            overlay[mask == 255].astype(np.float64) * 0.5
            + np.array([0, 255, 0], dtype=np.float64) * 0.5
        ).astype(np.uint8)
        # Mark seed point (red circle)
        cv2.circle(overlay, (seed_x, seed_y), 5, (0, 0, 255), -1)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(overlay, {
            "operation": "region_grow",
            "chapter": "Ch. 10.4",
            "params": {
                "seed_x": seed_x,
                "seed_y": seed_y,
                "tolerance": tolerance,
            },
            "region_pixels": int(np.count_nonzero(mask)),
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("region_grow failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 6. POST /watershed — Watershed Segmentation
# ---------------------------------------------------------------------------

def _apply_watershed(img: np.ndarray) -> np.ndarray:
    """
    Marker-based watershed segmentation.
    Automatic marker generation via distance transform + thresholding.
    Reference: Gonzalez & Woods, Ch. 10.5
    """
    gray = _to_gray(img)

    # Otsu threshold to get binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background — dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground — distance transform + threshold
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1            # background = 1 (not 0)
    markers[unknown == 255] = 0      # unknown = 0

    # Watershed
    img_color = img.copy()
    if len(img_color.shape) == 2:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # Colorize result
    # Boundaries → red, regions → random colors
    output = img_color.copy()
    output[markers == -1] = [0, 0, 255]  # boundary in red

    # Generate random colours for each region
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_labels + 2, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]    # background black
    colors[1] = [0, 0, 0]    # bg label

    colored = np.zeros_like(img_color)
    for label_id in range(2, num_labels + 1):
        colored[markers == label_id] = colors[label_id]
    colored[markers == -1] = [0, 0, 255]

    # Blend: 60% original + 40% colored overlay
    blended = cv2.addWeighted(img_color, 0.6, colored, 0.4, 0)
    blended[markers == -1] = [0, 0, 255]

    return blended, num_labels - 1  # subtract background label


@router.post("/watershed")
async def watershed(file: UploadFile = File(...)):
    """
    Marker-based watershed segmentation.
    Automatically generates markers via distance transform, then applies
    OpenCV's watershed. Returns the original overlaid with coloured regions
    and red boundaries.

    Reference: Gonzalez & Woods, Ch. 10.5
    """
    try:
        start = time.time()
        img = await _read_image(file)
        result, num_segments = _apply_watershed(img)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "watershed",
            "chapter": "Ch. 10.5",
            "params": {},
            "segments_found": num_segments,
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("watershed failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 7. POST /connected-components — Connected Component Labeling
# ---------------------------------------------------------------------------

@router.post("/connected-components")
async def connected_components(
    file: UploadFile = File(...),
    connectivity: int = Form(8),
):
    """
    Label connected components and assign each a unique random colour.

    Params:
        connectivity — 4 or 8 (pixel connectivity, default 8).

    Reference: Gonzalez & Woods, Ch. 10.4
    """
    try:
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")

        start = time.time()
        img = await _read_image(file)
        gray = _to_gray(img)

        # Threshold with Otsu
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Connected components
        num_labels, labels = cv2.connectedComponents(
            binary, connectivity=connectivity
        )

        # Assign random colours (label 0 = background → black)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # background

        colored = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for label_id in range(num_labels):
            colored[labels == label_id] = colors[label_id]

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(colored, {
            "operation": "connected_components",
            "chapter": "Ch. 10.4",
            "params": {"connectivity": connectivity},
            "num_components": int(num_labels - 1),  # exclude background
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("connected_components failed")
        raise HTTPException(status_code=500, detail=str(e))
