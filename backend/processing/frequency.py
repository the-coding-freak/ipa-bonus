"""
Frequency Domain Processing — Gonzalez & Woods, Ch. 4
Filtering in the Frequency Domain

Endpoints:
    POST /api/frequency/fft         Compute 2D FFT magnitude spectrum (Ch. 4.3)
    POST /api/frequency/filter      Apply frequency domain filter (Ch. 4.7–4.9)
    POST /api/frequency/inverse     Inverse FFT to reconstruct image (Ch. 4.3)
"""

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PilImage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/frequency", tags=["Frequency Processing"])


# ---------------------------------------------------------------------------
# Valid filter types
# ---------------------------------------------------------------------------

_VALID_FILTER_TYPES = {
    "ideal_low", "ideal_high",
    "butterworth_low", "butterworth_high",
    "gaussian_low", "gaussian_high",
    "notch",
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
    """Convert BGR image to single-channel grayscale float64."""
    if len(img.shape) == 2:
        return img.astype(np.float64)
    if img.shape[2] == 1:
        return img[:, :, 0].astype(np.float64)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def _compute_fft(gray: np.ndarray) -> np.ndarray:
    """
    Compute the 2D DFT and shift zero-frequency component to center.
    Returns the complex shifted FFT array.
    Reference: Gonzalez & Woods, Ch. 4.3
    """
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def _magnitude_spectrum(dft_shift: np.ndarray) -> np.ndarray:
    """
    Compute displayable magnitude spectrum from shifted FFT.
    Uses log transform: 20 * log10(1 + |F(u,v)|), normalized to [0, 255].
    Reference: Gonzalez & Woods, Ch. 4.3
    """
    magnitude = np.abs(dft_shift)
    # Log transform for visibility
    log_magnitude = np.log1p(magnitude)
    # Normalize to 0–255
    if log_magnitude.max() > 0:
        log_magnitude = (log_magnitude / log_magnitude.max()) * 255.0
    return np.clip(log_magnitude, 0, 255).astype(np.uint8)


def _encode_fft_base64(dft_shift: np.ndarray) -> str:
    """
    Encode the complex FFT array as base64 for later filtering / inverse.
    Stores shape + raw bytes of the complex128 array.
    """
    # Save as .npy bytes in memory
    buf = BytesIO()
    np.save(buf, dft_shift)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _decode_fft_base64(fft_data: str) -> np.ndarray:
    """Decode a base64-encoded complex FFT array back to numpy."""
    raw = base64.b64decode(fft_data)
    buf = BytesIO(raw)
    return np.load(buf, allow_pickle=False)


# ---------------------------------------------------------------------------
# Filter mask builders
# ---------------------------------------------------------------------------

def _distance_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Build a matrix D(u,v) = distance of each point from the center.
    Used by all frequency domain filters.
    """
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    return np.sqrt(u ** 2 + v ** 2)


def _ideal_lowpass(rows: int, cols: int, cutoff: float) -> np.ndarray:
    """
    Ideal low-pass filter: H(u,v) = 1 if D(u,v) <= D0, else 0.
    Reference: Gonzalez & Woods, Ch. 4.7
    """
    d0 = cutoff * (min(rows, cols) / 2.0)
    D = _distance_matrix(rows, cols)
    return (D <= d0).astype(np.float64)


def _ideal_highpass(rows: int, cols: int, cutoff: float) -> np.ndarray:
    """
    Ideal high-pass filter: H = 1 − H_LP.
    Reference: Gonzalez & Woods, Ch. 4.7
    """
    return 1.0 - _ideal_lowpass(rows, cols, cutoff)


def _butterworth_lowpass(
    rows: int, cols: int, cutoff: float, order: int
) -> np.ndarray:
    """
    Butterworth low-pass filter: H(u,v) = 1 / (1 + (D/D0)^(2n)).
    Reference: Gonzalez & Woods, Ch. 4.8
    """
    d0 = cutoff * (min(rows, cols) / 2.0)
    D = _distance_matrix(rows, cols)
    # Avoid division by zero
    d0 = max(d0, 1e-10)
    return 1.0 / (1.0 + (D / d0) ** (2 * order))


def _butterworth_highpass(
    rows: int, cols: int, cutoff: float, order: int
) -> np.ndarray:
    """
    Butterworth high-pass filter: H = 1 − H_LP.
    Reference: Gonzalez & Woods, Ch. 4.8
    """
    return 1.0 - _butterworth_lowpass(rows, cols, cutoff, order)


def _gaussian_lowpass(rows: int, cols: int, cutoff: float) -> np.ndarray:
    """
    Gaussian low-pass filter: H(u,v) = exp(−D²/(2·D0²)).
    Reference: Gonzalez & Woods, Ch. 4.9
    """
    d0 = cutoff * (min(rows, cols) / 2.0)
    d0 = max(d0, 1e-10)
    D = _distance_matrix(rows, cols)
    return np.exp(-(D ** 2) / (2.0 * d0 ** 2))


def _gaussian_highpass(rows: int, cols: int, cutoff: float) -> np.ndarray:
    """
    Gaussian high-pass filter: H = 1 − H_LP.
    Reference: Gonzalez & Woods, Ch. 4.9
    """
    return 1.0 - _gaussian_lowpass(rows, cols, cutoff)


def _notch_reject(
    rows: int, cols: int, cutoff: float,
    center_u: int = 0, center_v: int = 0,
) -> np.ndarray:
    """
    Notch reject filter — suppresses a symmetric pair of frequency spikes.
    Creates circular reject regions at (center_u, center_v) and its
    conjugate symmetric point.
    Reference: Gonzalez & Woods, Ch. 4.10
    """
    d0 = cutoff * (min(rows, cols) / 2.0)
    d0 = max(d0, 1e-10)
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows).reshape(-1, 1)
    v = np.arange(cols).reshape(1, -1)

    # Distance from notch center and its conjugate
    d1 = np.sqrt((u - (crow + center_u)) ** 2 + (v - (ccol + center_v)) ** 2)
    d2 = np.sqrt((u - (crow - center_u)) ** 2 + (v - (ccol - center_v)) ** 2)

    H = np.ones((rows, cols), dtype=np.float64)
    H[d1 <= d0] = 0.0
    H[d2 <= d0] = 0.0
    return H


def _build_filter_mask(
    rows: int,
    cols: int,
    filter_type: str,
    cutoff: float,
    order: int = 2,
    notch_center_u: int = 0,
    notch_center_v: int = 0,
) -> np.ndarray:
    """Dispatch to the correct filter builder based on type string."""
    if filter_type == "ideal_low":
        return _ideal_lowpass(rows, cols, cutoff)
    if filter_type == "ideal_high":
        return _ideal_highpass(rows, cols, cutoff)
    if filter_type == "butterworth_low":
        return _butterworth_lowpass(rows, cols, cutoff, order)
    if filter_type == "butterworth_high":
        return _butterworth_highpass(rows, cols, cutoff, order)
    if filter_type == "gaussian_low":
        return _gaussian_lowpass(rows, cols, cutoff)
    if filter_type == "gaussian_high":
        return _gaussian_highpass(rows, cols, cutoff)
    if filter_type == "notch":
        return _notch_reject(rows, cols, cutoff, notch_center_u, notch_center_v)

    raise ValueError(
        f"Unknown filter type: '{filter_type}'. "
        f"Must be one of: {', '.join(sorted(_VALID_FILTER_TYPES))}"
    )


# ---------------------------------------------------------------------------
# 1. POST /fft — 2D FFT Magnitude Spectrum
# ---------------------------------------------------------------------------

@router.post("/fft")
async def compute_fft(file: UploadFile = File(...)):
    """
    Compute the 2D FFT of an image and return:
      • magnitude spectrum as a displayable (log-transformed) image
      • raw shifted FFT encoded as base64 for later filtering / inverse
    Reference: Gonzalez & Woods, Ch. 4.3
    """
    try:
        start = time.time()
        img = await _read_image(file)

        # Convert to grayscale for FFT
        gray = _to_grayscale(img)

        # Compute FFT
        dft_shift = _compute_fft(gray)

        # Build displayable magnitude spectrum
        spectrum_img = _magnitude_spectrum(dft_shift)
        spectrum_bgr = cv2.cvtColor(spectrum_img, cv2.COLOR_GRAY2BGR)

        # Encode raw FFT for downstream use
        fft_base64 = _encode_fft_base64(dft_shift)

        elapsed = round((time.time() - start) * 1000, 2)

        # Encode spectrum image
        _, buffer = cv2.imencode('.png', spectrum_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "image": img_base64,
            "fft_data": fft_base64,
            "metadata": {
                "operation": "fft",
                "chapter": "Ch. 4.3",
                "params": {},
                "image_shape": list(gray.shape),
                "time_ms": elapsed,
            },
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("compute_fft failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /filter — Frequency Domain Filtering
# ---------------------------------------------------------------------------

@router.post("/filter")
async def frequency_filter(
    file: UploadFile = File(...),
    filter_type: str = Form("ideal_low"),
    cutoff: float = Form(0.1),
    order: int = Form(2),
    notch_center_u: int = Form(0),
    notch_center_v: int = Form(0),
):
    """
    Apply a frequency domain filter to an image.

    Params:
        filter_type — "ideal_low" | "ideal_high" | "butterworth_low" |
                       "butterworth_high" | "gaussian_low" | "gaussian_high" |
                       "notch"
        cutoff      — Normalized cutoff frequency (0–1). Maps to D0 relative
                       to half the smaller image dimension.
        order       — Filter order (Butterworth only, default 2).
        notch_center_u / notch_center_v — Notch offset from center (notch only).

    Returns the filtered image plus the filtered FFT data (base64) for
    optional inverse transform.
    Reference: Gonzalez & Woods, Ch. 4.7–4.10
    """
    try:
        if filter_type not in _VALID_FILTER_TYPES:
            raise ValueError(
                f"Invalid filter_type '{filter_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_FILTER_TYPES))}"
            )
        if not 0.0 < cutoff <= 1.0:
            raise ValueError("cutoff must be in the range (0, 1].")
        if order < 1:
            raise ValueError("order must be >= 1.")

        start = time.time()
        img = await _read_image(file)

        # Convert to grayscale
        gray = _to_grayscale(img)
        rows, cols = gray.shape

        # Forward FFT
        dft_shift = _compute_fft(gray)

        # Build and apply filter mask
        H = _build_filter_mask(
            rows, cols, filter_type, cutoff, order,
            notch_center_u, notch_center_v,
        )
        filtered_shift = dft_shift * H

        # Inverse FFT to get spatial result
        f_ishift = np.fft.ifftshift(filtered_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize to 0–255
        if img_back.max() > 0:
            img_back = (img_back / img_back.max()) * 255.0
        result = np.clip(img_back, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Also return the filtered FFT for optional inverse step
        filtered_fft_base64 = _encode_fft_base64(filtered_shift)

        # Build filter visualization (magnitude spectrum of filtered FFT)
        filtered_spectrum = _magnitude_spectrum(filtered_shift)
        filtered_spectrum_bgr = cv2.cvtColor(filtered_spectrum, cv2.COLOR_GRAY2BGR)
        _, spec_buf = cv2.imencode('.png', filtered_spectrum_bgr)
        filtered_spectrum_b64 = base64.b64encode(spec_buf).decode('utf-8')

        elapsed = round((time.time() - start) * 1000, 2)
        return JSONResponse({
            "image": base64.b64encode(
                cv2.imencode('.png', result_bgr)[1]
            ).decode('utf-8'),
            "filtered_spectrum": filtered_spectrum_b64,
            "fft_data": filtered_fft_base64,
            "metadata": {
                "operation": "frequency_filter",
                "chapter": "Ch. 4.7–4.10",
                "params": {
                    "filter_type": filter_type,
                    "cutoff": cutoff,
                    "order": order,
                    "notch_center_u": notch_center_u,
                    "notch_center_v": notch_center_v,
                },
                "image_shape": [rows, cols],
                "time_ms": elapsed,
            },
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("frequency_filter failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /inverse — Inverse FFT Reconstruction
# ---------------------------------------------------------------------------

@router.post("/inverse")
async def inverse_fft(
    fft_data: str = Form(...),
):
    """
    Reconstruct a spatial-domain image from a (possibly filtered) FFT.

    Input:
        fft_data — Base64-encoded complex FFT array (as returned by /fft or
                    /filter endpoints).

    Returns the reconstructed grayscale image.
    Reference: Gonzalez & Woods, Ch. 4.3
    """
    try:
        start = time.time()

        # Decode the FFT data
        dft_shift = _decode_fft_base64(fft_data)

        # Inverse shift + inverse FFT
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize to 0–255
        if img_back.max() > 0:
            img_back = (img_back / img_back.max()) * 255.0
        result = np.clip(img_back, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "inverse_fft",
            "chapter": "Ch. 4.3",
            "params": {},
            "image_shape": list(result.shape),
            "time_ms": elapsed,
        })
    except Exception as e:
        logger.exception("inverse_fft failed")
        raise HTTPException(status_code=500, detail=str(e))
