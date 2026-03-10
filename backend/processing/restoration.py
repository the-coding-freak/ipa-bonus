"""
Image Restoration — Gonzalez & Woods, Ch. 5
Noise Models, Denoising, and Blind/Non-blind Restoration

Endpoints:
    POST /api/restoration/add-noise        Add synthetic noise (Ch. 5.2)
    POST /api/restoration/denoise-spatial  Classical spatial denoising (Ch. 5.3)
    POST /api/restoration/wiener           Wiener filter deblurring (Ch. 5.8)
    POST /api/restoration/motion-deblur    Motion blur removal (Ch. 5.7–5.8)
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
from scipy.signal import wiener as scipy_wiener

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/restoration", tags=["Image Restoration"])


# ---------------------------------------------------------------------------
# Valid types
# ---------------------------------------------------------------------------

_VALID_NOISE_TYPES = {"gaussian", "salt_pepper", "speckle", "poisson"}
_VALID_DENOISE_TYPES = {
    "arithmetic_mean", "geometric_mean", "median",
    "adaptive_median", "contra_harmonic",
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


def _is_grayscale(img: np.ndarray) -> bool:
    """Check if an image is grayscale (single channel or all channels equal)."""
    if len(img.shape) == 2:
        return True
    if img.shape[2] == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

def _add_gaussian_noise(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Additive Gaussian noise: g(x,y) = f(x,y) + N(0, σ²)
    intensity controls σ (standard deviation), scaled to 0–255 range.
    Reference: Gonzalez & Woods, Ch. 5.2
    """
    sigma = intensity * 255.0
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_salt_pepper_noise(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Salt-and-pepper (impulse) noise.
    intensity is the probability of a pixel being corrupted (0–1).
    Reference: Gonzalez & Woods, Ch. 5.2
    """
    noisy = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    # Salt (white)
    num_salt = int(total_pixels * intensity / 2.0)
    salt_coords = (
        np.random.randint(0, img.shape[0], num_salt),
        np.random.randint(0, img.shape[1], num_salt),
    )
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Pepper (black)
    num_pepper = int(total_pixels * intensity / 2.0)
    pepper_coords = (
        np.random.randint(0, img.shape[0], num_pepper),
        np.random.randint(0, img.shape[1], num_pepper),
    )
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy


def _add_speckle_noise(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Multiplicative speckle noise: g = f + f * n, where n ~ N(0, σ²).
    Common in SAR / radar imagery.
    Reference: Gonzalez & Woods, Ch. 5.2
    """
    noise = np.random.normal(0, intensity, img.shape)
    noisy = img.astype(np.float64) + img.astype(np.float64) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _add_poisson_noise(img: np.ndarray, intensity: float) -> np.ndarray:
    """
    Poisson (shot) noise — signal-dependent.
    intensity scales the noise level (higher = more noise via lower photon count).
    Reference: Gonzalez & Woods, Ch. 5.2
    """
    # Scale factor: lower scale → noisier image
    scale = max(1.0 / max(intensity, 1e-6), 1.0)
    img_float = img.astype(np.float64) / 255.0
    noisy = np.random.poisson(img_float * scale) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Denoising filters
# ---------------------------------------------------------------------------

def _arithmetic_mean_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Arithmetic mean filter: ĝ(x,y) = (1/mn) Σ f(s,t).
    Smooths noise but blurs edges.
    Reference: Gonzalez & Woods, Ch. 5.3
    """
    return cv2.blur(img, (ksize, ksize))


def _geometric_mean_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Geometric mean filter: ĝ(x,y) = [Π f(s,t)]^(1/mn).
    Better at preserving detail than arithmetic mean.
    Reference: Gonzalez & Woods, Ch. 5.3
    """
    # Process each channel independently
    result = np.zeros_like(img, dtype=np.uint8)
    channels = 1 if _is_grayscale(img) else img.shape[2]

    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result_ch = _geometric_mean_single(gray.astype(np.float64), ksize)
        if len(img.shape) == 2:
            return cv2.cvtColor(result_ch, cv2.COLOR_GRAY2BGR)
        result[:, :, 0] = result_ch
        return cv2.cvtColor(result_ch, cv2.COLOR_GRAY2BGR)

    for c in range(channels):
        result[:, :, c] = _geometric_mean_single(
            img[:, :, c].astype(np.float64), ksize
        )
    return result


def _geometric_mean_single(channel: np.ndarray, ksize: int) -> np.ndarray:
    """Geometric mean filter on a single channel using log-domain averaging."""
    # Avoid log(0) by adding a small epsilon
    log_img = np.log(channel + 1e-10)
    # Average in log domain = geometric mean
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    log_mean = cv2.filter2D(log_img, cv2.CV_64F, kernel)
    geo_mean = np.exp(log_mean)
    return np.clip(geo_mean, 0, 255).astype(np.uint8)


def _median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Standard median filter — excellent for salt-and-pepper noise.
    Reference: Gonzalez & Woods, Ch. 5.3
    """
    return cv2.medianBlur(img, ksize)


def _adaptive_median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Adaptive median filter — variable window size for better edge preservation.
    Starts with 3×3 and grows up to ksize×ksize.
    Reference: Gonzalez & Woods, Ch. 5.3
    """
    max_ksize = ksize if ksize % 2 == 1 else ksize + 1
    # Work on each channel independently
    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result = _adaptive_median_single(gray, max_ksize)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _adaptive_median_single(img[:, :, c], max_ksize)
    return result


def _adaptive_median_single(channel: np.ndarray, max_ksize: int) -> np.ndarray:
    """Adaptive median on a single channel."""
    rows, cols = channel.shape
    output = channel.copy()
    pad = max_ksize // 2
    padded = np.pad(channel, pad, mode='reflect')

    for i in range(rows):
        for j in range(cols):
            win_size = 3
            while win_size <= max_ksize:
                half = win_size // 2
                # Extract window from padded image
                ci, cj = i + pad, j + pad
                window = padded[ci - half:ci + half + 1, cj - half:cj + half + 1]
                z_min = int(window.min())
                z_max = int(window.max())
                z_med = int(np.median(window))
                z_xy = int(channel[i, j])

                # Stage A: is median not an impulse?
                if z_min < z_med < z_max:
                    # Stage B: is current pixel an impulse?
                    if z_min < z_xy < z_max:
                        output[i, j] = z_xy  # keep original
                    else:
                        output[i, j] = z_med  # replace with median
                    break
                else:
                    win_size += 2  # grow window

            # If we exhausted all window sizes, use median of largest window
            if win_size > max_ksize:
                half = max_ksize // 2
                ci, cj = i + pad, j + pad
                window = padded[ci - half:ci + half + 1, cj - half:cj + half + 1]
                output[i, j] = int(np.median(window))

    return output


def _contra_harmonic_mean_filter(
    img: np.ndarray, ksize: int, q: float = 1.5
) -> np.ndarray:
    """
    Contra-harmonic mean filter:
        ĝ(x,y) = Σ f(s,t)^(Q+1) / Σ f(s,t)^Q

    Q > 0 eliminates pepper noise, Q < 0 eliminates salt noise.
    Reference: Gonzalez & Woods, Ch. 5.3
    """
    if _is_grayscale(img):
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        result = _contra_harmonic_single(gray.astype(np.float64), ksize, q)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    result = np.zeros_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _contra_harmonic_single(
            img[:, :, c].astype(np.float64), ksize, q
        )
    return result


def _contra_harmonic_single(
    channel: np.ndarray, ksize: int, q: float
) -> np.ndarray:
    """Contra-harmonic mean on a single channel."""
    kernel = np.ones((ksize, ksize), dtype=np.float64)
    # Avoid zero-division: add epsilon
    eps = 1e-10
    numerator = cv2.filter2D(
        np.power(channel + eps, q + 1), cv2.CV_64F, kernel
    )
    denominator = cv2.filter2D(
        np.power(channel + eps, q), cv2.CV_64F, kernel
    )
    result = numerator / (denominator + eps)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Motion blur PSF + Wiener deconvolution helpers
# ---------------------------------------------------------------------------

def _motion_blur_psf(length: int, angle: float, shape: tuple) -> np.ndarray:
    """
    Generate a motion blur point spread function (PSF) kernel,
    then embed it in a full-size array matching the image shape.
    Reference: Gonzalez & Woods, Ch. 5.7
    """
    # Build the small PSF kernel
    psf = np.zeros((length, length), dtype=np.float64)
    center = length // 2
    # Draw the motion line
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    for i in range(length):
        offset = i - center
        y = int(round(center + offset * sin_a))
        x = int(round(center + offset * cos_a))
        if 0 <= y < length and 0 <= x < length:
            psf[y, x] = 1.0
    psf /= psf.sum()  # normalize

    # Embed in full-size array (zero-padded)
    rows, cols = shape
    full_psf = np.zeros((rows, cols), dtype=np.float64)
    # Place at top-left corner (for DFT alignment)
    ph, pw = psf.shape
    full_psf[:ph, :pw] = psf
    # Circularly shift so center of PSF aligns with (0,0)
    full_psf = np.roll(full_psf, -center, axis=0)
    full_psf = np.roll(full_psf, -center, axis=1)
    return full_psf


def _wiener_deconvolve(
    img_f64: np.ndarray, psf: np.ndarray, noise_var: float
) -> np.ndarray:
    """
    Wiener deconvolution in the frequency domain:
        F̂(u,v) = [H*(u,v) / (|H(u,v)|² + K)] · G(u,v)

    where K = noise_variance / signal_variance (estimated).
    Reference: Gonzalez & Woods, Ch. 5.8
    """
    # DFTs
    G = np.fft.fft2(img_f64)
    H = np.fft.fft2(psf, s=img_f64.shape)

    # Wiener filter
    H_conj = np.conj(H)
    H_abs2 = np.abs(H) ** 2

    # Estimate K from noise variance
    K = noise_var if noise_var > 0 else 1e-6

    F_hat = (H_conj / (H_abs2 + K)) * G
    result = np.real(np.fft.ifft2(F_hat))
    return result


# ---------------------------------------------------------------------------
# 1. POST /add-noise — Add Synthetic Noise
# ---------------------------------------------------------------------------

@router.post("/add-noise")
async def add_noise(
    file: UploadFile = File(...),
    noise_type: str = Form("gaussian"),
    intensity: float = Form(0.05),
):
    """
    Add synthetic noise to an image for testing restoration algorithms.

    Params:
        noise_type — "gaussian" | "salt_pepper" | "speckle" | "poisson"
        intensity  — Noise strength (meaning varies by type):
                     gaussian:     σ as fraction of 255 (e.g. 0.1 → σ=25.5)
                     salt_pepper:  corruption probability (0–1)
                     speckle:      σ of multiplicative noise
                     poisson:      inverse scale factor (higher → noisier)

    Reference: Gonzalez & Woods, Ch. 5.2
    """
    try:
        if noise_type not in _VALID_NOISE_TYPES:
            raise ValueError(
                f"Invalid noise_type '{noise_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_NOISE_TYPES))}"
            )
        if intensity < 0:
            raise ValueError("intensity must be non-negative.")

        start = time.time()
        img = await _read_image(file)

        if noise_type == "gaussian":
            result = _add_gaussian_noise(img, intensity)
        elif noise_type == "salt_pepper":
            result = _add_salt_pepper_noise(img, intensity)
        elif noise_type == "speckle":
            result = _add_speckle_noise(img, intensity)
        elif noise_type == "poisson":
            result = _add_poisson_noise(img, intensity)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "add_noise",
            "chapter": "Ch. 5.2",
            "params": {"noise_type": noise_type, "intensity": intensity},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("add_noise failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /denoise-spatial — Classical Spatial Denoising
# ---------------------------------------------------------------------------

@router.post("/denoise-spatial")
async def denoise_spatial(
    file: UploadFile = File(...),
    denoise_type: str = Form("median"),
    kernel_size: int = Form(3),
    q: float = Form(1.5),
):
    """
    Apply a classical spatial denoising filter.

    Params:
        denoise_type — "arithmetic_mean" | "geometric_mean" | "median" |
                        "adaptive_median" | "contra_harmonic"
        kernel_size  — Filter window size (must be odd, default 3).
        q            — Order for contra-harmonic mean filter (default 1.5).
                        Q > 0 → eliminates pepper noise
                        Q < 0 → eliminates salt noise

    Reference: Gonzalez & Woods, Ch. 5.3
    """
    try:
        if denoise_type not in _VALID_DENOISE_TYPES:
            raise ValueError(
                f"Invalid denoise_type '{denoise_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_DENOISE_TYPES))}"
            )
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")

        # Force odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        start = time.time()
        img = await _read_image(file)

        if denoise_type == "arithmetic_mean":
            result = _arithmetic_mean_filter(img, kernel_size)
        elif denoise_type == "geometric_mean":
            result = _geometric_mean_filter(img, kernel_size)
        elif denoise_type == "median":
            result = _median_filter(img, kernel_size)
        elif denoise_type == "adaptive_median":
            result = _adaptive_median_filter(img, kernel_size)
        elif denoise_type == "contra_harmonic":
            result = _contra_harmonic_mean_filter(img, kernel_size, q)
        else:
            raise ValueError(f"Unsupported denoise type: {denoise_type}")

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "denoise_spatial",
            "chapter": "Ch. 5.3",
            "params": {
                "denoise_type": denoise_type,
                "kernel_size": kernel_size,
                "q": q if denoise_type == "contra_harmonic" else None,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("denoise_spatial failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 3. POST /wiener — Wiener Filter for Deblurring
# ---------------------------------------------------------------------------

@router.post("/wiener")
async def wiener_filter(
    file: UploadFile = File(...),
    noise_variance: float = Form(0.01),
):
    """
    Apply a Wiener filter for noise reduction / light deblurring.
    Uses scipy.signal.wiener applied per-channel.

    Params:
        noise_variance — Estimated noise power. Higher values produce
                          stronger smoothing (default 0.01).

    Reference: Gonzalez & Woods, Ch. 5.8
    """
    try:
        if noise_variance < 0:
            raise ValueError("noise_variance must be non-negative.")

        start = time.time()
        img = await _read_image(file)

        # Apply Wiener filter per channel
        result = np.zeros_like(img, dtype=np.uint8)
        noise_power = noise_variance * 255.0 * 255.0  # scale to pixel range

        if _is_grayscale(img):
            gray = img if len(img.shape) == 2 else img[:, :, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered = scipy_wiener(gray.astype(np.float64), noise=noise_power)
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        else:
            for c in range(img.shape[2]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    filtered = scipy_wiener(
                        img[:, :, c].astype(np.float64), noise=noise_power
                    )
                filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)
                result[:, :, c] = np.clip(filtered, 0, 255).astype(np.uint8)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "wiener_filter",
            "chapter": "Ch. 5.8",
            "params": {"noise_variance": noise_variance},
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("wiener_filter failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 4. POST /motion-deblur — Motion Blur Removal
# ---------------------------------------------------------------------------

@router.post("/motion-deblur")
async def motion_deblur(
    file: UploadFile = File(...),
    angle: float = Form(0.0),
    length: int = Form(15),
    noise_variance: float = Form(0.001),
):
    """
    Remove motion blur using Wiener deconvolution with an estimated
    motion blur PSF.

    Params:
        angle          — Motion direction in degrees (default 0 = horizontal).
        length         — Blur length in pixels (default 15).
        noise_variance — Noise-to-signal ratio K for Wiener filter
                          (default 0.001; increase for noisier images).

    Reference: Gonzalez & Woods, Ch. 5.7–5.8
    """
    try:
        if length < 1:
            raise ValueError("length must be >= 1.")

        start = time.time()
        img = await _read_image(file)

        # Process each channel through Wiener deconvolution
        if _is_grayscale(img):
            gray = (img if len(img.shape) == 2 else img[:, :, 0]).astype(
                np.float64
            )
            psf = _motion_blur_psf(length, angle, gray.shape)
            restored = _wiener_deconvolve(gray, psf, noise_variance)
            restored = np.clip(restored, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        else:
            result = np.zeros_like(img, dtype=np.uint8)
            for c in range(img.shape[2]):
                channel = img[:, :, c].astype(np.float64)
                psf = _motion_blur_psf(length, angle, channel.shape)
                restored = _wiener_deconvolve(channel, psf, noise_variance)
                result[:, :, c] = np.clip(restored, 0, 255).astype(np.uint8)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "motion_deblur",
            "chapter": "Ch. 5.7–5.8",
            "params": {
                "angle": angle,
                "length": length,
                "noise_variance": noise_variance,
            },
            "time_ms": elapsed,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("motion_deblur failed")
        raise HTTPException(status_code=500, detail=str(e))
