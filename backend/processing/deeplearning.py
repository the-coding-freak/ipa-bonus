"""
Deep Learning Inference — SRCNN & DnCNN
ONNX Runtime CPU-only inference (no GPU required at runtime).

Endpoints:
    POST /api/dl/super-resolution   SRCNN super-resolution ×2 or ×4 (DIV2K trained)
    POST /api/dl/denoise            DnCNN blind denoising (BSD68 trained)
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PilImage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dl", tags=["Deep Learning"])


# ---------------------------------------------------------------------------
# Module-level ONNX session loaders (per CONVENTIONS.md)
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent.parent / "models"

_srcnn_sessions: dict[int, ort.InferenceSession] = {}
_dncnn_session: ort.InferenceSession | None = None


def _get_srcnn_session(scale: int) -> ort.InferenceSession:
    """
    Lazy-load SRCNN ONNX session for a given scale factor.
    Models: srcnn_x2.onnx, srcnn_x4.onnx
    """
    global _srcnn_sessions
    if scale not in _srcnn_sessions:
        model_path = _MODELS_DIR / f"srcnn_x{scale}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"SRCNN ×{scale} model not found at {model_path}. "
                f"Place the ONNX file in backend/models/ to enable this endpoint."
            )
        _srcnn_sessions[scale] = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("SRCNN ×%d ONNX session loaded from %s", scale, model_path)
    return _srcnn_sessions[scale]


def _get_dncnn_session() -> ort.InferenceSession:
    """Lazy-load DnCNN ONNX session."""
    global _dncnn_session
    if _dncnn_session is None:
        model_path = _MODELS_DIR / "dncnn.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"DnCNN model not found at {model_path}. "
                f"Place the ONNX file in backend/models/ to enable this endpoint."
            )
        _dncnn_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("DnCNN ONNX session loaded from %s", model_path)
    return _dncnn_session


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


def _compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    Returns PSNR in dB. Higher is better.
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(255.0 ** 2 / mse))


# ---------------------------------------------------------------------------
# SRCNN preprocessing / postprocessing
# ---------------------------------------------------------------------------

def _srcnn_preprocess(img: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare image for SRCNN inference.
    1. Convert to YCrCb — SRCNN works on luminance (Y) only.
    2. Bicubic-upsample the Y channel by `scale`.
    3. Normalize to [0, 1] float32.
    4. Reshape to NCHW tensor: (1, 1, H, W).
    Returns (y_tensor, ycrcb_upscaled) so CbCr can be merged later.
    """
    # Convert BGR → YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale

    # Bicubic upsample full YCrCb (CbCr channels needed later)
    ycrcb_up = cv2.resize(ycrcb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Extract Y channel, normalize to [0, 1]
    y_channel = ycrcb_up[:, :, 0].astype(np.float32) / 255.0

    # Reshape to NCHW
    y_tensor = y_channel[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    return y_tensor, ycrcb_up


def _srcnn_postprocess(
    y_output: np.ndarray, ycrcb_up: np.ndarray
) -> np.ndarray:
    """
    Post-process SRCNN output.
    1. Squeeze output tensor → 2D.
    2. Clip to [0, 1], scale to [0, 255], cast to uint8.
    3. Replace Y channel in the upscaled YCrCb.
    4. Convert back to BGR.
    """
    # Squeeze: (1, 1, H, W) → (H, W)
    y_sr = y_output.squeeze()

    # Clip and scale
    y_sr = np.clip(y_sr, 0.0, 1.0)
    y_sr = (y_sr * 255.0).astype(np.uint8)

    # Replace Y channel
    ycrcb_up[:, :, 0] = y_sr

    # Convert back to BGR
    result = cv2.cvtColor(ycrcb_up, cv2.COLOR_YCrCb2BGR)
    return result


# ---------------------------------------------------------------------------
# DnCNN preprocessing / postprocessing
# ---------------------------------------------------------------------------

def _dncnn_preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare image for DnCNN inference.
    1. Convert to grayscale.
    2. Normalize to [0, 1] float32.
    3. Reshape to NCHW tensor: (1, 1, H, W).
    Returns (tensor, grayscale_uint8).
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img

    gray_f32 = gray.astype(np.float32) / 255.0
    tensor = gray_f32[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    return tensor, gray


def _dncnn_postprocess(
    noise_residual: np.ndarray, input_tensor: np.ndarray
) -> np.ndarray:
    """
    Post-process DnCNN output.
    DnCNN predicts the noise residual: denoised = input − residual.
    1. Squeeze tensors.
    2. Subtract residual from input.
    3. Clip to [0, 1], scale to [0, 255], cast to uint8.
    """
    residual = noise_residual.squeeze()
    clean = input_tensor.squeeze() - residual

    clean = np.clip(clean, 0.0, 1.0)
    return (clean * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. POST /super-resolution — SRCNN Upscaling
# ---------------------------------------------------------------------------

@router.post("/super-resolution")
async def super_resolution(
    file: UploadFile = File(...),
    scale: int = Form(2),
):
    """
    SRCNN super-resolution inference.
    Architecture: 3 convolutional layers (64 → 32 → 1 channels).
    Trained on DIV2K dataset, exported as ONNX.

    Params:
        scale — Upscaling factor: 2 or 4 (default 2).

    Pipeline:
        1. Convert to YCrCb, bicubic-upsample Y channel.
        2. Run SRCNN on upsampled Y (luminance) — ONNX inference.
        3. Merge enhanced Y with upsampled CbCr, convert to BGR.

    Returns upscaled image + metadata (scale, sizes, time_ms).
    """
    try:
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4.")

        # Load ONNX session (lazy, cached)
        try:
            session = _get_srcnn_session(scale)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

        start = time.time()
        img = await _read_image(file)
        orig_h, orig_w = img.shape[:2]

        # Preprocess
        y_tensor, ycrcb_up = _srcnn_preprocess(img, scale)

        # ONNX inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        y_output = session.run([output_name], {input_name: y_tensor})[0]

        # Postprocess
        result = _srcnn_postprocess(y_output, ycrcb_up)
        new_h, new_w = result.shape[:2]

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result, {
            "operation": "super_resolution",
            "model": f"srcnn_x{scale}",
            "params": {"scale": scale},
            "original_size": [orig_w, orig_h],
            "new_size": [new_w, new_h],
            "time_ms": elapsed,
        })
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("super_resolution failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 2. POST /denoise — DnCNN Blind Denoising
# ---------------------------------------------------------------------------

@router.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    """
    DnCNN blind denoising inference.
    Architecture: 17 layers (Conv + BatchNorm + ReLU).
    Trained on BSD68 dataset, exported as ONNX.

    Pipeline:
        1. Convert to grayscale, normalize to [0, 1].
        2. Run DnCNN — predicts the noise residual.
        3. Subtract residual from input to get clean image.
        4. Compute PSNR estimate (noisy vs. denoised).

    Returns denoised grayscale image + PSNR + metadata.
    """
    try:
        # Load ONNX session (lazy, cached)
        try:
            session = _get_dncnn_session()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

        start = time.time()
        img = await _read_image(file)

        # Preprocess
        input_tensor, gray_original = _dncnn_preprocess(img)

        # ONNX inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        noise_residual = session.run(
            [output_name], {input_name: input_tensor}
        )[0]

        # Postprocess
        denoised = _dncnn_postprocess(noise_residual, input_tensor)

        # PSNR: compare denoised vs original noisy input
        psnr_value = _compute_psnr(gray_original, denoised)

        # Convert grayscale result to BGR for consistent response format
        result_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        elapsed = round((time.time() - start) * 1000, 2)
        return _make_response(result_bgr, {
            "operation": "denoise",
            "model": "dncnn",
            "params": {},
            "image_shape": list(gray_original.shape),
            "psnr_db": round(psnr_value, 2) if psnr_value != float('inf') else "inf",
            "time_ms": elapsed,
        })
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("denoise failed")
        raise HTTPException(status_code=500, detail=str(e))
