"""
ResearchVisionPro — FastAPI Backend
====================================
Entry point for the FastAPI application.

Architecture: Next.js (3000) ↔ FastAPI (8000) ↔ ONNX / OpenCV / NumPy
Run with:  uvicorn main:app --reload
Docs:      http://localhost:8000/docs
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------------------
# Processing routers (one per Gonzalez & Woods chapter / domain)
# ---------------------------------------------------------------------------
from processing import spatial       # Ch. 3  — Spatial Domain
from processing import frequency     # Ch. 4  — Frequency Domain
from processing import restoration   # Ch. 5  — Image Restoration
from processing import color         # Ch. 6  — Color Processing
from processing import morphology    # Ch. 9  — Morphological Processing
from processing import segmentation  # Ch. 10 — Image Segmentation
from processing import deeplearning  # SRCNN + DnCNN ONNX inference
from processing import remote_sensing  # NDVI, band composite, HSI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("researchvisionpro")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/webp",
    "image/tiff", "image/gif",
}

# ---------------------------------------------------------------------------
# ONNX model presence check at startup
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).parent / "models"
_EXPECTED_MODELS = ["srcnn_x2.onnx", "srcnn_x4.onnx", "dncnn.onnx"]


def _check_models() -> None:
    """
    Warn if expected ONNX model files are missing.
    The app still starts — endpoints that need models raise 503 at call time.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name in _EXPECTED_MODELS:
        model_path = _MODELS_DIR / model_name
        if model_path.exists():
            logger.info("✅ Model found: %s", model_path.name)
        else:
            logger.warning(
                "⚠️  Model NOT found: %s  "
                "(place the ONNX file in backend/models/ before using DL endpoints)",
                model_path,
            )


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown events
# ---------------------------------------------------------------------------
async def _startup() -> None:
    """Run all startup tasks. Wrapped in a timeout to catch hangs."""
    _check_models()
    logger.info("📡 API docs available at http://localhost:8000/docs")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("🚀 ResearchVisionPro backend starting up ...")
    try:
        await asyncio.wait_for(_startup(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("❌ Startup timeout - a router or dependency is hanging")
        raise
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("🛑 ResearchVisionPro backend shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app creation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ResearchVisionPro API",
    description=(
        "Research-grade image processing backend. "
        "Implements techniques from Gonzalez & Woods — "
        "Digital Image Processing, 4th Edition."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Next.js dev server on port 3000
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return structured JSON for all HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "detail": str(exc.detail),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured JSON for request validation errors (422)."""
    errors = exc.errors()
    detail = "; ".join(
        f"{e.get('loc', ['?'])[-1]}: {e.get('msg', 'invalid')}" for e in errors
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": detail,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — never leak raw tracebacks."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Check server logs for details.",
        },
    )


# ---------------------------------------------------------------------------
# Middleware: Request logging + file validation
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with endpoint, method, content size, and duration."""
    start = time.time()
    path = request.url.path
    method = request.method

    # ── File type & size validation for POST endpoints with uploads ──
    if method == "POST" and path.startswith("/api/"):
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            # Check Content-Length header for early size rejection
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                size_mb = round(int(content_length) / (1024 * 1024), 1)
                elapsed = round((time.time() - start) * 1000, 2)
                logger.warning(
                    "⛔ %s %s — REJECTED (file too large: %sMB > %sMB) [%sms]",
                    method, path, size_mb, MAX_FILE_SIZE_MB, elapsed,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "File Too Large",
                        "detail": f"File size ({size_mb}MB) exceeds the {MAX_FILE_SIZE_MB}MB limit.",
                    },
                )

    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)

    # Log level based on status code
    status = response.status_code
    content_length = request.headers.get("content-length", "?")
    if status >= 500:
        logger.error(
            "❌ %s %s → %d  [%sms]  body=%sB",
            method, path, status, elapsed, content_length,
        )
    elif status >= 400:
        logger.warning(
            "⚠️  %s %s → %d  [%sms]  body=%sB",
            method, path, status, elapsed, content_length,
        )
    else:
        logger.info(
            "✅ %s %s → %d  [%sms]  body=%sB",
            method, path, status, elapsed, content_length,
        )

    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
_routers = [
    ("spatial", spatial.router),
    ("frequency", frequency.router),
    ("restoration", restoration.router),
    ("color", color.router),
    ("morphology", morphology.router),
    ("segmentation", segmentation.router),
    ("deeplearning", deeplearning.router),
    ("remote_sensing", remote_sensing.router),
]

for _name, _router in _routers:
    try:
        logger.info("Including router: %s", _name)
        app.include_router(_router)
        logger.info("✅ Router included: %s", _name)
    except Exception as _exc:
        logger.error("❌ Failed to include router %s: %s", _name, _exc, exc_info=True)
        raise

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple liveness probe.
    Returns { "status": "ok" } when the backend is running.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    """API root — redirects info to /docs."""
    return {
        "project": "ResearchVisionPro",
        "version": "0.1.0",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
    }
