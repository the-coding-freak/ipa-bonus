"""
ResearchVisionPro — Endpoint Integration Test Suite
=====================================================
Tests EVERY endpoint across all processing modules by sending
real HTTP requests to the running FastAPI server at localhost:8000.

Usage:
    1. Start the backend:  uvicorn main:app --reload
    2. Run this script:    python test_all_endpoints.py

Generates synthetic test images (no external files needed).
"""

import io
import sys
import time
from dataclasses import dataclass, field

# Force UTF-8 output on Windows (avoids cp1252 encoding errors with emoji)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import requests
import tifffile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
TIMEOUT = 60  # seconds per request


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    module: str
    endpoint: str
    status: str = "SKIP"
    time_ms: float = 0.0
    detail: str = ""


results: list[TestResult] = []


# ---------------------------------------------------------------------------
# Test image generators
# ---------------------------------------------------------------------------

def _make_color_image(width: int = 256, height: int = 256) -> bytes:
    """
    Create a synthetic 256×256 BGR test image with gradients, shapes,
    and edges — good for exercising most processing algorithms.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Horizontal gradient on Blue channel
    img[:, :, 0] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    # Vertical gradient on Green channel
    img[:, :, 1] = np.tile(
        np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width)
    )
    # Diagonal on Red
    for y in range(height):
        for x in range(width):
            img[y, x, 2] = (x + y) % 256

    # Draw shapes to create edges for edge detectors / Hough
    cv2.rectangle(img, (30, 30), (100, 100), (255, 255, 255), 2)
    cv2.circle(img, (180, 180), 40, (255, 255, 255), 2)
    cv2.line(img, (10, 200), (200, 10), (255, 255, 255), 2)

    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_grayscale_image(width: int = 256, height: int = 256) -> bytes:
    """Create a synthetic grayscale test image."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[:, :] = np.tile(np.linspace(20, 235, width, dtype=np.uint8), (height, 1))
    cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    cv2.circle(img, (128, 128), 30, 0, -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_multiband_tiff(bands: int = 4, width: int = 128, height: int = 128) -> bytes:
    """
    Create a synthetic multi-band TIFF (e.g. 4-band: B, G, R, NIR).
    Returns raw TIFF bytes.
    """
    data = np.zeros((bands, height, width), dtype=np.uint16)
    for b in range(bands):
        # Each band gets a different gradient + offset
        base = np.linspace(100 + b * 200, 500 + b * 300, width, dtype=np.float64)
        data[b] = np.tile(base.astype(np.uint16), (height, 1))
        # Add some spatial variation
        data[b] += np.random.randint(0, 50, (height, width), dtype=np.uint16)

    buf = io.BytesIO()
    tifffile.imwrite(buf, data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper to run a single test
# ---------------------------------------------------------------------------

def run_test(
    module: str,
    name: str,
    url: str,
    files: dict | None = None,
    data: dict | None = None,
    expect_status: int = 200,
) -> requests.Response | None:
    """POST to an endpoint, record pass/fail, return the response."""
    result = TestResult(module=module, endpoint=name)
    try:
        start = time.time()
        resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        result.time_ms = elapsed_ms

        if resp.status_code == expect_status:
            result.status = "PASS"
            result.detail = f"{resp.status_code}"
        else:
            result.status = "FAIL"
            try:
                detail = resp.json().get("detail", resp.text[:120])
            except Exception:
                detail = resp.text[:120]
            result.detail = f"HTTP {resp.status_code}: {detail}"

        results.append(result)
        _print_result(result)
        return resp

    except requests.ConnectionError:
        result.status = "FAIL"
        result.detail = "Connection refused — is the server running?"
        results.append(result)
        _print_result(result)
        return None
    except Exception as e:
        result.status = "FAIL"
        result.detail = str(e)[:120]
        results.append(result)
        _print_result(result)
        return None


def _print_result(r: TestResult):
    icon = "✅" if r.status == "PASS" else "❌"
    print(f"  {icon} {r.endpoint:<40s}  {r.time_ms:>8.1f} ms  {r.detail}")


# ---------------------------------------------------------------------------
# Module test functions
# ---------------------------------------------------------------------------

def test_health():
    print("\n🩺  Health Check")
    print("-" * 70)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        r = TestResult(module="core", endpoint="GET /health")
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            r.status = "PASS"
            r.detail = "200"
        else:
            r.status = "FAIL"
            r.detail = f"HTTP {resp.status_code}"
        results.append(r)
        _print_result(r)
    except Exception as e:
        r = TestResult(module="core", endpoint="GET /health", status="FAIL", detail=str(e)[:80])
        results.append(r)
        _print_result(r)
        print("\n⛔  Server is not reachable. Aborting.\n")
        sys.exit(1)


def test_spatial(img_bytes: bytes):
    print("\n📐  Spatial Processing  (/api/spatial)")
    print("-" * 70)
    url = f"{BASE_URL}/api/spatial"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}

    run_test("spatial", "POST /spatial/histogram-eq", f"{url}/histogram-eq", files=f())
    run_test("spatial", "POST /spatial/clahe", f"{url}/clahe", files=f(), data={"clip_limit": "2.0", "tile_size": "8"})
    run_test("spatial", "POST /spatial/contrast-stretch", f"{url}/contrast-stretch", files=f(), data={"low_percentile": "2", "high_percentile": "98"})
    run_test("spatial", "POST /spatial/gamma", f"{url}/gamma", files=f(), data={"gamma": "0.5"})
    run_test("spatial", "POST /spatial/log-transform", f"{url}/log-transform", files=f())
    for ftype in ["mean", "gaussian", "median", "laplacian", "sobel"]:
        run_test("spatial", f"POST /spatial/filter ({ftype})", f"{url}/filter", files=f(), data={"filter_type": ftype, "kernel_size": "3"})
    run_test("spatial", "POST /spatial/unsharp-mask", f"{url}/unsharp-mask", files=f(), data={"radius": "2.0", "amount": "1.5"})


def test_frequency(img_bytes: bytes):
    print("\n🌊  Frequency Processing  (/api/frequency)")
    print("-" * 70)
    url = f"{BASE_URL}/api/frequency"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}

    # FFT — capture fft_data for inverse test
    resp = run_test("frequency", "POST /frequency/fft", f"{url}/fft", files=f())
    fft_data = None
    if resp and resp.status_code == 200:
        fft_data = resp.json().get("fft_data")

    # Filter — all types
    for ftype in ["ideal_low", "ideal_high", "butterworth_low", "butterworth_high", "gaussian_low", "gaussian_high"]:
        run_test("frequency", f"POST /frequency/filter ({ftype})", f"{url}/filter", files=f(), data={"filter_type": ftype, "cutoff": "0.3", "order": "2"})
    run_test("frequency", "POST /frequency/filter (notch)", f"{url}/filter", files=f(), data={"filter_type": "notch", "cutoff": "0.1", "notch_center_u": "30", "notch_center_v": "30"})

    # Inverse FFT
    if fft_data:
        run_test("frequency", "POST /frequency/inverse", f"{url}/inverse", data={"fft_data": fft_data})
    else:
        r = TestResult(module="frequency", endpoint="POST /frequency/inverse", status="SKIP", detail="No fft_data from /fft")
        results.append(r)
        _print_result(r)


def test_restoration(img_bytes: bytes):
    print("\n🔧  Restoration  (/api/restoration)")
    print("-" * 70)
    url = f"{BASE_URL}/api/restoration"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}

    for ntype in ["gaussian", "salt_pepper", "speckle", "poisson"]:
        run_test("restoration", f"POST /restoration/add-noise ({ntype})", f"{url}/add-noise", files=f(), data={"noise_type": ntype, "intensity": "0.05"})

    for dtype in ["arithmetic_mean", "geometric_mean", "median", "adaptive_median", "contra_harmonic"]:
        run_test("restoration", f"POST /restoration/denoise-spatial ({dtype})", f"{url}/denoise-spatial", files=f(), data={"denoise_type": dtype, "kernel_size": "3", "q": "1.5"})

    run_test("restoration", "POST /restoration/wiener", f"{url}/wiener", files=f(), data={"noise_variance": "0.01"})
    run_test("restoration", "POST /restoration/motion-deblur", f"{url}/motion-deblur", files=f(), data={"angle": "30", "length": "15", "noise_variance": "0.001"})


def test_color(img_bytes: bytes, gray_bytes: bytes):
    print("\n🎨  Color Processing  (/api/color)")
    print("-" * 70)
    url = f"{BASE_URL}/api/color"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}
    g = lambda: {"file": ("test_gray.png", gray_bytes, "image/png")}

    for cs in ["HSV", "HSI", "LAB", "YCbCr", "GRAY", "RGB"]:
        run_test("color", f"POST /color/convert ({cs})", f"{url}/convert", files=f(), data={"target": cs})

    for cm in ["jet", "hot", "cool", "viridis", "rainbow", "plasma"]:
        run_test("color", f"POST /color/false-color ({cm})", f"{url}/false-color", files=g(), data={"colormap": cm})

    run_test("color", "POST /color/histogram-eq-color", f"{url}/histogram-eq-color", files=f())
    run_test("color", "POST /color/color-segment", f"{url}/color-segment", files=f(), data={"hue_min": "0.0", "hue_max": "0.3", "sat_min": "0.2", "sat_max": "1.0"})
    run_test("color", "POST /color/channel-split", f"{url}/channel-split", files=f())


def test_morphology(img_bytes: bytes):
    print("\n🔬  Morphology  (/api/morphology)")
    print("-" * 70)
    url = f"{BASE_URL}/api/morphology"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}
    common = {"kernel_size": "5", "kernel_shape": "rect", "iterations": "1"}

    run_test("morphology", "POST /morphology/erode", f"{url}/erode", files=f(), data=common)
    run_test("morphology", "POST /morphology/dilate", f"{url}/dilate", files=f(), data=common)
    run_test("morphology", "POST /morphology/open", f"{url}/open", files=f(), data=common)
    run_test("morphology", "POST /morphology/close", f"{url}/close", files=f(), data=common)
    run_test("morphology", "POST /morphology/gradient", f"{url}/gradient", files=f(), data={"kernel_size": "5", "kernel_shape": "ellipse"})
    run_test("morphology", "POST /morphology/tophat", f"{url}/tophat", files=f(), data={"kernel_size": "9", "kernel_shape": "rect"})
    run_test("morphology", "POST /morphology/blackhat", f"{url}/blackhat", files=f(), data={"kernel_size": "9", "kernel_shape": "cross"})
    run_test("morphology", "POST /morphology/skeleton", f"{url}/skeleton", files=f(), data={"kernel_shape": "cross"})


def test_segmentation(img_bytes: bytes):
    print("\n✂️  Segmentation  (/api/segmentation)")
    print("-" * 70)
    url = f"{BASE_URL}/api/segmentation"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}

    for tt in ["global", "otsu", "adaptive_mean", "adaptive_gaussian"]:
        run_test("segmentation", f"POST /segmentation/threshold ({tt})", f"{url}/threshold", files=f(), data={"threshold_type": tt, "threshold_value": "128", "block_size": "11", "c_value": "2"})

    for et in ["sobel", "prewitt", "roberts", "canny", "laplacian"]:
        run_test("segmentation", f"POST /segmentation/edge-detect ({et})", f"{url}/edge-detect", files=f(), data={"edge_type": et, "low_threshold": "50", "high_threshold": "150", "kernel_size": "3"})

    run_test("segmentation", "POST /segmentation/hough-lines", f"{url}/hough-lines", files=f(), data={"canny_low": "50", "canny_high": "150", "threshold": "50", "min_line_length": "30", "max_line_gap": "10"})
    run_test("segmentation", "POST /segmentation/hough-circles", f"{url}/hough-circles", files=f(), data={"dp": "1.2", "min_dist": "30", "param1": "100", "param2": "30", "min_radius": "10", "max_radius": "100"})
    run_test("segmentation", "POST /segmentation/region-grow", f"{url}/region-grow", files=f(), data={"seed_x": "128", "seed_y": "128", "tolerance": "20"})
    run_test("segmentation", "POST /segmentation/watershed", f"{url}/watershed", files=f())
    run_test("segmentation", "POST /segmentation/connected-components", f"{url}/connected-components", files=f(), data={"connectivity": "8"})


def test_deeplearning(img_bytes: bytes):
    print("\n🧠  Deep Learning  (/api/dl)")
    print("-" * 70)
    url = f"{BASE_URL}/api/dl"
    f = lambda: {"file": ("test.png", img_bytes, "image/png")}

    # Super-resolution (may return 503 if model files missing — still a valid test)
    for scale in [2, 4]:
        run_test("deeplearning", f"POST /dl/super-resolution (x{scale})", f"{url}/super-resolution", files=f(), data={"scale": str(scale)})

    run_test("deeplearning", "POST /dl/denoise", f"{url}/denoise", files=f())


def test_remote_sensing(tiff_bytes: bytes):
    print("\n🛰️  Remote Sensing  (/api/remote)")
    print("-" * 70)
    url = f"{BASE_URL}/api/remote"
    f = lambda: {"file": ("test.tif", tiff_bytes, "image/tiff")}

    run_test("remote", "POST /remote/ndvi", f"{url}/ndvi", files=f(), data={"nir_band": "3", "red_band": "2"})
    run_test("remote", "POST /remote/band-composite", f"{url}/band-composite", files=f(), data={"red_band": "2", "green_band": "1", "blue_band": "0"})
    run_test("remote", "POST /remote/band-stats", f"{url}/band-stats", files=f())
    run_test("remote", "POST /remote/band-viewer", f"{url}/band-viewer", files=f(), data={"band_index": "0"})
    run_test("remote", "POST /remote/enhance-satellite", f"{url}/enhance-satellite", files=f(), data={"clip_limit": "3.0", "tile_size": "16"})
    run_test("remote", "POST /remote/spectral-profile", f"{url}/spectral-profile", files=f(), data={"x": "64", "y": "64"})


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n")
    print("=" * 80)
    print("   RESEARCHVISIONPRO — TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    # Group by module
    modules = {}
    for r in results:
        modules.setdefault(r.module, []).append(r)

    print(f"\n{'Module':<16} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Total':>8}")
    print("-" * 56)
    for mod, mod_results in modules.items():
        mp = sum(1 for r in mod_results if r.status == "PASS")
        mf = sum(1 for r in mod_results if r.status == "FAIL")
        ms = sum(1 for r in mod_results if r.status == "SKIP")
        mt = len(mod_results)
        print(f"{mod:<16} {mp:>8} {mf:>8} {ms:>8} {mt:>8}")
    print("-" * 56)
    print(f"{'TOTAL':<16} {passed:>8} {failed:>8} {skipped:>8} {total:>8}")

    # Failures detail
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"\n❌ Failed tests ({len(failures)}):")
        for r in failures:
            print(f"   • [{r.module}] {r.endpoint}")
            print(f"     {r.detail}")

    avg_time = sum(r.time_ms for r in results) / max(total, 1)
    total_time = sum(r.time_ms for r in results) / 1000
    print(f"\n⏱  Avg response time: {avg_time:.0f} ms")
    print(f"⏱  Total test time:   {total_time:.1f} s")

    if failed == 0:
        print(f"\n🎉  ALL {passed} TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} / {total} tests FAILED.")

    print("=" * 80)
    return failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("   RESEARCHVISIONPRO — ENDPOINT INTEGRATION TESTS")
    print(f"   Target: {BASE_URL}")
    print("=" * 80)

    # Generate test images
    print("\n🖼️  Generating synthetic test images …")
    color_img = _make_color_image()
    gray_img = _make_grayscale_image()
    tiff_img = _make_multiband_tiff(bands=4)
    print(f"   Color PNG:  {len(color_img):,} bytes")
    print(f"   Gray PNG:   {len(gray_img):,} bytes")
    print(f"   4-band TIFF: {len(tiff_img):,} bytes")

    # Run tests
    test_health()
    test_spatial(color_img)
    test_frequency(color_img)
    test_restoration(color_img)
    test_color(color_img, gray_img)
    test_morphology(color_img)
    test_segmentation(color_img)
    test_deeplearning(color_img)
    test_remote_sensing(tiff_img)

    # Summary
    failed = print_summary()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
