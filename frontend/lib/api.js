/**
 * ResearchVisionPro — API Helper
 *
 * Base URL: http://localhost:8000
 * All endpoints accept POST with multipart/form-data.
 * All responses return { image: <base64_png>, metadata: { ... } }
 */

const BASE_URL = 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Generic request helper
// ---------------------------------------------------------------------------

/**
 * Send a multipart/form-data POST request to the backend.
 * @param {string}   endpoint  — API path (e.g. "/api/spatial/histogram-eq")
 * @param {File}     file      — Image file to upload
 * @param {Object}   params    — Additional form fields (key-value pairs)
 * @returns {Promise<{ image: string, metadata: object }>}
 */
async function request(endpoint, file, params = {}) {
    const formData = new FormData()
    if (file) {
        formData.append('file', file)
    }
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            formData.append(key, value)
        }
    })

    const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Processing failed' }))
        throw new Error(error.detail || 'Processing failed')
    }

    return response.json()
}

// ---------------------------------------------------------------------------
// Spatial Processing — /api/spatial/  (Ch. 3)
// ---------------------------------------------------------------------------

export const spatialAPI = {
    /** Histogram equalization (Ch. 3.3) */
    async histogramEq(file) {
        return request('/api/spatial/histogram-eq', file)
    },

    /** CLAHE — Contrast Limited Adaptive Histogram Equalization (Ch. 3.3) */
    async clahe(file, clipLimit = 2.0, tileSize = 8) {
        return request('/api/spatial/clahe', file, {
            clip_limit: clipLimit,
            tile_size: tileSize,
        })
    },

    /** Linear contrast stretching (Ch. 3.2) */
    async contrastStretch(file, lowPercentile = 2.0, highPercentile = 98.0) {
        return request('/api/spatial/contrast-stretch', file, {
            low_percentile: lowPercentile,
            high_percentile: highPercentile,
        })
    },

    /** Gamma (power-law) correction (Ch. 3.2) */
    async gamma(file, gamma = 1.0) {
        return request('/api/spatial/gamma', file, { gamma })
    },

    /** Logarithmic transformation (Ch. 3.2) */
    async logTransform(file) {
        return request('/api/spatial/log-transform', file)
    },

    /** Spatial filtering — mean, gaussian, median, laplacian, sobel (Ch. 3.5–3.6) */
    async filter(file, filterType = 'mean', kernelSize = 3) {
        return request('/api/spatial/filter', file, {
            filter_type: filterType,
            kernel_size: kernelSize,
        })
    },

    /** Unsharp masking (Ch. 3.6) */
    async unsharpMask(file, radius = 2.0, amount = 1.5) {
        return request('/api/spatial/unsharp-mask', file, { radius, amount })
    },
}

// ---------------------------------------------------------------------------
// Frequency Domain Processing — /api/frequency/  (Ch. 4)
// ---------------------------------------------------------------------------

export const frequencyAPI = {
    /** Compute 2D FFT magnitude spectrum (Ch. 4.3) */
    async fft(file) {
        return request('/api/frequency/fft', file)
    },

    /**
     * Apply frequency domain filter (Ch. 4.7–4.10)
     * filterType: "ideal_low" | "ideal_high" | "butterworth_low" |
     *             "butterworth_high" | "gaussian_low" | "gaussian_high" | "notch"
     */
    async filter(file, filterType = 'ideal_low', cutoff = 0.1, order = 2, notchCenterU = 0, notchCenterV = 0) {
        return request('/api/frequency/filter', file, {
            filter_type: filterType,
            cutoff,
            order,
            notch_center_u: notchCenterU,
            notch_center_v: notchCenterV,
        })
    },

    /** Inverse FFT reconstruction (Ch. 4.3) */
    async inverse(fftData) {
        const formData = new FormData()
        formData.append('fft_data', fftData)

        const response = await fetch(`${BASE_URL}/api/frequency/inverse`, {
            method: 'POST',
            body: formData,
        })

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Processing failed' }))
            throw new Error(error.detail || 'Processing failed')
        }

        return response.json()
    },
}

// ---------------------------------------------------------------------------
// Image Restoration — /api/restoration/  (Ch. 5)
// ---------------------------------------------------------------------------

export const restorationAPI = {
    /**
     * Add synthetic noise (Ch. 5.2)
     * noiseType: "gaussian" | "salt_pepper" | "speckle" | "poisson"
     */
    async addNoise(file, noiseType = 'gaussian', intensity = 0.05) {
        return request('/api/restoration/add-noise', file, {
            noise_type: noiseType,
            intensity,
        })
    },

    /**
     * Classical spatial denoising (Ch. 5.3)
     * denoiseType: "arithmetic_mean" | "geometric_mean" | "median" |
     *              "adaptive_median" | "contra_harmonic"
     */
    async denoiseSpatial(file, denoiseType = 'median', kernelSize = 3, q = 1.5) {
        return request('/api/restoration/denoise-spatial', file, {
            denoise_type: denoiseType,
            kernel_size: kernelSize,
            q,
        })
    },

    /** Wiener filter for deblurring (Ch. 5.8) */
    async wiener(file, noiseVariance = 0.01) {
        return request('/api/restoration/wiener', file, {
            noise_variance: noiseVariance,
        })
    },

    /** Motion blur removal via Wiener deconvolution (Ch. 5.7–5.8) */
    async motionDeblur(file, angle = 0.0, length = 15, noiseVariance = 0.001) {
        return request('/api/restoration/motion-deblur', file, {
            angle,
            length,
            noise_variance: noiseVariance,
        })
    },
}

// ---------------------------------------------------------------------------
// Color Image Processing — /api/color/  (Ch. 6)
// ---------------------------------------------------------------------------

export const colorAPI = {
    /**
     * Color space conversion (Ch. 6.2)
     * target: "HSV" | "HSI" | "LAB" | "YCbCr" | "GRAY" | "RGB"
     */
    async convert(file, target = 'HSV') {
        return request('/api/color/convert', file, { target })
    },

    /**
     * Pseudocolor / false-color mapping (Ch. 6.3)
     * colormap: "jet" | "hot" | "cool" | "viridis" | "rainbow" | "plasma"
     */
    async falseColor(file, colormap = 'jet') {
        return request('/api/color/false-color', file, { colormap })
    },

    /** Per-channel histogram equalization (Ch. 6.5) */
    async histogramEqColor(file) {
        return request('/api/color/histogram-eq-color', file)
    },

    /**
     * HSV-based color segmentation (Ch. 6.6)
     * All params normalized 0–1.
     */
    async colorSegment(file, hueMin = 0.0, hueMax = 1.0, satMin = 0.0, satMax = 1.0) {
        return request('/api/color/color-segment', file, {
            hue_min: hueMin,
            hue_max: hueMax,
            sat_min: satMin,
            sat_max: satMax,
        })
    },

    /** Split into individual B, G, R channels (Ch. 6.1) */
    async channelSplit(file) {
        return request('/api/color/channel-split', file)
    },
}

// ---------------------------------------------------------------------------
// Morphological Processing — /api/morphology/  (Ch. 9)
// ---------------------------------------------------------------------------

export const morphologyAPI = {
    /** Erosion (Ch. 9.1) */
    async erode(file, kernelSize = 3, kernelShape = 'rect', iterations = 1) {
        return request('/api/morphology/erode', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
            iterations,
        })
    },

    /** Dilation (Ch. 9.1) */
    async dilate(file, kernelSize = 3, kernelShape = 'rect', iterations = 1) {
        return request('/api/morphology/dilate', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
            iterations,
        })
    },

    /** Opening — erosion → dilation (Ch. 9.2) */
    async open(file, kernelSize = 3, kernelShape = 'rect', iterations = 1) {
        return request('/api/morphology/open', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
            iterations,
        })
    },

    /** Closing — dilation → erosion (Ch. 9.2) */
    async close(file, kernelSize = 3, kernelShape = 'rect', iterations = 1) {
        return request('/api/morphology/close', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
            iterations,
        })
    },

    /** Morphological gradient — dilation − erosion (Ch. 9.4) */
    async gradient(file, kernelSize = 3, kernelShape = 'rect') {
        return request('/api/morphology/gradient', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
        })
    },

    /** Top-hat (white) transform — original − opening (Ch. 9.4) */
    async topHat(file, kernelSize = 9, kernelShape = 'rect') {
        return request('/api/morphology/tophat', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
        })
    },

    /** Black-hat transform — closing − original (Ch. 9.4) */
    async blackHat(file, kernelSize = 9, kernelShape = 'rect') {
        return request('/api/morphology/blackhat', file, {
            kernel_size: kernelSize,
            kernel_shape: kernelShape,
        })
    },

    /** Skeletonization / thinning (Ch. 9.5) */
    async skeleton(file, kernelShape = 'cross') {
        return request('/api/morphology/skeleton', file, {
            kernel_shape: kernelShape,
        })
    },
}

// ---------------------------------------------------------------------------
// Image Segmentation — /api/segmentation/  (Ch. 10)
// ---------------------------------------------------------------------------

export const segmentationAPI = {
    /**
     * Thresholding (Ch. 10.3)
     * thresholdType: "global" | "otsu" | "adaptive_mean" | "adaptive_gaussian"
     */
    async threshold(file, thresholdType = 'otsu', thresholdValue = 128, blockSize = 11, cValue = 2) {
        return request('/api/segmentation/threshold', file, {
            threshold_type: thresholdType,
            threshold_value: thresholdValue,
            block_size: blockSize,
            c_value: cValue,
        })
    },

    /**
     * Edge detection (Ch. 10.2)
     * edgeType: "sobel" | "prewitt" | "roberts" | "canny" | "laplacian"
     */
    async edgeDetect(file, edgeType = 'canny', lowThreshold = 50.0, highThreshold = 150.0, kernelSize = 3) {
        return request('/api/segmentation/edge-detect', file, {
            edge_type: edgeType,
            low_threshold: lowThreshold,
            high_threshold: highThreshold,
            kernel_size: kernelSize,
        })
    },

    /** Hough line detection (Ch. 10.2) */
    async houghLines(file, cannyLow = 50.0, cannyHigh = 150.0, threshold = 100, minLineLength = 50, maxLineGap = 10) {
        return request('/api/segmentation/hough-lines', file, {
            canny_low: cannyLow,
            canny_high: cannyHigh,
            threshold,
            min_line_length: minLineLength,
            max_line_gap: maxLineGap,
        })
    },

    /** Hough circle detection (Ch. 10.2) */
    async houghCircles(file, dp = 1.2, minDist = 50, param1 = 100.0, param2 = 30.0, minRadius = 10, maxRadius = 200) {
        return request('/api/segmentation/hough-circles', file, {
            dp,
            min_dist: minDist,
            param1,
            param2,
            min_radius: minRadius,
            max_radius: maxRadius,
        })
    },

    /** Region growing segmentation (Ch. 10.4) */
    async regionGrow(file, seedX = 0, seedY = 0, tolerance = 15.0) {
        return request('/api/segmentation/region-grow', file, {
            seed_x: seedX,
            seed_y: seedY,
            tolerance,
        })
    },

    /** Marker-based watershed segmentation (Ch. 10.5) */
    async watershed(file) {
        return request('/api/segmentation/watershed', file)
    },

    /** Connected component labeling (Ch. 10.4) */
    async connectedComponents(file, connectivity = 8) {
        return request('/api/segmentation/connected-components', file, {
            connectivity,
        })
    },
}

// ---------------------------------------------------------------------------
// Deep Learning Inference — /api/dl/
// ---------------------------------------------------------------------------

export const deepLearningAPI = {
    /** SRCNN super-resolution ×2 or ×4 */
    async superResolution(file, scale = 2) {
        return request('/api/dl/super-resolution', file, { scale })
    },

    /** DnCNN blind denoising */
    async denoise(file) {
        return request('/api/dl/denoise', file)
    },
}

// ---------------------------------------------------------------------------
// Remote Sensing — /api/remote/
// ---------------------------------------------------------------------------

export const remoteAPI = {
    /** NDVI computation from NIR + Red bands */
    async ndvi(file, nirBand = 0, redBand = 1) {
        return request('/api/remote/ndvi', file, {
            nir_band: nirBand,
            red_band: redBand,
        })
    },

    /** False-color band composite */
    async bandComposite(file, redBand = 0, greenBand = 1, blueBand = 2) {
        return request('/api/remote/band-composite', file, {
            red_band: redBand,
            green_band: greenBand,
            blue_band: blueBand,
        })
    },

    /** Per-band statistics */
    async bandStats(file) {
        return request('/api/remote/band-stats', file)
    },

    /** Display a single band */
    async bandViewer(file, bandIndex = 0) {
        return request('/api/remote/band-viewer', file, {
            band_index: bandIndex,
        })
    },

    /** CLAHE for satellite imagery */
    async enhanceSatellite(file, clipLimit = 3.0, tileSize = 16) {
        return request('/api/remote/enhance-satellite', file, {
            clip_limit: clipLimit,
            tile_size: tileSize,
        })
    },

    /** Spectral profile at pixel (x, y) */
    async spectralProfile(file, x = 0, y = 0) {
        return request('/api/remote/spectral-profile', file, { x, y })
    },
}
