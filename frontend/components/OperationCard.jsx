'use client'

import { useState, useEffect } from 'react'
import ParameterSlider from './ParameterSlider'

// ─────────────────────────────────────────────────────────────
// Operation definitions — maps (category, operation) → config
// ─────────────────────────────────────────────────────────────

export const OPERATIONS = {
    // ═══════════════════════════════════════════════════════════
    // Spatial Processing — Ch. 3
    // ═══════════════════════════════════════════════════════════
    'spatial/histogram-eq': {
        title: 'Histogram Equalization',
        chapter: 'Ch. 3.3',
        description: 'Enhance contrast by equalizing the intensity histogram.',
        params: [],
    },
    'spatial/clahe': {
        title: 'CLAHE',
        chapter: 'Ch. 3.3',
        description: 'Contrast Limited Adaptive Histogram Equalization.',
        params: [
            { key: 'clipLimit', type: 'slider', label: 'Clip Limit', min: 1, max: 10, step: 0.5, default: 2.0 },
            { key: 'tileSize', type: 'select', label: 'Tile Size', options: [4, 8, 16, 32], default: 8 },
        ],
    },
    'spatial/contrast-stretch': {
        title: 'Contrast Stretch',
        chapter: 'Ch. 3.2',
        description: 'Linear contrast stretching with percentile limits.',
        params: [
            { key: 'lowPercentile', type: 'slider', label: 'Low Percentile', min: 0, max: 10, step: 0.5, default: 2.0, unit: '%' },
            { key: 'highPercentile', type: 'slider', label: 'High Percentile', min: 90, max: 100, step: 0.5, default: 98.0, unit: '%' },
        ],
    },
    'spatial/gamma': {
        title: 'Gamma Correction',
        chapter: 'Ch. 3.2',
        description: 'Power-law intensity transform. <1 brightens, >1 darkens.',
        params: [
            { key: 'gamma', type: 'slider', label: 'Gamma (γ)', min: 0.1, max: 3.0, step: 0.05, default: 1.0 },
        ],
    },
    'spatial/log-transform': {
        title: 'Log Transform',
        chapter: 'Ch. 3.2',
        description: 'Logarithmic transform: expands dark, compresses bright.',
        params: [],
    },
    'spatial/filter': {
        title: 'Spatial Filter',
        chapter: 'Ch. 3.5',
        description: 'Apply a spatial domain filter to the image.',
        params: [
            { key: 'filterType', type: 'select', label: 'Filter Type', options: ['mean', 'gaussian', 'median', 'laplacian', 'sobel'], default: 'mean' },
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
        ],
    },
    'spatial/unsharp-mask': {
        title: 'Unsharp Mask',
        chapter: 'Ch. 3.6',
        description: 'Sharpen by subtracting a blurred version.',
        params: [
            { key: 'radius', type: 'slider', label: 'Radius', min: 0.5, max: 10, step: 0.5, default: 2.0 },
            { key: 'amount', type: 'slider', label: 'Amount', min: 0.1, max: 5.0, step: 0.1, default: 1.5 },
        ],
    },

    // ═══════════════════════════════════════════════════════════
    // Frequency Domain — Ch. 4
    // ═══════════════════════════════════════════════════════════
    'frequency/fft': {
        title: 'FFT Spectrum',
        chapter: 'Ch. 4.3',
        description: 'Compute and display the 2D FFT magnitude spectrum.',
        params: [],
    },
    'frequency/filter': {
        title: 'Frequency Filter',
        chapter: 'Ch. 4.7',
        description: 'Apply a frequency domain filter.',
        params: [
            { key: 'filterType', type: 'select', label: 'Filter Type', options: ['ideal_low', 'ideal_high', 'butterworth_low', 'butterworth_high', 'gaussian_low', 'gaussian_high', 'notch'], default: 'ideal_low' },
            { key: 'cutoff', type: 'slider', label: 'Cutoff (D₀)', min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
            { key: 'order', type: 'select', label: 'Order (n)', options: [1, 2, 3, 4, 5], default: 2 },
            { key: 'notchCenterU', type: 'number', label: 'Notch Center U', min: -500, max: 500, default: 0 },
            { key: 'notchCenterV', type: 'number', label: 'Notch Center V', min: -500, max: 500, default: 0 },
        ],
    },
    'frequency/inverse': {
        title: 'Inverse FFT',
        chapter: 'Ch. 4.3',
        description: 'Reconstruct spatial image from FFT data.',
        params: [],
    },

    // ═══════════════════════════════════════════════════════════
    // Restoration — Ch. 5
    // ═══════════════════════════════════════════════════════════
    'restoration/add-noise': {
        title: 'Add Noise',
        chapter: 'Ch. 5.2',
        description: 'Add synthetic noise for testing restoration.',
        params: [
            { key: 'noiseType', type: 'select', label: 'Noise Type', options: ['gaussian', 'salt_pepper', 'speckle', 'poisson'], default: 'gaussian' },
            { key: 'intensity', type: 'slider', label: 'Intensity', min: 0.01, max: 0.5, step: 0.01, default: 0.05 },
        ],
    },
    'restoration/denoise-spatial': {
        title: 'Spatial Denoise',
        chapter: 'Ch. 5.3',
        description: 'Classical denoising via spatial filters.',
        params: [
            { key: 'denoiseType', type: 'select', label: 'Filter Type', options: ['arithmetic_mean', 'geometric_mean', 'median', 'adaptive_median', 'contra_harmonic'], default: 'median' },
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'q', type: 'slider', label: 'Q (contra-harmonic)', min: -3, max: 3, step: 0.5, default: 1.5 },
        ],
    },
    'restoration/wiener': {
        title: 'Wiener Filter',
        chapter: 'Ch. 5.8',
        description: 'Wiener deconvolution for noise reduction.',
        params: [
            { key: 'noiseVariance', type: 'slider', label: 'Noise Variance', min: 0.001, max: 0.1, step: 0.001, default: 0.01 },
        ],
    },
    'restoration/motion-deblur': {
        title: 'Motion Deblur',
        chapter: 'Ch. 5.7',
        description: 'Remove motion blur via Wiener deconvolution.',
        params: [
            { key: 'angle', type: 'slider', label: 'Angle', min: 0, max: 180, step: 1, default: 0, unit: '°' },
            { key: 'length', type: 'slider', label: 'Blur Length', min: 1, max: 50, step: 1, default: 15, unit: 'px' },
            { key: 'noiseVariance', type: 'slider', label: 'Noise Variance', min: 0.0001, max: 0.01, step: 0.0001, default: 0.001 },
        ],
    },

    // ═══════════════════════════════════════════════════════════
    // Color Processing — Ch. 6
    // ═══════════════════════════════════════════════════════════
    'color/convert': {
        title: 'Color Convert',
        chapter: 'Ch. 6.2',
        description: 'Convert to a different color space.',
        params: [
            { key: 'target', type: 'select', label: 'Target Space', options: ['HSV', 'HSI', 'LAB', 'YCbCr', 'GRAY', 'RGB'], default: 'HSV' },
        ],
    },
    'color/false-color': {
        title: 'False Color',
        chapter: 'Ch. 6.3',
        description: 'Apply pseudocolor mapping to grayscale.',
        params: [
            { key: 'colormap', type: 'select', label: 'Colormap', options: ['jet', 'hot', 'cool', 'viridis', 'rainbow', 'plasma'], default: 'jet' },
        ],
    },
    'color/histogram-eq-color': {
        title: 'Histogram Eq (Color)',
        chapter: 'Ch. 6.5',
        description: 'Per-channel histogram equalization.',
        params: [],
    },
    'color/color-segment': {
        title: 'Color Segment',
        chapter: 'Ch. 6.6',
        description: 'Segment image by HSV color range.',
        params: [
            { key: 'hueMin', type: 'slider', label: 'Hue Min', min: 0, max: 1, step: 0.01, default: 0.0 },
            { key: 'hueMax', type: 'slider', label: 'Hue Max', min: 0, max: 1, step: 0.01, default: 1.0 },
            { key: 'satMin', type: 'slider', label: 'Sat Min', min: 0, max: 1, step: 0.01, default: 0.0 },
            { key: 'satMax', type: 'slider', label: 'Sat Max', min: 0, max: 1, step: 0.01, default: 1.0 },
        ],
    },
    'color/channel-split': {
        title: 'Channel Split',
        chapter: 'Ch. 6.1',
        description: 'Split into individual B, G, R channels.',
        params: [],
    },

    // ═══════════════════════════════════════════════════════════
    // Morphology — Ch. 9
    // ═══════════════════════════════════════════════════════════
    'morphology/erode': {
        title: 'Erosion',
        chapter: 'Ch. 9.1',
        description: 'Shrink bright regions / expand dark ones.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
            { key: 'iterations', type: 'slider', label: 'Iterations', min: 1, max: 10, step: 1, default: 1 },
        ],
    },
    'morphology/dilate': {
        title: 'Dilation',
        chapter: 'Ch. 9.1',
        description: 'Expand bright regions / shrink dark ones.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
            { key: 'iterations', type: 'slider', label: 'Iterations', min: 1, max: 10, step: 1, default: 1 },
        ],
    },
    'morphology/open': {
        title: 'Opening',
        chapter: 'Ch. 9.2',
        description: 'Erosion → dilation. Removes small bright noise.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
            { key: 'iterations', type: 'slider', label: 'Iterations', min: 1, max: 10, step: 1, default: 1 },
        ],
    },
    'morphology/close': {
        title: 'Closing',
        chapter: 'Ch. 9.2',
        description: 'Dilation → erosion. Fills small dark holes.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
            { key: 'iterations', type: 'slider', label: 'Iterations', min: 1, max: 10, step: 1, default: 1 },
        ],
    },
    'morphology/gradient': {
        title: 'Gradient',
        chapter: 'Ch. 9.4',
        description: 'Dilation − erosion. Extracts edges.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11], default: 3 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
        ],
    },
    'morphology/tophat': {
        title: 'Top-Hat',
        chapter: 'Ch. 9.4',
        description: 'Original − opening. Extracts small bright features.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11, 15, 21], default: 9 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
        ],
    },
    'morphology/blackhat': {
        title: 'Black-Hat',
        chapter: 'Ch. 9.4',
        description: 'Closing − original. Extracts small dark features.',
        params: [
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7, 9, 11, 15, 21], default: 9 },
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['rect', 'ellipse', 'cross'], default: 'rect' },
        ],
    },
    'morphology/skeleton': {
        title: 'Skeleton',
        chapter: 'Ch. 9.5',
        description: 'Reduce shape to 1-pixel-wide medial axis.',
        params: [
            { key: 'kernelShape', type: 'select', label: 'Shape', options: ['cross', 'rect', 'ellipse'], default: 'cross' },
        ],
    },

    // ═══════════════════════════════════════════════════════════
    // Segmentation — Ch. 10
    // ═══════════════════════════════════════════════════════════
    'segmentation/threshold': {
        title: 'Threshold',
        chapter: 'Ch. 10.3',
        description: 'Segment into foreground / background.',
        params: [
            { key: 'thresholdType', type: 'select', label: 'Method', options: ['global', 'otsu', 'adaptive_mean', 'adaptive_gaussian'], default: 'otsu' },
            { key: 'thresholdValue', type: 'slider', label: 'Threshold Value', min: 0, max: 255, step: 1, default: 128 },
            { key: 'blockSize', type: 'select', label: 'Block Size', options: [3, 5, 7, 9, 11, 15, 21, 31], default: 11 },
            { key: 'cValue', type: 'slider', label: 'C Value', min: -10, max: 30, step: 1, default: 2 },
        ],
    },
    'segmentation/edge-detect': {
        title: 'Edge Detection',
        chapter: 'Ch. 10.2',
        description: 'Detect edges using various operators.',
        params: [
            { key: 'edgeType', type: 'select', label: 'Detector', options: ['canny', 'sobel', 'prewitt', 'roberts', 'laplacian'], default: 'canny' },
            { key: 'lowThreshold', type: 'slider', label: 'Low Threshold', min: 0, max: 255, step: 1, default: 50 },
            { key: 'highThreshold', type: 'slider', label: 'High Threshold', min: 0, max: 255, step: 1, default: 150 },
            { key: 'kernelSize', type: 'select', label: 'Kernel Size', options: [3, 5, 7], default: 3 },
        ],
    },
    'segmentation/hough-lines': {
        title: 'Hough Lines',
        chapter: 'Ch. 10.2',
        description: 'Detect lines via probabilistic Hough transform.',
        params: [
            { key: 'cannyLow', type: 'slider', label: 'Canny Low', min: 0, max: 255, step: 1, default: 50 },
            { key: 'cannyHigh', type: 'slider', label: 'Canny High', min: 0, max: 255, step: 1, default: 150 },
            { key: 'threshold', type: 'slider', label: 'Vote Threshold', min: 10, max: 300, step: 5, default: 100 },
            { key: 'minLineLength', type: 'slider', label: 'Min Length', min: 5, max: 200, step: 5, default: 50, unit: 'px' },
            { key: 'maxLineGap', type: 'slider', label: 'Max Gap', min: 1, max: 50, step: 1, default: 10, unit: 'px' },
        ],
    },
    'segmentation/hough-circles': {
        title: 'Hough Circles',
        chapter: 'Ch. 10.2',
        description: 'Detect circles via Hough Circle transform.',
        params: [
            { key: 'dp', type: 'slider', label: 'dp (resolution)', min: 0.5, max: 3, step: 0.1, default: 1.2 },
            { key: 'minDist', type: 'slider', label: 'Min Distance', min: 10, max: 200, step: 5, default: 50, unit: 'px' },
            { key: 'param1', type: 'slider', label: 'Canny Threshold', min: 10, max: 300, step: 5, default: 100 },
            { key: 'param2', type: 'slider', label: 'Accumulator', min: 5, max: 100, step: 1, default: 30 },
            { key: 'minRadius', type: 'slider', label: 'Min Radius', min: 0, max: 200, step: 5, default: 10, unit: 'px' },
            { key: 'maxRadius', type: 'slider', label: 'Max Radius', min: 10, max: 500, step: 10, default: 200, unit: 'px' },
        ],
    },
    'segmentation/region-grow': {
        title: 'Region Growing',
        chapter: 'Ch. 10.4',
        description: 'Grow region from seed by intensity similarity.',
        params: [
            { key: 'seedX', type: 'number', label: 'Seed X', min: 0, max: 4096, default: 0 },
            { key: 'seedY', type: 'number', label: 'Seed Y', min: 0, max: 4096, default: 0 },
            { key: 'tolerance', type: 'slider', label: 'Tolerance', min: 1, max: 100, step: 1, default: 15 },
        ],
    },
    'segmentation/watershed': {
        title: 'Watershed',
        chapter: 'Ch. 10.5',
        description: 'Marker-based watershed segmentation.',
        params: [],
    },
    'segmentation/connected-components': {
        title: 'Connected Components',
        chapter: 'Ch. 10.4',
        description: 'Label and colorize connected regions.',
        params: [
            { key: 'connectivity', type: 'select', label: 'Connectivity', options: [4, 8], default: 8 },
        ],
    },

    // ═══════════════════════════════════════════════════════════
    // Deep Learning — ONNX
    // ═══════════════════════════════════════════════════════════
    'dl/super-resolution': {
        title: 'Super Resolution (SRCNN)',
        chapter: 'SRCNN',
        description: 'AI upscaling via SRCNN ONNX model.',
        params: [
            { key: 'scale', type: 'select', label: 'Scale Factor', options: [2, 4], default: 2 },
        ],
    },
    'dl/denoise': {
        title: 'AI Denoise (DnCNN)',
        chapter: 'DnCNN',
        description: 'Blind denoising via DnCNN ONNX model.',
        params: [],
    },

    // ═══════════════════════════════════════════════════════════
    // Remote Sensing
    // ═══════════════════════════════════════════════════════════
    'remote/ndvi': {
        title: 'NDVI',
        chapter: 'RS',
        description: 'Compute NDVI from NIR and Red bands.',
        params: [
            { key: 'nirBand', type: 'number', label: 'NIR Band Index', min: 0, max: 20, default: 0 },
            { key: 'redBand', type: 'number', label: 'Red Band Index', min: 0, max: 20, default: 1 },
        ],
    },
    'remote/band-composite': {
        title: 'Band Composite',
        chapter: 'RS',
        description: 'Create false/true color composite.',
        params: [
            { key: 'redBand', type: 'number', label: 'Red Band', min: 0, max: 20, default: 0 },
            { key: 'greenBand', type: 'number', label: 'Green Band', min: 0, max: 20, default: 1 },
            { key: 'blueBand', type: 'number', label: 'Blue Band', min: 0, max: 20, default: 2 },
        ],
    },
    'remote/band-stats': {
        title: 'Band Statistics',
        chapter: 'RS',
        description: 'Compute per-band statistics and histograms.',
        params: [],
    },
    'remote/band-viewer': {
        title: 'Band Viewer',
        chapter: 'RS',
        description: 'Display a single band from multi-band image.',
        params: [
            { key: 'bandIndex', type: 'number', label: 'Band Index', min: 0, max: 20, default: 0 },
        ],
    },
    'remote/enhance-satellite': {
        title: 'Enhance Satellite',
        chapter: 'RS',
        description: 'CLAHE optimized for satellite imagery.',
        params: [
            { key: 'clipLimit', type: 'slider', label: 'Clip Limit', min: 1, max: 10, step: 0.5, default: 3.0 },
            { key: 'tileSize', type: 'select', label: 'Tile Size', options: [8, 16, 32, 64], default: 16 },
        ],
    },
    'remote/spectral-profile': {
        title: 'Spectral Profile',
        chapter: 'RS',
        description: 'Extract spectral signature at pixel (x, y).',
        params: [
            { key: 'x', type: 'number', label: 'Pixel X', min: 0, max: 4096, default: 0 },
            { key: 'y', type: 'number', label: 'Pixel Y', min: 0, max: 4096, default: 0 },
        ],
    },
}

// ─────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────

/**
 * @param {{
 *   category: string,
 *   operation: string,
 *   onProcess: (params: object) => void,
 *   isLoading: boolean,
 *   hasImage: boolean,
 * }} props
 */
export default function OperationCard({
    category,
    operation,
    onProcess,
    isLoading,
    hasImage,
}) {
    const opKey = `${category}/${operation}`
    const config = OPERATIONS[opKey]

    // Build initial param state from defaults
    const getDefaults = () => {
        if (!config) return {}
        const defaults = {}
        config.params.forEach((p) => {
            defaults[p.key] = p.default
        })
        return defaults
    }

    const [paramValues, setParamValues] = useState(getDefaults)

    // Reset params when operation changes
    useEffect(() => {
        setParamValues(getDefaults())
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [category, operation])

    if (!config) {
        return (
            <div className="p-4 text-center text-gray-500 text-sm">
                Select an operation from the panel.
            </div>
        )
    }

    const updateParam = (key, value) => {
        setParamValues((prev) => ({ ...prev, [key]: value }))
    }

    const handleApply = () => {
        onProcess(paramValues)
    }

    return (
        <div className="flex flex-col bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
            {/* ── Header ─────────────────────────────────────── */}
            <div className="px-4 py-3 bg-gray-750 border-b border-gray-700">
                <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">{config.title}</h3>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-600/20 text-blue-400 border border-blue-500/30">
                        {config.chapter}
                    </span>
                </div>
                <p className="text-[11px] text-gray-500 mt-1 leading-snug">
                    {config.description}
                </p>
            </div>

            {/* ── Parameters ─────────────────────────────────── */}
            <div className="px-4 py-3 space-y-3 flex-1">
                {config.params.length === 0 ? (
                    <p className="text-xs text-gray-600 italic text-center py-2">
                        No parameters — click Apply to run.
                    </p>
                ) : (
                    config.params.map((p) => {
                        const value = paramValues[p.key] ?? p.default

                        // ── Slider ──
                        if (p.type === 'slider') {
                            return (
                                <ParameterSlider
                                    key={p.key}
                                    label={p.label}
                                    min={p.min}
                                    max={p.max}
                                    step={p.step}
                                    value={value}
                                    unit={p.unit}
                                    onChange={(v) => updateParam(p.key, v)}
                                />
                            )
                        }

                        // ── Select / dropdown ──
                        if (p.type === 'select') {
                            return (
                                <div key={p.key} className="flex flex-col gap-1">
                                    <label className="text-xs text-gray-400 font-medium">
                                        {p.label}
                                    </label>
                                    <select
                                        value={value}
                                        onChange={(e) => {
                                            const v = e.target.value
                                            // Parse as number if the options are numbers
                                            updateParam(
                                                p.key,
                                                typeof p.options[0] === 'number' ? Number(v) : v
                                            )
                                        }}
                                        className="w-full bg-gray-700 border border-gray-600 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
                                    >
                                        {p.options.map((opt) => (
                                            <option key={opt} value={opt}>
                                                {opt}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            )
                        }

                        // ── Number input ──
                        if (p.type === 'number') {
                            return (
                                <div key={p.key} className="flex flex-col gap-1">
                                    <label className="text-xs text-gray-400 font-medium">
                                        {p.label}
                                    </label>
                                    <input
                                        type="number"
                                        min={p.min}
                                        max={p.max}
                                        value={value}
                                        onChange={(e) => updateParam(p.key, Number(e.target.value))}
                                        className="w-full bg-gray-700 border border-gray-600 text-white text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
                                    />
                                </div>
                            )
                        }

                        return null
                    })
                )}
            </div>

            {/* ── Apply button ───────────────────────────────── */}
            <div className="px-4 py-3 border-t border-gray-700/50">
                <button
                    onClick={handleApply}
                    disabled={!hasImage || isLoading}
                    className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${!hasImage || isLoading
                            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm shadow-blue-500/20 hover:shadow-blue-500/30 active:scale-[0.98]'
                        }`}
                >
                    {isLoading ? (
                        <>
                            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                            </svg>
                            Processing…
                        </>
                    ) : (
                        <>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Apply
                        </>
                    )}
                </button>
                {!hasImage && (
                    <p className="text-[10px] text-gray-600 text-center mt-1.5">
                        Upload an image first
                    </p>
                )}
            </div>
        </div>
    )
}
