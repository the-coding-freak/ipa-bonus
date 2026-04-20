'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import Navbar from '../components/Navbar'
import ImageCanvas from '../components/ImageCanvas'
import ToolPanel from '../components/ToolPanel'
import OperationCard, { OPERATIONS } from '../components/OperationCard'
import ToastContainer, { useToast } from '../components/Toast'
import {
    spatialAPI,
    frequencyAPI,
    restorationAPI,
    colorAPI,
    morphologyAPI,
    segmentationAPI,
    deepLearningAPI,
    remoteAPI,
} from '../lib/api'

// ─────────────────────────────────────────────────────────────
// Maps category/operation → api.js function call
// Each entry receives (file, params) and calls the right API fn
// ─────────────────────────────────────────────────────────────

const API_MAP = {
    // Spatial
    'spatial/histogram-eq': (f) => spatialAPI.histogramEq(f),
    'spatial/clahe': (f, p) => spatialAPI.clahe(f, p.clipLimit, p.tileSize),
    'spatial/contrast-stretch': (f, p) => spatialAPI.contrastStretch(f, p.lowPercentile, p.highPercentile),
    'spatial/gamma': (f, p) => spatialAPI.gamma(f, p.gamma),
    'spatial/log-transform': (f) => spatialAPI.logTransform(f),
    'spatial/filter': (f, p) => spatialAPI.filter(f, p.filterType, p.kernelSize),
    'spatial/unsharp-mask': (f, p) => spatialAPI.unsharpMask(f, p.radius, p.amount),

    // Frequency
    'frequency/fft': (f) => frequencyAPI.fft(f),
    'frequency/filter': (f, p) => frequencyAPI.filter(f, p.filterType, p.cutoff, p.order, p.notchCenterU, p.notchCenterV),
    'frequency/inverse': (_f, p) => frequencyAPI.inverse(p.fftData),

    // Restoration
    'restoration/add-noise': (f, p) => restorationAPI.addNoise(f, p.noiseType, p.intensity),
    'restoration/denoise-spatial': (f, p) => restorationAPI.denoiseSpatial(f, p.denoiseType, p.kernelSize, p.q),
    'restoration/wiener': (f, p) => restorationAPI.wiener(f, p.noiseVariance),
    'restoration/motion-deblur': (f, p) => restorationAPI.motionDeblur(f, p.angle, p.length, p.noiseVariance),

    // Color
    'color/convert': (f, p) => colorAPI.convert(f, p.target),
    'color/false-color': (f, p) => colorAPI.falseColor(f, p.colormap),
    'color/histogram-eq-color': (f) => colorAPI.histogramEqColor(f),
    'color/color-segment': (f, p) => colorAPI.colorSegment(f, p.hueMin, p.hueMax, p.satMin, p.satMax),
    'color/channel-split': (f) => colorAPI.channelSplit(f),

    // Morphology
    'morphology/erode': (f, p) => morphologyAPI.erode(f, p.kernelSize, p.kernelShape, p.iterations),
    'morphology/dilate': (f, p) => morphologyAPI.dilate(f, p.kernelSize, p.kernelShape, p.iterations),
    'morphology/open': (f, p) => morphologyAPI.open(f, p.kernelSize, p.kernelShape, p.iterations),
    'morphology/close': (f, p) => morphologyAPI.close(f, p.kernelSize, p.kernelShape, p.iterations),
    'morphology/gradient': (f, p) => morphologyAPI.gradient(f, p.kernelSize, p.kernelShape),
    'morphology/tophat': (f, p) => morphologyAPI.topHat(f, p.kernelSize, p.kernelShape),
    'morphology/blackhat': (f, p) => morphologyAPI.blackHat(f, p.kernelSize, p.kernelShape),
    'morphology/skeleton': (f, p) => morphologyAPI.skeleton(f, p.kernelShape),

    // Segmentation
    'segmentation/threshold': (f, p) => segmentationAPI.threshold(f, p.thresholdType, p.thresholdValue, p.blockSize, p.cValue),
    'segmentation/edge-detect': (f, p) => segmentationAPI.edgeDetect(f, p.edgeType, p.lowThreshold, p.highThreshold, p.kernelSize),
    'segmentation/hough-lines': (f, p) => segmentationAPI.houghLines(f, p.cannyLow, p.cannyHigh, p.threshold, p.minLineLength, p.maxLineGap),
    'segmentation/hough-circles': (f, p) => segmentationAPI.houghCircles(f, p.dp, p.minDist, p.param1, p.param2, p.minRadius, p.maxRadius),
    'segmentation/region-grow': (f, p) => segmentationAPI.regionGrow(f, p.seedX, p.seedY, p.tolerance),
    'segmentation/watershed': (f) => segmentationAPI.watershed(f),
    'segmentation/connected-components': (f, p) => segmentationAPI.connectedComponents(f, p.connectivity),

    // Deep Learning
    'dl/super-resolution': (f, p) => deepLearningAPI.superResolution(f, p.scale),
    'dl/denoise': (f) => deepLearningAPI.denoise(f),

    // Remote Sensing
    'remote/ndvi': (f, p) => remoteAPI.ndvi(f, p.nirBand, p.redBand),
    'remote/band-composite': (f, p) => remoteAPI.bandComposite(f, p.redBand, p.greenBand, p.blueBand),
    'remote/band-stats': (f) => remoteAPI.bandStats(f),
    'remote/band-viewer': (f, p) => remoteAPI.bandViewer(f, p.bandIndex),
    'remote/enhance-satellite': (f, p) => remoteAPI.enhanceSatellite(f, p.clipLimit, p.tileSize),
    'remote/spectral-profile': (f, p) => remoteAPI.spectralProfile(f, p.x, p.y),
}

// ─────────────────────────────────────────────────────────────
// Backend health check interval (ms)
// ─────────────────────────────────────────────────────────────
const HEALTH_CHECK_INTERVAL = 15000

export default function Home() {
    // ── Core state (per CONVENTIONS.md) ───────────────────────
    const [originalFile, setOriginalFile] = useState(null)
    const [originalImage, setOriginalImage] = useState(null)
    const [processedImage, setProcessedImage] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [metadata, setMetadata] = useState(null)
    const [error, setError] = useState(null)

    // ── Tool selection state ─────────────────────────────────
    const [activeCategory, setActiveCategory] = useState(null)
    const [activeOperation, setActiveOperation] = useState(null)

    // ── Backend health state ─────────────────────────────────
    const [backendOnline, setBackendOnline] = useState(true)

    // ── Toast system ─────────────────────────────────────────
    const { toasts, addToast, removeToast } = useToast()

    // ── Backend health check (on mount + periodic poll) ──────
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch('http://localhost:8000/health', {
                    signal: AbortSignal.timeout(3000),
                })
                if (res.ok) {
                    setBackendOnline((prev) => {
                        if (!prev) addToast('Backend reconnected', 'success')
                        return true
                    })
                } else {
                    setBackendOnline(false)
                }
            } catch {
                setBackendOnline(false)
            }
        }

        checkHealth()
        const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL)
        return () => clearInterval(interval)
    }, [addToast])

    // ── Ctrl+Z undo — restore original image ─────────────────
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z' && processedImage) {
                e.preventDefault()
                setProcessedImage(null)
                setMetadata(null)
                addToast('Restored original image (Ctrl+Z)', 'info')
            }
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [processedImage, addToast])

    // ── File selection handler ────────────────────────────────
    const handleFileSelect = useCallback((file) => {
        setOriginalFile(file)
        setOriginalImage(URL.createObjectURL(file))
        setProcessedImage(null)
        setMetadata(null)
        setError(null)
    }, [])

    // ── Clear / remove image ───────────────────────────────
    const handleClearImage = useCallback(() => {
        setOriginalFile(null)
        setOriginalImage(null)
        setProcessedImage(null)
        setMetadata(null)
        setError(null)
        addToast('Image removed', 'info')
    }, [addToast])

    // ── Operation selection handler ──────────────────────────
    const handleOperationSelect = useCallback((category, operation) => {
        setActiveCategory(category)
        setActiveOperation(operation)
        setError(null)
    }, [])

    // ── Standard process function (CONVENTIONS.md pattern) ────
    const handleProcess = useCallback(
        async (apiFunction, ...params) => {
            if (!originalFile) return
            setIsLoading(true)
            setError(null)
            try {
                const result = await apiFunction(originalFile, ...params)
                setProcessedImage(`data:image/png;base64,${result.image}`)
                setMetadata(result.metadata)
                // Show success toast with processing time
                const timeMs = result.metadata?.time_ms
                const opName = result.metadata?.operation || 'Operation'
                addToast(
                    timeMs
                        ? `${opName} completed in ${timeMs}ms`
                        : `${opName} completed`,
                    'success'
                )
            } catch (err) {
                setError(err.message)
                addToast(err.message || 'Processing failed', 'error')
            } finally {
                setIsLoading(false)
            }
        },
        [originalFile, addToast]
    )

    // ── Dispatch from OperationCard → correct API function ───
    const handleOperationProcess = useCallback(
        (params) => {
            if (!activeCategory || !activeOperation) return
            const key = `${activeCategory}/${activeOperation}`
            const apiFn = API_MAP[key]
            if (!apiFn) {
                setError(`No API mapping for ${key}`)
                addToast(`No API mapping for ${key}`, 'error')
                return
            }
            // Wrap in handleProcess pattern
            const wrappedFn = (file) => apiFn(file, params)
            handleProcess(wrappedFn)
        },
        [activeCategory, activeOperation, handleProcess, addToast]
    )

    // ── Get current operation display name ───────────────────
    const currentOperationName = activeCategory && activeOperation
        ? OPERATIONS[`${activeCategory}/${activeOperation}`]?.title ?? null
        : null

    // ── Dismiss error ────────────────────────────────────────
    const handleDismissError = useCallback(() => setError(null), [])

    return (
        <div className="flex flex-col h-screen overflow-hidden bg-gray-900">
            {/* ── Backend offline banner ──────────────────────── */}
            {!backendOnline && (
                <div className="flex items-center justify-center gap-2 px-4 py-2 bg-red-900/80 border-b border-red-700/50 text-red-200 text-sm font-medium">
                    <svg className="w-4 h-4 text-red-400 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M18.364 5.636a9 9 0 010 12.728M5.636 18.364a9 9 0 010-12.728M15.536 8.464a5 5 0 010 7.072M8.464 15.536a5 5 0 010-7.072"
                        />
                    </svg>
                    Backend offline — make sure the FastAPI server is running on port 8000
                </div>
            )}

            {/* ── Header ─────────────────────────────────────── */}
            <Navbar />

            {/* ── Toast notifications ─────────────────────────── */}
            <ToastContainer toasts={toasts} onRemove={removeToast} />

            {/* ── Main content ───────────────────────────────── */}
            <div className="flex flex-1 min-h-0">
                {/* ── Left sidebar ────────────────────────────── */}
                <aside className="w-[280px] flex-shrink-0 bg-gray-800 border-r border-gray-700 flex flex-col min-h-0">
                    {/* Tool list (scrollable) */}
                    <div className="flex-1 overflow-y-auto min-h-0">
                        <ToolPanel
                            onOperationSelect={handleOperationSelect}
                            activeCategory={activeCategory}
                            activeOperation={activeOperation}
                        />
                    </div>

                    {/* Operation parameter card (scrollable) */}
                    {activeCategory && activeOperation && (
                        <div className="flex-shrink-0 border-t border-gray-700 overflow-y-auto max-h-[45%]">
                            <OperationCard
                                category={activeCategory}
                                operation={activeOperation}
                                onProcess={handleOperationProcess}
                                isLoading={isLoading}
                                hasImage={!!originalFile}
                            />
                        </div>
                    )}

                </aside>

                {/* ── Right: Image canvas ─────────────────────── */}
                <main className="flex-1 min-w-0 p-3">
                    <ImageCanvas
                        originalFile={originalFile}
                        originalImage={originalImage}
                        processedImage={processedImage}
                        metadata={metadata}
                        isLoading={isLoading}
                        onFileSelect={handleFileSelect}
                        onClearImage={handleClearImage}
                        operationName={currentOperationName}
                        error={error}
                        onDismissError={handleDismissError}
                    />
                </main>
            </div>
        </div>
    )
}
