'use client'

import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'

/**
 * ImageCanvas — Two-panel side-by-side image viewer.
 *
 * @param {{
 *   originalFile: File | null,
 *   originalImage: string | null,
 *   processedImage: string | null,
 *   metadata: object | null,
 *   isLoading: boolean,
 *   onFileSelect: (file: File) => void,
 *   operationName: string | null,
 *   error: string | null,
 *   onDismissError: () => void,
 * }} props
 */
export default function ImageCanvas({
    originalFile,
    originalImage,
    processedImage,
    metadata,
    isLoading,
    onFileSelect,
    onClearImage = () => {},
    operationName = null,
    error = null,
    onDismissError = () => { },
}) {
    const [zoom, setZoom] = useState(1)
    const [origDimensions, setOrigDimensions] = useState(null)
    const [procDimensions, setProcDimensions] = useState(null)

    // ── Hidden file input for "Change Image" button ────────────
    const changeInputRef = useRef(null)

    const handleChangeImage = () => {
        if (changeInputRef.current) changeInputRef.current.click()
    }

    const handleChangeInputChange = (e) => {
        const file = e.target.files?.[0]
        if (file) {
            onFileSelect(file)
            setZoom(1)
            setOrigDimensions(null)
            setProcDimensions(null)
        }
        // Reset so the same file can be re-selected
        e.target.value = ''
    }

    const handleRemoveImage = () => {
        setZoom(1)
        setOrigDimensions(null)
        setProcDimensions(null)
        onClearImage()
    }

    // ── Dropzone ──────────────────────────────────────────────
    const onDrop = useCallback(
        (acceptedFiles) => {
            if (acceptedFiles.length > 0) {
                onFileSelect(acceptedFiles[0])
                setZoom(1)
            }
        },
        [onFileSelect]
    )

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff'],
        },
        maxFiles: 1,
        maxSize: 20 * 1024 * 1024, // 20 MB
    })

    // ── Zoom ──────────────────────────────────────────────────
    const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 5))
    const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.25))
    const handleZoomReset = () => setZoom(1)

    // ── Download processed image ──────────────────────────────
    const handleDownload = () => {
        if (!processedImage) return
        const link = document.createElement('a')
        link.href = processedImage
        link.download = `processed_${Date.now()}.png`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    // ── Format file size ──────────────────────────────────────
    const formatSize = (bytes) => {
        if (!bytes) return '—'
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
    }

    return (
        <div className="flex flex-col h-full relative">
            {/* ── Toolbar ──────────────────────────────────────── */}
            <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700 rounded-t-lg">
                <div className="flex items-center gap-2">
                    {/* Zoom controls */}
                    <button
                        onClick={handleZoomOut}
                        className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
                        title="Zoom Out"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
                        </svg>
                    </button>

                    <button
                        onClick={handleZoomReset}
                        className="px-2 py-0.5 text-xs font-mono text-gray-300 bg-gray-700 rounded hover:bg-gray-600 transition-colors min-w-[48px]"
                        title="Reset Zoom"
                    >
                        {Math.round(zoom * 100)}%
                    </button>

                    <button
                        onClick={handleZoomIn}
                        className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors"
                        title="Zoom In"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                        </svg>
                    </button>
                </div>

                {/* Right-side actions */}
                <div className="flex items-center gap-2">

                    {/* ── Change / Remove — only when image is loaded ── */}
                    {originalImage && (
                        <>
                            {/* Change Image */}
                            <button
                                onClick={handleChangeImage}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors
                                           bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white"
                                title="Load a different image"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 12V4m0 0l-3 3m3-3l3 3" />
                                </svg>
                                Change
                            </button>

                            {/* Remove Image */}
                            <button
                                onClick={handleRemoveImage}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors
                                           bg-red-900/60 hover:bg-red-800/80 text-red-300 hover:text-red-100"
                                title="Remove image and clear canvas"
                            >
                                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                                Remove
                            </button>
                        </>
                    )}

                    {/* Download processed image */}
                    <button
                        onClick={handleDownload}
                        disabled={!processedImage}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                            processedImage
                                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                        title="Download Processed Image"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download
                    </button>
                </div>

                {/* Hidden file input for Change Image */}
                <input
                    ref={changeInputRef}
                    type="file"
                    accept=".jpeg,.jpg,.png,.bmp,.webp,.tif,.tiff"
                    className="hidden"
                    onChange={handleChangeInputChange}
                />
            </div>

            {/* ── Error toast ─────────────────────────────────── */}
            {error && (
                <div className="absolute top-14 right-6 z-20 max-w-sm animate-slide-in">
                    <div className="flex items-start gap-2 px-4 py-3 rounded-lg bg-red-900/90 border border-red-700/60 shadow-lg shadow-red-900/30 backdrop-blur-sm">
                        <svg className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div className="flex-1 min-w-0">
                            <p className="text-xs font-medium text-red-300">Processing Error</p>
                            <p className="text-[11px] text-red-400/80 mt-0.5 break-words">{error}</p>
                        </div>
                        <button
                            onClick={onDismissError}
                            className="p-0.5 text-red-500 hover:text-red-300 transition-colors flex-shrink-0"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                </div>
            )}

            {/* ── Dual Panel ───────────────────────────────────── */}
            <div className="flex flex-1 min-h-0 bg-gray-900 rounded-b-lg">
                {/* ── Left: Original ──────────────────────────────── */}
                <div className="flex-1 min-w-0 flex flex-col border-r border-gray-700">
                    <div className="px-3 py-1.5 bg-gray-800/50 border-b border-gray-700/50">
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Original</span>
                    </div>

                    <div className="flex-1 min-h-0 overflow-auto relative">
                        {originalImage ? (
                            <div className="min-w-full min-h-full p-4 grid place-items-center">
                                {/* TIFF files can't be rendered natively in browsers */}
                                {originalFile && /\.tiff?$/i.test(originalFile.name) ? (
                                    <div className="flex flex-col items-center gap-3 text-center">
                                        <div className="w-16 h-16 rounded-xl bg-indigo-900/40 border border-indigo-500/30 flex items-center justify-center">
                                            <svg className="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                                                    d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                            </svg>
                                        </div>
                                        <div>
                                            <p className="text-sm font-semibold text-indigo-300">{originalFile.name}</p>
                                            <p className="text-xs text-gray-500 mt-1">Multi-band TIFF · {formatSize(originalFile.size)}</p>
                                            <p className="text-[11px] text-gray-600 mt-2 max-w-[200px] leading-snug">
                                                TIFF format cannot be previewed in the browser.<br />
                                                Apply an operation to see the processed output.
                                            </p>
                                        </div>
                                    </div>
                                ) : (
                                    <img
                                        src={originalImage}
                                        alt="Original"
                                        className="max-w-none"
                                        style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
                                        onLoad={(e) =>
                                            setOrigDimensions({
                                                width: e.target.naturalWidth,
                                                height: e.target.naturalHeight,
                                            })
                                        }
                                    />
                                )}
                            </div>
                        ) : (
                            /* ── Empty state / dropzone ── */
                            <div
                                {...getRootProps()}
                                className={`flex flex-col items-center justify-center h-full cursor-pointer transition-colors ${isDragActive
                                    ? 'bg-blue-900/20 border-2 border-dashed border-blue-500'
                                    : 'hover:bg-gray-800/30'
                                    }`}
                            >
                                <input {...getInputProps()} />
                                <div className="flex flex-col items-center gap-3 text-gray-500">
                                    {/* Upload icon */}
                                    <svg className="w-16 h-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={1.5}
                                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                        />
                                    </svg>
                                    <div className="text-center">
                                        <p className="text-sm font-medium text-gray-400">
                                            {isDragActive ? 'Drop image here…' : 'Drag & drop an image'}
                                        </p>
                                        <p className="text-xs text-gray-600 mt-1">
                                            or click to browse · JPG, PNG, BMP, WebP, TIFF · max 20 MB
                                        </p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Info bar */}
                    <div className="px-3 py-1 bg-gray-800/50 border-t border-gray-700/50 flex items-center gap-3 text-[10px] text-gray-500 font-mono">
                        {origDimensions && (
                            <>
                                <span>{origDimensions.width} × {origDimensions.height}</span>
                                <span className="text-gray-700">|</span>
                            </>
                        )}
                        {originalFile && <span>{formatSize(originalFile.size)}</span>}
                    </div>
                </div>

                {/* ── Right: Processed ────────────────────────────── */}
                <div className="flex-1 min-w-0 flex flex-col">
                    <div className="px-3 py-1.5 bg-gray-800/50 border-b border-gray-700/50">
                        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Processed</span>
                    </div>

                    <div className="flex-1 min-h-0 overflow-auto relative">
                        {/* Loading overlay */}
                        {isLoading && (
                            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-gray-900/80 backdrop-blur-sm">
                                <div className="w-10 h-10 border-3 border-blue-500 border-t-transparent rounded-full animate-spin" />
                                <p className="mt-3 text-sm text-gray-400">Processing…</p>
                            </div>
                        )}

                        {processedImage ? (
                            <div className="min-w-full min-h-full p-4 grid place-items-center">
                                <img
                                    src={processedImage}
                                    alt="Processed"
                                    className="max-w-none"
                                    style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
                                    onLoad={(e) =>
                                        setProcDimensions({
                                            width: e.target.naturalWidth,
                                            height: e.target.naturalHeight,
                                        })
                                    }
                                />
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full text-gray-600">
                                <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={1.5}
                                        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                                    />
                                </svg>
                                <p className="mt-2 text-xs">Result will appear here</p>
                            </div>
                        )}
                    </div>

                    {/* ── Rich metadata bar ──────────────────────── */}
                    <div className="px-3 py-1.5 bg-gray-800/70 border-t border-gray-700/50 flex items-center gap-2 text-[10px] text-gray-500 font-mono flex-wrap">
                        {/* Operation name */}
                        {operationName && metadata && (
                            <span className="text-gray-300 font-semibold font-sans text-[11px]">
                                {operationName}
                            </span>
                        )}

                        {/* Processing time */}
                        {metadata?.time_ms != null && (
                            <>
                                <span className="text-gray-700">·</span>
                                <span className="text-green-500">
                                    <svg className="w-2.5 h-2.5 inline mr-0.5 -mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    {metadata.time_ms} ms
                                </span>
                            </>
                        )}

                        {/* Chapter reference */}
                        {metadata?.chapter && (
                            <>
                                <span className="text-gray-700">·</span>
                                <span className="text-blue-400">{metadata.chapter}</span>
                            </>
                        )}

                        {/* Model name (for DL operations) */}
                        {metadata?.model && (
                            <>
                                <span className="text-gray-700">·</span>
                                <span className="text-purple-400">
                                    🧠 {metadata.model}
                                </span>
                            </>
                        )}

                        {/* Spacer */}
                        <span className="flex-1" />

                        {/* Output dimensions */}
                        {procDimensions && (
                            <span className="text-gray-400">
                                {procDimensions.width} × {procDimensions.height}
                            </span>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
