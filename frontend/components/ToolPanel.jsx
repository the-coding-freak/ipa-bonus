'use client'

import { useState } from 'react'

// ─────────────────────────────────────────────────────────────
// Tool category definitions — one entry per backend module
// ─────────────────────────────────────────────────────────────

const TOOL_CATEGORIES = [
    {
        id: 'spatial',
        label: 'Spatial Processing',
        icon: '🎨',
        chapter: 'Ch. 3',
        operations: [
            { id: 'histogram-eq', label: 'Histogram Equalization' },
            { id: 'clahe', label: 'CLAHE' },
            { id: 'contrast-stretch', label: 'Contrast Stretch' },
            { id: 'gamma', label: 'Gamma Correction' },
            { id: 'log-transform', label: 'Log Transform' },
            { id: 'filter', label: 'Spatial Filter' },
            { id: 'unsharp-mask', label: 'Unsharp Mask' },
        ],
    },
    {
        id: 'frequency',
        label: 'Frequency Domain',
        icon: '〰️',
        chapter: 'Ch. 4',
        operations: [
            { id: 'fft', label: 'FFT Spectrum' },
            { id: 'filter', label: 'Frequency Filter' },
            { id: 'inverse', label: 'Inverse FFT' },
        ],
    },
    {
        id: 'restoration',
        label: 'Restoration',
        icon: '🔧',
        chapter: 'Ch. 5',
        operations: [
            { id: 'add-noise', label: 'Add Noise' },
            { id: 'denoise-spatial', label: 'Spatial Denoise' },
            { id: 'wiener', label: 'Wiener Filter' },
            { id: 'motion-deblur', label: 'Motion Deblur' },
        ],
    },
    {
        id: 'color',
        label: 'Color Processing',
        icon: '🌈',
        chapter: 'Ch. 6',
        operations: [
            { id: 'convert', label: 'Color Convert' },
            { id: 'false-color', label: 'False Color' },
            { id: 'histogram-eq-color', label: 'Histogram Eq (Color)' },
            { id: 'color-segment', label: 'Color Segment' },
            { id: 'channel-split', label: 'Channel Split' },
        ],
    },
    {
        id: 'morphology',
        label: 'Morphology',
        icon: '🔷',
        chapter: 'Ch. 9',
        operations: [
            { id: 'erode', label: 'Erosion' },
            { id: 'dilate', label: 'Dilation' },
            { id: 'open', label: 'Opening' },
            { id: 'close', label: 'Closing' },
            { id: 'gradient', label: 'Gradient' },
            { id: 'tophat', label: 'Top-Hat' },
            { id: 'blackhat', label: 'Black-Hat' },
            { id: 'skeleton', label: 'Skeleton' },
        ],
    },
    {
        id: 'segmentation',
        label: 'Segmentation',
        icon: '✂️',
        chapter: 'Ch. 10',
        operations: [
            { id: 'threshold', label: 'Threshold' },
            { id: 'edge-detect', label: 'Edge Detection' },
            { id: 'hough-lines', label: 'Hough Lines' },
            { id: 'hough-circles', label: 'Hough Circles' },
            { id: 'region-grow', label: 'Region Growing' },
            { id: 'watershed', label: 'Watershed' },
            { id: 'connected-components', label: 'Connected Comps' },
        ],
    },
    {
        id: 'dl',
        label: 'Deep Learning',
        icon: '🤖',
        chapter: 'ONNX',
        operations: [
            { id: 'super-resolution', label: 'Super Resolution' },
            { id: 'denoise', label: 'AI Denoise' },
        ],
    },
    {
        id: 'remote',
        label: 'Remote Sensing',
        icon: '🛰️',
        chapter: 'RS',
        operations: [
            { id: 'ndvi', label: 'NDVI' },
            { id: 'band-composite', label: 'Band Composite' },
            { id: 'band-stats', label: 'Band Statistics' },
            { id: 'band-viewer', label: 'Band Viewer' },
            { id: 'enhance-satellite', label: 'Enhance Satellite' },
            { id: 'spectral-profile', label: 'Spectral Profile' },
        ],
    },
]

// ─────────────────────────────────────────────────────────────
// Chevron icon (rotates when open)
// ─────────────────────────────────────────────────────────────

function ChevronIcon({ open }) {
    return (
        <svg
            className={`w-3.5 h-3.5 text-gray-500 transition-transform duration-200 ${open ? 'rotate-90' : ''
                }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
        >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
    )
}

// ─────────────────────────────────────────────────────────────
// ToolPanel
// ─────────────────────────────────────────────────────────────

/**
 * @param {{
 *   onOperationSelect: (category: string, operation: string) => void,
 *   activeCategory: string | null,
 *   activeOperation: string | null,
 * }} props
 */
export default function ToolPanel({
    onOperationSelect,
    activeCategory = null,
    activeOperation = null,
}) {
    const [openSections, setOpenSections] = useState([])

    const toggleSection = (id) => {
        setOpenSections((prev) =>
            prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
        )
    }

    return (
        <nav className="flex flex-col h-full">
            {/* Header */}
            <div className="px-4 pt-4 pb-3">
                <h2 className="text-[11px] font-semibold text-gray-500 uppercase tracking-widest">
                    Operations
                </h2>
            </div>

            {/* Scrollable list */}
            <div className="flex-1 overflow-y-auto px-2 pb-4 space-y-0.5">
                {TOOL_CATEGORIES.map((cat) => {
                    const isOpen = openSections.includes(cat.id)
                    const isCatActive = activeCategory === cat.id

                    return (
                        <div key={cat.id}>
                            {/* ── Category header ── */}
                            <button
                                onClick={() => toggleSection(cat.id)}
                                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors
                  ${isCatActive
                                        ? 'bg-blue-600/15 text-blue-400'
                                        : 'text-gray-300 hover:bg-gray-700/60 hover:text-white'
                                    }
                `}
                            >
                                <span className="text-base leading-none">{cat.icon}</span>
                                <span className="flex-1 text-left font-medium truncate">
                                    {cat.label}
                                </span>
                                <span className="text-[9px] text-gray-600 font-mono mr-1">
                                    {cat.chapter}
                                </span>
                                <ChevronIcon open={isOpen} />
                            </button>

                            {/* ── Operations list (animated) ── */}
                            <div
                                className={`overflow-hidden transition-all duration-200 ease-in-out ${isOpen ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'
                                    }`}
                            >
                                <div className="ml-4 pl-3 border-l border-gray-700/60 py-1 space-y-0.5">
                                    {cat.operations.map((op) => {
                                        const isActive =
                                            activeCategory === cat.id && activeOperation === op.id

                                        return (
                                            <button
                                                key={op.id}
                                                onClick={() => onOperationSelect(cat.id, op.id)}
                                                className={`w-full text-left px-3 py-1.5 rounded-md text-[13px] transition-colors ${isActive
                                                        ? 'bg-blue-600 text-white font-medium shadow-sm shadow-blue-500/20'
                                                        : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                                                    }`}
                                            >
                                                {op.label}
                                            </button>
                                        )
                                    })}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* Footer — operation count */}
            <div className="px-4 py-2 border-t border-gray-700/50 text-[10px] text-gray-600 font-mono">
                {TOOL_CATEGORIES.reduce((sum, c) => sum + c.operations.length, 0)} operations · 8 modules
            </div>
        </nav>
    )
}
