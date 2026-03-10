'use client'

import { useState, useEffect, useCallback, useRef } from 'react'

// ─────────────────────────────────────────────────────────────
// Toast Hook — manages a queue of notifications
// ─────────────────────────────────────────────────────────────

let _toastCounter = 0

/**
 * Custom hook providing toast management.
 * @returns {{ toasts, addToast, removeToast }}
 */
export function useToast() {
    const [toasts, setToasts] = useState([])

    const addToast = useCallback((message, type = 'info', duration = 4000) => {
        const id = ++_toastCounter
        setToasts((prev) => [...prev, { id, message, type, duration }])
        return id
    }, [])

    const removeToast = useCallback((id) => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
    }, [])

    return { toasts, addToast, removeToast }
}


// ─────────────────────────────────────────────────────────────
// Single Toast Item
// ─────────────────────────────────────────────────────────────

function ToastItem({ toast, onRemove }) {
    const [exiting, setExiting] = useState(false)
    const timerRef = useRef(null)

    useEffect(() => {
        timerRef.current = setTimeout(() => {
            setExiting(true)
            setTimeout(() => onRemove(toast.id), 300)
        }, toast.duration)

        return () => clearTimeout(timerRef.current)
    }, [toast.id, toast.duration, onRemove])

    const handleClose = () => {
        clearTimeout(timerRef.current)
        setExiting(true)
        setTimeout(() => onRemove(toast.id), 300)
    }

    const styles = {
        success: {
            bg: 'bg-emerald-900/90 border-emerald-500/40',
            icon: (
                <svg className="w-4 h-4 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
            ),
        },
        error: {
            bg: 'bg-red-900/90 border-red-500/40',
            icon: (
                <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            ),
        },
        info: {
            bg: 'bg-blue-900/90 border-blue-500/40',
            icon: (
                <svg className="w-4 h-4 text-blue-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            ),
        },
    }

    const s = styles[toast.type] || styles.info

    return (
        <div
            className={`flex items-start gap-2.5 px-3.5 py-2.5 rounded-lg border backdrop-blur-sm shadow-lg
                ${s.bg}
                ${exiting ? 'animate-fade-out' : 'animate-slide-in'}
                max-w-xs text-sm text-white`}
        >
            {s.icon}
            <p className="flex-1 leading-snug text-[13px]">{toast.message}</p>
            <button
                onClick={handleClose}
                className="text-gray-400 hover:text-white transition-colors flex-shrink-0 mt-0.5"
            >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
    )
}


// ─────────────────────────────────────────────────────────────
// Toast Container — renders at top-right
// ─────────────────────────────────────────────────────────────

export default function ToastContainer({ toasts, onRemove }) {
    if (!toasts.length) return null

    return (
        <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-auto">
            {toasts.map((t) => (
                <ToastItem key={t.id} toast={t} onRemove={onRemove} />
            ))}
        </div>
    )
}
