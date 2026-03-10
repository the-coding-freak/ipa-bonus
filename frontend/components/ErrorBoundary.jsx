'use client'

import React from 'react'

/**
 * React class-based Error Boundary.
 * Catches render errors and shows a fallback UI instead of a blank screen.
 */
export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props)
        this.state = { hasError: false, error: null }
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error }
    }

    componentDidCatch(error, errorInfo) {
        console.error('[ErrorBoundary] Caught error:', error, errorInfo)
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null })
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex items-center justify-center min-h-screen bg-gray-900 p-8">
                    <div className="max-w-md w-full bg-gray-800 border border-gray-700 rounded-xl p-8 text-center shadow-xl">
                        {/* Icon */}
                        <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-red-900/40 border border-red-500/30 flex items-center justify-center">
                            <svg className="w-7 h-7 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"
                                />
                            </svg>
                        </div>

                        <h2 className="text-lg font-semibold text-white mb-2">
                            Something went wrong
                        </h2>
                        <p className="text-sm text-gray-400 mb-6 leading-relaxed">
                            An unexpected error occurred in the application.
                            Click below to try again.
                        </p>

                        {/* Error detail (collapsed) */}
                        {this.state.error && (
                            <p className="text-xs text-red-400/70 bg-red-900/20 border border-red-800/30 rounded-lg px-3 py-2 mb-5 font-mono break-all">
                                {this.state.error.toString()}
                            </p>
                        )}

                        <button
                            onClick={this.handleRetry}
                            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg
                                       transition-all shadow-sm shadow-blue-500/20 hover:shadow-blue-500/30 active:scale-[0.98]"
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}
