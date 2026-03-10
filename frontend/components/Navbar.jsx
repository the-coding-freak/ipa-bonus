export default function Navbar() {
    return (
        <header className="flex items-center justify-between px-6 py-3 bg-gray-800 border-b border-gray-700 select-none">
            {/* ── Brand ────────────────────────────────────────── */}
            <div className="flex items-center gap-3">
                {/* Microscope / eye icon */}
                <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/20">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                        />
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                        />
                    </svg>
                </div>

                <div>
                    <h1 className="text-lg font-bold text-white leading-tight tracking-tight">
                        ResearchVision
                        <span className="text-blue-400"> Pro</span>
                    </h1>
                    <p className="text-[10px] text-gray-500 tracking-wide uppercase">
                        Digital Image Processing Research Tool
                    </p>
                </div>
            </div>

            {/* ── Right side — placeholder for future controls ── */}
            <div className="flex items-center gap-3 text-xs text-gray-500">
                <span className="hidden sm:inline">Gonzalez &amp; Woods, 4th Ed.</span>
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" title="Backend Online" />
            </div>
        </header>
    )
}
