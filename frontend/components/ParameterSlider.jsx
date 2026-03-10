'use client'

/**
 * ParameterSlider — Reusable slider with label + live value display.
 *
 * @param {{
 *   label: string,
 *   min: number,
 *   max: number,
 *   step: number,
 *   value: number,
 *   onChange: (value: number) => void,
 *   unit?: string,
 * }} props
 */
export default function ParameterSlider({
    label,
    min,
    max,
    step,
    value,
    onChange,
    unit = '',
}) {
    return (
        <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
                <label className="text-xs text-gray-400 font-medium">{label}</label>
                <span className="text-xs font-mono text-blue-400 bg-gray-700/60 px-1.5 py-0.5 rounded">
                    {typeof value === 'number' && !Number.isInteger(step)
                        ? value.toFixed(2)
                        : value}
                    {unit && <span className="text-gray-500 ml-0.5">{unit}</span>}
                </span>
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:h-3.5
          [&::-webkit-slider-thumb]:bg-blue-500
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:hover:bg-blue-400
          [&::-webkit-slider-thumb]:transition-colors
          [&::-moz-range-thumb]:w-3.5
          [&::-moz-range-thumb]:h-3.5
          [&::-moz-range-thumb]:bg-blue-500
          [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:border-0
          [&::-moz-range-thumb]:cursor-pointer"
            />
            <div className="flex justify-between text-[9px] text-gray-600 font-mono">
                <span>{min}</span>
                <span>{max}</span>
            </div>
        </div>
    )
}
