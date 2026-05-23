interface SliderProps {
  label?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  formatValue?: (v: number) => string;
  hint?: string;
  accent?: string;
}

export default function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  formatValue,
  hint,
  accent,
}: SliderProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {(label || formatValue) && (
        <div className="flex items-baseline justify-between">
          {label && (
            <div className="text-sm text-ink-200 flex items-center gap-2">
              {accent && (
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ background: accent }}
                />
              )}
              <span>{label}</span>
            </div>
          )}
          {formatValue && (
            <div className="text-sm font-mono text-ink-100">
              {formatValue(value)}
            </div>
          )}
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <div className="text-[11px] text-ink-400">{hint}</div>}
    </div>
  );
}
